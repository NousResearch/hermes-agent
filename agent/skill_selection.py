"""Per-turn skill shortlist: up to three skills that fit THIS request, named on the current user message.

The ``<available_skills>`` index lives in the cached system prompt with descriptions clipped to 60
characters, and picking from it is the model's judgment alone; on a large library the model follows
the task and skips the scan (#113469). With ``skills.selection: shortlist`` every turn ranks the
request against the skills that index already offers — their full description plus optional
``metadata.hermes.situations`` phrases, which only this module reads and the index never renders —
and names at most three on the per-turn note channel (the ``api_content`` sidecar), behind the cached
prefix. It suggests; it never loads a skill.

Precision beats recall here: a wrong line costs every turn's attention, a missing one costs nothing
the catalog did not already cost. So small talk, slash/injected messages and pasted blobs get no
line, a candidate must share two content words with the request (or one strong one), and a
suggestion set is shown once per session. Measured by ``evals/skill_selection``.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from agent.message_content import flatten_message_text

logger = logging.getLogger(__name__)

SELECTION_CATALOG = "catalog"
SELECTION_SHORTLIST = "shortlist"
MAX_CANDIDATES = 3
# A runner-up below this share of the leader's score is noise riding on one shared word.
RUNNER_UP_SCORE_RATIO = 0.5
_MIN_CONTENT_TERMS = 2
_PASTE_HEAD_TAIL_CHARS = 300
_PASTE_THRESHOLD_CHARS = 1500

NOTE_PREFIX = ("[System: Skills that may fit this request — suggestions only. Load one with "
               "skill_view(name) if it applies; otherwise ignore this note.")
_ROW = "    - "

_ASCII_RE = re.compile(r"[A-Za-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+")
_FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_SKIPPED_LINE_RE = re.compile(r"^(?:\s*>|    |\t)")  # quoted reply, indented code/log
# Function words plus request filler every skill description shares ("use when", "help me").
_STOPWORDS = frozenset("""
a about after again all also am an and any are as at be been before being both but by can could
did do does doing done for from get got had has have having he her here hers him his how i if in
into is it its just let lets like me more most my no not now of off on once only or other our out
over please same she should so some such than that the their them then there these they this those
through to too under until up us very was we were what when where which while who whom why will
with would you your yours yourself hi hello hey thanks thank ok okay yes yeah sure cool great nice
lol good morning evening night today tomorrow tonight weekend day week time really think know want
need help use using used make made new one two way thing things something anything stuff lot bit
skill skills user users task tasks work working tell said say ask asks question answer go going
""".split())


def selection_mode(skills_cfg: Optional[dict[str, Any]]) -> str:
    """``shortlist`` or the legacy ``catalog`` (unknown values fall back to ``catalog``)."""
    raw = str(skills_cfg.get("selection") or "").strip().lower() if isinstance(skills_cfg, dict) else ""
    return SELECTION_SHORTLIST if raw == SELECTION_SHORTLIST else SELECTION_CATALOG


def _terms(text: str) -> list[str]:
    """Stemmed ASCII content words plus CJK character bigrams (a lone CJK character stays a unigram).
    CJK has no spaces between words; bigrams keep a two-character word like 周报 one unit where
    per-character tokens would match any text sharing a common character."""
    from tools.tool_search_catalog import _stem
    out = [_stem(w) for w in (t.lower() for t in _ASCII_RE.findall(text)) if len(w) > 1 and w not in _STOPWORDS]
    for run in _CJK_RUN_RE.findall(text):
        out += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    return out


def request_text(user_message: Any) -> str:
    """The part of the turn that states the ask: fenced code, quoted replies and indented
    code/log lines are pasted material, not the request. A long residue is a pasted blob whose ask
    sits at its head or tail."""
    raw = user_message if isinstance(user_message, str) else flatten_message_text(user_message)
    text = _FENCE_RE.sub(" ", raw)
    text = "\n".join(line for line in text.splitlines() if not _SKIPPED_LINE_RE.match(line)).strip()
    if len(text) > _PASTE_THRESHOLD_CHARS:
        text = text[:_PASTE_HEAD_TAIL_CHARS] + "\n" + text[-_PASTE_HEAD_TAIL_CHARS:]
    return text


def _explicit_invocation(text: str, names: Iterable[str]) -> bool:
    """A slash command, an injected ``[...]`` note, or a request naming a skill outright (hyphenated
    or ``plugin:skill`` names only — single words like ``plan`` are ordinary vocabulary)."""
    if text.startswith(("/", "[")):
        return True
    lowered = text.lower()
    return any(("-" in n or ":" in n) and re.search(rf"(?<![\w-]){re.escape(n.lower())}(?![\w-])", lowered)
               for n in names)


class _Doc:
    """One skill's search document: name x3, situations x2, full description x1. ``name`` and
    ``_tokens`` are the duck type ``tools.tool_search_catalog._corpus_stats`` reads; ``strong`` holds
    the name and situation terms, the words that say what the skill is FOR."""
    __slots__ = ("_tokens", "name", "strong")

    def __init__(self, name: str, description: str, situations: tuple[str, ...]):
        name_terms, situation_terms = _terms(re.sub(r"[-_:]", " ", name)), _terms(" ".join(situations))
        self.name = name
        self._tokens = name_terms * 3 + situation_terms * 2 + _terms(description)
        self.strong = frozenset(name_terms + situation_terms)


@lru_cache(maxsize=8)
def _corpus(docs: tuple[tuple[str, str, tuple[str, ...]], ...]):
    from tools.tool_search_catalog import _corpus_stats
    corpus = [_Doc(*doc) for doc in docs]
    return corpus, _corpus_stats(corpus)  # type: ignore[arg-type]  # duck-typed CatalogEntry


def shortlist(query: Any, docs: tuple[tuple[str, str, tuple[str, ...]], ...]) -> list[str]:
    """At most :data:`MAX_CANDIDATES` skill names for ``query`` from ``docs`` (``(name, full
    description, situations)``), best first; ``[]`` when nothing clears the admission gate."""
    text = request_text(query)
    if not docs or not text or _explicit_invocation(text, (n for n, _, _ in docs)):
        return []
    query_terms = _terms(text)
    if len(set(query_terms)) < _MIN_CONTENT_TERMS:
        return []
    from tools.tool_search_catalog import _bm25_score
    corpus, stats = _corpus(docs)
    scored = sorted(((_bm25_score(query_terms, d._tokens, *stats), d) for d in corpus),
                    key=lambda row: (-row[0], row[1].name))
    # Admission (head-3 only): two distinct shared content words, or one that sits in the skill's
    # name or situations. One shared description word ("explain", "answer") is how small talk and
    # general questions reached a skill in the eval.
    picked = []
    for score, doc in scored[:MAX_CANDIDATES]:
        anchors = set(query_terms) & set(doc._tokens)
        if score > 0 and (len(anchors) >= 2 or anchors & doc.strong) and (
                not picked or score >= RUNNER_UP_SCORE_RATIO * scored[0][0]):
            picked.append(doc.name)
    return picked


# ── live wiring ──────────────────────────────────────────────────────────────


@lru_cache(maxsize=4096)
def _skill_file_doc(path: str, _sig: tuple) -> tuple[str, str, tuple[str, ...]]:
    """``(declared or directory name, full description, situations)`` for one SKILL.md, cached per
    file signature; an unreadable file has no name."""
    from agent.skill_utils import _config_str_list, _hermes_metadata, _normalize_skill_description, parse_frontmatter
    try:
        frontmatter, _ = parse_frontmatter(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError):
        return "", "", ()
    meta = _hermes_metadata(frontmatter)
    # Retrieval-only (the index never renders either); a malformed value (``triggers: true``) reads
    # as no phrases instead of raising and silencing every suggestion.
    phrases = [p for key in ("situations", "triggers") for p in _config_str_list(meta.get(key, frontmatter.get(key)))]
    name = str(frontmatter.get("name") or "").strip() or Path(path).parent.name
    return name, _normalize_skill_description(frontmatter), tuple(phrases)


def skill_docs_from_roots(roots: Iterable[tuple[int, Path]], index: dict[str, str]) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """A document for every skill ``index`` (``{name: index row}``) offers. The files under ``roots``
    (``(tier, dir)``) are resolved by the shared ``resolve_skill_catalog``, so each document sits under
    the ``load_name`` the index lists — a same-tier duplicate's exact path included. Skills with no
    file (plugin skills) fall back to their index row."""
    from agent.skill_utils import TIER_PROJECT, iter_project_skill_files, iter_skill_index_files, resolve_skill_catalog
    from utils import file_signature
    scanned: list[dict[str, Any]] = []
    for tier, root in roots:
        if not root.is_dir():
            continue
        files = iter_project_skill_files(root) if tier == TIER_PROJECT else iter_skill_index_files(root, "SKILL.md")
        for skill_file in files:
            try:
                sig = tuple(file_signature(skill_file.stat()))
            except OSError:
                continue
            name, desc, situations = _skill_file_doc(str(skill_file), sig)
            if name:
                scanned.append({"name": name, "tier": tier, "root": root, "path": skill_file, "doc": (desc, situations)})
    found = {e["load_name"]: e["doc"] for e in resolve_skill_catalog(scanned) if e["load_name"]}
    return tuple(
        (name, *found.get(name, (row.split(": ", 1)[1] if ": " in row else "", ())))
        for name, row in sorted(index.items())
    )


def _note(names: list[str], index: dict[str, str]) -> str:
    return NOTE_PREFIX + "\n" + "\n".join(_ROW + (index.get(n, "").strip()[2:] or n) for n in names) + "]"


def _shown_sets(conversation_history: Any) -> set[frozenset]:
    """Every suggestion set already delivered in this session, read back from the transcript: a
    gateway builds a fresh agent per turn, so agent state alone would repeat the line."""
    shown: set[frozenset] = set()
    for msg in conversation_history or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        sidecar = msg.get("api_content")
        text = (sidecar if isinstance(sidecar, str) else "") + "\n" + flatten_message_text(msg.get("content"))
        for chunk in text.split(NOTE_PREFIX)[1:]:
            rows = []
            for line in chunk.split("\n")[1:]:
                if not line.startswith(_ROW):
                    break
                rows.append(line[len(_ROW):].removesuffix("]").split(": ", 1)[0].strip())
            shown.add(frozenset(rows))
    return shown


def _loaded_skills(conversation_history: Any) -> set[str]:
    """Skills this session already carries in full — ``skills.auto_load``, a slash/bundle or gateway
    auto-load turn, a skill_view call: suggesting them again is noise."""
    from agent.skill_commands import resolve_auto_load_skills, skills_loaded_by_scaffold
    loaded = set(resolve_auto_load_skills())
    for msg in conversation_history or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            loaded.update(skills_loaded_by_scaffold(msg.get("content")))
        for tc in (msg.get("tool_calls") or []) if isinstance(msg, dict) else []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict) and fn.get("name") == "skill_view":
                try:
                    loaded.add(str(json.loads(fn.get("arguments") or "{}").get("name") or ""))
                except (TypeError, ValueError, AttributeError):
                    continue
    return loaded


def stage_skill_shortlist_note(agent: Any, prompt: Any, conversation_history: Any, user_message: Any) -> bool:
    """Stage this turn's shortlist on ``agent._skill_shortlist_note`` (consumed with the other
    per-turn notes). Returns whether it staged. Off unless ``skills.selection: shortlist``; skipped
    for internal forks (no user request) and for MoA/codex_app_server turns, which never stamp
    the sidecar the once-per-session check reads back."""
    from agent.skill_utils import _skills_cfg, get_skill_search_roots
    from agent.skills_index_delta import index_entries
    from agent.system_prompt import _agent_skills_dir, _skills_prompt
    if selection_mode(_skills_cfg()) != SELECTION_SHORTLIST or getattr(agent, "_persist_disabled", False):
        return False
    if getattr(agent, "provider", None) == "moa" or getattr(agent, "api_mode", None) == "codex_app_server":
        return False
    if not index_entries(prompt):
        return False
    try:
        # What the session can load NOW (the index stage_skills_index_note compares the prompt
        # with), so a skill removed since the prompt was built is never suggested.
        index = index_entries(_skills_prompt(agent))
        loaded = _loaded_skills(conversation_history)
        roots = get_skill_search_roots(_agent_skills_dir(agent))
        docs = tuple(d for d in skill_docs_from_roots(roots, index) if d[0] not in loaded)
        names = shortlist(user_message, docs)
    except Exception:
        logger.debug("skill shortlist: ranking failed", exc_info=True)
        return False
    if not names or frozenset(names) in _shown_sets(conversation_history):
        return False
    agent._skill_shortlist_note = _note(names, index)
    logger.info("Session %s: skill shortlist for this turn: %s", getattr(agent, "session_id", None), ", ".join(names))
    return True
