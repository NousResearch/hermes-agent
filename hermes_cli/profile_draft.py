"""Draft a bot profile from a plain-language description.

Same shape as ``profile_describer``: lazy aux client import, one tool-less call, lenient JSON
parse, and expected failures come back as ``ok=False`` instead of raising. Nothing is written
here — callers (``profiles.draft`` RPC, ``hermes profile create --from-prompt``) show the draft
and save only after the user confirms.

Capabilities are picked from a catalog the caller builds on the machine the bot will live on
(``build_catalog``); names the model invents are dropped and reported in ``dropped``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# (key, heading) in SOUL.md order. The heading text is the on-disk contract.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("role", "Role"),
    ("voice", "Voice"),
    ("does", "Does"),
    ("doesnt", "Doesn't"),
    ("silence", "When to stay silent"),
)
_HEADING_TO_KEY = {title.lower(): key for key, title in SECTIONS}
FIELDS = ("name", "title", "description")
CAPABILITY_KINDS = ("skills", "toolsets", "mcp_servers")
MAX_CATALOG_ITEMS = 120
_SECTION_CHARS = 1500
_HEADING_RE = re.compile(r"^##[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE | re.IGNORECASE)
_SLUG_BAD = re.compile(r"[^a-z0-9_-]+")

_SYSTEM_PROMPT = """You design persistent assistant bots for a user's personal agent install.

From the user's description, write one bot as a single JSON object:

{
  "name": "<short lowercase handle: letters, digits, '-' or '_'>",
  "title": "<2-4 word job title>",
  "description": "<one sentence a teammate reads to know when to hand this bot work>",
  "sections": {
    "role": "<who the bot is and the outcome it owns, 2-4 sentences>",
    "voice": "<how it writes: tone, length, formatting habits>",
    "does": "<markdown bullet list of concrete duties>",
    "doesnt": "<markdown bullet list of things it refuses or leaves to others>",
    "silence": "<when it should say nothing or reply with a bare acknowledgement>"
  },
  "skills": ["<names from AVAILABLE SKILLS only>"],
  "toolsets": ["<names from AVAILABLE TOOLSETS only>"],
  "mcp_servers": ["<names from AVAILABLE MCP SERVERS only>"]
}

Rules:
  - Pick capabilities only from the lists you are given, copied exactly. Pick few: what the
    job needs, not everything that could help. An empty list is fine.
  - Write the sections in second person ("You ...") as instructions to the bot.
  - Keep each section under 120 words. No preamble, no code fences: output only the JSON.
"""

_REVISE_RULES = """
You are revising the CURRENT DRAFT below. Apply the user's change request and keep every
field the request does not touch word-for-word identical. When it has saved_soul_md, that is
the bot's whole saved instruction file: read it for what the bot already does (the sections
above are only the parts of it under the five known headings)."""


@dataclass
class Catalog:
    """Capability names installed on the machine the bot will live on."""

    skills: list[str] = field(default_factory=list)
    toolsets: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)

    def names(self, kind: str) -> list[str]:
        return list(getattr(self, kind))


@dataclass
class ProfileDraft:
    ok: bool
    reason: str = ""
    name: str = ""
    title: str = ""
    description: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)
    toolsets: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)

    @property
    def soul(self) -> str:
        return compose_soul(self.title or self.name, self.sections)

    def to_wire(self) -> dict:
        return {
            "ok": self.ok, "reason": self.reason, "name": self.name, "title": self.title,
            "description": self.description,
            "sections": [{"key": k, "title": t, "text": self.sections.get(k, "")} for k, t in SECTIONS],
            "soul": self.soul if self.ok else "",
            "skills": self.skills, "toolsets": self.toolsets, "mcp_servers": self.mcp_servers,
            "dropped": self.dropped,
        }


# ── SOUL.md sections ─────────────────────────────────────────────────────────────────────────


def split_soul(text: str) -> dict[str, str]:
    """The known ``## <heading>`` sections of ``text`` keyed by section key (other text ignored)."""
    out: dict[str, str] = {}
    matches = list(_HEADING_RE.finditer(text or ""))
    for i, m in enumerate(matches):
        key = _HEADING_TO_KEY.get(m.group(1).strip().lower())
        if key is None or key in out:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[key] = text[m.end():end].strip()
    return out


def compose_soul(heading: str, sections: dict[str, str]) -> str:
    """``# heading`` then each non-empty section as ``## Title`` in canonical order."""
    parts = [f"# {heading.strip()}"] if heading and heading.strip() else []
    parts += [f"## {title}\n\n{sections[key].strip()}" for key, title in SECTIONS
              if (sections.get(key) or "").strip()]
    return "\n\n".join(parts) + ("\n" if parts else "")


def merge_sections(soul: str, sections: dict[str, str]) -> str:
    """``soul`` with each changed section of ``sections`` written over its ``## Title`` block in
    place (appended when absent); a section given as empty text removes its block, a key that is
    absent from ``sections`` is left alone. Every other byte — the heading, custom sections,
    plain text — is kept, so revising a saved SOUL.md never drops what the draft cannot see."""
    out = soul or ""
    for key, title in SECTIONS:
        if key not in sections:
            continue
        body = (sections[key] or "").strip()
        if body == split_soul(out).get(key, "").strip():
            continue
        m = next((h for h in _HEADING_RE.finditer(out) if _HEADING_TO_KEY.get(h.group(1).strip().lower()) == key), None)
        if m is None:
            out = f"{out.rstrip()}\n\n## {title}\n\n{body}\n" if out.strip() else f"## {title}\n\n{body}\n"
            continue
        nxt = _HEADING_RE.search(out, m.end())
        tail = out[nxt.start():] if nxt else ""
        if not body:
            head = out[:m.start()]
            out = head + tail if tail else (f"{head.rstrip()}\n" if head.strip() else "")
            continue
        out = f"{out[:m.end()]}\n\n{body}\n" + (f"\n{tail}" if tail else "")
    return out


# ── catalog ──────────────────────────────────────────────────────────────────────────────────


def _mcp_names(profile_dir: Path) -> list[str]:
    """MCP servers the profile's runtime would see: its own config.yaml plus the managed overlay."""
    from hermes_cli.config_effective import load_user_config_effective
    servers = load_user_config_effective(profile_dir / "config.yaml").get("mcp_servers")
    return sorted(str(n) for n in servers) if isinstance(servers, dict) else []


def _toolset_names() -> list[str]:
    from hermes_cli.tools_config import _get_effective_configurable_toolsets, _toolset_allowed_for_platform
    return [ts for ts, _label, _desc in _get_effective_configurable_toolsets()
            if _toolset_allowed_for_platform(ts, "cli")]


def build_catalog(profile_dir: Path) -> Catalog:
    """Installed skills + configured MCP servers of ``profile_dir`` (read by explicit path) and the
    toolsets of the CURRENT profile scope: plugin toolsets are discovered per profile home, so
    callers bind ``profile_dir``'s scope around this call (and only this call — the aux model
    call that follows bills the caller)."""
    from hermes_cli.profile_describer import _collect_skills, _sample_skills
    try:
        toolsets = _toolset_names()
    except Exception:  # health: allow BLE001 -- plugin toolset discovery runs third-party code; the draft still works without toolsets
        logger.debug("draft: toolset catalog unavailable", exc_info=True)
        toolsets = []
    return Catalog(skills=_sample_skills(_collect_skills(Path(profile_dir)))[:MAX_CATALOG_ITEMS],
                   toolsets=toolsets[:MAX_CATALOG_ITEMS], mcp_servers=_mcp_names(Path(profile_dir)))


def _filter_to_catalog(kind: str, raw, allowed: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """``raw`` names that exist in ``allowed`` (case-insensitive; a bare skill name matches its
    ``category/name`` entry), in catalog spelling, deduped; the rest come back as dropped."""
    exact = {a.lower(): a for a in allowed}
    leaf = {a.rsplit("/", 1)[-1].lower(): a for a in allowed}
    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    for item in raw if isinstance(raw, list) else []:
        name = str(item or "").strip()
        if not name:
            continue
        hit = exact.get(name.lower()) or leaf.get(name.rsplit("/", 1)[-1].lower())
        if hit is None:
            dropped.append({"kind": kind, "name": name})
        elif hit not in kept:
            kept.append(hit)
    return kept, dropped


# ── the call ─────────────────────────────────────────────────────────────────────────────────


def slugify(text: str) -> str:
    slug = _SLUG_BAD.sub("-", (text or "").strip().lower()).strip("-_")[:64].strip("-_")
    return slug if slug and slug[0].isalnum() else ""


def _clip(value, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _as_current(current) -> Optional[dict]:
    """``current`` with ``sections`` as a key → text mapping. Also takes the list of
    ``{key, title, text}`` that ``to_wire`` returns, so a client can send a result straight back."""
    if not isinstance(current, dict):
        return None
    raw = current.get("sections")
    if isinstance(raw, list):
        raw = {str(s["key"]): str(s.get("text") or "") for s in raw if isinstance(s, dict) and s.get("key")}
    return current | {"sections": raw if isinstance(raw, dict) else {}}


def _current_block(current: dict) -> str:
    import json
    body = {k: current.get(k, "") for k in FIELDS} | {"sections": current["sections"]}
    body |= {k: current.get(k) or [] for k in CAPABILITY_KINDS}
    if str(current.get("soul") or "").strip():
        body["saved_soul_md"] = current["soul"]
    return "CURRENT DRAFT:\n" + json.dumps(body, ensure_ascii=False, indent=2)


def _user_message(description: str, catalog: Catalog, current: Optional[dict], note: str, only: str) -> str:
    def listing(kind: str) -> str:
        return "\n".join(f"  - {n}" for n in catalog.names(kind)) or "  (none installed)"
    lines = [f"DESCRIPTION:\n{description.strip()}"]
    if current:
        lines.append(_current_block(current))
    if note.strip():
        lines.append(f"CHANGE REQUEST:\n{note.strip()}")
    if only:
        lines.append(f"Change only '{only}'.")
    lines += [f"AVAILABLE SKILLS:\n{listing('skills')}", f"AVAILABLE TOOLSETS:\n{listing('toolsets')}",
              f"AVAILABLE MCP SERVERS:\n{listing('mcp_servers')}"]
    return "\n\n".join(lines)


def _parse_reply(raw: str) -> tuple[Optional[dict], str]:
    from hermes_cli.kanban_specify import _extract_json_blob
    parsed = _extract_json_blob(raw, _FENCE_RE)
    if parsed is not None:
        return parsed, ""
    if not raw.strip():
        return None, "the model returned an empty reply"
    return None, "the model's reply was not a usable draft (malformed or cut off) — try again"


def _draft_from(parsed: dict, catalog: Catalog) -> ProfileDraft:
    raw = parsed.get("sections")
    sections_raw: dict = raw if isinstance(raw, dict) else {}
    draft = ProfileDraft(
        ok=True, reason="drafted", name=slugify(str(parsed.get("name") or "")),
        title=_clip(parsed.get("title"), 80), description=_clip(parsed.get("description"), 280),
        sections={k: _clip(sections_raw[k], _SECTION_CHARS) for k, _ in SECTIONS if k in sections_raw})
    for kind in CAPABILITY_KINDS:
        kept, dropped = _filter_to_catalog(kind, parsed.get(kind), catalog.names(kind))
        setattr(draft, kind, kept)
        draft.dropped += dropped
    return draft


def _keep_omitted(draft: ProfileDraft, parsed: dict, current: dict) -> None:
    """Revise mode: a part the reply leaves out stays as ``current`` has it, while one it sends
    back empty stays empty — that is a removal ``merge_sections`` applies to the saved SOUL.md."""
    draft.sections = {k: draft.sections.get(k, str(current["sections"].get(k) or "")) for k, _ in SECTIONS}
    for key in FIELDS:
        if key not in parsed:
            setattr(draft, key, str(current.get(key) or ""))


def _keep_untouched(draft: ProfileDraft, current: dict, only: str) -> None:
    """Regenerate-one-part: everything except ``only`` is restored from ``current`` verbatim."""
    cur_sections = current["sections"]
    for key in FIELDS:
        if key != only:
            setattr(draft, key, str(current.get(key) or ""))
    draft.sections = {k: (draft.sections.get(k, "") if k == only else str(cur_sections.get(k) or ""))
                      for k, _ in SECTIONS}
    if only != "capabilities":
        for kind in CAPABILITY_KINDS:
            setattr(draft, kind, [str(n) for n in current.get(kind) or []])
        draft.dropped = []


def draft_profile(
    description: str, *, catalog: Catalog, current: Optional[dict] = None, note: str = "",
    only: str = "", timeout: Optional[int] = None,
) -> ProfileDraft:
    """One aux call (``auxiliary.profile_describer``) → a validated draft; never raises.

    ``current`` + ``note`` revise an existing draft; ``only`` (a section key, a field name or
    ``"capabilities"``) keeps every other part of ``current`` untouched."""
    if not (description or "").strip() and not (note or "").strip():
        return ProfileDraft(False, "describe the bot first")
    if only and only not in {k for k, _ in SECTIONS} | set(FIELDS) | {"capabilities"}:
        return ProfileDraft(False, f"unknown part '{only}'")
    current = _as_current(current)
    try:
        from agent.auxiliary_client import call_llm
    except ImportError:
        logger.debug("draft: auxiliary client import failed", exc_info=True)
        return ProfileDraft(False, "auxiliary model unavailable")
    system = _SYSTEM_PROMPT + (_REVISE_RULES if current else "")
    try:
        resp = call_llm(
            task="profile_describer",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": _user_message(description or "", catalog, current, note, only)}],
            temperature=0.4, max_tokens=1800, timeout=timeout or 90,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:  # health: allow BLE001 -- provider/transport errors are an expected outcome reported to the user, never a crash
        logger.info("draft: auxiliary call failed", exc_info=True)
        return ProfileDraft(False, f"model call failed: {type(exc).__name__}")
    parsed, why = _parse_reply(raw)
    if parsed is None:
        return ProfileDraft(False, why)
    draft = _draft_from(parsed, catalog)
    if current:
        _keep_omitted(draft, parsed, current)
    if current and only:
        _keep_untouched(draft, current, only)
    if not any(draft.sections.values()):
        return ProfileDraft(False, "the draft came back without any instructions — try again")
    return draft
