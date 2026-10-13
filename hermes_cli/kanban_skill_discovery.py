"""Pre-dispatch skill discovery hook (ADR-001 Phase 1).

Runs two mandatory searches (session FTS over the state DB + Obsidian
``simple_search`` with the FAISS ``vault_search`` fallback) against every
task claim, BEFORE the worker process is spawned. Canonical skill names
from the hits are written into ``tasks.skills`` (a force-load mechanism —
``Task.from_row`` coerces to ``[str]``, ``_worker_argv`` builds one
``--skills <name>`` pair per entry), the full evidence stays in the
claimed run's ``metadata.discovery`` (``_end_run`` merges it, so this
survives the run's terminal transition).

Failure behaviour is DEGRADED-FIRST (Spec v1.0 §1(4), the correction over
the hard-block card body): one failed source may not freeze the pipeline
(~140 claims/week), so a claim with ≥1 responsive source dispatches
normally; only when ALL THREE sources (session, obsidian, vault fallback)
fail within their timeout does the claim get rolled back and the spawn
skipped for this tick (``skipped_discovery_failed``) — the task stays
``ready`` without touching the ``consecutive_failures`` breaker.

Kill switch: ``config.yaml → kanban.skill_discovery.enabled`` (default
true), deliberately NOT an env var so flipping it is a durable, auditable
config change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Optional

_log = logging.getLogger("hermes_cli.kanban_db")

# Bounded stopword list for the vault query's TF keywords (no LLM in the
# hot path; deterministic is the requirement). EN+DE covers the board's
# real task bodies.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "be", "are", "at", "by", "from", "as", "it", "this", "that",
    "der", "die", "das", "und", "oder", "im", "am", "auf",
    "für", "mit", "von", "aus", "zu", "zum", "zur", "bei", "ist", "sind",
    "ein", "eine", "einem", "einen", " werden", "wird", "werden",
})

_KANBAN_WORKER_BUILTIN = "kanban-worker"


@dataclass
class DiscoveryHit:
    skill: Optional[str]
    score: float
    source: str            # "session" | "vault"
    ref: str
    snippet: str
    score_raw: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "skill": self.skill, "score": round(self.score, 4),
            "source": self.source, "ref": self.ref, "snippet": self.snippet[:160],
        }


@dataclass
class DiscoveryResult:
    status: str                                  # ok | degraded:<source> | total_failure
    queries: dict = field(default_factory=dict)  # {"session": Q1, "vault": Q2}
    hits: list = field(default_factory=list)     # DiscoveryHit dicts, ≤ max_hits
    skills: list = field(default_factory=list)   # canonical names, ≤ max_skills
    source_results: dict = field(default_factory=dict)  # per-source: ok/failed/empty, hit count, ms
    audit: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "status": self.status,
            "ts": int(time.time()),
            "queries": self.queries,
            "hits": self.hits,
            "skills_written": list(self.skills),
            "source_results": self.source_results,
            "audit": self.audit,
        }

    @property
    def responsive_sources(self) -> int:
        return sum(1 for s in self.source_results.values() if s.get("status") == "ok")

    @property
    def total_failure(self) -> bool:
        return self.responsive_sources == 0


# ---------------------------------------------------------------------------
# Query construction (Spec §1(5): title + body only — tasks has no tags column)
# ---------------------------------------------------------------------------

def build_queries(title: str, body: Optional[str], session_cfg: dict, vault_cfg: dict) -> dict:
    """Deterministic ≤200-char queries. Q1(session) = normalized title +
    first 240 whitespace-normalized body chars; Q2(vault) = title + top-8
    TF (term frequency) keywords from the body. Truncation cuts on word
    boundaries."""
    body_plain = " ".join((body or "").split())
    title_plain = " ".join((title or "").split())

    q1_parts = [title_plain, body_plain[:240]]
    q1 = " ".join(p for p in q1_parts if p).strip()
    q1 = _truncate_on_word(q1, session_cfg.get("query_max_chars", 200))

    freq: dict[str, int] = {}
    for token in re.findall(r"[\w-]+", body_plain.lower()):
        if token not in _STOPWORDS and len(token) >= 3:
            freq[token] = freq.get(token, 0) + 1
    keywords = [w for w, _n in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))][: vault_cfg.get("max_keywords", 8)]

    q2_parts = [title_plain] + keywords
    q2 = " ".join(p for p in q2_parts if p).strip()
    q2 = _truncate_on_word(q2, vault_cfg.get("query_max_chars", 200))
    return {"session": q1, "vault": q2}


def _truncate_on_word(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space >= limit * 0.6:
        return cut[:space].rstrip()
    return cut.rstrip()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# ---------------------------------------------------------------------------
# Skill-name registry (vault mapping needs exact token match, Spec §1(3))
# ---------------------------------------------------------------------------

def collect_skill_names(home: Optional[str] = None) -> frozenset:
    """Skill identifiers from all profile skill dirs (``<home>/skills``,
    ``<home>/optional-skills`` — the dirs ``build_skills_system_prompt``
    scans, at HERMES_DEFAULT_HOME scope so the profile layering is out of
    scope for Phase 1). Names only — a hit may propose a skill the
    consumer renders as "missing" gracefully."""
    from pathlib import Path
    from hermes_constants import _get_platform_default_hermes_home
    try:
        root = Path(home) if home is not None else _get_platform_default_hermes_home()
        skills_dir, optional_dir = root / "skills", root / "optional-skills"
    except Exception as exc:
        _log.debug("skill discovery: skill dirs unavailable: %s", exc)
        return frozenset()
    names: list[str] = []
    for base in (skills_dir, optional_dir):
        try:
            names.extend(p.name for p in base.iterdir() if p.is_dir())
        except Exception:
            continue
    return frozenset(n for n in (s.strip() for s in names) if n)


def extract_skill_name(text: str, registry: frozenset) -> Optional[str]:
    """The registered skill name that appears as a whole word in *text*
    (case-insensitive), longest first so ``vault-semantic-search`` wins
    over ``vault-search``."""
    if not text or not registry:
        return None
    low = text.lower()
    for name in sorted(registry, key=len, reverse=True):
        if re.search(fr"\b{re.escape(name.lower())}\b", low):
            return name
    return None


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _timeout_wrap(seconds: float, fn, *args, **kwargs):
    """Small thread-based deadline; the sources are read-only lookups with
    no shared state, so daemon-thread abandonment is safe."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=seconds), None
        except concurrent.futures.TimeoutError:
            return None, f"timeout after {seconds}s"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"


def session_search(query: str, limit: int) -> tuple[list, Optional[str]]:
    """FTS over the session DB. Discovery shape returns sessions: one has a
    session_id, when, source, title, snippet etc. Rows carry their ``skills``
    only via a second kanban lookup the dispatcher does (by task_id
    extracted from the title). Returns (rows, error)."""
    from tools.session_search_tool import session_search as _session_search_impl, check_session_search_requirements

    if not check_session_search_requirements():
        return [], "session DB unavailable"
    raw, err = _timeout_wrap(_source_timeout(), _session_search_impl, query=query, limit=limit)
    if err is not None:
        return [], err
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError) as exc:
        return [], f"unparseable session_search payload: {exc}"
    if not isinstance(payload, dict) or not payload.get("success"):
        return [], str(payload.get("error") if isinstance(payload, dict) else "session_search failed")
    rows = []
    results = payload.get("results") or []
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            continue
        snippet = r.get("snippet") or r.get("preview") or ""
        rows.append({
            "title": r.get("title") or "",
            "snippet": " ".join(str(snippet).split()),
            # Rank-scaled: best FTS hit = 1.0, descending.
            "score": round(1.0 / (1 + i * 0.25), 4),
            "session_id": r.get("session_id") or "",
        })
    return rows, None


def search_vault_fallback(query: str, limit: int) -> tuple[list, Optional[str]]:
    """FAISS CLI fallback (`vault_search.py --json`), independent of the
    Obsidian app. CLI path from ``kanban.skill_discovery.vault_search_cli``
    (operator-workspace deploy path by default) — never hardcoded only."""
    import os
    import sys as _sys
    from pathlib import Path as _Path

    cli = _Path(os.environ.get(
        "HERMES_VAULT_SEARCH_CLI",
        "/Users/minimetro/.hermes/profiles/operator/workspace/"
        "04-Monitoring-Data/vault-semantic-search/vault_search.py",
    ))
    if not cli.exists():
        return [], f"vault_search CLI missing: {cli}"
    py = _sys.executable or "python3"
    raw, err = _timeout_wrap(
        _source_timeout(),
        subprocess_run_json, py, str(cli), query, limit)
    if err is not None:
        return [], err
    if not raw.get("success"):
        return [], str(raw.get("error") or "vault_search failed")
    rows = []
    for r in raw.get("results") or []:
        if not isinstance(r, dict):
            continue
        rows.append({
            "title": r.get("title") or "",
            "snippet": r.get("excerpt") or r.get("section") or "",
            "score": float(r.get("score") or 0.0),
            "path": r.get("file_path") or "",
        })
    return rows, None


def subprocess_run_json(py: str, cli: str, query: str, limit: int) -> dict:
    import subprocess
    proc = subprocess.run(
        [py, cli, query, "--k", str(limit), "--json"],
        capture_output=True, text=True, timeout=30, check=False,
        cwd=str(__import__("pathlib").Path(cli).parent),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"vault_search exit {proc.returncode}: {proc.stderr.strip()[:200]}")
    return json.loads(proc.stdout)


def search_obsidian(query: str, limit: int, timeout_s: float) -> tuple[list, Optional[str]]:
    """Direct Local REST API client (mcp-obsidian patch semantics,
    src/mcp_obsidian/obsidian.py:97): POST /search/simple/, Bearer key
    from the profile .env. Called directly — NOT via the MCP stdio client
    — to stay process- and app-independent. Response rows:
    ``{filename, score, matches[{context}]}``."""
    import os
    import urllib.error
    import urllib.parse
    import urllib.request
    import ssl
    from pathlib import Path

    env_candidates = [
        Path(os.environ.get("HERMES_PROFILE_HOME", "")) / ".env",
        Path(os.environ.get("HERMES_HOME", "")) / ".env",
    ]
    api_key = None
    for p in env_candidates:
        try:
            if p.exists():
                for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("OBSIDIAN_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
            if api_key:
                break
        except Exception:
            continue
    if not api_key:
        api_key = os.environ.get("OBSIDIAN_API_KEY")
    if not api_key:
        return [], "OBSIDIAN_API_KEY not configured"

    host = os.environ.get("OBSIDIAN_HOST", "127.0.0.1")
    port = os.environ.get("OBSIDIAN_PORT", "27124")
    url = f"https://{host}:{port}/search/simple/"
    params = urllib.parse.urlencode({"query": query, "contextLength": 100})
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        f"{url}?{params}", method="POST", data=b"",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    def _do():
        with urllib.request.urlopen(req, timeout=max(timeout_s, 5), context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    raw, err = _timeout_wrap(timeout_s, _do)
    if err is not None:
        return [], err
    if not isinstance(raw, list):
        return [], f"unexpected simple_search shape: {type(raw).__name__}"
    rows = []
    for r in raw[:limit]:
        if not isinstance(r, dict):
            continue
        matches = r.get("matches") or []
        context = " ".join(m.get("context", "") for m in matches[:2])
        rows.append({
            "title": r.get("filename") or "",
            "snippet": " ".join(context.split()),
            "score": float(r.get("score") or 0.0),
            "path": r.get("filename") or "",
        })
    return rows, None


def _source_timeout() -> float:
    return float(_discovery_config().get("timeout_s", 5))


# ---------------------------------------------------------------------------
# Config (kill-switch, Spec §1(7))
# ---------------------------------------------------------------------------

_DISCOVERY_DEFAULTS = {
    "enabled": True,
    "max_skills": 3,
    "max_hits": 8,
    "timeout_s": 5,
    "max_keywords": 8,
    "session_limit": 3,
    "vault_limit": 5,
    "query_max_chars": 200,
    "total_failure_backoff_s": 10,
}


def _discovery_config() -> dict:
    try:
        from hermes_cli.config_effective import load_user_config_effective
        kanban = (load_user_config_effective(fail_closed=True) or {}).get("kanban", {})
    except Exception as exc:
        _log.debug("skill discovery: config read failed (%s) — defaults", exc)
        return dict(_DISCOVERY_DEFAULTS)
    block = kanban.get("skill_discovery") if isinstance(kanban, dict) else None
    if block is None:
        return dict(_DISCOVERY_DEFAULTS)
    merged = dict(_DISCOVERY_DEFAULTS)
    if isinstance(block, dict):
        merged.update({k: v for k, v in block.items() if k in _DISCOVERY_DEFAULTS})
    return merged


# ---------------------------------------------------------------------------
# Main hook
# ---------------------------------------------------------------------------

def run_skill_discovery(
    task, registry: Optional[frozenset] = None, cfg: Optional[dict] = None,
) -> DiscoveryResult:
    """The Phase-1 pre-dispatch hook: run both mandatory searches against
    title+body, map hits to canonical skill names, return the structured
    result. NEVER raises — every source error degrades into the result.
    """
    cfg = cfg or _discovery_config()
    if not cfg.get("enabled", True):
        # Disabled ⇒ nothing forced, no evidence record: the caller skips
        # the hook entirely. This branch is only for direct callers.
        return DiscoveryResult(status="disabled")
    if registry is None:
        registry = collect_skill_names()

    q = build_queries(task.title, task.body, cfg, cfg)
    max_hits = int(cfg.get("max_hits", 8))
    session_limit = int(cfg.get("session_limit", 3))
    vault_limit = int(cfg.get("vault_limit", 5))

    result = DiscoveryResult(
        status="ok", queries=q,
        audit={"q1_sha256": _sha(q["session"]), "q2_sha256": _sha(q["vault"])},
    )

    # Mandatory 1: session FTS.
    t0 = time.monotonic()
    try:
        s_rows, s_err = session_search(q["session"], session_limit)
    except Exception as exc:
        s_rows, s_err = [], f"{type(exc).__name__}: {exc}"
    result.source_results["session"] = {
        "status": "ok" if s_err is None else "failed",
        "error": s_err, "hit_count": len(s_rows) if s_err is None else 0,
        "ms": int((time.monotonic() - t0) * 1000),
    }
    failed = s_err is not None

    # Mandatory 2: Obsidian simple_search.
    t0 = time.monotonic()
    try:
        v_rows, v_err = search_obsidian(q["vault"], vault_limit, float(cfg.get("timeout_s", 5)))
    except Exception as exc:
        v_rows, v_err = [], f"{type(exc).__name__}: {exc}"
    result.source_results["vault"] = {
        "status": "ok" if v_err is None else "failed",
        "error": v_err, "hit_count": len(v_rows) if v_err is None else 0,
        "ms": int((time.monotonic() - t0) * 1000),
    }
    if v_err is not None:
        failed = True
        # Degraded fallback: FAISS vault_search (app-independent).
        t0 = time.monotonic()
        fb_rows, fb_err = search_vault_fallback(q["vault"], vault_limit)
        result.source_results["vault_fallback"] = {
            "status": "ok" if fb_err is None else "failed",
            "error": fb_err, "hit_count": len(fb_rows) if fb_err is None else 0,
            "ms": int((time.monotonic() - t0) * 1000),
        }
        if fb_err is None:
            v_rows = fb_rows

    if result.responsive_sources == 0:
        result.status = "total_failure"
        return result
    if failed:
        result.status = (
            "degraded:session" if result.source_results["session"]["status"] != "ok"
            else "degraded:vault"
        )

    # --- mapping (Spec §1(3)) --------------------------------------------
    hits: list[dict] = []
    session_skill_scores: dict[str, float] = {}
    for row in s_rows:
        # Session hits are past tasks: adopt that task's skills (reuse).
        for name in _session_task_skills(row):
            if name != _KANBAN_WORKER_BUILTIN:
                session_skill_scores[name] = max(
                    session_skill_scores.get(name, 0.0), row["score"])
        hits.append(DiscoveryHit(
            skill=None, score=row["score"], source="session",
            ref=row.get("session_id") or "session",
            snippet=row["snippet"]).to_dict())
    vault_skill_scores: dict[str, float] = {}
    for row in v_rows:
        blob = f"{row['title']}\n{row['snippet']}"
        name = extract_skill_name(blob, registry)
        if name and name != _KANBAN_WORKER_BUILTIN:
            vault_skill_scores[name] = max(vault_skill_scores.get(name, 0.0), row["score"])
        hits.append(DiscoveryHit(
            skill=name, score=row["score"], source="vault",
            ref=row.get("path") or "vault",
            snippet=row["snippet"]).to_dict())

    result.hits = hits[:max_hits]

    # tasks.skills: dedup, score-desc, top max_skills (Spec §1(3)).
    max_skills = int(cfg.get("max_skills", 3))
    combined_scores = {**session_skill_scores}
    for name, score in vault_skill_scores.items():
        combined_scores[name] = max(combined_scores.get(name, 0.0), score)
    # Dedup against the profile's ``skills.auto_load`` (Spec §1(3)): force-
    # loading a skill that is already auto-loaded every session is waste.
    try:
        from agent.skill_commands import resolve_auto_load_skills
        auto_load = set(resolve_auto_load_skills())
    except Exception as exc:
        _log.debug("skill discovery: auto_load unreadable (%s) — no dedup", exc)
        auto_load = set()
    result.skills = [
        name for name, _s in sorted(combined_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if name not in auto_load
    ][:max_skills]
    return result


def _kanban_conn() -> sqlite3.Connection:
    """A read-only connection to the LIVE board's kanban.db (HERMES_KANBAN_DB
    env pin → current board → default). Read-only: the dispatcher owns the
    writes; this lookup must never race one."""
    from pathlib import Path

    from hermes_cli.kanban_db import kanban_db_path
    path = Path(kanban_db_path()).resolve()
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _session_task_skills(row: dict) -> list:
    """Skills of the past task a session-search hit belongs to.

    Phase-1 heuristic (keeps the hot path zero-extra-IO): session titles on
    this board are the task title verbatim and kanban task ids surface in
    the snippet as ``t_xxxxxxxx``; a hit whose snippet carries a task id
    gets that task's ``tasks.skills`` from the kanban DB. Non-kanban
    sessions contribute no skill names."""
    snippet = row.get("snippet") or ""
    title = row.get("title") or ""
    m = re.search(r"\bt_[0-9a-f]{8}\b", f"{title} {snippet}")
    if m is None:
        return []
    try:
        conn = _kanban_conn()
        trow = conn.execute(
            "SELECT skills FROM tasks WHERE id = ?", (m.group(0),)
        ).fetchone()
    except Exception as exc:
        _log.debug("skill discovery: session task lookup failed: %s", exc)
        return []
    if trow is None:
        return []
    return [s for s in _parse_json_list(trow["skills"]) if isinstance(s, str) and s.strip()]


def _parse_json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (TypeError, ValueError):
            return []
    return []


def write_skills(conn: sqlite3.Connection, task_id: str, skills: list) -> None:
    """Persist ``tasks.skills`` (raw column, JSON array of names). Called
    under the caller's txn by the dispatcher; an empty list means
    "discovery ran, no relevant skill" and clears prior suggestions for
    this claim (idempotent overwrite, Spec §1 Re-Dispatch)."""
    conn.execute(
        "UPDATE tasks SET skills = ? WHERE id = ?",
        (json.dumps(list(skills), ensure_ascii=False), task_id),
    )


def attach_to_run(conn: sqlite3.Connection, run_id: int, discovery: DiscoveryResult) -> None:
    """Write ``discovery`` into the CLAIMED run's metadata column. Run rows
    are merged at their terminal transition by :func:`_end_run`, so this
    survives completion/reclaim."""
    if run_id is None:
        return
    row = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    try:
        meta = json.loads(row["metadata"]) if (row and row["metadata"]) else {}
    except (TypeError, ValueError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {"_legacy_metadata": meta}
    meta["discovery"] = discovery.to_metadata()
    conn.execute(
        "UPDATE task_runs SET metadata = ? WHERE id = ?",
        (json.dumps(meta, ensure_ascii=False), run_id),
    )