"""
Dream scheduler — enqueue, check, and run consolidation cycles.

State lives in {HERMES_HOME}/dreams/:
  staging.jsonl        one candidate per line, written by on_session_end
  state.json           last_dream_at, sessions_since_dream
  lock                 present while a cycle is running
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from . import _config, _diary, _score

_DEFAULT_MIN_HOURS = 24
_DEFAULT_MIN_SESSIONS = 5


def _dreams_dir(hermes_home: str | None = None) -> Path:
    base = Path(hermes_home or os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    d = base / "dreams"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _staging_path(hermes_home: str | None = None) -> Path:
    return _dreams_dir(hermes_home) / "staging.jsonl"


def _state_path(hermes_home: str | None = None) -> Path:
    return _dreams_dir(hermes_home) / "state.json"


def _lock_path(hermes_home: str | None = None) -> Path:
    return _dreams_dir(hermes_home) / "lock"


def _read_state(hermes_home: str | None = None) -> dict:
    p = _state_path(hermes_home)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"last_dream_at": 0.0, "sessions_since_dream": 0}


def _write_state(state: dict, hermes_home: str | None = None) -> None:
    _state_path(hermes_home).write_text(json.dumps(state))


def enqueue_session(
    transcript: list[dict[str, str]],
    *,
    hermes_home: str | None = None,
    count_session: bool = True,
) -> int:
    """
    Extract candidate memories from a session transcript and append to staging.jsonl.
    Returns the number of candidates written.
    """
    now = time.time()
    candidates: list[dict] = []
    seen: set[str] = set()

    for turn in transcript:
        role = turn.get("role", "")
        content = turn.get("content", "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        # Simple heuristic: sentences that are short, declarative, and not questions.
        for sentence in _split_sentences(content):
            if len(sentence) < 20 or sentence.endswith("?"):
                continue
            key = hashlib.sha1(sentence.lower().encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "text": sentence,
                "hash": key,
                "role": role,
                "created_at": now,
                "frequency": 1,
                "query_count": 1,
                "word_count": len(sentence.split()),
            })

    if not candidates:
        return 0

    staging = _staging_path(hermes_home)
    with staging.open("a", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c) + "\n")

    if count_session:
        mark_session_complete(hermes_home=hermes_home)
    return len(candidates)


def mark_session_complete(*, hermes_home: str | None = None) -> None:
    """Increment the completed-session counter after transcript turns are staged."""
    state = _read_state(hermes_home)
    state["sessions_since_dream"] = state.get("sessions_since_dream", 0) + 1
    _write_state(state, hermes_home)


def dream_check(
    *,
    hermes_home: str | None = None,
    min_hours: float = _DEFAULT_MIN_HOURS,
    min_sessions: int = _DEFAULT_MIN_SESSIONS,
) -> bool:
    """Return True if conditions are met to run a dream cycle."""
    if _lock_path(hermes_home).exists():
        return False
    staging = _staging_path(hermes_home)
    if not staging.exists() or staging.stat().st_size == 0:
        return False
    state = _read_state(hermes_home)
    hours_since = (time.time() - state["last_dream_at"]) / 3600
    return (
        hours_since >= min_hours
        and state.get("sessions_since_dream", 0) >= min_sessions
    )


def dream_run(
    *,
    hermes_home: str | None = None,
    force: bool = False,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """
    Run one full dream cycle. Returns a summary dict.
    Raises RuntimeError if a lock is already held and force=False.
    """
    lock = _lock_path(hermes_home)
    if lock.exists() and not force:
        raise RuntimeError("dream cycle already running")
    lock.touch()

    cfg = cfg or _config.load_config()
    promote_threshold = _config.promote_threshold(cfg)

    try:
        # --- Light Sleep: load, deduplicate, score ---
        staging = _staging_path(hermes_home)
        raw: list[dict] = []
        seen_hashes: set[str] = set()

        if staging.exists():
            for line in staging.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                h = rec.get("hash", "")
                if h in seen_hashes:
                    # merge frequency
                    for existing in raw:
                        if existing.get("hash") == h:
                            existing["frequency"] = existing.get("frequency", 1) + 1
                            existing["query_count"] = existing.get("query_count", 1) + 1
                    continue
                seen_hashes.add(h)
                raw.append(rec)

        now = time.time()
        scored = [(c, _score.score(c, now)) for c in raw]
        scored.sort(key=lambda x: x[1], reverse=True)

        # --- REM: theme extraction + narrative ---
        narrative = _rem_narrative([c for c, _ in scored[:30]], cfg=cfg)

        # --- Deep Sleep: promote to MEMORY.md ---
        eligible: list[str] = []
        skipped_meta: list[str] = []
        to_promote = [c for c, s in scored if s >= promote_threshold]

        for candidate in to_promote:
            text = candidate["text"]
            if _score.is_meta_entry(text):
                skipped_meta.append(text)
            else:
                eligible.append(text)

        promoted = _promote_to_memory(eligible)

        if skipped_meta:
            hermes_base = Path(
                hermes_home or os.environ.get("HERMES_HOME", Path.home() / ".hermes")
            )
            _write_to_skill(skipped_meta, hermes_base / "SKILL.md")

        # --- Write diary entry ---
        _diary.append_entry(narrative, promoted, skipped_meta, hermes_home=hermes_home)

        # --- Reset staging and state ---
        staging.write_text("", encoding="utf-8")
        state = _read_state(hermes_home)
        state["last_dream_at"] = now
        state["sessions_since_dream"] = 0
        _write_state(state, hermes_home)

        return {
            "candidates_scanned": len(raw),
            "promoted": len(promoted),
            "skipped_meta": len(skipped_meta),
            "narrative_length": len(narrative),
        }

    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass


def _rem_narrative(candidates: list[dict], *, cfg: dict | None = None) -> str:
    """Ask a local LLM for theme extraction. Falls back to a structured summary."""
    texts = [c["text"] for c in candidates if c.get("text")]
    if not texts:
        return "No candidates surfaced this cycle."

    try:
        return _ollama_narrative(texts, cfg=cfg or _config.load_config())
    except Exception:
        pass

    # Fallback: structured summary without LLM
    lines = ["**Themes surfaced this cycle:**\n"]
    for i, text in enumerate(texts[:10], 1):
        lines.append(f"{i}. {text[:120]}")
    return "\n".join(lines)


def _ollama_narrative(texts: list[str], *, cfg: dict) -> str:
    import urllib.request

    rem = _config.rem(cfg)
    prompt = (
        "You are a memory consolidation assistant. The following facts were observed "
        "across recent sessions. Identify 3-5 recurring themes and write a brief "
        "narrative paragraph (2-4 sentences) summarising what matters most. "
        "Be concrete, not abstract.\n\nFacts:\n"
        + "\n".join(f"- {t}" for t in texts[:20])
    )
    payload = json.dumps({
        "model": rem["model"],
        "prompt": prompt,
        "stream": False,
    }).encode()
    url = rem["base_url"] + "/api/generate"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    return body.get("response", "").strip()


def _promote_to_memory(entries: list[str]) -> list[str]:
    """Write through Hermes's bounded, locked, threat-scanning memory store."""
    from tools.memory_tool import MemoryStore

    store = MemoryStore()
    promoted: list[str] = []
    for entry in entries:
        result = store.add("memory", entry)
        if result.get("success"):
            promoted.append(entry)
    return promoted


def _write_to_skill(entries: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n<!-- dreaming: meta-entries -->\n")
        for e in entries:
            fh.write(f"- {e[:200]}\n")


def _split_sentences(text: str) -> list[str]:
    import re
    parts = re.split(r"(?<=[.!])\s+", text)
    return [p.strip() for p in parts if p.strip()]
