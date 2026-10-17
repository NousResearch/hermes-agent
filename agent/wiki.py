"""Phase-2 WikiSkill bridge.

This module deliberately keeps the Wiki layer separate from the active skill
library. Background review records a small, redacted event; curator maintenance
turns unprocessed events into pending proposals. No function here edits
``$HERMES_HOME/skills`` or memory files.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home
from utils import atomic_json_write, atomic_write_text

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)((?:api[_ -]?key|token|secret|password)\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)
_SKILL_RE = re.compile(r"skill\s+['`\"]?([A-Za-z0-9][A-Za-z0-9_-]{1,63})", re.I)


def _wiki_root() -> Path:
    return get_hermes_home() / "wiki"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if not isinstance(value, str):
        return value
    result = value
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", result)
    return result


def _safe_slug(value: str, fallback: str = "event") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value or "").strip("-.")
    return (slug[:80] or fallback)


def _parse_time(now: str | None) -> datetime:
    if now:
        parsed = datetime.fromisoformat(now.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def record_review_event(
    *,
    actions: Iterable[str],
    session_id: str | None,
    outcome: str,
    evidence: Iterable[str] = (),
    now: str | None = None,
) -> Path:
    """Write one redacted background-review event under the active profile."""
    timestamp = _parse_time(now).astimezone(timezone.utc)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    month = timestamp.strftime("%Y-%m")
    sid = _safe_slug(session_id or "no-session")
    path = _wiki_root() / "raw" / month / f"{stamp}-{sid}.json"
    payload = {
        "event_type": "background_review",
        "recorded_at": timestamp.isoformat(),
        "session_id": session_id or "",
        "outcome": outcome,
        "actions": list(actions),
        "evidence": list(evidence),
    }
    atomic_json_write(path, _redact(payload), indent=2, sort_keys=True)
    return path


def _event_paths() -> list[Path]:
    root = _wiki_root() / "raw"
    return sorted(root.rglob("*.json")) if root.exists() else []


def _load_events() -> list[tuple[Path, dict[str, Any]]]:
    events: list[tuple[Path, dict[str, Any]]] = []
    for path in _event_paths():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("event_type") == "background_review":
            events.append((path, data))
    return events


def _proposal_target(event: dict[str, Any]) -> str | None:
    for action in event.get("actions", []) or []:
        match = _SKILL_RE.search(str(action))
        if match:
            return match.group(1)
    return None


def _proposal_path(event_path: Path, target: str) -> Path:
    proposal_id = f"{event_path.stem}-{_safe_slug(target)}"
    return _wiki_root() / "proposals" / f"{proposal_id}.md"


def _render_proposal(event_path: Path, event: dict[str, Any], target: str) -> str:
    actions = "\n".join(f"- {_redact(str(x))}" for x in event.get("actions", []) or [])
    evidence = "\n".join(f"- {_redact(str(x))}" for x in event.get("evidence", []) or []) or "- No explicit evidence reference recorded."
    return (
        "---\n"
        f"proposal_id: {event_path.stem}-{_safe_slug(target)}\n"
        "status: pending\n"
        f"target_skill: {target}\n"
        f"created: {event.get('recorded_at', '')[:10]}\n"
        "source: phase-2-wiki-bridge\n"
        "---\n\n"
        f"# Review proposal for `{target}`\n\n"
        "## Evidence\n"
        f"{evidence}\n\n"
        "## Observed review actions\n"
        f"{actions or '- No action summary recorded.'}\n\n"
        "## Intended Minimal Change\n"
        "A human or later validation pass should inspect the evidence and propose the smallest change to the target skill. This proposal has not modified the active skill.\n\n"
        "## Validation Plan\n"
        "- [ ] Record baseline behavior\n"
        "- [ ] Test candidate behavior\n"
        "- [ ] Check for regression\n"
        "- [ ] Obtain approval before applying any skill change\n\n"
        "## Risk / Rollback\n"
        "Keep the active skill unchanged until validation passes; use the existing skill backup/archive mechanism for rollback.\n"
    )


def _refresh_index(patterns: int, proposals: list[Path]) -> None:
    root = _wiki_root()
    pattern_files = sorted((root / "patterns").glob("*.md")) if (root / "patterns").exists() else []
    proposal_files = sorted((root / "proposals").glob("*.md")) if (root / "proposals").exists() else []
    all_proposals = sorted({*proposal_files, *proposals})
    index_path = root / "index.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    has_proposal_heading = "## Pending proposals" in existing or "## Proposal status" in existing
    if "## Patterns" not in existing or not has_proposal_heading:
        lines = [
            "# Hermes Experience Wiki\n",
            "\nProfile-local, proposal-only knowledge compiled from Hermes work.\n",
            "\n## Status vocabulary\n",
            "\n- `observed`: supported by a session or file reference, remedy not fully verified.\n- `verified`: remedy was exercised successfully with evidence.\n- `pending`: proposed skill change, not applied.\n",
            "\n## Patterns\n",
            "\n_No patterns recorded yet._",
            "\n\n## Pending proposals\n",
            "\n_No proposals recorded yet._",
        ]
        existing = "".join(lines) + "\n"

    missing_patterns = [
        f"- [{p.stem}]({p.relative_to(root).as_posix()})"
        for p in pattern_files
        if f"({p.relative_to(root).as_posix()})" not in existing
    ]
    missing_proposals = [
        f"- [{p.stem}]({p.relative_to(root).as_posix()})"
        for p in all_proposals
        if f"({p.relative_to(root).as_posix()})" not in existing
    ]
    if missing_patterns:
        marker = "\n\n## Pending proposals" if "## Pending proposals" in existing else "\n\n## Proposal status"
        insertion = "\n" + "\n".join(missing_patterns) + "\n"
        existing = existing.replace(marker, insertion + marker, 1)
    if missing_proposals:
        existing = existing.rstrip() + "\n" + "\n".join(missing_proposals) + "\n"
    atomic_write_text(index_path, existing)


def maintain_wiki() -> dict[str, int]:
    """Convert unprocessed review events into pending, idempotent proposals."""
    root = _wiki_root()
    proposals: list[Path] = []
    created = 0
    for event_path, event in _load_events():
        target = _proposal_target(event)
        if not target:
            continue
        proposal = _proposal_path(event_path, target)
        proposals.append(proposal)
        if proposal.exists():
            continue
        atomic_write_text(proposal, _render_proposal(event_path, event, target))
        created += 1

    root.joinpath("proposals").mkdir(parents=True, exist_ok=True)
    root.joinpath("patterns").mkdir(parents=True, exist_ok=True)
    _refresh_index(0, proposals)
    log_path = root / "logs.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Wiki Evolution Log\n"
    entry = f"\n## {datetime.now(timezone.utc).date().isoformat()}\n\n- Maintained Wiki from {len(_load_events())} background-review event(s); created {created} pending proposal(s).\n"
    atomic_write_text(log_path, existing.rstrip() + "\n" + entry)
    return {"events": len(_load_events()), "proposals_created": created, "proposals_total": len(proposals)}
