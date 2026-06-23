"""Autopilot decision log — an append-only ADR (Architecture Decision Record)
trail of every judgment autopilot makes on a run.

When autopilot decides whether a goal is complete, whether to keep going, or how
to auto-answer a ``clarify`` question, that decision is made by an INDEPENDENT
reviewer (the Hermes Council when available, otherwise a single auxiliary
reviewer). Those decisions are exactly the moments a human would normally be in
the loop, so this module records each one to a human-readable markdown file the
user can review after an unattended run:

    * what was sent for verification (the goal + work context + candidate result),
    * what the reviewer returned (verdict, confidence, the gap it found, and the
      specific checks it said were required to reach a passing state),
    * which options were on the table and which path autopilot took.

It is OFF by default. Enable it with ``autopilot.adr: true`` in config (the CLI
bridges this to ``HERMES_AUTOPILOT_ADR=1``) or ``HERMES_AUTOPILOT_ADR=1`` in the
environment. The file is a local artifact under the workspace; it is never part
of a request to a model and never shipped anywhere.

Design rules:
    * Pure append. Every decision is one new markdown section; nothing is ever
      rewritten, so a record is lossless and cheap.
    * Fail-soft. Any IO or formatting error is logged at debug and swallowed —
      an ADR problem must never break an autopilot run.
    * Self-contained. No imports from the rest of autopilot, so it can be unit
      tested in isolation.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def adr_enabled(agent: Any = None) -> bool:
    """True when the autopilot ADR decision log is turned on.

    Reads the per-agent attribute first (set from ``config.autopilot.adr`` by the
    CLI bridge), then the ``HERMES_AUTOPILOT_ADR`` environment variable.
    """
    if agent is not None:
        val = getattr(agent, "_autopilot_adr", None)
        if val is not None:
            return bool(val)
    return os.environ.get("HERMES_AUTOPILOT_ADR", "").strip().lower() in _TRUTHY


def _session_id(agent: Any) -> str:
    for attr in ("session_id", "_session_id", "_autopilot_session_id"):
        val = getattr(agent, attr, "") if agent is not None else ""
        if val:
            return str(val)[:40]
    return "session"


def adr_path(agent: Any = None) -> Path:
    """Resolve the ADR file path.

    Priority: explicit per-agent attribute / ``AUTOPILOT_ADR_PATH`` env override,
    else ``<workspace>/.hermes/autopilot/adr/AUTOPILOT-<session>-<YYYYMMDD>.md``.
    The workspace root falls back to the current working directory.
    """
    override = ""
    if agent is not None:
        override = str(getattr(agent, "_autopilot_adr_path", "") or "")
    override = override or os.environ.get("AUTOPILOT_ADR_PATH", "").strip()
    if override:
        return Path(override).expanduser()

    root = os.environ.get("HERMES_WORKSPACE", "").strip() or os.getcwd()
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path(root) / ".hermes" / "autopilot" / "adr" / f"AUTOPILOT-{_session_id(agent)}-{day}.md"


def _trunc(text: Any, limit: int) -> str:
    s = "" if text is None else str(text).strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f" …[+{len(s) - limit} chars]"


def _fmt_options(options: Optional[Sequence[str]]) -> str:
    opts = [str(o).strip() for o in (options or []) if str(o).strip()]
    if not opts:
        return "_(open-ended; no preset options)_"
    return "\n".join(f"  - {o}" for o in opts)


def _ensure_header(path: Path) -> None:
    """Write the file header once, on first record."""
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    path.write_text(
        f"# Autopilot decision log\n\n"
        f"Started {started}. Each section below is one decision made by the "
        f"independent reviewer (Hermes Council when available, otherwise the "
        f"auxiliary reviewer / options fallback). Autopilot always took the "
        f"recommended path; this log lets you review what the alternatives were "
        f"and why each path was chosen.\n",
        encoding="utf-8",
    )


def record_decision(
    agent: Any,
    *,
    kind: str,
    goal: str = "",
    sent_for_verification: str = "",
    options: Optional[Sequence[str]] = None,
    chosen: str = "",
    verdict: str = "",
    confidence: float = 0.0,
    gap: str = "",
    required_checks: str = "",
    rationale: str = "",
    source: str = "",
) -> Optional[Path]:
    """Append one decision record to the ADR file. Returns the path, or None.

    ``kind`` is one of ``completion`` | ``continue`` | ``clarify``. Every field is
    optional so both the Council lane (rich verdict + gap + checks) and the
    fallback lane (options + recommended choice) record uniformly. Never raises.
    """
    try:
        if not adr_enabled(agent):
            return None
        path = adr_path(agent)
        _ensure_header(path)

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        lines: list[str] = [f"\n## {ts} — {kind}"]
        lines.append(f"- reviewer: {source or 'unknown'}")
        if verdict:
            conf = f" (confidence {confidence:.2f})" if confidence else ""
            lines.append(f"- verdict: {verdict}{conf}")
        if goal:
            lines.append(f"- goal: {_trunc(goal, 600)}")
        if sent_for_verification:
            lines.append(
                "- sent for verification:\n\n```\n"
                + _trunc(sent_for_verification, 2000)
                + "\n```"
            )
        if gap:
            lines.append(f"- gap found / why not passing: {_trunc(gap, 800)}")
        if required_checks:
            lines.append(f"- required to pass: {_trunc(required_checks, 800)}")
        if options is not None:
            lines.append("- options considered:\n" + _fmt_options(options))
        if chosen:
            lines.append(f"- chosen path: {_trunc(chosen, 600)}")
        if rationale:
            lines.append(f"- rationale: {_trunc(rationale, 600)}")

        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return path
    except Exception as exc:  # noqa: BLE001 — ADR must never break a run
        logger.debug("autopilot: ADR record failed (%s)", exc)
        return None
