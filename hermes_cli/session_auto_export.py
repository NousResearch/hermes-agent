"""Export a finished session to Markdown when ``session.auto_export`` is on.

``hermes sessions export`` already renders Markdown/QMD, redacts secrets and
appends a manifest entry — but it has to be run by hand. People who keep a
notes vault (Obsidian, Logseq, ...) end up with only the conversations they
remembered to export, which is precisely the wrong subset: the transcript you
want later is the one you did not think was worth keeping at the time.

This module is the unattended counterpart. Same renderer, same manifest, same
redaction — driven from the session-finalize hook instead of argv.

It stays off by default. Writing full transcripts to disk on every exit is a
privacy-relevant side effect, and a user who has not asked for it should never
discover their conversations already on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_FORMAT = "md"
SUPPORTED_FORMATS = ("md", "qmd")


def resolve_settings(config: Optional[dict[str, Any]] = None) -> tuple[bool, Path, str]:
    """Return ``(enabled, output_dir, fmt)`` for auto-export.

    Reads ``session.auto_export*``. An unset ``auto_export_dir`` resolves to
    ``$HERMES_HOME/session-exports`` — the same default ``hermes sessions
    export`` uses, so manual and automatic exports land in one directory
    sharing one manifest rather than drifting into two half-complete sets.
    """
    if config is None:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()

    session_cfg = config.get("session") if isinstance(config, dict) else None
    if not isinstance(session_cfg, dict):
        session_cfg = {}

    enabled = bool(session_cfg.get("auto_export", False))

    raw_dir = str(session_cfg.get("auto_export_dir") or "").strip()
    if raw_dir:
        output_dir = Path(raw_dir).expanduser()
    else:
        from hermes_constants import get_hermes_home

        output_dir = get_hermes_home() / "session-exports"

    fmt = str(session_cfg.get("auto_export_format") or DEFAULT_FORMAT).strip().lower()
    if fmt not in SUPPORTED_FORMATS:
        # Don't fail the export over a typo in the config — a transcript in the
        # wrong-but-valid format beats no transcript at shutdown, where nobody
        # is watching for an error message.
        logger.warning(
            "Unknown session.auto_export_format %r; falling back to %r",
            fmt,
            DEFAULT_FORMAT,
        )
        fmt = DEFAULT_FORMAT

    return enabled, output_dir, fmt


def export_finalized_session(
    session_id: str,
    *,
    config: Optional[dict[str, Any]] = None,
    db: Any = None,
) -> Optional[Path]:
    """Export one finalized session, or return ``None`` if nothing was written.

    ``None`` covers every "not applicable" case — auto-export disabled, unknown
    session, empty transcript — so the caller cannot tell a skip from a failure
    and does not need to: failures raise, and the finalize hook swallows them.

    ``db`` is injectable for tests; when omitted a short-lived ``SessionDB`` is
    opened and closed here.
    """
    session_id = str(session_id or "").strip()
    if not session_id:
        return None

    enabled, output_dir, fmt = resolve_settings(config)
    if not enabled:
        return None

    owns_db = db is None
    if owns_db:
        from hermes_state import SessionDB

        db = SessionDB()

    try:
        data = db.export_session(session_id)
        if not data:
            return None

        from hermes_cli.session_export_md import (
            append_manifest_entry,
            redact_session_data,
            write_session_markdown,
        )

        # A `hermes` invocation that never produced a turn is not a
        # conversation. Exporting it would bury the real transcripts under
        # empty files nobody asked for. (``export_session`` returns a flat
        # dict — only ``export_session_lineage`` builds ``segments`` — so a
        # plain message count is the whole story here.)
        if not (data.get("messages") or []):
            return None

        # Always redact. The manual command makes this a flag because the
        # operator is right there to judge; an unattended write into a synced
        # notes vault has no such supervision, so it takes the safe branch.
        data = redact_session_data(data)

        # force=True: a resumed session finalizes again with a longer
        # transcript, and the newer file is a superset of the older one.
        # Refusing to overwrite would freeze the export at the first exit.
        path = write_session_markdown(data, output_dir, fmt=fmt, force=True)
        append_manifest_entry(output_dir, data, path, fmt=fmt)
        return path
    finally:
        if owns_db:
            try:
                db.close()
            except Exception:
                logger.debug("Auto-export session DB close failed", exc_info=True)
