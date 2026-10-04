"""Session-id minting and filesystem-safe artifact naming.

stdlib-only on purpose: ``agent/``, ``cli.py``, ``gateway/`` and ``tui_gateway/`` all mint ids and
must not pull the SessionDB import graph in to do it. ``hermes_cli/session_lost_and_found.py``
classifies schema-less salvage rows by ``SESSION_ID_PATTERN``, so a shape change here is a
recovery-classification change — keep the prefix stable. Session ids are otherwise opaque logical
keys; filesystem sinks must derive a component here rather than interpret the key as a path.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

SESSION_ID_PATTERN = re.compile(r"^\d{8}_\d{6}_")

# Interactive surfaces (CLI, TUI, agent, branches, imports) share 6 hex chars — the Desktop's
# session-id candidate regex is pinned to that width. Gateway keys are 8, portability imports 12
# (many rows minted in the same second).
DEFAULT_HEX_LEN = 6


def new_session_id(now: Optional[datetime] = None, *, hex_len: int = DEFAULT_HEX_LEN) -> str:
    """``<timestamp>_<random hex>`` for a fresh session; ``now`` pins the timestamp to a clock the
    caller already captured (``agent.session_start``) so the id and the row agree to the second."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:hex_len]}"


def session_id_storage_name(session_id: object) -> str:
    """Stable, single-component storage name for an opaque session id.

    Existing single-component word-character ids keep their historical filenames. Any id that
    needs normalization receives a content hash so normalization does not collapse ordinary
    distinct logical ids onto one artifact.
    """
    raw = "" if session_id is None else str(session_id)
    sanitized = re.sub(r"[^\w-]", "_", raw).strip("._")[:96] or "session"
    if raw and sanitized == raw:
        return sanitized
    digest = hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    return f"{sanitized}_{digest}"


def session_artifact_path(
    sessions_dir: Path, session_id: object, *, prefix: str = "", suffix: str = "",
) -> Path:
    """Build one session artifact path and fail closed unless it resolves directly under the root."""
    root = Path(sessions_dir)
    path = root / f"{prefix}{session_id_storage_name(session_id)}{suffix}"
    if path.resolve(strict=False).parent != root.resolve(strict=False):
        raise ValueError("session artifact path must remain directly inside the sessions directory")
    return path
