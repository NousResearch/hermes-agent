"""Session-id minting: the ONE place that knows the ``YYYYMMDD_HHMMSS_<hex>`` shape.

stdlib-only on purpose: ``agent/``, ``cli.py``, ``gateway/`` and ``tui_gateway/`` all mint ids and
must not pull the SessionDB import graph in to do it. ``hermes_cli/session_lost_and_found.py``
classifies schema-less salvage rows by ``SESSION_ID_PATTERN``, so a shape change here is a
recovery-classification change — keep the prefix stable.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
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
