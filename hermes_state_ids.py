"""Session-id minting: the ONE place that knows the ``YYYYMMDD_HHMMSS_<hex>`` shape.

stdlib-only on purpose: ``agent/``, ``cli.py``, ``gateway/`` and ``tui_gateway/`` all mint ids and
must not pull the SessionDB import graph in to do it. ``hermes_cli/session_lost_and_found.py``
classifies schema-less salvage rows by ``is_known_session_id``, so a shape change here is a
recovery-classification change — keep the prefix stable, and add any NEW derived-id shape to
``SESSION_ID_RECOGNIZERS`` or salvage will drop those sessions and mis-infer the table's layout.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Optional

SESSION_ID_PATTERN = re.compile(r"^\d{8}_\d{6}_")

# Not every session id is minted by ``new_session_id``. Several first-class surfaces derive an id
# from something they already have (a job id, a clock, a room hash) so the id itself carries the
# provenance. Salvage classifies schema-less rows by these shapes, so a surface missing here (a)
# has its sessions DROPPED by ``hermes sessions recover`` (classify returns None) and (b) poisons
# layout inference for the WHOLE table via ``parent_session_id``: an ordinary child session of a
# cron job classifies fine but carries the unrecognised parent id, and one bad sampled value vetoes
# every candidate layout, which drops recovery onto the positional-guessing fallback. Measured on a
# real 3,373-session store: 379 rows (11.2%) -- 375 cron + 1 bg + junk -- were silently discarded.
#
# Keep each entry anchored on the parts the minting site actually fixes; these are layout sentinels,
# so a loose pattern costs real wrong-column-mapping safety.
SESSION_ID_RECOGNIZERS = (
    SESSION_ID_PATTERN,
    # cron/scheduler.py: f"cron_{job_id}_{now:%Y%m%d_%H%M%S}". job_id is uuid4().hex[:12] for jobs
    # minted by cron/jobs.py, but user/legacy job ids ("job-1") are real and still in live stores.
    re.compile(r"^cron_[A-Za-z0-9][A-Za-z0-9._-]*_\d{8}_\d{6}$"),
    # hermes_cli/cli_commands_mixin.py (/bg): f"bg_{now:%H%M%S}_{uuid4().hex[:6]}" -- time only, no date.
    re.compile(r"^bg_\d{6}_[0-9a-f]{6}$"),
    # gateway/platforms/api_server_room_dispatch.py: f"room_{sha256(seed)[:32]}".
    re.compile(r"^room_[0-9a-f]{32}$"),
)


def is_known_session_id(value: str) -> bool:
    """True when ``value`` has the shape of an id one of this repo's surfaces really mints.

    The recognition side of the minting contract. Salvage uses this (not
    ``SESSION_ID_PATTERN`` alone) to tell a session id from an arbitrary cell.
    """
    return any(pattern.match(value) for pattern in SESSION_ID_RECOGNIZERS)

# Interactive surfaces (CLI, TUI, agent, branches, imports) share 6 hex chars — the Desktop's
# session-id candidate regex is pinned to that width. Gateway keys are 8, portability imports 12
# (many rows minted in the same second).
DEFAULT_HEX_LEN = 6


def new_session_id(now: Optional[datetime] = None, *, hex_len: int = DEFAULT_HEX_LEN) -> str:
    """``<timestamp>_<random hex>`` for a fresh session; ``now`` pins the timestamp to a clock the
    caller already captured (``agent.session_start``) so the id and the row agree to the second."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:hex_len]}"
