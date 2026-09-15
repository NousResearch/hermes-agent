from __future__ import annotations

import contextvars
from pathlib import Path
from typing import Optional

import hermes_state

from ._model import StateDBAdmissionError, StateDBFileIdentity


EXPECTED_GENERATION: contextvars.ContextVar[
    Optional[tuple[Path, StateDBFileIdentity]]
] = contextvars.ContextVar("gateway_state_db_expected_generation", default=None)
PENDING_FAILURE: contextvars.ContextVar[Optional[StateDBAdmissionError]] = (
    contextvars.ContextVar("gateway_state_db_pending_failure", default=None)
)

ORIGINAL_SESSION_DB = getattr(
    hermes_state.SessionDB,
    "_gateway_state_db_original_class",
    hermes_state.SessionDB,
)
ORIGINAL_TRACKED_CONNECT = getattr(
    hermes_state._connect_tracked_db,
    "__wrapped__",
    hermes_state._connect_tracked_db,
)
