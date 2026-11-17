"""Stable, dependency-free constants shared by the state-store layers.

Kept separate from ``hermes_state`` (the heavy SessionDB facade) and from
``hermes_state_common`` (which pulls agent modules): the worker persistence facade
(``agent/runtime_session_store.py``) imports this at module level on CLI/worker
startup paths, where test seams replace ``hermes_state.SessionDB`` and a
heavyweight import would ride every worker start. Do not add imports here.
"""

# How long a transcript write may wait out a sibling writer before the store
# reports busy (``SessionDB._TRANSCRIPT_WRITE_PATIENCE_S``). The worker receipt
# budget is sourced from this same value: the gateway applies a receipt as such
# a write, so the client must never time out before the server's own patience.
TRANSCRIPT_WRITE_PATIENCE_S = 60.0
