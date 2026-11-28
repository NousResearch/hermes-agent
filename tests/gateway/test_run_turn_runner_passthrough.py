"""The gateway turn projection must carry the proxy-reported models to the footer (#54864)."""
from __future__ import annotations

from gateway.run_turn_runner import _LEDGER_PASSTHROUGH_KEYS


def test_projection_passes_through_footer_model_fields():
    # run_turn_persistence builds the runtime footer from agent_result["requested_model"] and
    # ["served_model"] (set by agent.turn_finalizer). The gateway projection copies only these
    # keys out of the run_conversation result, so without them the opt-in `served_model`
    # footer field is always empty on the gateway path.
    assert {"requested_model", "served_model"} <= set(_LEDGER_PASSTHROUGH_KEYS)
