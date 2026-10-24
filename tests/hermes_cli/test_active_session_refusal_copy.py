"""Refusal reconstruction must preserve the machine-readable admission contract."""
import copy
import pickle

import pytest

from hermes_cli.active_sessions import (
    MAX_CONCURRENT_SESSIONS,
    SESSION_COORDINATION_UNAVAILABLE,
    SESSION_NOT_OWNED,
    ActiveSessionRefusal,
)


@pytest.mark.parametrize("reason", [
    MAX_CONCURRENT_SESSIONS, SESSION_NOT_OWNED, SESSION_COORDINATION_UNAVAILABLE,
])
@pytest.mark.parametrize("operation", ["copy", "deepcopy", "pickle"])
def test_refusal_reconstruction_preserves_message_and_reason(reason, operation):
    refusal = ActiveSessionRefusal("Session unavailable.", reason)
    if operation == "pickle":
        # Round-trip only this locally constructed value, never untrusted input.
        restored = pickle.loads(pickle.dumps(refusal))
    else:
        restored = getattr(copy, operation)(refusal)

    assert isinstance(restored, ActiveSessionRefusal)
    assert restored == "Session unavailable."
    assert restored.reason == reason
