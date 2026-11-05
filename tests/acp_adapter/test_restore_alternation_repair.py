"""ACP session resume heals durable alternation violations before live replay.

A turn that persists a user row but no assistant row leaves a ``user;user`` pair in the store; the restored history
becomes the resumed agent's working conversation, so it must be alternation-clean or the pre-request repair re-fires
on every turn.
"""

import itertools

from acp_adapter.session import SessionManager
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


def _seed_wedged_acp_session(store, session_id="acp1"):
    seed_session(store, session_id, "acp")
    seed_messages(store, session_id,
                  {"role": "user", "content": "first ask"}, {"role": "assistant", "content": "first reply"},
                  {"role": "user", "content": "unanswered turn"}, {"role": "user", "content": "next turn"},
                  {"role": "assistant", "content": "next reply"})


def test_acp_restore_heals_alternation_for_live_replay(tmp_path):
    """acp_adapter.SessionManager._restore feeds LIVE REPLAY: the loaded history
    becomes the resumed agent's working conversation. It must be alternation-
    clean so the pre-request repair doesn't re-fire every turn."""
    store = open_persistence(profile_scope(home=tmp_path))
    _seed_wedged_acp_session(store, "acp1")

    class _StubAgent:
        model = "stub"

    mgr = SessionManager(agent_factory=lambda: _StubAgent(), db=store)
    state = mgr._restore("acp1")

    assert state is not None
    roles = [m["role"] for m in state.history]
    # No consecutive user turns: the durable user;user wedge was healed.
    assert roles == ["user", "assistant", "user", "assistant"], roles
    for a, b in itertools.pairwise(roles):
        assert not (a == "user" and b == "user"), "unhealed user;user in ACP live replay"
    # No user input lost: both user texts survive, merged in order.
    merged = state.history[2]["content"]
    assert "unanswered turn" in merged and "next turn" in merged
