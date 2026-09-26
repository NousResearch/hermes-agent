"""ACP permission mode must survive save/load (process restart or session/load).

``set_session_mode`` / ``set_config_option`` write ``state.mode`` then call
``save_session()``, but the persistence layer never included the mode field in
the stored ``model_config`` JSON — so after a restart/restore the session
silently fell back to the default mode ("default"/ask), while ACP clients kept
showing the user-selected "Don't Ask" in their UI. Invariants:

1. Saving a session whose mode was switched must carry ``mode`` in the
   persisted model_config metadata.
2. Restoring such a session from the DB brings the mode back, and the
   edit-approval policy derived from it is NOT the "ask" fallback.
3. Legacy rows without a ``mode`` key restore cleanly to the default
   ("ask") — no crash, no invented value.
4. fork_session inherits the original session's mode.
"""

import asyncio
import json

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager

from tests.acp_adapter.test_acp_commands import FakeAgent


class RecordingDb:
    """Minimal SessionDB stand-in that records what _persist writes and can
    hand rows back to _restore."""

    def __init__(self):
        self.rows = {}  # session_id -> row dict

    def get_session(self, session_id, *_args, **_kwargs):
        return self.rows.get(session_id)

    def create_session(self, session_id, source, model=None, model_config=None, **_kwargs):
        meta_json = json.dumps(model_config) if model_config is not None else None
        self.rows[session_id] = {
            "id": session_id,
            "source": source,
            "model": model,
            "model_config": meta_json,
        }
        return session_id

    def update_session_meta(self, session_id, model_config_json, model=None):
        row = self.rows.get(session_id)
        if row is not None:
            row["model_config"] = model_config_json

    def replace_messages(self, *_args, **_kwargs):
        return None

    def get_messages_as_conversation(self, *_args, **_kwargs):
        return []


def _make_manager(db):
    fake = FakeAgent()
    return SessionManager(agent_factory=lambda **_kwargs: fake, db=db)


def _make_live_session(manager):
    """A session with one history message: contentless editor probes stay
    ephemeral in _persist by design, so persistence tests need history."""
    state = manager.create_session(cwd=".")
    state.history.append({"role": "user", "content": "hello"})
    return state


def test_persist_carries_mode_in_model_config():
    """set_session_mode -> save_session must write mode into stored metadata."""
    db = RecordingDb()
    manager = _make_manager(db)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _make_live_session(manager)
    manager._persist(state)  # materialize the row (a turn would normally do this)

    asyncio.run(acp_agent.set_session_mode(mode_id="dont_ask", session_id=state.session_id))

    row = db.get_session(state.session_id)
    meta = json.loads(row["model_config"])
    assert meta.get("mode") == "dont_ask", (
        "mode missing from persisted model_config — set_session_mode's "
        "save_session() would again persist nothing"
    )


def test_restore_recovers_mode_and_policy():
    """After memory is dropped (simulated restart), the restored session keeps
    dont_ask and the edit-approval policy is the auto-allow one, not ask."""
    db = RecordingDb()
    manager = _make_manager(db)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _make_live_session(manager)
    manager._persist(state)

    asyncio.run(acp_agent.set_session_mode(mode_id="dont_ask", session_id=state.session_id))

    # Simulate process restart: wipe in-memory sessions only, DB row survives.
    manager._sessions.clear()

    restored = manager.get_session(state.session_id)
    assert restored is not None
    assert restored.mode == "dont_ask"

    policy, _cwd = acp_agent._edit_approval_policy_for_state(restored)
    assert policy == "session", f"expected dont_ask->session policy, got {policy!r}"

    modes = acp_agent._session_modes(restored)
    assert modes.current_mode_id == "dont_ask"


def test_accept_edits_mode_round_trips():
    db = RecordingDb()
    manager = _make_manager(db)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _make_live_session(manager)
    manager._persist(state)

    asyncio.run(acp_agent.set_session_mode(mode_id="accept_edits", session_id=state.session_id))
    manager._sessions.clear()

    restored = manager.get_session(state.session_id)
    assert restored.mode == "accept_edits"
    policy, _ = acp_agent._edit_approval_policy_for_state(restored)
    assert policy == "workspace_session"


def test_legacy_row_without_mode_key_restores_default():
    """Pre-fix rows (no mode in model_config) must restore to the default ask
    behavior without raising."""
    db = RecordingDb()
    manager = _make_manager(db)
    acp_agent = HermesACPAgent(session_manager=manager)

    session_id = "legacy-session"
    db.rows[session_id] = {
        "id": session_id,
        "source": "acp",
        "model": "fake-model",
        "model_config": json.dumps({"cwd": "/tmp"}),  # no "mode" key at all
    }

    restored = manager.get_session(session_id)
    assert restored is not None
    assert restored.mode == ""
    policy, cwd = acp_agent._edit_approval_policy_for_state(restored)
    assert policy == "ask"
    assert cwd == "/tmp"


def test_fork_session_inherits_mode():
    db = RecordingDb()
    manager = _make_manager(db)
    acp_agent = HermesACPAgent(session_manager=manager)
    state = _make_live_session(manager)
    manager._persist(state)

    asyncio.run(acp_agent.set_session_mode(mode_id="dont_ask", session_id=state.session_id))

    forked = manager.fork_session(state.session_id, cwd=".")
    assert forked is not None
    assert forked.mode == "dont_ask"
    # and the fork's own DB row carries it too
    fork_meta = json.loads(db.get_session(forked.session_id)["model_config"])
    assert fork_meta.get("mode") == "dont_ask"
