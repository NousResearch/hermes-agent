"""Regression tests for #133955: ACP must pin HERMES_HOME for prompt builds.

An overlay home (e.g. Multica) symlinks ``<home>/state.db`` at a per-conversation
store. The state registry stores the RESOLVED path, so ``_agent_home``'s db-parent
fallback derives the store folder as the agent home and SOUL.md/memories/skills
load from the wrong tree. The turn body and the slash-command dispatch both pin
the launch home so every system-prompt build reads the overlay home.
"""

import pytest

from acp_adapter.session import SessionState


@pytest.fixture()
def overlay_home(tmp_path, monkeypatch):
    """An overlay HERMES_HOME whose state.db is a symlink into a store folder."""
    overlay = tmp_path / "hermes-home"
    overlay.mkdir()
    store = tmp_path / "store"
    store.mkdir()
    real_db = store / "state.db"
    real_db.write_bytes(b"")
    (overlay / "state.db").symlink_to(real_db)
    monkeypatch.setenv("HERMES_HOME", str(overlay))
    return overlay, store


class _StubAgent:
    """Mimics the resolved-registry shape (``_session_db.db_path`` is the store
    path the symlink resolves to) and records what the turn saw."""

    session_id = "sess"

    def __init__(self, resolved_db_path):
        self._session_db = type("Db", (), {"db_path": resolved_db_path})()
        self.seen = {}

    def run_conversation(self, **kwargs):
        from agent.system_prompt import _agent_home
        from hermes_constants import get_hermes_home_override

        self.seen["override"] = get_hermes_home_override()
        self.seen["home"] = _agent_home(self)
        return {"final_response": "ok", "messages": []}


class TestTurnBindsHomeOverride:
    def test_agent_home_is_the_overlay_not_the_store(self, overlay_home):
        from acp_adapter.server import HermesACPAgent

        overlay, store = overlay_home
        agent = _StubAgent(store / "state.db")
        state = SessionState(session_id="sess", agent=agent, cwd=".", model="m")
        server = HermesACPAgent.__new__(HermesACPAgent)

        result = server._run_agent_turn(
            state=state,
            session_id="sess",
            user_text="hi",
            user_content="hi",
            conn=None,
            loop=None,
            approval_cb=None,
            edit_approval_requester=None,
        )

        assert result["final_response"] == "ok"
        assert agent.seen["override"] == str(overlay)
        assert agent.seen["home"] == overlay

    def test_db_parent_fallback_without_override_reads_the_store(self, overlay_home):
        # The fallback's own contract, locked separately: with no override it follows
        # the resolved symlink into the store folder — which is why ACP must pin.
        overlay, store = overlay_home
        from agent.system_prompt import _agent_home

        assert _agent_home(_StubAgent(store / "state.db")) == store


class TestSlashCommandBindsHomeOverride:
    def test_dispatch_pins_home_for_prompt_rebuilds(self, overlay_home):
        from acp_adapter.commands import SlashCommandsMixin

        overlay, store = overlay_home
        captured = {}

        class _Host(SlashCommandsMixin):
            _COMMANDS = {"probe": ("probe", "probe", None)}

            def _cmd_probe(self, args, state):
                from hermes_constants import get_hermes_home_override

                captured["override"] = get_hermes_home_override()
                return "probed"

        host = _Host()
        state = SessionState(
            session_id="sess", agent=_StubAgent(store / "state.db"), cwd=".", model="m"
        )
        assert host._handle_slash_command("/probe", state) == "probed"
        assert captured["override"] == str(overlay)
