"""ACP configOptions advertisement: model picker + reasoning effort for JetBrains Air (#136084).

Air builds its model dropdown and reasoning-effort selector from ``configOptions``
(``SessionConfigOptionSelect``), not from the legacy ``models``/``modes`` fields, so a session
response without ``config_options`` leaves both controls unrendered. The adapter now advertises
a ``model`` select (ids are the same ``provider:model`` choices ``_switch_model`` accepts) and a
``reasoning_effort`` select over ``EFFORT_LADDER``, and ``session/set_config_option`` interprets
those two ids instead of stashing them as opaque strings.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from acp.schema import SessionConfigOptionSelect

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


@pytest.fixture()
def agent():
    manager = SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))
    return HermesACPAgent(session_manager=manager)


def _picker_payload() -> dict:
    return {
        "providers": [
            {"slug": "anthropic", "name": "Anthropic", "models": ["claude-sonnet-4-6"]},
            {
                "slug": "openai-codex",
                "name": "OpenAI Codex",
                "models": [{"id": "gpt-5.4"}],
            },
        ]
    }


class TestConfigOptionsAdvertisement:
    def test_new_session_advertises_model_and_effort_selects(self, agent):
        state = agent.session_manager.create_session(cwd="/tmp")
        state.agent.model, state.agent.provider = "gpt-5.4", "openai-codex"
        state.model = "gpt-5.4"

        with (
            patch(
                "hermes_cli.inventory.load_picker_context",
                return_value=MagicMock(
                    with_overrides=MagicMock(return_value=MagicMock())
                ),
            ),
            patch(
                "hermes_cli.inventory.build_models_payload",
                return_value=_picker_payload(),
            ),
        ):
            options = agent._session_config_options(state)

        assert options, "session responses must advertise configOptions"
        by_id = {opt.id: opt for opt in options}
        assert set(by_id) >= {"model", "reasoning_effort"}

        model = by_id["model"]
        assert isinstance(model, SessionConfigOptionSelect)
        assert model.type == "select"
        assert model.category == "model"
        assert model.current_value == "openai-codex:gpt-5.4"
        values = [choice.value for choice in model.options]
        assert (
            "openai-codex:gpt-5.4" in values and "anthropic:claude-sonnet-4-6" in values
        )

        effort = by_id["reasoning_effort"]
        assert isinstance(effort, SessionConfigOptionSelect)
        assert effort.category == "thought_level"
        ladder = [choice.value for choice in effort.options]
        assert (
            "none" in ladder
            and "low" in ladder
            and "medium" in ladder
            and "high" in ladder
        )
        assert effort.current_value in ladder

    def test_effort_current_value_reflects_agent_setting(self, agent):
        state = agent.session_manager.create_session(cwd="/tmp")
        state.agent.reasoning_config = {"enabled": True, "effort": "high"}

        options = agent._session_config_options(state)
        effort = next(opt for opt in options if opt.id == "reasoning_effort")
        assert effort.current_value == "high"

    def test_effort_current_value_defaults_when_unset(self, agent):
        state = agent.session_manager.create_session(cwd="/tmp")
        state.agent.reasoning_config = None

        options = agent._session_config_options(state)
        effort = next(opt for opt in options if opt.id == "reasoning_effort")
        assert effort.current_value == "medium"

    @pytest.mark.asyncio
    async def test_new_session_response_carries_config_options(self, agent):
        with patch.object(HermesACPAgent, "_session_config_options", return_value=[]):
            resp = await agent.new_session(cwd="/tmp")
        assert resp.config_options == []

        with patch.object(
            HermesACPAgent,
            "_session_config_options",
            return_value=[MagicMock(spec=SessionConfigOptionSelect)],
        ):
            resp = await agent.new_session(cwd="/tmp")
        assert resp.config_options is not None


class TestSetConfigOptionRouting:
    @pytest.mark.asyncio
    async def test_model_option_routes_to_switch_model(self, agent):
        resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(resp.session_id)
        state.agent.model, state.agent.provider, state.model = (
            "old-model",
            "openrouter",
            "old-model",
        )

        with patch.object(agent, "_switch_model") as switch:
            switch.return_value = ("openrouter", "anthropic", "claude-sonnet-4-6")
            update = await agent.set_config_option(
                "model", resp.session_id, "anthropic:claude-sonnet-4-6"
            )

        switch.assert_called_once()
        assert switch.call_args.args[0] is state
        assert switch.call_args.args[1] == "anthropic:claude-sonnet-4-6"
        assert switch.call_args.kwargs.get("keep_endpoint") is True
        assert update is not None

    @pytest.mark.asyncio
    async def test_effort_option_applies_reasoning_config(self, agent):
        resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(resp.session_id)
        state.agent.reasoning_config = None

        update = await agent.set_config_option(
            "reasoning_effort", resp.session_id, "high"
        )

        assert state.agent.reasoning_config == {"enabled": True, "effort": "high"}
        assert update is not None

    @pytest.mark.asyncio
    async def test_effort_none_disables_reasoning(self, agent):
        resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(resp.session_id)

        await agent.set_config_option("reasoning_effort", resp.session_id, "none")

        assert state.agent.reasoning_config == {"enabled": False}

    @pytest.mark.asyncio
    async def test_unknown_option_id_keeps_opaque_stash(self, agent):
        resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(resp.session_id)

        update = await agent.set_config_option(
            "some_client_option", resp.session_id, "value1"
        )

        assert getattr(state, "config_options", None) == {
            "some_client_option": "value1"
        }
        assert update is not None

    @pytest.mark.asyncio
    async def test_response_returns_full_option_set(self, agent):
        resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(resp.session_id)

        with patch.object(
            agent,
            "_session_config_options",
            return_value=[MagicMock(spec=SessionConfigOptionSelect, id="model")],
        ) as build:
            update = await agent.set_config_option(
                "reasoning_effort", resp.session_id, "low"
            )

        build.assert_called_once_with(state)
        assert update.config_options is not None


class TestReasoningEffortLifecycle:
    """An explicit effort picked through configOptions is a session setting: it must survive a
    process restart (``session/load`` from a real SessionDB) and an agent rebuild on model switch,
    instead of falling back to the config default the fresh agent is seeded with."""

    @staticmethod
    def _factory():
        return SimpleNamespace(
            model="fixture",
            provider="openrouter",
            reasoning_config={"enabled": True, "effort": "medium"},
        )

    @pytest.mark.asyncio
    async def test_effort_survives_restart_from_session_db(self, tmp_path):
        db = SessionDB(tmp_path / "state.db")
        server = HermesACPAgent(
            session_manager=SessionManager(db=db, agent_factory=self._factory)
        )
        resp = await server.new_session(cwd=str(tmp_path))
        state = server.session_manager.get_session(resp.session_id)
        # Content is what mints the row; empty sessions stay ephemeral.
        state.history.append({"role": "user", "content": "hello"})

        await server.set_config_option("reasoning_effort", resp.session_id, "high")

        # A new adapter process: nothing in memory, the agent is rebuilt from config.
        reloaded = HermesACPAgent(
            session_manager=SessionManager(db=db, agent_factory=self._factory)
        )
        restored = reloaded.session_manager.get_session(resp.session_id)
        assert restored is not None
        assert restored.reasoning_effort == "high"
        assert restored.agent.reasoning_config == {"enabled": True, "effort": "high"}
        effort = reloaded._effort_config_option(restored)
        assert effort.current_value == "high"
        db.close()

    @pytest.mark.asyncio
    async def test_effort_survives_model_switch(self, tmp_path):
        server = HermesACPAgent(
            session_manager=SessionManager(agent_factory=self._factory)
        )
        resp = await server.new_session(cwd=str(tmp_path))
        state = server.session_manager.get_session(resp.session_id)
        await server.set_config_option("reasoning_effort", resp.session_id, "none")
        old_agent = state.agent

        result = SimpleNamespace(
            success=True, target_provider="openrouter", new_model="other-model"
        )
        with (
            patch("hermes_cli.model_switch.switch_model", return_value=result),
            patch("hermes_cli.config.load_config", return_value={}),
            patch(
                "hermes_cli.config.get_compatible_custom_providers", return_value=[]
            ),
            patch(
                "hermes_cli.observability.shared_metrics_events.record_model_switch"
            ),
        ):
            server._switch_model(state, "openrouter:other-model")

        assert state.agent is not old_agent, "the switch rebuilds the agent"
        assert state.model == "other-model"
        assert state.agent.reasoning_config == {"enabled": False}

    @pytest.mark.asyncio
    async def test_effort_survives_model_switch_then_restart(self, tmp_path):
        """The reviewer's full chain: pick ``high``, switch model, reload in a fresh manager."""
        db = SessionDB(tmp_path / "state.db")
        server = HermesACPAgent(
            session_manager=SessionManager(db=db, agent_factory=self._factory)
        )
        resp = await server.new_session(cwd=str(tmp_path))
        state = server.session_manager.get_session(resp.session_id)
        state.history.append({"role": "user", "content": "hello"})
        await server.set_config_option("reasoning_effort", resp.session_id, "high")

        result = SimpleNamespace(
            success=True, target_provider="openrouter", new_model="other-model"
        )
        with (
            patch("hermes_cli.model_switch.switch_model", return_value=result),
            patch("hermes_cli.config.load_config", return_value={}),
            patch(
                "hermes_cli.config.get_compatible_custom_providers", return_value=[]
            ),
            patch(
                "hermes_cli.observability.shared_metrics_events.record_model_switch"
            ),
        ):
            server._switch_model(state, "openrouter:other-model")

        reloaded = HermesACPAgent(
            session_manager=SessionManager(db=db, agent_factory=self._factory)
        )
        restored = reloaded.session_manager.get_session(resp.session_id)
        assert restored.model == "other-model"
        assert restored.agent.reasoning_config == {"enabled": True, "effort": "high"}
        assert reloaded._effort_config_option(restored).current_value == "high"
        db.close()

    @pytest.mark.asyncio
    async def test_fork_keeps_effort(self, tmp_path):
        server = HermesACPAgent(
            session_manager=SessionManager(agent_factory=self._factory)
        )
        resp = await server.new_session(cwd=str(tmp_path))
        await server.set_config_option("reasoning_effort", resp.session_id, "low")

        forked = server.session_manager.fork_session(resp.session_id, cwd=str(tmp_path))

        assert forked.reasoning_effort == "low"
        assert forked.agent.reasoning_config == {"enabled": True, "effort": "low"}
