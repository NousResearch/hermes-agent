"""Tests for the ACP ``configOptions`` model selector (category: model).

Newer ACP clients (e.g. Zed) render their per-session model picker from
``configOptions`` rather than the legacy ``SessionModelState`` field. These
tests assert that Hermes advertises a ``category: "model"`` select option on
session creation and that ``session/set_config_option`` performs a
session-scoped model switch without mutating the global default.
"""

from types import SimpleNamespace

import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager

from tests.acp_adapter.test_acp_commands import (
    CaptureConn,
    FakeAgent,
    NoopDb,
    make_agent_and_state,
)


def _install_model_state(monkeypatch, current_id="fake-provider:model-a"):
    """Force a deterministic model inventory for _build_model_state."""
    from acp.schema import ModelInfo, SessionModelState

    def fake_build(self, state):
        return SessionModelState(
            available_models=[
                ModelInfo(
                    model_id="fake-provider:model-a",
                    name="Fake Provider · model-a",
                    description="Provider: Fake Provider • current",
                ),
                ModelInfo(
                    model_id="fake-provider:model-b",
                    name="Fake Provider · model-b",
                    description="Provider: Fake Provider",
                ),
            ],
            current_model_id=current_id,
        )

    monkeypatch.setattr(HermesACPAgent, "_build_model_state", fake_build)


@pytest.mark.asyncio
async def test_new_session_advertises_model_config_option(monkeypatch):
    _install_model_state(monkeypatch)
    fake = FakeAgent()
    manager = SessionManager(agent_factory=lambda **kwargs: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)

    response = await acp_agent.new_session(cwd=".")

    assert response.config_options, "expected configOptions on new_session"
    model_opt = next(
        (o for o in response.config_options if o.id == "model"), None
    )
    assert model_opt is not None
    assert model_opt.category == "model"
    assert model_opt.type == "select"
    assert model_opt.current_value == "fake-provider:model-a"
    assert [o.value for o in model_opt.options] == [
        "fake-provider:model-a",
        "fake-provider:model-b",
    ]


@pytest.mark.asyncio
async def test_new_session_still_sends_legacy_models_field(monkeypatch):
    """The legacy ``models`` field must remain for older ACP clients."""
    _install_model_state(monkeypatch)
    fake = FakeAgent()
    manager = SessionManager(agent_factory=lambda **kwargs: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)

    response = await acp_agent.new_session(cwd=".")

    assert response.models is not None
    assert response.models.available_models


@pytest.mark.asyncio
async def test_set_model_config_option_switches_session_model(monkeypatch):
    _install_model_state(monkeypatch)
    acp_agent, state, _fake, _conn = make_agent_and_state()

    # Route _resolve_model_selection deterministically (avoid live catalogs).
    monkeypatch.setattr(
        HermesACPAgent,
        "_resolve_model_selection",
        staticmethod(lambda raw, prov: ("fake-provider", raw.split(":", 1)[-1])),
    )

    response = await acp_agent.set_config_option(
        session_id=state.session_id,
        config_id="model",
        value="fake-provider:model-b",
    )

    # Session state reflects the switch.
    assert state.model == "model-b"
    # Response echoes the full configOptions list (spec requires complete set).
    assert response.config_options
    assert any(o.id == "model" for o in response.config_options)


@pytest.mark.asyncio
async def test_set_model_config_option_does_not_write_global_config(monkeypatch):
    """A model configOption switch must never persist to config.yaml."""
    _install_model_state(monkeypatch)
    acp_agent, state, _fake, _conn = make_agent_and_state()

    monkeypatch.setattr(
        HermesACPAgent,
        "_resolve_model_selection",
        staticmethod(lambda raw, prov: ("fake-provider", raw.split(":", 1)[-1])),
    )

    import hermes_cli.config as hermes_config

    saved = {"called": False}

    def _tripwire(*_args, **_kwargs):
        saved["called"] = True
        raise AssertionError("config.yaml must not be written on session model switch")

    # Any attempt to persist global config during the switch fails the test.
    if hasattr(hermes_config, "save_config"):
        monkeypatch.setattr(hermes_config, "save_config", _tripwire, raising=False)
    if hasattr(hermes_config, "save_config_value"):
        monkeypatch.setattr(hermes_config, "save_config_value", _tripwire, raising=False)

    await acp_agent.set_config_option(
        session_id=state.session_id,
        config_id="model",
        value="fake-provider:model-b",
    )

    assert saved["called"] is False
    assert state.model == "model-b"


@pytest.mark.asyncio
async def test_set_edit_approval_policy_still_returns_config_options(monkeypatch):
    """Non-model config ids keep working and now echo the full option set."""
    _install_model_state(monkeypatch)
    acp_agent, state, _fake, _conn = make_agent_and_state()

    response = await acp_agent.set_config_option(
        session_id=state.session_id,
        config_id="edit_approval_policy",
        value="accept_edits",
    )

    assert response.config_options is not None


@pytest.mark.asyncio
async def test_config_option_response_always_includes_model_option(monkeypatch):
    """The set_config_option response must carry the complete option set.

    Zed and other ACP clients refresh their selectors from the full
    configOptions list returned by session/set_config_option, so the model
    option must be present after both a model switch and a non-model
    (edit-approval) update.
    """
    _install_model_state(monkeypatch)
    acp_agent, state, _fake, _conn = make_agent_and_state()

    monkeypatch.setattr(
        HermesACPAgent,
        "_resolve_model_selection",
        staticmethod(lambda raw, prov: ("fake-provider", raw.split(":", 1)[-1])),
    )

    after_model = await acp_agent.set_config_option(
        session_id=state.session_id,
        config_id="model",
        value="fake-provider:model-b",
    )
    assert any(o.id == "model" for o in after_model.config_options)

    after_edit = await acp_agent.set_config_option(
        session_id=state.session_id,
        config_id="edit_approval_policy",
        value="accept_edits",
    )
    assert any(o.id == "model" for o in after_edit.config_options)
