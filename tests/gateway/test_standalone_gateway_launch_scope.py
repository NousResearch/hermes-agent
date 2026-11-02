"""A standalone gateway (``multiplex_profiles`` off) keeps resolving the launch profile's credentials
after a native hosted room activated the process-wide secret guard (#112878).

``tui_gateway.launch_profile_policy.activate_multi_profile_hosting`` runs inside the messaging
gateway process when a hosted room serves a second profile; ``get_secret`` then fails closed for
every unscoped read. The standalone gateway's turns and handler entry points must bind the launch
profile's OWN scope (``.env`` over the env frozen at activation) — not skip binding because the
config flag is off, and not rebuild from ``.env`` alone (systemd / ``op run`` injection has no file).
"""
import asyncio
from contextlib import nullcontext
from unittest import mock

import pytest

from agent import secret_scope
from agent.secret_scope import UnscopedSecretError, current_secret_scope, get_secret
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from tui_gateway import launch_profile_policy

INJECTED = "HOSTEDROOM_TEST_INJECTED_KEY"


@pytest.fixture
def standalone(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / ".env").write_text("OPENAI_API_KEY=launch-dotenv-key\n", encoding="utf-8")
    secondary = tmp_path / "profiles" / "roomie"
    secondary.mkdir(parents=True)
    (secondary / ".env").write_text("OPENAI_API_KEY=secondary-key\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv(INJECTED, "launch-env-injected")  # systemd / `op run` style injection
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    return runner, secondary


def _keys():
    return get_secret("OPENAI_API_KEY"), get_secret(INJECTED)


def test_standalone_turn_binds_launch_profile_scope_after_hosted_activation(standalone):
    runner, secondary = standalone
    from tui_gateway.server import _session_profile_runtime_scope

    # A native hosted room ran a second profile: real activation, real secondary scope entered and left.
    launch_profile_policy.activate_multi_profile_hosting()
    with _session_profile_runtime_scope({"profile_home": str(secondary)}):
        assert get_secret("OPENAI_API_KEY") == "secondary-key"
        assert get_secret(INJECTED) is None  # the launch env never leaks into a secondary
    assert secret_scope.is_multiplex_active()

    source = mock.MagicMock(profile=None)
    with runner._profile_scope_for_source(source):
        assert _keys() == ("launch-dotenv-key", "launch-env-injected")
    runner._handle_message = mock.AsyncMock(side_effect=lambda event: _keys())
    assert asyncio.run(runner._primary_message_handler()(mock.MagicMock(source=source))) == (
        "launch-dotenv-key", "launch-env-injected")
    with pytest.raises(UnscopedSecretError):  # the guard itself is not weakened
        get_secret("OPENAI_API_KEY")


def test_standalone_without_hosted_activation_stays_unscoped(standalone):
    runner, _secondary = standalone
    source = mock.MagicMock(profile=None)
    assert isinstance(runner._profile_scope_for_source(source), nullcontext)
    runner._handle_message = mock.AsyncMock(side_effect=lambda event: current_secret_scope())
    assert asyncio.run(runner._primary_message_handler()(mock.MagicMock(source=source))) is None
