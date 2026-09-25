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
from hermes_constants import get_hermes_home
from tools.terminal_scope import terminal_env
from tui_gateway import launch_profile_policy

INJECTED = "HOSTEDROOM_TEST_INJECTED_KEY"


@pytest.fixture
def standalone(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / ".env").write_text("OPENAI_API_KEY=launch-dotenv-key\n", encoding="utf-8")
    (launch / "config.yaml").write_text(
        "terminal:\n  ssh_host: launch.example\n", encoding="utf-8"
    )
    secondary = tmp_path / "profiles" / "roomie"
    secondary.mkdir(parents=True)
    (secondary / ".env").write_text("OPENAI_API_KEY=secondary-key\n", encoding="utf-8")
    (secondary / "config.yaml").write_text(
        "terminal:\n  ssh_host: secondary.example\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv(INJECTED, "launch-env-injected")  # systemd / `op run` style injection
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_authority", None)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    return runner, secondary


def _keys():
    return get_secret("OPENAI_API_KEY"), get_secret(INJECTED)


def _scope_values():
    return *_keys(), terminal_env("TERMINAL_SSH_HOST"), str(get_hermes_home())


def test_launch_authority_never_reobserves_process_home(monkeypatch):
    """A stripped snapshot fails closed instead of mixing in a later live home."""
    import hermes_constants

    calls = []

    def _resolve(env=None):
        calls.append(env)
        if env is not None:
            raise ValueError("snapshot has no platform home")
        return "/late-process-home"

    monkeypatch.setattr(launch_profile_policy, "_authority", None)
    monkeypatch.setattr(hermes_constants, "get_process_hermes_home", _resolve)

    with pytest.raises(ValueError, match="snapshot has no platform home"):
        launch_profile_policy.capture_launch_authority()
    assert len(calls) == 1 and calls[0] is not None


def test_captured_launch_authority_is_immediately_authoritative(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    poison = tmp_path / "poison"
    launch.mkdir()
    poison.mkdir()
    monkeypatch.setattr(launch_profile_policy, "_authority", None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("CAPTURED_ONLY", "launch")

    captured = launch_profile_policy.capture_launch_authority()
    monkeypatch.setenv("HERMES_HOME", str(poison))
    monkeypatch.setenv("CAPTURED_ONLY", "poison")

    observed = launch_profile_policy.launch_authority()
    assert observed is captured
    assert observed.home == launch
    assert observed.env["CAPTURED_ONLY"] == "launch"


def test_runtime_scope_keeps_legacy_home_argument_but_ignores_it(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    poison = tmp_path / "poison"
    launch.mkdir()
    poison.mkdir()
    monkeypatch.setattr(launch_profile_policy, "_authority", None)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    launch_profile_policy.capture_launch_authority()

    with launch_profile_policy.launch_profile_runtime_scope(poison):
        assert get_hermes_home() == launch


def test_standalone_turn_binds_launch_profile_scope_after_hosted_activation(
    standalone, monkeypatch
):
    runner, secondary = standalone
    from tui_gateway.server import _session_profile_runtime_scope

    # A native hosted room ran a second profile: real activation, real secondary scope entered and left.
    launch_profile_policy.activate_multi_profile_hosting()
    with _session_profile_runtime_scope({"profile_home": str(secondary)}):
        assert get_secret("OPENAI_API_KEY") == "secondary-key"
        assert get_secret(INJECTED) is None  # the launch env never leaks into a secondary
    assert secret_scope.is_multiplex_active()

    # Secondary activity can poison process-global identity/policy after activation. A standalone
    # turn still belongs to the launch authority captured above, never the live process home.
    monkeypatch.setenv("HERMES_HOME", str(secondary))
    monkeypatch.setenv("TERMINAL_SSH_HOST", "live-poison.example")

    source = mock.MagicMock(profile=None)
    with runner._profile_scope_for_source(source):
        assert _scope_values() == (
            "launch-dotenv-key", "launch-env-injected", "launch.example", str(secondary.parent.parent / "launch")
        )
    runner._handle_message = mock.AsyncMock(side_effect=lambda event: _scope_values())
    assert asyncio.run(runner._primary_message_handler()(mock.MagicMock(source=source))) == (
        "launch-dotenv-key", "launch-env-injected", "launch.example", str(secondary.parent.parent / "launch"))
    with pytest.raises(UnscopedSecretError):  # the guard itself is not weakened
        get_secret("OPENAI_API_KEY")


def test_standalone_without_hosted_activation_stays_unscoped(standalone):
    runner, _secondary = standalone
    source = mock.MagicMock(profile=None)
    assert isinstance(runner._profile_scope_for_source(source), nullcontext)
    runner._handle_message = mock.AsyncMock(side_effect=lambda event: current_secret_scope())
    assert asyncio.run(runner._primary_message_handler()(mock.MagicMock(source=source))) is None
