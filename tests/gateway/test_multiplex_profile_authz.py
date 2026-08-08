"""Regression tests for multiplex profile-aware own-policy authorization."""

import asyncio
import json
import socket
import urllib.error
import urllib.request

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "WECOM_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "WECOM_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_multiplex_runner(monkeypatch):
    """Runner with default allowlist WeCom and secondary open-policy WeCom."""
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    default_adapter = SimpleNamespace(
        send=AsyncMock(),
        enforces_own_access_policy=True,
        _dm_policy="allowlist",
        _group_policy="pairing",
    )
    secondary_adapter = SimpleNamespace(
        send=AsyncMock(),
        enforces_own_access_policy=True,
        _dm_policy="open",
        _group_policy="open",
    )

    runner.adapters = {Platform.WECOM: default_adapter}
    runner._profile_adapters = {
        "coder": {Platform.WECOM: secondary_adapter},
    }
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner, default_adapter, secondary_adapter


def test_default_profile_still_trusts_own_allowlist(monkeypatch):
    """Default-profile allowlist trust is unchanged when profile is unstamped."""
    runner, _default_adapter, _secondary_adapter = _make_multiplex_runner(monkeypatch)

    source = SessionSource(
        platform=Platform.WECOM,
        user_id="allowed-user",
        chat_id="dm-chat",
        user_name="allowed-user",
        chat_type="dm",
        profile=None,
    )

    assert runner._is_user_authorized(source) is True


def test_active_profile_stamp_resolves_primary_adapter(monkeypatch):
    """A single-profile gateway stamps its active profile but stores adapters as primary."""
    runner, default_adapter, _secondary_adapter = _make_multiplex_runner(monkeypatch)
    runner._active_profile_name = lambda: "dev"

    assert runner._authorization_adapter(Platform.WECOM, profile="dev") is default_adapter


def test_scoped_secondary_profile_still_uses_profile_adapters(monkeypatch):
    """Runtime scope must not redirect secondary authz to primary adapters.

    ``_make_profile_message_handler`` wraps ``_handle_message`` in
    ``_profile_runtime_scope``, which overrides HERMES_HOME so
    ``get_active_profile_name()`` equals the secondary profile for that turn.
    Authorization must still read ``_profile_adapters[profile]``, not the
    empty primary ``self.adapters`` map — otherwise upstream-auth platforms
    such as A2A default-deny an already-authenticated peer (#80884).
    """
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False

    secondary = SimpleNamespace(
        authorization_is_upstream=True,
        enforces_own_access_policy=False,
    )
    runner._profile_adapters = {"beta": {Platform("a2a"): secondary}}
    # Simulate the scoped turn: active profile name collapses to the secondary.
    runner._active_profile_name = lambda: "beta"

    assert runner._authorization_adapter(Platform("a2a"), profile="beta") is secondary

    source = SessionSource(
        platform=Platform("a2a"),
        chat_id="a2a-context",
        user_id="alpha",
        user_name="alpha",
        chat_type="dm",
        profile="beta",
    )
    assert runner._is_user_authorized(source) is True


@pytest.mark.asyncio
async def test_secondary_a2a_listener_keeps_upstream_authorization(
    monkeypatch, tmp_path
):
    """A real secondary listener accepts its token through gateway authz."""
    from agent.secret_scope import set_multiplex_active
    from gateway.run import GatewayRunner
    from plugins.platforms.a2a.adapter import A2AAdapter
    from plugins.platforms.a2a import protocol

    _clear_auth_env(monkeypatch)
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("A2A_PEER_TOKENS", "default:default-token")
    monkeypatch.setenv("A2A_REPLY_TIMEOUT", "2")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setenv("A2A_PORT", str(port))

    profile_home = tmp_path / "coder"
    profile_home.mkdir()
    (profile_home / ".env").write_text(
        "A2A_PEER_TOKENS=alpha:secondary-token\nA2A_HOST=127.0.0.1\n",
        encoding="utf-8",
    )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.session_store = None
    runner._busy_text_mode = "queue"
    runner._handle_active_session_busy_message = None
    runner._recover_telegram_topic_thread_id = None
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner._profile_name_for_source = lambda source: None

    a2a = Platform("a2a")
    profile_config = GatewayConfig(multiplex_profiles=True)
    profile_config.platforms = {a2a: PlatformConfig(enabled=True)}
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: profile_config)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda profile: profile_home
    )
    monkeypatch.setattr(
        runner,
        "_create_adapter",
        lambda platform, config: A2AAdapter(config),
    )

    async def connect(adapter, platform):
        return await adapter.connect()

    monkeypatch.setattr(runner, "_connect_initial_adapter_with_timeout", connect)

    observed = {}

    async def handle_message(event):
        observed["profile"] = event.source.profile
        observed["authorized"] = runner._is_user_authorized(event.source)
        await runner._profile_adapters["coder"][a2a].send(
            event.source.chat_id,
            "authorized" if observed["authorized"] else "unauthorized",
            metadata={"notify": True},
        )

    runner._handle_message = handle_message

    def post(token):
        message = protocol.text_message(protocol.ROLE_USER, "profile auth")
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {"message": message},
            }
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode())

    set_multiplex_active(True)
    adapter = None
    try:
        assert await runner._start_one_profile_adapters(
            "coder", profile_home, {}
        ) == 1
        adapter = runner._profile_adapters["coder"][a2a]

        response = await asyncio.to_thread(post, "secondary-token")
        assert response["result"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert observed == {"profile": "coder", "authorized": True}

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            await asyncio.to_thread(post, "default-token")
        assert exc_info.value.code == 401
    finally:
        if adapter is not None:
            await adapter.disconnect()
        set_multiplex_active(False)


def test_secondary_allowlist_dm_behavior_ignores_unauthorized(monkeypatch):
    """Unauthorized-DM behavior must read the secondary adapter's dm_policy."""
    runner, _default_adapter, secondary_adapter = _make_multiplex_runner(monkeypatch)
    secondary_adapter._dm_policy = "allowlist"

    assert runner._get_unauthorized_dm_behavior(
        Platform.WECOM,
        profile="coder",
    ) == "ignore"
    assert runner._get_unauthorized_dm_behavior(Platform.WECOM) == "ignore"


def test_adapter_auth_check_stamps_secondary_profile(monkeypatch):
    """The adapter auth-check callback must stamp its own secondary profile.

    Regression for the gap where ``_make_adapter_auth_check`` built a
    profile-less ``SessionSource``, so a secondary adapter's external-context
    authorization (e.g. Slack/Discord thread-reply lookups) silently
    resolved the *active* profile's allowlist scope instead of its own.
    """
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    captured: dict = {}

    def fake_is_user_authorized(source):
        captured["profile"] = source.profile
        return True

    runner._is_user_authorized = fake_is_user_authorized

    check = runner._make_adapter_auth_check(Platform.WECOM, profile_name="coder")
    assert check("some-user", "dm", "dm-chat") is True
    assert captured["profile"] == "coder"


def test_secondary_open_policy_fails_startup_guard(monkeypatch):
    """Secondary profiles must pass the same open-policy startup guard."""
    from gateway.run import _own_policy_open_startup_violation

    _clear_auth_env(monkeypatch)

    secondary_cfg = GatewayConfig(multiplex_profiles=True)
    secondary_cfg.platforms = {
        Platform.WECOM: PlatformConfig(
            enabled=True,
            extra={"dm_policy": "open"},
        ),
    }

    violation = _own_policy_open_startup_violation(secondary_cfg)
    assert violation is not None
    assert "wecom" in violation
    assert "open policy" in violation
