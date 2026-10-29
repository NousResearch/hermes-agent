"""The Webapp's gateway socket runs in the served profile's scope on a multiplexed gateway.

A browser tab's ``/api/ws`` reaches the gateway through the dashboard mount (``chat_ws.gateway_ws``),
not the native ticket path (``gateway/run_api.py``) that wraps the socket in ``owner_scope``. Legacy
handlers it falls back to (``billing.state``, ``subscription.state``, ``usage.bars``) bind no scope of
their own, so every Webapp billing/usage read on a multiplexed gateway resolved the Nous Portal
override with no profile secret scope: ``UnscopedSecretError`` -> "no profile secret scope on a
multiplexed call" -> the override was dropped and the token went to the default Portal.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

LAUNCH_PORTAL = "https://launch-portal.example.invalid"
B_PORTAL = "https://b-portal.example.invalid"


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    launch = tmp_path / "hermes"
    b = launch / "profiles" / "b"
    b.mkdir(parents=True)
    (launch / ".env").write_text(f"HERMES_PORTAL_BASE_URL={LAUNCH_PORTAL}\n", encoding="utf-8")
    (b / ".env").write_text(f"HERMES_PORTAL_BASE_URL={B_PORTAL}\n", encoding="utf-8")
    for home in (launch, b):
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", LAUNCH_PORTAL)  # the launch process loaded its own .env
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    return launch, b


def _gateway(launch: Path, b: Path):
    """A multiplexed runner serving both homes, and its launch authority."""
    by_home = {}
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    runner.session_authorities = SimpleNamespace(for_home=lambda home: by_home.get(Path(home).resolve()))
    for home in (launch, b):
        by_home[home.resolve()] = SimpleNamespace(profile_id=str(home), instance_id=f"i-{home.name}", runner=runner)
    return by_home[launch.resolve()]


def _socket(authority, ticket_home=None):
    async def close(code=None):
        raise AssertionError(f"socket closed with {code}")
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(session_authority=authority)), scope={},
        _hermes_auth_identity={"user_id": "u", "provider": "session-token"},
        _hermes_ticket_profile_home=None if ticket_home is None else str(ticket_home), close=close)


def _portal_seen_by_socket(monkeypatch, ws) -> list:
    """Run ``gateway_ws`` with a handler that resolves the Portal override the way a legacy billing
    handler does: on a worker thread copied from the socket's context."""
    from hermes_cli import mcp_startup
    from tui_gateway import ws as tui_ws
    from hermes_cli.web_routers import chat_ws

    async def allowed(*_a, **_k):
        return True

    seen = []

    async def handler(*_a, **_k):
        from hermes_cli.auth_nous import _nous_portal_env_override
        seen.append(await asyncio.to_thread(_nous_portal_env_override))

    monkeypatch.setattr(chat_ws, "_close_unless_sidecar_allowed", allowed)
    monkeypatch.setattr(mcp_startup, "start_deferred_mcp_discovery_now", lambda: None)
    monkeypatch.setattr(tui_ws, "handle_ws", handler)
    asyncio.run(chat_ws.gateway_ws(ws))
    return seen


def test_webapp_socket_reads_the_launch_profiles_portal_override_in_its_scope(two_homes, monkeypatch, caplog):
    launch, b = two_homes
    authority = _gateway(launch, b)

    assert _portal_seen_by_socket(monkeypatch, _socket(authority)) == [LAUNCH_PORTAL]
    assert "no profile secret scope" not in caplog.text


def test_webapp_socket_for_a_served_profile_reads_that_profiles_override(two_homes, monkeypatch, caplog):
    launch, b = two_homes
    authority = _gateway(launch, b)

    seen = [_portal_seen_by_socket(monkeypatch, _socket(authority, home)) for home in (launch, b, launch)]

    assert seen == [[LAUNCH_PORTAL], [B_PORTAL], [LAUNCH_PORTAL]]
    assert "no profile secret scope" not in caplog.text
