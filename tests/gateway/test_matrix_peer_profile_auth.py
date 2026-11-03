"""Matrix budget admission must authorize the event's final routed profile."""

import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.profile_routing import parse_profile_routes
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.fixture(params=[False, True], ids=["thread-route", "server-thread-route"])
def routed_matrix(tmp_path, monkeypatch, request):
    from gateway.run import GatewayRunner

    home = tmp_path / "hh"
    secondary = home / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve",
                        lambda **kw: [("default", home), ("secondary", secondary)])
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir",
                        lambda name: home if name == "default" else secondary)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    # Keep secret discovery out of this routing/pairing test. Authorization itself is real.
    monkeypatch.setattr("gateway.run._profile_runtime_scope", lambda *a, **kw: nullcontext())
    monkeypatch.setattr("gateway.authz_mixin._auth_env", lambda name, default="": default)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.profile_routes = parse_profile_routes([
        {"platform": "matrix", "chat_id": "!room:example.org",
         "thread_id": "$routed", "profile": "secondary",
         **({"guild_id": "example.org"} if request.param else {})},
    ])
    runner._primary_profile_name = "default"
    runner._profile_adapters = {"secondary": {}}
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {"default": runner.pairing_store,
                             "secondary": PairingStore(profile="secondary")}
    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={
        "homeserver": "https://example.org", "user_id": "@hermes:example.org",
        "peer_agent_ids": ["@yomi:example.org"],
        "peer_reply_budget_per_human_message": 2,
    }))
    runner.adapters = {Platform.MATRIX: adapter}
    runner.config.platforms = {Platform.MATRIX: adapter.config}
    adapter.gateway_runner = runner
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.MATRIX))
    adapter._startup_ts = time.time() - 10
    adapter._require_mention = adapter._thread_require_mention = False
    adapter._text_batch_delay_seconds = 0
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._resolve_room_identity = AsyncMock(return_value=SimpleNamespace(
        display_name="Room", room_topic="", server_name="example.org"))
    adapter._get_display_name = AsyncMock(return_value="Sender")
    adapter._background_read_receipt = Mock()
    adapter._threads = Mock()
    adapter._threads.__contains__ = Mock(return_value=False)
    adapter._threads.mark_async = AsyncMock()
    adapter.handle_message = AsyncMock()
    return runner, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("synthetic", [False, True], ids=["real-thread", "synthetic-thread"])
@pytest.mark.parametrize("grant_profile", ["default", "secondary"])
@pytest.mark.parametrize("peer", [True, False], ids=["peer-spend", "human-replenish"])
async def test_budget_uses_final_thread_pairing_grant(routed_matrix, synthetic, grant_profile, peer):
    runner, adapter = routed_matrix
    room = "!room:example.org"
    sender = "@yomi:example.org" if peer else "@alice:example.org"
    store = runner.pairing_stores[grant_profile]
    store._save_json(store._approved_path("matrix"), {sender: {}})
    adapter._matrix_session_scope = "thread" if synthetic else "room"
    adapter._peer_reply_budget_remaining[room] = 2 if peer else 0
    content = {"msgtype": "m.text", "body": "@hermes:example.org help",
               "m.mentions": {"user_ids": ["@hermes:example.org"]}}
    if not synthetic:
        content["m.relates_to"] = {"rel_type": "m.thread", "event_id": "$routed"}
    event = SimpleNamespace(room_id=room, sender=sender, event_id="$routed" if synthetic else "$reply",
                            timestamp=int(time.time() * 1000), content=content)

    await adapter._on_room_message(event)

    authorized = grant_profile == "secondary"
    expected_budget = (1 if authorized else 2) if peer else (2 if authorized else 0)
    assert adapter._peer_reply_budget_remaining[room] == expected_budget
    assert adapter.handle_message.await_count == (int(authorized) if peer else 1)
    if adapter.handle_message.await_count:
        source = adapter.handle_message.call_args.args[0].source
        assert source.thread_id == "$routed"
        assert source.guild_id == "example.org"
        assert runner._canonicalize(source).runtime_profile == "secondary"
        assert runner._is_user_authorized_for_source(source) is authorized
