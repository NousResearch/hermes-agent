"""Bounded security characterization: central denial precedes profile/seen I/O.

Target: stock 2237be355906fbe6065ce1815711eee52b2d646e.
Run under the stock tests/gateway conftests in an isolated test-only copy.
No connection, credentials, model turn, or live policy is needed.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.run import GatewayRunner
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz = load_plugin_adapter("buzz")
SENDER = "a" * 64
OTHER_ALLOWED_USER = "b" * 64
SELF = "c" * 64
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"


@pytest.fixture(params=["empty-local", "stale-snapshot-grant", "adapter-config-grant"])
def denied_intake(request, monkeypatch, tmp_path):
    # Constructor snapshots local policy BEFORE the central allowlist is set.
    for name in tuple(os.environ):
        if name.startswith("BUZZ_") or name in (
            "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
        ):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_buzz, "_DEFAULT_CREDENTIALS_DIR", tmp_path / "no-creds")
    extra = {"relay_url": "https://buzz.invalid", "require_mention": False}
    if request.param != "empty-local":
        extra["allowed_users"] = [SENDER]
    # Live mention admission must reach the authorization boundary under test.
    # Match the fixture's permissive constructor intent in the real owner config,
    # before construction captures the transport home (no resolver monkeypatch).
    home = tmp_path / "combined-auth-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "buzz:\n  require_mention: false\n  thread_require_mention: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = _buzz.BuzzAdapter(PlatformConfig(enabled=True, extra=extra))
    adapter._self_pubkey = SELF
    adapter._self_npub = _buzz.hex_to_npub(SELF)
    adapter._display_name = "CharacterizationBot"
    if request.param == "stale-snapshot-grant":
        # Operator policy no longer grants this sender, but the adapter's
        # constructor-time parsed list remains stale. Do not patch the predicate.
        adapter.config.extra["allowed_users"] = []
    assert bool(adapter._allowed_pubkeys) == (request.param != "empty-local")
    if request.param != "empty-local":
        assert SENDER in adapter._allowed_pubkeys

    previous = platform_registry.get("buzz")
    platform_registry.register(PlatformEntry(
        name="buzz", label="Buzz", adapter_factory=_buzz.BuzzAdapter,
        check_fn=lambda: True, allowed_users_env="BUZZ_ALLOWED_USERS",
        allow_all_env="BUZZ_ALLOW_ALL_USERS",
    ))
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", OTHER_ALLOWED_USER)
    monkeypatch.setenv("BUZZ_ALLOW_ALL_USERS", "false")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "false")

    # Same bare-runner construction used by stock test_buzz_authz.py;
    # use the actual callback factory AND central authorization implementation.
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.pairing_store = None
    runner.adapters = {adapter.platform: adapter}
    runner._profile_adapters = {}
    authority = runner._make_adapter_auth_check(adapter.platform)
    adapter.set_authorization_check(authority)

    # Precondition/control proves the callback is connected to a real policy,
    # not a blanket False mock or an unknown/throwing authorization callback.
    assert authority(SENDER, "group", CHANNEL) is False
    assert authority(OTHER_ALLOWED_USER, "group", CHANNEL) is True
    assert adapter._is_sender_authorized(SENDER, "group", CHANNEL) is False

    verdicts = []

    async def deny_at_runner_boundary(event):
        # Keep base handle_message/background scheduling real. Replace only the
        # downstream runner turn with a central-policy check and no response.
        verdicts.append(runner._is_user_authorized(event.source))
        return None

    adapter.set_message_handler(deny_at_runner_boundary)
    # Fake transport only: name resolution itself is deliberately NOT replaced.
    cli = AsyncMock(return_value=(0, '[]', ''))
    monkeypatch.setattr(adapter, "_run_cli", cli)
    reaction = AsyncMock(return_value=True)
    monkeypatch.setattr(adapter, "send_reaction", reaction)
    monkeypatch.setattr(adapter, "send_typing", AsyncMock())
    resolver = AsyncMock(wraps=adapter._resolve_user_name)
    monkeypatch.setattr(adapter, "_resolve_user_name", resolver)
    adapter._user_names.clear()
    try:
        yield SimpleNamespace(
            adapter=adapter, cli=cli, reaction=reaction,
            resolver=resolver, verdicts=verdicts,
        )
    finally:
        platform_registry.unregister("buzz")
        if previous is not None:
            platform_registry.register(previous)


async def _dispatch_denied_text(probe):
    adapter = probe.adapter
    event = {
        "id": "d" * 64, "kind": 9, "pubkey": SENDER,
        "content": "hello", "created_at": 1, "tags": [["h", CHANNEL]],
    }
    try:
        await adapter._handle_event(CHANNEL, adapter._new_channel_state("group"), event)
        tasks = tuple(adapter._session_tasks.values())
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        # Early denial may suppress downstream intake altogether. If dispatch
        # reached the runner boundary, it must still be a literal central denial.
        assert all(verdict is False for verdict in probe.verdicts)
    finally:
        tasks = tuple(adapter._session_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_unauthorized_dispatch_does_not_resolve_user_profile(denied_intake):
    await _dispatch_denied_text(denied_intake)
    profile_calls = [
        call for call in denied_intake.cli.call_args_list
        if list(call.args[0][:2]) == ["users", "get"]
    ]
    assert not profile_calls, "central-denied sender triggered a Buzz users get profile fetch"
    denied_intake.resolver.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_dispatch_does_not_emit_seen_reaction(denied_intake):
    await _dispatch_denied_text(denied_intake)
    denied_intake.reaction.assert_not_called()


@pytest.mark.asyncio
async def test_central_allowed_sender_keeps_profile_dispatch_and_seen(denied_intake, monkeypatch):
    probe = denied_intake
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", SENDER)
    assert probe.adapter._is_sender_authorized(SENDER, "group", CHANNEL) is True
    event = {"id": "e" * 64, "kind": 9, "pubkey": SENDER,
             "content": "hello", "created_at": 1, "tags": [["h", CHANNEL]]}
    await probe.adapter._handle_event(CHANNEL, probe.adapter._new_channel_state("group"), event)
    tasks = tuple(probe.adapter._session_tasks.values())
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    assert probe.verdicts == [True]
    probe.resolver.assert_awaited_once_with(SENDER)
    assert any(list(call.args[0][:2]) == ["users", "get"] for call in probe.cli.call_args_list)
    probe.reaction.assert_awaited_once_with(CHANNEL, event["id"], "👀")


@pytest.mark.asyncio
@pytest.mark.parametrize("authority", ["missing", "throws", "truthy", "integer", "falsey"])
async def test_unknown_authority_suppresses_io_but_preserves_central_denial(denied_intake, authority):
    def throws(*_args):
        raise RuntimeError("test authority unavailable")
    callbacks = {"missing": None, "throws": throws, "truthy": lambda *_: "AUTHORIZED",
                 "integer": lambda *_: 1, "falsey": lambda *_: 0}
    probe = denied_intake
    probe.adapter.set_authorization_check(callbacks[authority])
    assert probe.adapter._is_sender_authorized(SENDER, "group", CHANNEL) is None
    await _dispatch_denied_text(probe)
    probe.resolver.assert_not_called()
    probe.reaction.assert_not_called()
    probe.cli.assert_not_called()
    assert probe.verdicts == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [False, True])
async def test_reaction_only_sender_obeys_central_authority(denied_intake, monkeypatch, allowed):
    probe = denied_intake
    probe.adapter._allowed_pubkeys = {OTHER_ALLOWED_USER}
    probe.adapter._reaction_only_pubkeys = {SENDER}
    if allowed:
        monkeypatch.setenv("BUZZ_ALLOWED_USERS", SENDER)
    event = {"id": "f" * 64, "kind": 9, "pubkey": SENDER,
             "content": "@CharacterizationBot hello", "created_at": 1,
             "tags": [["h", CHANNEL], ["p", SELF]]}
    assert probe.adapter._is_mentioned(event["content"])
    await probe.adapter._handle_event(CHANNEL, probe.adapter._new_channel_state("group"), event)
    if allowed:
        probe.reaction.assert_awaited_once_with(CHANNEL, event["id"], "👀")
    else:
        probe.reaction.assert_not_called()
    probe.resolver.assert_not_called()
    assert probe.verdicts == []
