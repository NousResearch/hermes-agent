"""Native Telegram account enrollment/order, without an agent or real OAuth traffic."""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from hermes_cli.commands import resolve_command, should_bypass_active_session
from hermes_constants import get_hermes_home


def _jwt(account):
    payload = {"sub": account, "exp": 4102444800,
               "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    part = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return "e30." + part + ".test-signature"


def _event(text, profile="a", chat_type="dm", user_id="7"):
    return MessageEvent(text=text, message_id="13", source=SessionSource(
        platform=Platform.TELEGRAM, chat_id="7", user_id=user_id,
        chat_type=chat_type, thread_id="11", profile=profile))


def _fixture(tmp_path, monkeypatch):
    default = tmp_path / ".hermes"
    default.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default))
    homes = {name: default / "profiles" / name for name in ("a", "b")}
    for name, home in homes.items():
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "model:\n  provider: openai-codex\nauth:\n  codex_login_flow: browser\n",
            encoding="utf-8",
        )
        entry = {"id": "original", "label": name + " original", "auth_type": "oauth",
                 "source": "manual:device_code", "priority": 0,
                 "access_token": _jwt(name + "-old"), "refresh_token": name + "-refresh",
                 "last_status": "exhausted", "last_error_reset_at": 4102444800}
        (home / "auth.json").write_text(json.dumps({"version": 1,
            "active_provider": "openai-codex", "credential_pool": {"openai-codex": [entry]}}), encoding="utf-8")
        _entries(home)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True, platforms={
        Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-only",
                                         extra={"allow_admin_from": ["7"]})})
    deliveries = []

    class Adapter:
        config = runner.config.platforms[Platform.TELEGRAM]

        async def send(self, chat_id, text, metadata=None):
            deliveries.append((chat_id, text, metadata))
            return SimpleNamespace(success=True)

    adapter = Adapter()
    runner._delivery_adapter_for = lambda source: adapter
    runner._resolve_profile_home_for_source = lambda source: homes[source.profile]
    runner._thread_metadata_for_source = lambda source: {"thread_id": source.thread_id}
    runner._background_tasks = set()
    return runner, homes, deliveries


def _entries(home):
    with _profile_runtime_scope(home):
        from agent.credential_pool import load_pool
        return load_pool("openai-codex").entries()


@pytest.mark.asyncio
async def test_native_enrollment_and_order_use_the_requesting_profile(tmp_path, monkeypatch, capsys):
    runner, homes, delivered = _fixture(tmp_path, monkeypatch)
    import hermes_cli.auth_codex as codex
    count = 0
    observed_homes = []

    def request(issuer, client_id):
        return {"user_code": "TEST-DEVICE-CODE", "device_auth_id": "fake-device", "interval": 3}

    def poll(issuer, **kwargs):
        assert any("TEST-DEVICE-CODE" in text for _, text, _ in delivered)
        observed_homes.append(get_hermes_home())
        return {"authorization_code": "test-code", "code_verifier": "test-verifier"}

    def exchange(issuer, client_id, code):
        nonlocal count
        count += 1
        return {"access_token": _jwt(f"new-{count}"), "refresh_token": f"new-refresh-{count}"}

    monkeypatch.setattr(codex, "_codex_request_device_code", request)
    monkeypatch.setattr(codex, "_codex_poll_authorization_code", poll)
    monkeypatch.setattr(codex, "_codex_exchange_authorization_code", exchange)
    command = resolve_command("auth")
    assert command is not None
    assert should_bypass_active_session(command.name)
    handler = runner._gateway_plain_command_handlers()[command.name]

    for profile, label, priority in (("a", "Work Pro", 0), ("b", "Team Reserve", 1), ("a", "Travel", 1)):
        before_other = (homes["b" if profile == "a" else "a"] / "auth.json").read_bytes()
        started = await handler(_event(f'/auth add openai-codex --label "{label}" --priority {priority}', profile))
        assert "started" in started.lower()
        await asyncio.gather(*list(runner._background_tasks))
        entries = _entries(homes[profile])
        assert entries[priority].label == label
        assert next(e for e in entries if e.id == "original").last_status == "exhausted"
        assert (homes["b" if profile == "a" else "a"] / "auth.json").read_bytes() == before_other

    reordered = await handler(_event('/auth priority openai-codex "Travel" 0'))
    assert "priority 0" in reordered
    assert _entries(homes["a"])[0].label == "Travel"
    listing = await handler(_event('/auth list openai-codex'))
    assert listing.index("Travel") < listing.index("Work Pro")
    assert "fill_first" in listing
    assert observed_homes == [homes["a"], homes["b"], homes["a"]]
    assert all(chat == "7" and meta["thread_id"] == "11" for chat, _, meta in delivered)
    assert all("new-refresh" not in text and ".test-signature" not in text for _, text, _ in delivered)
    assert "TEST-DEVICE-CODE" not in capsys.readouterr().out
    assert not (tmp_path / ".hermes" / "auth.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["group", "non_admin", "missing_user", "unknown_flag", "invalid_label", "delivery_failure", "cancel"])
async def test_auth_never_mutates_without_private_admission_and_delivered_consent(tmp_path, monkeypatch, case):
    runner, homes, delivered = _fixture(tmp_path, monkeypatch)
    before = {name: (home / "auth.json").read_bytes() for name, home in homes.items()}
    import hermes_cli.auth_codex as codex
    poll_calls = []
    monkeypatch.setattr(codex, "_codex_request_device_code", lambda *args: {
        "user_code": "TEST-DEVICE-CODE", "device_auth_id": "fake", "interval": 3})
    def no_poll(*args, **kwargs):
        poll_calls.append(True)
        raise AssertionError("Consent was not delivered: polling must not start")
    monkeypatch.setattr(codex, "_codex_poll_authorization_code", no_poll)
    default_text = '/auth add openai-codex --label "Work" --priority 0'
    overrides = {
        "group": {"chat_type": "group"},
        "non_admin": {"user_id": "other"},
        "missing_user": {"user_id": None},
    }
    inputs = {
        "unknown_flag": '/auth add openai-codex --label Work --api-key do-not-echo',
        "invalid_label": '/auth add openai-codex --label "bad\x00name"',
    }
    event = _event(inputs.get(case, default_text), **overrides.get(case, {}))
    if case == "delivery_failure":
        adapter = runner._delivery_adapter_for(event.source)
        async def fail(*args, **kwargs):
            return SimpleNamespace(success=False, error="do-not-echo")
        adapter.send = fail

    reply = await runner._handle_auth_command(event)
    assert isinstance(reply, str) and reply
    assert "do-not-echo" not in reply
    if case == "cancel":
        cancel = await runner._handle_auth_command(_event('/auth cancel'))
        assert "cancel" in cancel.lower()
    await asyncio.gather(*list(runner._background_tasks))
    assert not poll_calls
    assert {name: (home / "auth.json").read_bytes() for name, home in homes.items()} == before
    assert not (tmp_path / ".hermes" / "auth.json").exists()
