"""Profile-scoped Codex enrollment through the shared gateway /login command (#74728)."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import anon_auth, auth_codex
from hermes_cli.commands import resolve_command


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("model:\n  provider: anthropic\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    config = GatewayConfig(multiplex_profiles=True, platforms={
        Platform.TELEGRAM: PlatformConfig(enabled=True, extra={"allow_admin_from": ["owner"]})})
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner._profile_configs = {"coder": config}
    runner._resolve_profile_home_for_source = lambda source: profile
    adapter = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(success=True)))
    runner._delivery_adapter_for = lambda source: adapter
    runner._thread_metadata_for_source = lambda source: {}
    runner._deliver_platform_notice = AsyncMock(wraps=runner._deliver_platform_notice)
    monkeypatch.setattr(anon_auth, "current_nous_state", lambda: {"access_token": "test-nous"})
    return runner, adapter, root, profile


def _event(*, chat_type="dm", user_id="owner", platform=Platform.TELEGRAM, args="codex"):
    return MessageEvent(text=f"/login {args}", source=SessionSource(
        platform=platform, chat_id="test-chat", user_id=user_id,
        chat_type=chat_type, profile="coder"))


async def _finish(runner):
    await asyncio.gather(*list(runner._background_tasks))
    runner._login_exec.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [False, True])
async def test_device_flow_delivers_before_poll_and_appends_only_to_source_profile(
    setup, monkeypatch, capsys, owned,
):
    runner, adapter, root, profile = setup
    existing = {"id": "existing", "label": "test-account", "auth_type": "oauth",
                "priority": 0, "source": "manual:device_code",
                "access_token": "test-existing", "refresh_token": "test-existing-refresh"}
    root_entry = dict(existing, id="root", access_token="test-root", refresh_token="test-root-refresh")
    (root / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {"openai-codex": [root_entry]}}))
    root_bytes = (root / "auth.json").read_bytes()
    if owned:
        (profile / "auth.json").write_text(json.dumps({
            "version": 1, "providers": {}, "active_provider": "anthropic",
            "credential_pool": {"openai-codex": [existing]}}))
    stages = []
    responses = iter([
        {"user_code": "TEST-CODE", "device_auth_id": "test-device", "interval": 5},
        {"authorization_code": "test-auth-code", "code_verifier": "test-verifier"},
        {"access_token": "test-new-access", "refresh_token": "test-new-refresh"},
    ] * 2)

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            if url.endswith("deviceauth/token"):
                assert [c.args[1] for c in adapter.send.await_args_list][-3:-1] == [
                    "https://auth.openai.com/codex/device", "TEST-CODE"]
                stages.append("poll")
            return SimpleNamespace(status_code=200, json=lambda: next(responses))

    monkeypatch.setattr(auth_codex, "_codex_http_client", lambda **kwargs: Client())
    monkeypatch.setattr(auth_codex.time, "sleep", lambda seconds: None)
    for _ in range(2):
        result = await runner._handle_login_command(_event())
        assert "Codex sign-in started" in result
        await _finish(runner)
    payload = json.loads((profile / "auth.json").read_text())
    rows = payload["credential_pool"]["openai-codex"]
    assert [row["access_token"] for row in rows] == (
        ["test-existing"] if owned else []) + ["test-new-access"] * 2
    assert len({row["id"] for row in rows}) == len(rows)
    assert payload["active_provider"] == ("anthropic" if owned else "openai-codex")
    assert payload["providers"] == {}
    assert (root / "auth.json").read_bytes() == root_bytes
    assert (profile / "config.yaml").read_text() == "model:\n  provider: anthropic\n"
    assert stages == ["poll", "poll"]
    assert "TEST-CODE" not in capsys.readouterr().out
    notices = " ".join(c.args[1] for c in runner._deliver_platform_notice.await_args_list)
    assert "account was added to this profile" in notices
    assert "TEST-CODE" in notices and "test-new-access" not in notices


@pytest.mark.asyncio
async def test_codex_uses_shared_gates_and_busy_rejection(setup, monkeypatch):
    runner, adapter, _, _ = setup
    login = MagicMock()
    monkeypatch.setattr("hermes_cli.auth._codex_device_code_login", login)
    for event in (_event(chat_type="group"), _event(platform="ntfy"), _event(user_id="member")):
        assert await runner._handle_login_command(event) in {
            anon_auth.LOGIN_DM_ONLY, anon_auth.LOGIN_NOT_ALLOWED}
    assert await runner._handle_login_command(_event(args="unknown")) == "Usage: /login [nous|codex]"
    event = _event()
    result = await runner._dispatch_busy_slash_command(event, resolve_command("login"), "test-key", event.source)
    assert "can't run mid-turn" in result
    login.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_background_handler_keeps_login_slot_until_worker_exits(setup, monkeypatch):
    runner, _, _, profile = setup
    entered, release = threading.Event(), threading.Event()
    workers = []
    run_blocking = runner._run_login_blocking

    async def tracking(func):
        workers.append(asyncio.current_task())
        return await run_blocking(func)

    runner._run_login_blocking = tracking

    def login(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return {"tokens": {"access_token": "test-access", "refresh_token": "test-refresh"}}

    monkeypatch.setattr("hermes_cli.auth._codex_device_code_login", login)
    assert "Codex sign-in started" in await runner._handle_login_command(_event())
    assert await asyncio.to_thread(entered.wait, 5)
    handler = next(iter(runner._background_tasks))
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    assert "already active" in await runner._handle_login_command(_event())
    assert await runner._handle_login_command(_event(args="nous")) == anon_auth.LOGIN_BUSY_ELSEWHERE
    release.set()
    await workers[0]
    runner._login_exec.shutdown(wait=True)
    assert runner._login_attempts == {}
    assert json.loads((profile / "auth.json").read_text())["credential_pool"]["openai-codex"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_private_delivery_failure_aborts_poll_without_leaking_exception(setup, monkeypatch, caplog, raises):
    runner, adapter, _, profile = setup
    adapter.send.return_value = SimpleNamespace(success=False)
    if raises:
        adapter.send.side_effect = RuntimeError("test-sensitive-error")
    monkeypatch.setattr(auth_codex, "_codex_request_device_code", lambda *args: {
        "user_code": "TEST-CODE", "device_auth_id": "test-device", "interval": 5})
    poll = MagicMock(side_effect=RuntimeError("test-sensitive-error"))
    monkeypatch.setattr(auth_codex, "_codex_poll_authorization_code", poll)
    assert "Codex sign-in started" in await runner._handle_login_command(_event())
    await _finish(runner)
    poll.assert_not_called()
    assert not (profile / "auth.json").exists()
    assert not runner._login_attempts
    assert "test-sensitive-error" not in caplog.text
    assert "did not complete" in runner._deliver_platform_notice.await_args.args[1]
