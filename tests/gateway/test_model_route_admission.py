"""Turn admission follows the selected route, not another chat's provider I/O."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_messaging_session_picks import _compressed_chat


def _turn(runner, source, route, session_id):
    from gateway.run_turn_runner import TurnRunner
    turn = object.__new__(TurnRunner)
    turn._runner = runner
    turn._ctx = SimpleNamespace(session_id=session_id, session_key=route, source=source)
    return turn


def _observe_mount(live, monkeypatch):
    entered = asyncio.Event()
    original = live.mutation_lock

    @asynccontextmanager
    async def observed():
        async with original:
            entered.set()
            yield

    monkeypatch.setattr(live, "mutation_lock", observed())
    return entered


@pytest.mark.asyncio
@pytest.mark.parametrize(("entry", "stage"), [
    ("typed", "resolution"), ("typed", "publication"),
    ("picker", "resolution"), ("picker", "publication"), ("confirmed", "publication"),
])
@pytest.mark.parametrize("scope", ["session", "global"])
async def test_switch_only_blocks_affected_turn_admission(tmp_path, monkeypatch, entry, stage, scope):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_authorities import active_authority
    from hermes_cli.model_switch import ModelSwitchResult
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, source, route, viewer, root, _ = await _compressed_chat(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    tasks = []
    try:
        authority = active_authority(runner)
        other_source = SessionSource(platform=Platform.TELEGRAM, chat_id="5151", user_id="u-5151", chat_type="dm")
        other_id = authority.register(other_source).session_id
        other_route = runner._session_key_for_source(other_source)
        other = _turn(runner, other_source, other_route, other_id)
        affected = _turn(runner, source, route, "seg-1")
        result = ModelSwitchResult(success=True, new_model="picked", target_provider="openrouter",
                                   base_url="https://openrouter.ai/api/v1")

        async def resolve(*args):
            if stage == "resolution":
                entered.set()
                await release.wait()
            return result, None

        original = runner._session_db.update_session_model

        async def publish(*args, **kwargs):
            await original(*args, **kwargs)
            if stage == "publication":
                entered.set()
                await release.wait()

        monkeypatch.setattr(runner, "_perform_model_switch", resolve)
        monkeypatch.setattr(runner._session_db, "update_session_model", publish)
        monkeypatch.setattr(runner, "_switch_cached_agent_model", lambda *a: None)
        monkeypatch.setattr(runner, "_record_switch_metrics", lambda *a: None)
        monkeypatch.setattr(runner, "_model_switch_confirmation", AsyncMock(return_value="ok"))
        monkeypatch.setattr(runner, "_model_selection_guard_reply", AsyncMock(return_value=(False, None)))
        event = MessageEvent(text=f"/model picked --{scope}", source=source)
        if entry == "picker":
            callback = []

            class PickerAdapter:
                async def send_model_picker(self):
                    pass

            async def show_picker(*args):
                callback.append(args[-1])
                return True

            monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: PickerAdapter())
            monkeypatch.setattr(runner, "_send_model_picker", show_picker)
            assert await runner._handle_model_command(
                MessageEvent(text=f"/model --{scope}", source=source)) is None
            pending = callback[0](source.chat_id, "picked", "openrouter")
        elif entry == "confirmed":
            from gateway.slash_commands_model import _ModelSwitchContext
            ctx = _ModelSwitchContext(session_key=route, source=source,
                                      config_path=tmp_path / "config.yaml", persist_global=scope == "global")
            pending = runner._commit_model_switch(result, ctx, source=source)
        else:
            pending = runner._handle_model_command(event)
        switch = asyncio.create_task(pending)
        tasks.append(switch)
        await asyncio.wait_for(entered.wait(), 5)
        affected_entered = _observe_mount(authority.sessions[root], monkeypatch)
        affected_capture = asyncio.create_task(affected.capture_route_sync_fence())
        tasks.append(affected_capture)
        await asyncio.wait_for(affected_entered.wait(), 5)
        other_entered = _observe_mount(authority.sessions[other_id], monkeypatch)
        other_capture = asyncio.create_task(other.capture_route_sync_fence())
        tasks.append(other_capture)
        await asyncio.wait_for(other_entered.wait(), 5)
        if scope == "session":
            await asyncio.wait_for(asyncio.shield(other_capture), 5)
            assert other._route_sync_fence[0] == other_id
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(affected_capture), 2)
        else:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(other_capture), 2)
            assert not affected_capture.done()
        release.set()
        assert await asyncio.wait_for(switch, 5) == "ok"
        await asyncio.wait_for(asyncio.gather(affected_capture, other_capture), 5)
        assert runner.session_store._db.get_session("seg-1")["model"] == "picked"
        if scope == "session":
            assert runner._session_model_override(route)["model"] == "picked"
        else:
            assert runner._session_model_override(route) is None
            assert "picked" in (tmp_path / "config.yaml").read_text()
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_turn_without_route_key_captures_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, source, _route, viewer, root, _ = await _compressed_chat(tmp_path)
    try:
        turn = _turn(runner, source, None, "seg-1")
        await asyncio.wait_for(turn.capture_route_sync_fence(), 5)
        assert turn._route_sync_fence[0] == root
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_reset_remains_route_scoped_with_global_default_policy(tmp_path, monkeypatch):
    import threading
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_authorities import active_authority
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("model:\n  default: old\n  persist_switch_by_default: true\n")
    runner, source, _route, viewer, _root, _ = await _compressed_chat(tmp_path)
    entered, release = threading.Event(), threading.Event()
    tasks = []
    try:
        authority = active_authority(runner)
        other_source = SessionSource(platform=Platform.TELEGRAM, chat_id="5151", user_id="u-5151", chat_type="dm")
        other_id = authority.register(other_source).session_id
        other = _turn(runner, other_source, runner._session_key_for_source(other_source), other_id)

        def default(*args):
            entered.set()
            assert release.wait(20)
            return "old", "openrouter"

        monkeypatch.setattr(runner, "_channel_default_route", default)
        switch = asyncio.create_task(runner._handle_model_command(MessageEvent(text="/model reset", source=source)))
        tasks.append(switch)
        assert await asyncio.to_thread(entered.wait, 5)
        capture = asyncio.create_task(other.capture_route_sync_fence())
        tasks.append(capture)
        await asyncio.wait_for(asyncio.shield(capture), 5)
        assert other._route_sync_fence[0] == other_id
        release.set()
        await asyncio.wait_for(switch, 5)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("persist_by_default", [False, True])
@pytest.mark.parametrize("route_scope", ["config", "channel", "session"])
async def test_model_picker_delivery_does_not_block_turn_admission(tmp_path, monkeypatch, persist_by_default, route_scope):
    from gateway.config import ChannelOverride, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_authorities import active_authority
    import gateway.run as gateway_run
    from hermes_cli import model_switch_providers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: old\n  provider: openrouter\n"
        f"  persist_switch_by_default: {str(persist_by_default).lower()}\n")
    runner, source, route, viewer, root, _ = await _compressed_chat(tmp_path)
    expected_model = "old"
    if route_scope in {"channel", "session"}:
        runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(
            enabled=True,
            channel_overrides={source.chat_id: ChannelOverride(model="channel-model")},
        )
        expected_model = "channel-model"
    if route_scope == "session":
        runner._session_model_overrides[route] = {"model": "session-model", "provider": "openrouter"}
        expected_model = "session-model"
    entered, release = asyncio.Event(), asyncio.Event()
    tasks = []
    try:
        authority = active_authority(runner)
        other_source = SessionSource(platform=Platform.TELEGRAM, chat_id="5151", user_id="u-5151", chat_type="dm")
        other_id = authority.register(other_source).session_id
        other = _turn(runner, other_source, runner._session_key_for_source(other_source), other_id)
        own = _turn(runner, source, route, "seg-1")

        class PickerAdapter:
            async def send_model_picker(self, **kwargs):
                assert kwargs["current_model"] == expected_model
                entered.set()
                await release.wait()
                return SimpleNamespace(success=True)

        monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: PickerAdapter())
        monkeypatch.setattr(model_switch_providers, "list_picker_providers", lambda **kwargs: [
            {"provider": "openrouter", "models": [{"id": "old"}]}])
        switch = AsyncMock(side_effect=AssertionError("Opening a picker must not switch models"))
        monkeypatch.setattr(runner, "_perform_model_switch", switch)
        listing = asyncio.create_task(runner._handle_model_command(MessageEvent(text="/model", source=source)))
        tasks.append(listing)
        await asyncio.wait_for(entered.wait(), 5)
        other_capture = asyncio.create_task(other.capture_route_sync_fence())
        tasks.append(other_capture)
        await asyncio.wait_for(asyncio.shield(other_capture), 5)
        assert other._route_sync_fence[0] == other_id
        own_capture = asyncio.create_task(own.capture_route_sync_fence())
        tasks.append(own_capture)
        await asyncio.wait_for(asyncio.shield(own_capture), 5)
        assert own._route_sync_fence[0] == root
        switch.assert_not_awaited()
        release.set()
        assert await asyncio.wait_for(listing, 5) is None
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await viewer.close()
        runner.session_store.close_all_db_handles()
