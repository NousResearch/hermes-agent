"""Transport continuity and fail-closed legacy delivery for #101335."""
import asyncio
from unittest.mock import AsyncMock, patch
from contextlib import nullcontext

import pytest

from gateway.config import Platform
from gateway.session import SessionContext, SessionSource
from gateway.session_identity import canonical_identity
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from tests.gateway.test_kanban_routed_transport import (
    RecordingAdapter, collect, deliver, setup_runner, unseen,
)
from tools.kanban_tools import _maybe_auto_subscribe


def test_auto_subscription_keeps_receiving_primary_bot_and_runtime_owner(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch)
    primary = runner.adapters[Platform.DISCORD]
    secondary = RecordingAdapter()
    runner._profile_adapters["yuki"][Platform.DISCORD] = secondary
    source = SessionSource(platform=Platform.DISCORD, chat_id="post", thread_id="post",
                           chat_type="thread", user_id="creator", guild_id="guild",
                           parent_chat_id="parent")
    canonical_identity(source, runner=runner, adapter=primary, transport_profile="default")
    tokens = runner._set_session_env(SessionContext(source=source, connected_platforms=[], home_channels={}))
    try:
        with kbc.connect() as conn, patch("tools.kanban_tools.load_config", return_value={}):
            task = kb.create_task(conn, title="routed child", assignee="worker")
            assert _maybe_auto_subscribe(conn, task)
            sub = kbn.list_notify_subs(conn, task)[0]
            assert sub["notifier_profile"] == "yuki"
            assert sub["delivery_metadata"]["transport_profile"] == "default"
            kb.complete_task(conn, task, result="finished")
    finally:
        runner._clear_session_env(tokens)
    asyncio.run(deliver(runner, collect(runner)))
    assert len(primary.sent) == len(primary.handled) == 1
    assert primary.handled[0].source.profile == "yuki"
    assert not secondary.sent and not secondary.handled
    assert not unseen(task)
    assert not collect(runner)


@pytest.mark.parametrize("creation", ["absent", "raises"])
def test_configured_credential_without_adapter_never_claims_primary(tmp_path, monkeypatch, creation):
    from gateway.config import GatewayConfig, PlatformConfig
    from tests.gateway.test_kanban_routed_transport import completion

    runner = setup_runner(tmp_path, monkeypatch)
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fixture-token")})
    runner._load_secondary_profile_config = AsyncMock(return_value=config)
    monkeypatch.setattr("gateway.run._profile_runtime_scope", lambda *a, **kw: nullcontext())
    monkeypatch.setattr("gateway.run._platform_has_bot_credential", lambda *a: True)
    def create(*args):
        if creation == "raises":
            raise RuntimeError("fixture factory failure")
    runner._create_adapter = create
    assert asyncio.run(runner._start_one_profile_adapters("yuki", tmp_path / "yuki", {})) == 0
    assert not runner._profile_adapters["yuki"]
    assert not (getattr(runner, "_profile_failed_platforms", {}) or {}).get("yuki")
    task = completion()
    assert not collect(runner)
    assert unseen(task)
    assert not runner.adapters[Platform.DISCORD].sent
    # A later rescan that proves no credential clears only this platform's old intent.
    monkeypatch.setattr("gateway.run._platform_has_bot_credential", lambda *a: False)
    assert asyncio.run(runner._start_one_profile_adapters("yuki", tmp_path / "yuki", {})) == 0
    asyncio.run(deliver(runner, collect(runner)))
    assert len(runner.adapters[Platform.DISCORD].sent) == 1
    assert not unseen(task)


def test_inferred_primary_identity_is_not_persisted_as_known_transport(tmp_path, monkeypatch):
    from gateway.session_context import get_session_env
    runner = setup_runner(tmp_path, monkeypatch)
    source = SessionSource(platform=Platform.DISCORD, chat_id="post", profile="yuki")
    identity = canonical_identity(source, runner=runner)
    assert identity is not None and identity.transport_inferred
    tokens = runner._set_session_env(SessionContext(source=source, connected_platforms=[], home_channels={}))
    try:
        assert get_session_env("HERMES_SESSION_TRANSPORT_PROFILE") == ""
    finally:
        runner._clear_session_env(tokens)


def test_unserve_forgets_configured_credential_boundary(tmp_path, monkeypatch):
    from gateway.runtime_ownership import process_ownership
    from hermes_constants import mark_named_profile_deleted
    from tests.gateway.test_multiplex_hot_serve import _runner, _mkprofile

    runner, home = _runner(tmp_path, monkeypatch)
    profile = _mkprofile(home, "satellite")
    runner._profile_configured_platforms = {"satellite": {Platform.DISCORD}}
    async def remove():
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            await runner._start_secondary_profile_adapters()
            mark_named_profile_deleted(profile)
            result = await runner.reconcile_served_profiles()
        assert result["removed"] == ["satellite"]
        assert "satellite" not in runner._profile_configured_platforms
    try:
        asyncio.run(remove())
    finally:
        for reserved in process_ownership.homes:
            process_ownership.release(reserved)
