"""Persisted transport authority never bypasses current routing or changes bots on retry."""
import asyncio

import pytest

from gateway.config import Platform
from gateway.profile_routing import parse_profile_routes
from tests.gateway.test_kanban_routed_transport import (
    RecordingAdapter, collect, completion, deliver, setup_runner, unseen,
)


def stamped(transport="default", **kwargs):
    return completion(metadata={"scope_id": "guild", "parent_chat_id": "parent",
                                "transport_profile": transport}, **kwargs)


@pytest.mark.parametrize("after_claim", [False, True])
def test_known_primary_does_not_switch_to_live_secondary_when_disconnected(tmp_path, monkeypatch, after_claim):
    runner = setup_runner(tmp_path, monkeypatch)
    primary = runner.adapters[Platform.DISCORD]
    secondary = RecordingAdapter()
    runner._profile_adapters["yuki"][Platform.DISCORD] = secondary
    task = stamped()
    rows = collect(runner) if after_claim else []
    runner.adapters.clear()
    if after_claim:
        assert len(rows) == 1
        asyncio.run(deliver(runner, rows))
    else:
        assert not collect(runner)
    assert not primary.sent and not secondary.sent
    assert unseen(task)
    runner.adapters[Platform.DISCORD] = primary
    asyncio.run(deliver(runner, collect(runner)))
    assert len(primary.sent) == len(primary.handled) == 1
    assert not secondary.sent and not unseen(task)


@pytest.mark.parametrize("transport", ["yuki", "unknown"])
def test_known_secondary_cannot_borrow_shared_primary(tmp_path, monkeypatch, transport):
    runner = setup_runner(tmp_path, monkeypatch)
    task = stamped(transport)
    assert not collect(runner)
    assert unseen(task)
    assert not runner.adapters[Platform.DISCORD].sent


@pytest.mark.parametrize("after_claim", [False, True])
def test_known_primary_still_obeys_runtime_route_reassignment(tmp_path, monkeypatch, after_claim):
    runner = setup_runner(tmp_path, monkeypatch)
    runner._profile_adapters["yuki"][Platform.DISCORD] = RecordingAdapter()
    task = stamped()
    rows = collect(runner) if after_claim else []
    runner.config.profile_routes = parse_profile_routes([
        dict(platform="discord", guild_id="guild", chat_id="parent", profile="other")])
    if after_claim:
        assert len(rows) == 1
        asyncio.run(deliver(runner, rows))
    else:
        assert not collect(runner)
    assert unseen(task)
    assert not runner.adapters[Platform.DISCORD].sent
    assert not runner._profile_adapters["yuki"][Platform.DISCORD].sent


def test_known_transport_does_not_skip_missing_sender_authorization(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch)
    runner._profile_adapters["yuki"][Platform.DISCORD] = RecordingAdapter()
    runner.config.profile_routes = parse_profile_routes([
        dict(platform="discord", user_id="creator", profile="other"),
        dict(platform="discord", chat_id="parent", profile="yuki")])
    task = stamped(user=None)
    assert not collect(runner) and unseen(task)


def test_secondary_transport_can_route_runtime_to_primary_without_changing_bot(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch)
    secondary = RecordingAdapter()
    runner._profile_adapters["yuki"][Platform.DISCORD] = secondary
    runner.config.profile_routes = parse_profile_routes([
        dict(platform="discord", bot_profile="yuki", chat_id="parent", profile="default")])
    task = stamped("yuki", profile="default")
    asyncio.run(deliver(runner, collect(runner)))
    assert len(secondary.sent) == len(secondary.handled) == 1
    assert secondary.handled[0].source.profile == "default"
    assert not runner.adapters[Platform.DISCORD].sent and not unseen(task)


def test_raw_primary_legacy_row_cannot_override_runtime_route(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch)
    task = completion(profile="default")
    assert not collect(runner) and unseen(task)
    # With no competing runtime route, the primary's own legacy destination is still valid.
    runner.config.profile_routes = []
    asyncio.run(deliver(runner, collect(runner)))
    assert len(runner.adapters[Platform.DISCORD].sent) == 1
    assert not unseen(task)


@pytest.mark.parametrize("configured", [set(), {Platform.TELEGRAM}])
def test_other_platform_credential_does_not_block_legacy_shared_route(tmp_path, monkeypatch, configured):
    runner = setup_runner(tmp_path, monkeypatch)
    runner._profile_configured_platforms = {"yuki": configured}
    task = completion()
    asyncio.run(deliver(runner, collect(runner)))
    assert len(runner.adapters[Platform.DISCORD].sent) == 1
    assert not unseen(task)


def test_configured_credential_appearing_after_claim_rewinds_legacy_row(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch)
    task = completion()
    rows = collect(runner)
    assert len(rows) == 1
    runner._profile_configured_platforms = {"yuki": {Platform.DISCORD}}
    asyncio.run(deliver(runner, rows))
    assert unseen(task)
    assert not runner.adapters[Platform.DISCORD].sent


def test_verified_primary_stamp_overrides_runtime_credential_intent(tmp_path, monkeypatch):
    runner = setup_runner(tmp_path, monkeypatch)
    runner._profile_configured_platforms = {"yuki": {Platform.DISCORD}}
    task = stamped()
    asyncio.run(deliver(runner, collect(runner)))
    assert len(runner.adapters[Platform.DISCORD].sent) == 1
    assert not unseen(task)
