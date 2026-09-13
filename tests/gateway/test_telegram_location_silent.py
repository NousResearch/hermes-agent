"""Telegram location updates are profile-scoped telemetry, never agent turns."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.profile_routing import ProfileRouteRejected
from gateway.session import SessionSource
from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


_BASE_TIME = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _location_update(
    *,
    update_id=100,
    offset: int = 0,
    chat_id: int = 101,
    thread_id: int | None = None,
    chat_type: str = "private",
):
    location = SimpleNamespace(
        latitude=0.125 + offset / 10_000,
        longitude=-0.25 - offset / 10_000,
        horizontal_accuracy=3.0,
        heading=90,
        speed=1.5,
        live_period=600,
    )
    message = SimpleNamespace(
        location=location,
        venue=None,
        chat=SimpleNamespace(
            id=chat_id,
            type=chat_type,
            title="Synthetic Test Chat",
            full_name=None,
            is_forum=chat_type == "supergroup",
        ),
        from_user=SimpleNamespace(
            id=202,
            username="synthetic-user",
            full_name="Synthetic User",
            is_bot=False,
        ),
        sender_chat=None,
        edit_date=_BASE_TIME + timedelta(seconds=offset),
        date=None,
        message_thread_id=thread_id,
        message_id=303,
        is_topic_message=thread_id is not None,
        text=None,
        caption=None,
        entities=[],
        caption_entities=[],
        reply_to_message=None,
    )
    return SimpleNamespace(
        update_id=update_id,
        effective_message=message,
        message=message,
        edited_message=message,
    )


def _adapter(*, authorized: bool, extra: dict | None = None) -> TelegramAdapter:
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="synthetic-test-token", extra=extra or {})
    )
    adapter.set_authorization_check(lambda *_args, **_kwargs: authorized)
    adapter.set_message_handler(AsyncMock(return_value=None))
    adapter.send = AsyncMock()
    return adapter


def _snapshot_path(home: Path, profile: str | None = None) -> Path:
    base = home / "profiles" / profile if profile else home
    return base / "location" / "latest.json"


def _prepare_profile(home: Path, profile: str) -> None:
    (home / "profiles" / profile).mkdir(parents=True)


def test_unknown_sender_cannot_write_dispatch_or_log_location_identity(
    tmp_path, monkeypatch, caplog
):
    """Changing strict auth to the pairing-friendly prefilter must fail this test."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(authorized=False)

    with caplog.at_level("WARNING"):
        asyncio.run(adapter._handle_location_message(_location_update(), None))

    assert not _snapshot_path(tmp_path).exists()
    adapter._message_handler.assert_not_awaited()
    adapter.send.assert_not_awaited()
    assert adapter._active_sessions == {}
    assert adapter._pending_messages == {}
    assert "Rejected location telemetry from unauthorized source" in caplog.text
    assert "user 202" not in caplog.text
    assert "chat 101" not in caplog.text


def test_routed_location_writes_only_routed_profile(tmp_path, monkeypatch, caplog):
    """Using adapter ownership instead of the event route must fail this test."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(authorized=True)
    adapter._owner_profile = "transport-owner"
    _prepare_profile(tmp_path, "routed-profile")
    adapter.gateway_runner = SimpleNamespace(
        _profile_name_for_source=lambda _source: "routed-profile"
    )

    with caplog.at_level("INFO"):
        asyncio.run(adapter._handle_location_message(_location_update(), None))

    routed = _snapshot_path(tmp_path, "routed-profile")
    payload = json.loads(routed.read_text(encoding="utf-8"))
    assert set(payload) == {
        "accuracy_m", "chat_id", "heading", "is_live", "latitude",
        "live_period", "longitude", "message_id", "message_thread_id",
        "profile", "source", "source_timestamp", "speed_mps", "update_id",
        "updated_at", "user_id",
    }
    assert payload["profile"] == "routed-profile"
    assert payload["latitude"] == 0.125
    assert payload["longitude"] == -0.25
    assert not _snapshot_path(tmp_path).exists()
    assert not _snapshot_path(tmp_path, "transport-owner").exists()
    adapter._message_handler.assert_not_awaited()
    adapter.send.assert_not_awaited()
    assert "0.125" not in caplog.text
    assert "-0.25" not in caplog.text


def test_forum_general_location_uses_canonical_thread_profile_route(
    tmp_path, monkeypatch
):
    """General-topic routing must use the same synthetic thread id as group gating."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(authorized=True, extra={"require_mention": False})
    _prepare_profile(tmp_path, "general-profile")
    adapter.gateway_runner = SimpleNamespace(
        _profile_name_for_source=lambda source: (
            "general-profile" if source.thread_id == "1" else None
        )
    )

    asyncio.run(
        adapter._handle_location_message(
            _location_update(chat_type="supergroup", thread_id=None), None
        )
    )

    assert _snapshot_path(tmp_path, "general-profile").exists()
    assert not _snapshot_path(tmp_path).exists()
    adapter._message_handler.assert_not_awaited()


def test_rejected_profile_route_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(authorized=True)

    def _reject(_source):
        raise ProfileRouteRejected("synthetic-route")

    adapter.gateway_runner = SimpleNamespace(_profile_name_for_source=_reject)
    asyncio.run(adapter._handle_location_message(_location_update(), None))

    assert not list(tmp_path.rglob("latest.json"))
    adapter._message_handler.assert_not_awaited()


def test_failed_profile_resolution_cannot_fall_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(authorized=True)

    def _fail(_source):
        raise RuntimeError("synthetic routing failure")

    adapter.gateway_runner = SimpleNamespace(_profile_name_for_source=_fail)
    asyncio.run(adapter._handle_location_message(_location_update(), None))

    assert not list(tmp_path.rglob("latest.json"))
    adapter._message_handler.assert_not_awaited()


def test_group_and_topic_gates_apply_without_observed_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(
        authorized=True,
        extra={
            "allowed_chats": ["101"],
            "group_allowed_chats": ["101"],
            "allowed_topics": ["7"],
            "require_mention": True,
            "observe_unmentioned_group_messages": True,
        },
    )
    store = Mock()
    adapter.set_session_store(store)

    asyncio.run(
        adapter._handle_location_message(
            _location_update(chat_type="supergroup", thread_id=7), None
        )
    )
    assert _snapshot_path(tmp_path).exists()
    store.append_to_transcript.assert_not_called()

    _snapshot_path(tmp_path).unlink()
    asyncio.run(
        adapter._handle_location_message(
            _location_update(chat_type="supergroup", chat_id=999, thread_id=7), None
        )
    )
    asyncio.run(
        adapter._handle_location_message(
            _location_update(chat_type="supergroup", thread_id=8), None
        )
    )
    assert not _snapshot_path(tmp_path).exists()
    store.append_to_transcript.assert_not_called()


def test_persistence_failure_and_dispatch_boundary_both_fail_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(authorized=True)
    monkeypatch.setattr(
        telegram_adapter,
        "persist_location_snapshot",
        Mock(side_effect=OSError("synthetic write failure")),
        raising=False,
    )

    asyncio.run(adapter._handle_location_message(_location_update(), None))

    event = MessageEvent(
        text="must never escape",
        message_type=MessageType.LOCATION,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="101",
            chat_type="dm",
            user_id="202",
        ),
    )
    asyncio.run(adapter.handle_message(event))
    adapter._message_handler.assert_not_awaited()
    adapter.send.assert_not_awaited()
    assert adapter._active_sessions == {}
    assert adapter._session_tasks == {}
    assert not _snapshot_path(tmp_path).exists()


def test_edited_location_is_not_published_as_platform_event():
    adapter = _adapter(authorized=True)
    update = _location_update()

    assert adapter._normalize_platform_event(update) is None


def _payload(order: int, *, update_id=None) -> dict:
    return {
        "accuracy_m": 3.0,
        "heading": 90,
        "is_live": True,
        "latitude": 0.125 + order / 10_000,
        "live_period": 600,
        "longitude": -0.25 - order / 10_000,
        "message_id": 303,
        "message_thread_id": 7,
        "profile": "routed-profile",
        "source": "telegram",
        "source_timestamp": (_BASE_TIME + timedelta(seconds=order)).isoformat(),
        "speed_mps": 1.5,
        "update_id": order if update_id is None else update_id,
        "updated_at": (_BASE_TIME + timedelta(seconds=order)).isoformat(),
        "user_id": 202,
        "chat_id": 101,
    }


def _process_persist(home: str, barrier, results) -> None:
    import os

    os.environ["HERMES_HOME"] = home
    from plugins.platforms.telegram.location_ingest import persist_location_snapshot

    barrier.wait(timeout=10)
    results.put(persist_location_snapshot(_payload(5), profile="routed-profile"))


def test_snapshot_rejects_malformed_stale_duplicate_and_string_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram.location_ingest import persist_location_snapshot

    _prepare_profile(tmp_path, "routed-profile")
    path = _snapshot_path(tmp_path, "routed-profile")
    assert persist_location_snapshot(_payload(2), profile="routed-profile") is True
    original = path.read_bytes()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert persist_location_snapshot(_payload(2), profile="routed-profile") is False
    assert persist_location_snapshot(_payload(1), profile="routed-profile") is False
    assert path.read_bytes() == original

    path.write_text("{malformed", encoding="utf-8")
    assert persist_location_snapshot(_payload(3), profile="routed-profile") is False
    assert path.read_text(encoding="utf-8") == "{malformed"

    path.write_text(
        json.dumps({**_payload(3), "chat_title": "unexpected state"}),
        encoding="utf-8",
    )
    malformed_schema = path.read_bytes()
    assert persist_location_snapshot(_payload(4), profile="routed-profile") is False
    assert path.read_bytes() == malformed_schema

    path.unlink()
    assert (
        persist_location_snapshot(
            _payload(3, update_id="3"), profile="routed-profile"
        )
        is False
    )
    assert not path.exists()


def test_snapshot_rejects_invalid_live_shape_and_unknown_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram.location_ingest import persist_location_snapshot

    _prepare_profile(tmp_path, "routed-profile")
    invalid_live = {**_payload(1), "live_period": None}
    assert persist_location_snapshot(invalid_live, profile="routed-profile") is False

    extra_metadata = {**_payload(1), "chat_title": "must not persist"}
    assert persist_location_snapshot(extra_metadata, profile="routed-profile") is False
    assert not _snapshot_path(tmp_path, "routed-profile").exists()


def test_thread_and_process_races_converge_on_one_latest_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.telegram.location_ingest import persist_location_snapshot

    _prepare_profile(tmp_path, "routed-profile")
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(
            pool.map(
                lambda order: persist_location_snapshot(
                    _payload(order), profile="routed-profile"
                ),
                range(1, 17),
            )
        )
    assert any(outcomes)
    latest = json.loads(
        _snapshot_path(tmp_path, "routed-profile").read_text(encoding="utf-8")
    )
    assert latest["update_id"] == 16

    _snapshot_path(tmp_path, "routed-profile").unlink()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    results = ctx.Queue()
    processes = [
        ctx.Process(target=_process_persist, args=(str(tmp_path), barrier, results))
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(20)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    assert sorted(results.get(timeout=2) for _ in processes) == [False, True]
