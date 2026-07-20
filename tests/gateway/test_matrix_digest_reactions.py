"""Tests for Matrix 🧾 digest-detail reactions."""

import asyncio
import sys
import types
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock


def _stub_mautrix():
    stub = types.ModuleType("mautrix")
    for sub in (
        "mautrix.types",
        "mautrix.client",
        "mautrix.client.api",
        "mautrix.errors",
        "mautrix.crypto",
        "mautrix.util",
        "mautrix.util.config",
    ):
        sys.modules.setdefault(sub, types.ModuleType(sub))
    sys.modules.setdefault("mautrix", stub)
    m = sys.modules["mautrix.types"]

    class EventType:
        ROOM_MESSAGE = "m.room.message"
        REACTION = "m.reaction"
        ROOM_ENCRYPTED = "m.room.encrypted"
        ROOM_NAME = "m.room.name"

    class PaginationDirection:
        BACKWARD = "b"
        FORWARD = "f"

    class PresenceState:
        ONLINE = "online"
        OFFLINE = "offline"
        UNAVAILABLE = "unavailable"

    class RoomCreatePreset:
        PRIVATE = "private_chat"
        PUBLIC = "public_chat"
        TRUSTED_PRIVATE = "trusted_private_chat"

    class TrustState:
        UNVERIFIED = 0
        VERIFIED = 1

    for attr in ("ContentURI", "EventID", "RoomID", "SyncToken", "UserID"):
        setattr(m, attr, str)
    m.EventType = EventType
    m.PaginationDirection = PaginationDirection
    m.PresenceState = PresenceState
    m.RoomCreatePreset = RoomCreatePreset
    m.TrustState = TrustState


_stub_mautrix()

from plugins.platforms.matrix.adapter import MatrixAdapter  # noqa: E402


def _make_adapter(allowed_user_ids=None):
    adapter = object.__new__(MatrixAdapter)
    adapter._user_id = "@bot:matrix.org"
    adapter._allowed_user_ids = set(allowed_user_ids or [])
    adapter._processed_events = deque(maxlen=512)
    adapter._processed_events_set = set()
    adapter._approval_prompts_by_event = {}
    adapter._approval_prompt_by_session = {}
    adapter._model_picker_prompts_by_event = {}
    adapter._choice_picker_prompts_by_event = {}
    adapter._digest_detail_prompts_by_event = {}
    adapter._approval_require_sender = True
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="$selection"))
    adapter._send_reaction = AsyncMock(return_value="$reaction")
    return adapter


def _make_reaction(sender, reacts_to, key="🧾", event_id="$reaction-from-user"):
    return SimpleNamespace(
        sender=sender,
        event_id=event_id,
        room_id="!room:example.org",
        content={"m.relates_to": {"event_id": reacts_to, "key": key}},
    )


def _write_source_output(tmp_path, job_id, response):
    output_dir = tmp_path / "cron" / "output" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "2026-06-28_08-00-00.md"
    path.write_text(
        "# Cron Job\n\n## Prompt\ninternal prompt\n\n## Response\n" + response,
        encoding="utf-8",
    )
    return path


def test_digest_reaction_sends_single_source_detail_reply(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    from cron.digest_reactions import register_digest_delivery

    _write_source_output(tmp_path, "source-a", "📌 **Befund**\n- detail A\n")
    register_digest_delivery(
        room_id="!room:example.org",
        event_id="$digest",
        digest_job={"id": "digest-job", "name": "Morning Digest"},
        source_job_ids=["source-a"],
        source_names={"source-a": "Source A"},
    )

    adapter = _make_adapter(allowed_user_ids=["@alice:example.org"])
    asyncio.run(adapter._on_reaction(_make_reaction("@alice:example.org", "$digest")))

    adapter.send.assert_awaited_once()
    args, kwargs = adapter.send.await_args
    assert args[0] == "!room:example.org"
    assert "**🧾 Einzelbericht: Source A**" in args[1]
    assert "detail A" in args[1]
    assert "internal prompt" not in args[1]
    assert kwargs["reply_to"] == "$digest"


def test_digest_reaction_offers_selection_for_multiple_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    from cron.digest_reactions import register_digest_delivery

    _write_source_output(tmp_path, "source-a", "detail A")
    _write_source_output(tmp_path, "source-b", "detail B")
    register_digest_delivery(
        room_id="!room:example.org",
        event_id="$digest",
        digest_job={"id": "digest-job", "name": "Morning Digest"},
        source_job_ids=["source-a", "source-b"],
        source_names={"source-a": "Source A", "source-b": "Source B"},
    )

    adapter = _make_adapter(allowed_user_ids=["@alice:example.org"])
    asyncio.run(adapter._on_reaction(_make_reaction("@alice:example.org", "$digest")))

    args, kwargs = adapter.send.await_args
    assert "Mehrere Einzelberichte verfügbar" in args[1]
    assert "1️⃣ Source A" in args[1]
    assert "2️⃣ Source B" in args[1]
    assert kwargs["reply_to"] == "$digest"
    assert adapter._send_reaction.await_count == 2

    asyncio.run(
        adapter._on_reaction(
            _make_reaction("@alice:example.org", "$selection", key="2️⃣", event_id="$select-b")
        )
    )

    second_args, second_kwargs = adapter.send.await_args
    assert "**🧾 Einzelbericht: Source B**" in second_args[1]
    assert "detail B" in second_args[1]
    assert second_kwargs["reply_to"] == "$digest"


def test_digest_reaction_denies_unlisted_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    from cron.digest_reactions import register_digest_delivery

    _write_source_output(tmp_path, "source-a", "detail A")
    register_digest_delivery(
        room_id="!room:example.org",
        event_id="$digest",
        digest_job={"id": "digest-job", "name": "Morning Digest"},
        source_job_ids=["source-a"],
        source_names={"source-a": "Source A"},
    )

    adapter = _make_adapter(allowed_user_ids=["@alice:example.org"])
    asyncio.run(adapter._on_reaction(_make_reaction("@mallory:example.org", "$digest")))

    args, kwargs = adapter.send.await_args
    assert "Only an authorized Matrix user" in args[1]
    assert kwargs["reply_to"] == "$digest"
