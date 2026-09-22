"""Non-chat subscriptions must reach local consumers through both transports."""

import asyncio
import json
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.secret_scope import set_multiplex_active
from gateway.config import PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.feishu import adapter as fa


EVENT_KEYS = (
    "minutes.minute.generated_v1",
    "vc.meeting.participant_meeting_ended_v1",
    "vc.meeting.participant_meeting_started_v1",
    "vc.meeting.participant_meeting_joined_v1",
    "task.task.update_user_access_v2",
    "task.task.update_tenant_v1",
    "approval.task.status_changed_v4",
    "approval.instance.status_changed_v4",
    "vc.recording.recording_started_v1",
    "vc.recording.recording_ended_v1",
    "vc.recording.recording_transcript_generated_v1",
    "vc.meeting.all_meeting_started_v1",
    "vc.meeting.all_meeting_ended_v1",
)


def _dispatch(adapter, transport, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if transport == "websocket":
        pytest.importorskip("lark_oapi")
        fa._load_lark_oapi()
        # This is the SDK dispatch entry point used by its WebSocket client.
        adapter._build_event_handler()._do_without_validation(body)
    else:
        pytest.importorskip("aiohttp")

        async def readexactly(size):
            raise asyncio.IncompleteReadError(body, size)

        request = SimpleNamespace(
            remote="127.0.0.1", content_length=len(body), headers={},
            content=SimpleNamespace(readexactly=readexactly),
        )
        response = asyncio.run(adapter._handle_webhook_request(request))
        assert response.status == 200


def _payload(event_key, event_id):
    return {
        "schema": "2.0",
        "header": {
            "event_type": event_key, "event_id": event_id,
            "create_time": "1234567890", "token": "test-verification-token",
        },
        "event": {"items": [{"title": "会议纪要", "done": False}], "revision": 2},
    }


@pytest.fixture
def multiplex():
    set_multiplex_active(True)
    try:
        yield
    finally:
        set_multiplex_active(False)


@pytest.mark.usefixtures("multiplex")
@pytest.mark.parametrize("transport", ["websocket", "webhook"])
@pytest.mark.parametrize("event_key", EVENT_KEYS)
def test_registered_event_preserves_payload_in_owning_profile(tmp_path, transport, event_key):
    homes = [tmp_path / "a", tmp_path / "b"]
    expected = {home: [] for home in homes}
    for index, home in enumerate([homes[0], homes[1], homes[0]]):
        token = set_hermes_home_override(str(home))
        try:
            adapter = fa.FeishuAdapter(PlatformConfig(extra={
                "verification_token": "test-verification-token",
            }))
            payload = _payload(event_key, f"test-event-{index}")
            _dispatch(adapter, transport, payload)
            expected[home].append(payload)
        finally:
            reset_hermes_home_override(token)

    for home, payloads in expected.items():
        lines = (home / "feishu-user" / "agent-events.jsonl").read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert len(records) == len(payloads)
        for record, payload in zip(records, payloads):
            assert record["event_key"] == payload["header"]["event_type"]
            assert record["header"] == {
                key: payload["header"][key] for key in ("event_id", "event_type", "create_time")
            }
            assert record["payload"] == payload["event"]
            assert isinstance(record["ts"], int)


@pytest.fixture
def socket_home():
    # Keep the path within AF_UNIX's limit, independently of pytest's test-name suffix.
    with tempfile.TemporaryDirectory(prefix="fe-") as directory:
        yield Path(directory)


@pytest.mark.parametrize("_host", [
    pytest.param("macos", marks=pytest.mark.macos_only),
    pytest.param("linux", marks=pytest.mark.linux_only),
])
def test_socket_notification_follows_journal_and_io_failures_are_visible(socket_home, caplog, _host):
    token = set_hermes_home_override(str(socket_home))
    try:
        adapter = fa.FeishuAdapter(PlatformConfig())
        base = socket_home / "feishu-user"
        base.mkdir()
        journal = base / "agent-events.jsonl"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(base / "agent-events.sock"))
            server.listen(1)
            server.settimeout(2)
            _dispatch(adapter, "websocket", _payload(EVENT_KEYS[0], "first"))
            assert journal.stat().st_mode & 0o777 == 0o600
            with server.accept()[0] as connection:
                connection.settimeout(2)
                with connection.makefile("rb") as stream:
                    assert stream.readline() == journal.read_bytes()

        # A stale socket must not lose the record or escape into SDK dispatch.
        _dispatch(adapter, "websocket", _payload(EVENT_KEYS[0], "second"))
        assert len(journal.read_text(encoding="utf-8").splitlines()) == 2
        assert "persisted" in caplog.text and "notification failed" in caplog.text

        # A filesystem error must be observable, not mistaken for successful forwarding.
        journal.unlink()
        journal.mkdir()
        caplog.clear()
        _dispatch(adapter, "websocket", _payload(EVENT_KEYS[0], "third"))
        assert "Failed to persist" in caplog.text
        assert "notification failed" not in caplog.text
    finally:
        reset_hermes_home_override(token)
