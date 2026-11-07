"""Late media must reach the durable FIFO while retaining the provider reply anchor."""
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_ingress_context import native_callback
from gateway.session_envelope import restore_native
from hermes_constants import get_hermes_home
from hermes_state_runtime import list_session_admissions
from tests.gateway.bluebubbles_test_support import Request, adapter


@pytest.mark.asyncio
async def test_same_message_late_media_has_distinct_durable_admissions(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEBUBBLES_ALLOWED_USERS", "sender@example.test")
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    instance = adapter()
    instance.gateway_runner = runner
    runner.adapters[Platform.BLUEBUBBLES] = instance
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="bluebubbles-media")
    runner._wire_adapter_handlers(instance)
    events, downloads = [], []

    async def admit(event):
        with native_callback(runner, event, get_hermes_home()):
            await authority.admit_native(event)
        events.append(event)

    async def download(guid, metadata):
        downloads.append(guid)
        path = tmp_path / (guid + ".txt")
        path.write_text("synthetic attachment " + guid)
        return str(path)

    monkeypatch.setattr(instance, "handle_message", admit)
    monkeypatch.setattr(instance, "_download_attachment", download)
    try:
        assert (await instance._handle_webhook(Request())).status == 200
        for guid in ("first-file", "second-file"):
            request = Request(event="updated-message", attachments=[{
                "guid": guid, "mimeType": "text/plain"}])
            assert (await instance._handle_webhook(request)).status == 200
            assert (await instance._handle_webhook(request)).status == 200
        rows = [row for sid in authority.sessions for row in list_session_admissions(
            authority.db, session_id=sid, pending_only=False)]
        assert len(rows) == 3
        assert len({row["request_id"] for row in rows}) == 3
        assert [event.message_id for event in events] == ["same-message"] * 3
        assert [event.text for event in events] == ["caption", "(attachment)", "(attachment)"]
        assert downloads == ["first-file", "second-file"]
        retained = [restore_native(row["payload"]).media_urls or [] for row in rows]
        assert [len(media) for media in retained] == [0, 1, 1]
        assert [Path(media[0]).read_text() for media in retained[1:]] == [
            "synthetic attachment first-file", "synthetic attachment second-file"]
    finally:
        authority.db.close()
