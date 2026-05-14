"""Tests for outbound file methods on TeamsAdapter (Task 6).

Covers:

* Public ``send_document`` / ``send_video`` / ``send_voice``.
* DM dispatch via FileConsent card + ``_pending_uploads`` bookkeeping.
* Channel dispatch via Graph SharePoint upload + FileDownload card.
* ``_is_channel_chat`` heuristic — conv_ref override beats id-shape.
* ``_send_attachment`` activity_sender vs send selection on conv_ref presence.

The fixture builds a ``TeamsAdapter`` *without* calling ``connect()`` so we
do not need the real SDK app, aiohttp listener, or MSAL credentials. The
``self._app`` slot is filled with an ``AsyncMock`` whose ``send`` and
``activity_sender.send`` methods record their arguments.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.teams.adapter import TeamsAdapter
from plugins.platforms.teams.cards import (
    FILE_CONSENT_CONTENT_TYPE,
    FILE_DOWNLOAD_INFO_CONTENT_TYPE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter(monkeypatch) -> TeamsAdapter:
    """Build a TeamsAdapter with mocked SDK app, no live network."""
    # Make sure stale env vars don't seep into _sharepoint_site_id / etc.
    for var in (
        "TEAMS_CLIENT_ID",
        "TEAMS_CLIENT_SECRET",
        "TEAMS_TENANT_ID",
        "TEAMS_SHAREPOINT_SITE_ID",
        "TEAMS_SHAREPOINT_FOLDER",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "client_id": "fake-client",
            "client_secret": "fake-secret",
            "tenant_id": "fake-tenant",
        },
    )
    a = TeamsAdapter(cfg)

    # Replace the SDK App with a Mock that has the two send entry points
    # the adapter dispatches through.
    fake_app = MagicMock()
    fake_app.send = AsyncMock(return_value=Mock(id="sent-msg-id"))
    fake_app.activity_sender = MagicMock()
    fake_app.activity_sender.send = AsyncMock(return_value=Mock(id="conv-msg-id"))
    a._app = fake_app
    return a


@pytest.fixture
def doc_file(tmp_path: Path) -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"X" * 100)
    return p


# ---------------------------------------------------------------------------
# Public send_* surface — input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_document_missing_file_returns_failure(adapter):
    result = await adapter.send_document("19:foo@unq.gbl.spaces", "/nonexistent/file.txt")
    assert result.success is False
    assert result.retryable is False
    assert "not found" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_send_document_unreadable_file_returns_failure(adapter, tmp_path):
    # Pass a directory path — open() on it raises IsADirectoryError.
    bad = tmp_path / "as_dir"
    bad.mkdir()
    result = await adapter.send_document("19:foo@unq.gbl.spaces", str(bad))
    assert result.success is False
    # "not found" path treats directories as non-files; either way it's a hard fail.
    assert result.retryable is False


# ---------------------------------------------------------------------------
# DM (FileConsent) dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_document_to_dm_uses_file_consent_card(adapter, doc_file):
    chat_id = "a:dm-thread-id"  # DM-shaped — no @thread.
    result = await adapter.send_document(chat_id, str(doc_file))

    assert result.success is True
    # The send happened via _app.send (no conv_ref stored).
    assert adapter._app.send.await_count == 1
    activity = adapter._app.send.await_args.args[1]
    # The Attachment carries the FileConsentCard contentType.
    atts = activity.attachments
    assert len(atts) == 1
    assert atts[0].content_type == FILE_CONSENT_CONTENT_TYPE
    # Pending uploads bookkeeping populated.
    assert len(adapter._pending_uploads) == 1
    upload_id, payload = next(iter(adapter._pending_uploads.items()))
    assert payload["filename"] == "doc.pdf"
    assert payload["bytes"] == b"X" * 100
    assert payload["chat_id"] == chat_id


@pytest.mark.asyncio
async def test_pending_upload_recorded_with_correct_payload(adapter, doc_file):
    chat_id = "a:dm-thread-id"
    await adapter.send_document(
        chat_id, str(doc_file), caption="hello there", reply_to="parent-msg-1"
    )

    assert len(adapter._pending_uploads) == 1
    payload = next(iter(adapter._pending_uploads.values()))
    assert set(payload.keys()) >= {"filename", "bytes", "chat_id", "caption", "reply_to"}
    assert payload["filename"] == "doc.pdf"
    assert payload["bytes"] == b"X" * 100
    assert payload["chat_id"] == chat_id
    assert payload["caption"] == "hello there"
    assert payload["reply_to"] == "parent-msg-1"


@pytest.mark.asyncio
async def test_send_local_file_uses_basename_not_full_path(adapter, tmp_path):
    nested = tmp_path / "deep" / "deeper"
    nested.mkdir(parents=True)
    file = nested / "report.pdf"
    file.write_bytes(b"PDFDATA")
    chat_id = "a:dm"
    await adapter.send_document(chat_id, str(file))

    payload = next(iter(adapter._pending_uploads.values()))
    assert payload["filename"] == "report.pdf"
    # The card's content also carries the bare basename.
    activity = adapter._app.send.await_args.args[1]
    att = activity.attachments[0]
    assert att.name == "report.pdf"


# ---------------------------------------------------------------------------
# Channel (Graph SharePoint) dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_document_to_channel_uploads_via_graph(adapter, doc_file):
    chat_id = "19:abc@thread.tacv2"
    adapter._sharepoint_site_id = "site1"

    # Stub the lazy-built GraphClient so no MSAL / Graph SDK is touched.
    fake_graph = MagicMock()
    fake_graph.upload_to_sharepoint = AsyncMock(
        return_value="https://sp.example.com/x.pdf"
    )

    async def _ensure():
        adapter._graph = fake_graph
        return fake_graph

    adapter._ensure_graph = _ensure  # type: ignore[assignment]

    result = await adapter.send_document(chat_id, str(doc_file))

    assert result.success is True
    fake_graph.upload_to_sharepoint.assert_awaited_once()
    kwargs = fake_graph.upload_to_sharepoint.await_args.kwargs
    assert kwargs["site_id"] == "site1"
    assert kwargs["filename"] == "doc.pdf"
    assert kwargs["content"] == b"X" * 100
    # Folder is "<base>/<sanitized chat id>".
    assert kwargs["folder_path"] == "hermes/19_abc_at_thread.tacv2"
    # _app.send received an activity carrying a FileDownload attachment.
    adapter._app.send.assert_awaited_once()
    activity = adapter._app.send.await_args.args[1]
    assert activity.attachments[0].content_type == FILE_DOWNLOAD_INFO_CONTENT_TYPE


@pytest.mark.asyncio
async def test_send_document_to_channel_without_sharepoint_config_fails(
    adapter, doc_file
):
    adapter._sharepoint_site_id = ""
    result = await adapter.send_document("19:abc@thread.tacv2", str(doc_file))
    assert result.success is False
    assert result.retryable is False
    assert "SHAREPOINT" in (result.error or "").upper()


@pytest.mark.asyncio
async def test_send_document_to_channel_when_upload_fails(adapter, doc_file):
    adapter._sharepoint_site_id = "site1"
    fake_graph = MagicMock()
    fake_graph.upload_to_sharepoint = AsyncMock(return_value=None)

    async def _ensure():
        return fake_graph

    adapter._ensure_graph = _ensure  # type: ignore[assignment]
    result = await adapter.send_document("19:abc@thread.tacv2", str(doc_file))
    assert result.success is False
    assert result.retryable is True
    assert "upload failed" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# Public send_video / send_voice — they just delegate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_video_dispatches_through_send_local_file(adapter, doc_file):
    sentinel = SendResult(success=True, message_id="ok-video")
    adapter._send_local_file = AsyncMock(return_value=sentinel)  # type: ignore[assignment]
    result = await adapter.send_video("a:dm", str(doc_file), caption="demo")
    assert result is sentinel
    adapter._send_local_file.assert_awaited_once()
    args = adapter._send_local_file.await_args.args
    # (chat_id, path, caption, reply_to)
    assert args[0] == "a:dm"
    assert args[1] == str(doc_file)
    assert args[2] == "demo"


@pytest.mark.asyncio
async def test_send_voice_dispatches_through_send_local_file(adapter, doc_file):
    sentinel = SendResult(success=True, message_id="ok-voice")
    adapter._send_local_file = AsyncMock(return_value=sentinel)  # type: ignore[assignment]
    result = await adapter.send_voice("a:dm", str(doc_file))
    assert result is sentinel
    adapter._send_local_file.assert_awaited_once()


# ---------------------------------------------------------------------------
# _is_channel_chat heuristic
# ---------------------------------------------------------------------------


def test_is_channel_chat_conv_ref_says_channel(adapter):
    chat_id = "any-id-shape"
    ref = Mock()
    ref.conversation = Mock(conversation_type="channel")
    adapter._conv_refs[chat_id] = ref
    assert adapter._is_channel_chat(chat_id) is True


def test_is_channel_chat_conv_ref_says_personal_overrides_id_shape(adapter):
    # ID looks like a channel (19:...@thread.) but conv_ref says personal.
    chat_id = "19:foo@thread.bar"
    ref = Mock()
    ref.conversation = Mock(conversation_type="personal")
    adapter._conv_refs[chat_id] = ref
    assert adapter._is_channel_chat(chat_id) is False


def test_is_channel_chat_no_conv_ref_thread_substring_treated_as_channel(adapter):
    assert adapter._is_channel_chat("19:abc@thread.tacv2") is True


def test_is_channel_chat_no_conv_ref_dm_shape_returns_false(adapter):
    assert adapter._is_channel_chat("a:dm-thread-id") is False
    assert adapter._is_channel_chat("19:user@unq.gbl.spaces") is False


# ---------------------------------------------------------------------------
# _send_attachment dispatch — uses activity_sender when conv_ref present.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_attachment_uses_conv_ref_when_available(adapter, doc_file):
    chat_id = "a:dm-with-ref"
    ref = Mock()
    ref.conversation = Mock(conversation_type="personal")
    adapter._conv_refs[chat_id] = ref
    result = await adapter.send_document(chat_id, str(doc_file))
    assert result.success is True
    adapter._app.activity_sender.send.assert_awaited_once()
    adapter._app.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_attachment_forwards_content_url_to_sdk_attachment(adapter):
    """``_send_attachment`` must forward ``contentUrl`` to the SDK Attachment.

    FileInfoCard / FileDownloadInfoCard responses to a FileConsent invoke have
    ``contentUrl`` at the *top level* of the Bot Framework attachment dict. The
    SDK ``Attachment`` model exposes ``content_url`` as a real field — dropping
    it makes Bot Framework reject the activity with HTTP 400 and the consent
    card buttons unfreeze with no visible result. Pin the four-field wrap.
    """
    chat_id = "a:dm-content-url"
    attachment_dict = {
        "contentType": "application/vnd.microsoft.teams.file.download.info",
        "contentUrl": "https://example.sharepoint.com/sites/x/file.pdf",
        "name": "Canvas Clinical Call.pdf",
        "content": {"uniqueId": "abc-123", "fileType": "pdf"},
    }

    captured: dict = {}

    async def fake_send(_chat_id, activity):
        # MessageActivityInput.add_attachments stashes the SDK Attachment on
        # the activity — grab the first one so the assertion can inspect it.
        captured["attachment"] = activity.attachments[0]
        return Mock(id="mid-1")

    adapter._app.send = AsyncMock(side_effect=fake_send)

    result = await adapter._send_attachment(chat_id, attachment_dict)

    assert result.success is True
    att = captured["attachment"]
    assert att.content_type == attachment_dict["contentType"]
    assert att.content_url == attachment_dict["contentUrl"], (
        "Attachment.content_url must be forwarded from attachment_dict['contentUrl']; "
        "Bot Framework returns 400 for FileInfoCard responses without it."
    )
    assert att.name == attachment_dict["name"]
    assert att.content == attachment_dict["content"]


# ---------------------------------------------------------------------------
# Constructor — extra config & env plumbing
# ---------------------------------------------------------------------------


def test_constructor_picks_up_sharepoint_config_from_extra(monkeypatch):
    for var in ("TEAMS_SHAREPOINT_SITE_ID", "TEAMS_SHAREPOINT_FOLDER"):
        monkeypatch.delenv(var, raising=False)
    cfg = PlatformConfig(
        enabled=True,
        extra={
            "client_id": "x",
            "client_secret": "y",
            "tenant_id": "z",
            "sharepoint_site_id": "site-from-config",
            "sharepoint_folder": "shared/hermes",
        },
    )
    a = TeamsAdapter(cfg)
    assert a._sharepoint_site_id == "site-from-config"
    assert a._sharepoint_folder == "shared/hermes"
    assert a._pending_uploads == {}
    assert a._graph is None
    assert a._token_provider is None


def test_constructor_falls_back_to_env_vars(monkeypatch):
    monkeypatch.setenv("TEAMS_SHAREPOINT_SITE_ID", "site-from-env")
    monkeypatch.setenv("TEAMS_SHAREPOINT_FOLDER", "env/folder")
    cfg = PlatformConfig(
        enabled=True,
        extra={"client_id": "x", "client_secret": "y", "tenant_id": "z"},
    )
    a = TeamsAdapter(cfg)
    assert a._sharepoint_site_id == "site-from-env"
    assert a._sharepoint_folder == "env/folder"


def test_constructor_default_folder_is_hermes(monkeypatch):
    for var in ("TEAMS_SHAREPOINT_SITE_ID", "TEAMS_SHAREPOINT_FOLDER"):
        monkeypatch.delenv(var, raising=False)
    cfg = PlatformConfig(
        enabled=True,
        extra={"client_id": "x", "client_secret": "y", "tenant_id": "z"},
    )
    a = TeamsAdapter(cfg)
    assert a._sharepoint_site_id == ""
    assert a._sharepoint_folder == "hermes"


# ---------------------------------------------------------------------------
# _pending_uploads memory bounds + send-failure cleanup (quality fixes)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_upload_dropped_when_send_fails(
    adapter, doc_file, monkeypatch, caplog
):
    """If FileConsent send fails, the stashed bytes must be popped."""
    monkeypatch.setattr(
        adapter,
        "_send_attachment",
        AsyncMock(
            return_value=SendResult(success=False, error="boom", retryable=True)
        ),
    )
    chat_id = "a:dm-thread-id"
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.teams.adapter"):
        result = await adapter.send_document(chat_id, str(doc_file))

    assert result.success is False
    # Crucially: no leak of bytes when the user never got a card.
    assert len(adapter._pending_uploads) == 0
    assert any(
        "FileConsent send failed" in rec.getMessage()
        for rec in caplog.records
        if rec.levelno == logging.WARNING
    )


@pytest.mark.asyncio
async def test_pending_uploads_bounded_at_max(
    adapter, doc_file, monkeypatch, caplog
):
    """Beyond the cap, oldest entries are evicted FIFO with a warning."""
    monkeypatch.setattr(adapter, "_PENDING_UPLOAD_MAX", 3)

    captured_ids: list[str] = []

    # Send 4 documents, capturing the upload_id registered for each.
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.teams.adapter"):
        for i in range(4):
            before = set(adapter._pending_uploads.keys())
            await adapter.send_document(f"a:dm-{i}", str(doc_file))
            after = set(adapter._pending_uploads.keys())
            new_ids = after - before
            # The send may both add and (on the 4th) evict; what we want
            # is the id that *got registered* this round.
            if new_ids:
                captured_ids.append(next(iter(new_ids)))
            else:
                # Registration + immediate eviction in same call shouldn't
                # happen at MAX=3 with one new send (cap holds 3, we add
                # the 4th, the oldest goes), so this branch is unexpected.
                captured_ids.append("<missing>")

    # Cap holds exactly MAX entries.
    assert len(adapter._pending_uploads) == 3
    # The first registered upload_id is gone.
    assert captured_ids[0] not in adapter._pending_uploads
    # The last 3 are present.
    for uid in captured_ids[1:]:
        assert uid in adapter._pending_uploads
    # The eviction emitted a warning.
    assert any(
        "evicted oldest" in rec.getMessage()
        for rec in caplog.records
        if rec.levelno == logging.WARNING
    )


@pytest.mark.asyncio
async def test_pending_uploads_evict_stale_by_ttl(adapter, doc_file):
    """Entries older than TTL are swept on the next DM send."""
    # First send: register a normal entry.
    await adapter.send_document("a:dm-old", str(doc_file))
    assert len(adapter._pending_uploads) == 1
    old_id = next(iter(adapter._pending_uploads))

    # Manually backdate the entry's timestamp past the TTL (2h ago).
    adapter._pending_uploads[old_id]["ts"] = time.monotonic() - 7200

    # Second send triggers _evict_stale_pending_uploads at the top of
    # _send_dm_file_consent, sweeping the stale entry before registering
    # the new one.
    await adapter.send_document("a:dm-new", str(doc_file))

    assert old_id not in adapter._pending_uploads
    # Only the fresh entry remains.
    assert len(adapter._pending_uploads) == 1


def test_register_pending_upload_sets_timestamp(adapter):
    """_register_pending_upload stamps a monotonic ts on the entry."""
    before = time.monotonic()
    adapter._register_pending_upload(
        "upload-xyz",
        {
            "filename": "f.pdf",
            "bytes": b"x",
            "chat_id": "a:dm",
            "caption": None,
            "reply_to": None,
        },
    )
    after = time.monotonic()

    entry = adapter._pending_uploads["upload-xyz"]
    assert "ts" in entry
    assert isinstance(entry["ts"], float)
    assert before <= entry["ts"] <= after


# ---------------------------------------------------------------------------
# fileConsent/invoke handler — Allow / Decline lifecycle (Task 7)
# ---------------------------------------------------------------------------


def _make_consent_activity(
    *,
    action: str = "accept",
    context: object = None,
    upload_info=None,
):
    """Build a minimal FileConsentInvokeActivity for handler tests.

    Uses ``model_construct`` to skip pydantic validation — the activity
    in production carries a full Bot Framework envelope (from_, conversation,
    id, recipient) that's irrelevant to the handler logic we're exercising.
    """
    from microsoft_teams.api.activities.invoke.file_consent import (
        FileConsentInvokeActivity,
    )
    from microsoft_teams.api.models import FileConsentCardResponse

    response = FileConsentCardResponse.model_construct(
        action=action,
        context=context,
        upload_info=upload_info,
    )
    return FileConsentInvokeActivity.model_construct(value=response)


def _make_ctx(activity, *, conversation_id: str = "dm:user1"):
    """Wrap an activity in the minimal ``ctx`` shape the handler reads.

    Also attaches a ``conversation`` with ``id`` to the activity (so the
    handler can read ``ctx.activity.conversation.id`` for consent-card
    cleanup) and an ``api.conversations.activities(conv_id).delete(...)``
    chain whose ``delete`` is an ``AsyncMock`` tests can inspect.
    """
    ctx = MagicMock()
    ctx.activity = activity
    # Attach conversation.id without blocking pydantic model_construct shape.
    if not hasattr(activity, "conversation") or activity.conversation is None:
        try:
            activity.conversation = MagicMock(id=conversation_id)
        except (AttributeError, TypeError):
            # Pydantic model with frozen fields — skip; tests targeting
            # delete will use _make_ctx_with_api below instead.
            pass
    # api.conversations.activities(conv_id) returns an object with async delete.
    delete_mock = AsyncMock()
    activity_ops = MagicMock()
    activity_ops.delete = delete_mock
    ctx.api = MagicMock()
    ctx.api.conversations.activities = MagicMock(return_value=activity_ops)
    # Stash the delete mock on ctx for direct test inspection.
    ctx._delete_mock = delete_mock
    return ctx


@pytest.fixture
def mock_aiohttp_put(monkeypatch):
    """Patch aiohttp.ClientSession to capture PUT calls and return a
    configurable status. Returns a recorder dict that tests can mutate
    (status, body) and inspect (calls).
    """
    recorder: dict = {"status": 200, "body": "", "calls": [], "raise_on_put": None}

    class _MockResp:
        def __init__(self, status, body):
            self.status, self._body = status, body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def text(self):
            return self._body

    class _MockSession:
        def __init__(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def put(self, url, data=None, headers=None):
            recorder["calls"].append(
                {"url": url, "data": data, "headers": headers}
            )
            if recorder["raise_on_put"] is not None:
                raise recorder["raise_on_put"]
            return _MockResp(recorder["status"], recorder["body"])

    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", _MockSession)
    return recorder


def _register(adapter, upload_id: str, *, filename="r.pdf", data=b"PDFCONTENT", chat_id="dm:user1", activity_id="consent-msg-id"):
    adapter._register_pending_upload(
        upload_id,
        {
            "filename": filename,
            "bytes": data,
            "chat_id": chat_id,
            "caption": None,
            "reply_to": None,
            "activity_id": activity_id,
        },
    )


@pytest.mark.asyncio
async def test_file_consent_invoke_decline_pops_entry_no_put(
    adapter, mock_aiohttp_put, caplog
):
    _register(adapter, "u1")
    activity = _make_consent_activity(
        action="decline", context={"upload_id": "u1"}
    )

    with caplog.at_level(logging.INFO, logger="plugins.platforms.teams.adapter"):
        result = await adapter._handle_file_consent_invoke(_make_ctx(activity))

    assert result is None
    assert adapter._pending_uploads == {}
    assert mock_aiohttp_put["calls"] == []
    assert any(
        "declined" in rec.getMessage().lower() for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_file_consent_invoke_unknown_upload_id_logs_and_returns(
    adapter, mock_aiohttp_put, caplog
):
    activity = _make_consent_activity(
        action="accept", context={"upload_id": "missing"}
    )

    with caplog.at_level(logging.INFO, logger="plugins.platforms.teams.adapter"):
        result = await adapter._handle_file_consent_invoke(_make_ctx(activity))

    assert result is None
    assert mock_aiohttp_put["calls"] == []
    assert any("unknown upload_id" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_file_consent_invoke_missing_value_returns_none(
    adapter, mock_aiohttp_put, caplog
):
    from microsoft_teams.api.activities.invoke.file_consent import (
        FileConsentInvokeActivity,
    )

    activity = FileConsentInvokeActivity.model_construct(value=None)
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.teams.adapter"):
        result = await adapter._handle_file_consent_invoke(_make_ctx(activity))

    assert result is None
    assert mock_aiohttp_put["calls"] == []
    assert any("without value" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_file_consent_invoke_accept_no_upload_url_warns(
    adapter, mock_aiohttp_put, caplog
):
    from microsoft_teams.api.models import FileUploadInfo

    _register(adapter, "u-no-url")
    upload_info = FileUploadInfo.model_construct(
        upload_url=None, content_url="https://x/y", unique_id="id1", file_type="pdf"
    )
    activity = _make_consent_activity(
        action="accept",
        context={"upload_id": "u-no-url"},
        upload_info=upload_info,
    )

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.teams.adapter"):
        result = await adapter._handle_file_consent_invoke(_make_ctx(activity))

    assert result is None
    # Entry was popped (consumed) even though we couldn't PUT.
    assert adapter._pending_uploads == {}
    assert mock_aiohttp_put["calls"] == []
    assert any(
        "missing upload_info.upload_url" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_file_consent_invoke_accept_happy_path(
    adapter, mock_aiohttp_put, monkeypatch
):
    from microsoft_teams.api.models import FileUploadInfo

    _register(
        adapter,
        "u-happy",
        filename="r.pdf",
        data=b"PDFCONTENT",
        chat_id="dm:user1",
    )
    upload_info = FileUploadInfo.model_construct(
        upload_url="https://onedrive.example/upload-session/123",
        content_url="https://onedrive.example/files/r.pdf",
        unique_id="graph-drive-item-abc",
        file_type="pdf",
        name="r.pdf",
    )
    activity = _make_consent_activity(
        action="accept",
        context={"upload_id": "u-happy"},
        upload_info=upload_info,
    )

    # Capture the FileInfoCard payload sent via _send_attachment.
    send_attachment_mock = AsyncMock(return_value=SendResult(success=True, message_id="m1"))
    monkeypatch.setattr(adapter, "_send_attachment", send_attachment_mock)

    result = await adapter._handle_file_consent_invoke(_make_ctx(activity))

    assert result is None
    assert adapter._pending_uploads == {}

    # Exactly one PUT, with the expected single-shot OneDrive headers.
    assert len(mock_aiohttp_put["calls"]) == 1
    call = mock_aiohttp_put["calls"][0]
    assert call["url"] == "https://onedrive.example/upload-session/123"
    assert call["data"] == b"PDFCONTENT"
    headers = call["headers"]
    assert headers["Content-Type"] == "application/octet-stream"
    assert headers["Content-Length"] == "10"
    assert headers["Content-Range"] == "bytes 0-9/10"

    # FileInfoCard follow-up sent to the same chat_id.
    send_attachment_mock.assert_awaited_once()
    args, kwargs = send_attachment_mock.await_args
    assert args[0] == "dm:user1"
    card = args[1]
    assert card["contentType"] == "application/vnd.microsoft.teams.card.file.info"
    # The Graph drive-item id is echoed through, not a fresh uuid.
    assert card["content"]["uniqueId"] == "graph-drive-item-abc"
    assert card["content"]["fileType"] == "pdf"


@pytest.mark.asyncio
async def test_file_consent_invoke_accept_put_500_no_followup(
    adapter, mock_aiohttp_put, monkeypatch, caplog
):
    from microsoft_teams.api.models import FileUploadInfo

    _register(adapter, "u-500")
    upload_info = FileUploadInfo.model_construct(
        upload_url="https://onedrive.example/u",
        content_url="https://x/y",
        unique_id="id",
        file_type="pdf",
    )
    activity = _make_consent_activity(
        action="accept",
        context={"upload_id": "u-500"},
        upload_info=upload_info,
    )
    mock_aiohttp_put["status"] = 500
    mock_aiohttp_put["body"] = "internal error"

    send_attachment_mock = AsyncMock(return_value=SendResult(success=True))
    monkeypatch.setattr(adapter, "_send_attachment", send_attachment_mock)

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.teams.adapter"):
        result = await adapter._handle_file_consent_invoke(_make_ctx(activity))

    assert result is None
    assert len(mock_aiohttp_put["calls"]) == 1
    send_attachment_mock.assert_not_awaited()
    assert any(
        "PUT failed status=500" in rec.getMessage() for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_file_consent_invoke_accept_put_transport_error_no_followup(
    adapter, mock_aiohttp_put, monkeypatch, caplog
):
    import aiohttp
    from microsoft_teams.api.models import FileUploadInfo

    _register(adapter, "u-boom")
    upload_info = FileUploadInfo.model_construct(
        upload_url="https://onedrive.example/u",
        content_url="https://x/y",
        unique_id="id",
        file_type="pdf",
    )
    activity = _make_consent_activity(
        action="accept",
        context={"upload_id": "u-boom"},
        upload_info=upload_info,
    )
    mock_aiohttp_put["raise_on_put"] = aiohttp.ClientError("boom")

    send_attachment_mock = AsyncMock(return_value=SendResult(success=True))
    monkeypatch.setattr(adapter, "_send_attachment", send_attachment_mock)

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.teams.adapter"):
        result = await adapter._handle_file_consent_invoke(_make_ctx(activity))

    assert result is None
    send_attachment_mock.assert_not_awaited()
    assert any(
        "transport error" in rec.getMessage() for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_file_consent_invoke_accept_followup_failure_logs(
    adapter, mock_aiohttp_put, monkeypatch, caplog
):
    from microsoft_teams.api.models import FileUploadInfo

    _register(adapter, "u-followup")
    upload_info = FileUploadInfo.model_construct(
        upload_url="https://onedrive.example/u",
        content_url="https://x/y",
        unique_id="id",
        file_type="pdf",
    )
    activity = _make_consent_activity(
        action="accept",
        context={"upload_id": "u-followup"},
        upload_info=upload_info,
    )
    # PUT succeeds, but the FileInfoCard send fails.
    send_attachment_mock = AsyncMock(
        return_value=SendResult(success=False, error="x", retryable=True)
    )
    monkeypatch.setattr(adapter, "_send_attachment", send_attachment_mock)

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.teams.adapter"):
        result = await adapter._handle_file_consent_invoke(_make_ctx(activity))

    assert result is None
    send_attachment_mock.assert_awaited_once()
    assert any(
        "FileInfoCard follow-up failed" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_file_consent_invoke_evicts_stale_before_lookup(
    adapter, mock_aiohttp_put
):
    from microsoft_teams.api.models import FileUploadInfo

    _register(adapter, "u-stale")
    # Backdate so the entry is stale by the time the invoke arrives.
    adapter._pending_uploads["u-stale"]["ts"] = time.monotonic() - 7200

    upload_info = FileUploadInfo.model_construct(
        upload_url="https://onedrive.example/u",
        content_url="https://x/y",
        unique_id="id",
        file_type="pdf",
    )
    activity = _make_consent_activity(
        action="accept",
        context={"upload_id": "u-stale"},
        upload_info=upload_info,
    )
    result = await adapter._handle_file_consent_invoke(_make_ctx(activity))

    assert result is None
    # Eviction happened before lookup → unknown-id path → no PUT.
    assert mock_aiohttp_put["calls"] == []
    assert "u-stale" not in adapter._pending_uploads


@pytest.mark.asyncio
async def test_file_consent_invoke_context_not_dict_returns_unknown(
    adapter, mock_aiohttp_put, caplog
):
    activity = _make_consent_activity(
        action="accept", context="stale-context-from-old-card"
    )

    with caplog.at_level(logging.INFO, logger="plugins.platforms.teams.adapter"):
        result = await adapter._handle_file_consent_invoke(_make_ctx(activity))

    assert result is None
    assert mock_aiohttp_put["calls"] == []
    assert any("unknown upload_id" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Cross-loop bridge: gateway-loop capture
#
# The Microsoft Teams SDK's ``App`` caches asyncio primitives bound to the
# loop that constructed it. Tools dispatched via model_tools._run_async
# run on a *worker* loop in a sidecar thread — awaiting App methods from
# there raises::
#
#     RuntimeError: <Event ... [unset]> is bound to a different event loop
#
# tools/send_message_tool._send_teams bridges over this by hopping to
# ``adapter._loop`` via ``asyncio.run_coroutine_threadsafe``. That bridge
# only works if connect() actually captures the loop. These tests pin the
# capture/clear contract.
# ---------------------------------------------------------------------------


def test_loop_attribute_is_none_pre_connect(adapter):
    """Fresh adapter: ``_loop`` is the documented sentinel (None)."""
    assert adapter._loop is None


@pytest.mark.asyncio
async def test_connect_captures_running_loop(monkeypatch, adapter):
    """Stub the heavy bits of connect() (App init, aiohttp listener) and
    confirm the running gateway loop lands in ``self._loop``."""
    import asyncio as _asyncio

    # Stub out the SDK App + its initialize() coro. The fixture already
    # filled ``adapter._app`` with a mock; we replace the App() *class*
    # used inside connect() so the new instance also matches.
    fake_app = MagicMock()
    fake_app.initialize = AsyncMock()
    fake_app.on_message = lambda fn: fn
    fake_app.on_card_action = lambda fn: fn
    fake_app.on_file_consent = lambda fn: fn

    monkeypatch.setattr(
        "plugins.platforms.teams.adapter.App", lambda **kw: fake_app
    )
    monkeypatch.setattr(
        "plugins.platforms.teams.adapter.TEAMS_SDK_AVAILABLE", True
    )
    monkeypatch.setattr(
        "plugins.platforms.teams.adapter.AIOHTTP_AVAILABLE", True
    )

    # Stub out the aiohttp listener so we don't bind a port.
    fake_runner = MagicMock()
    fake_runner.setup = AsyncMock()
    fake_runner.cleanup = AsyncMock()
    monkeypatch.setattr(
        "plugins.platforms.teams.adapter.web.AppRunner", lambda app: fake_runner
    )

    fake_site = MagicMock()
    fake_site.start = AsyncMock()
    monkeypatch.setattr(
        "plugins.platforms.teams.adapter.web.TCPSite",
        lambda runner, host, port: fake_site,
    )

    ok = await adapter.connect()
    try:
        assert ok is True
        assert adapter._loop is _asyncio.get_running_loop(), (
            "connect() must capture the running event loop so cross-loop "
            "tool calls can bridge back via run_coroutine_threadsafe"
        )
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_disconnect_clears_loop(adapter):
    """``disconnect()`` clears ``_loop`` so a stale tool call doesn't try
    to schedule on a torn-down loop."""
    import asyncio as _asyncio

    # Pretend connect() ran: stamp the loop manually rather than going
    # through the full stub path.
    adapter._loop = _asyncio.get_running_loop()
    adapter._runner = None  # the fixture leaves this unset; disconnect tolerates it

    await adapter.disconnect()
    assert adapter._loop is None


# ---------------------------------------------------------------------------
# Consent card cleanup — both Accept and Decline must delete the card so
# Teams' UI doesn't leave the buttons re-enabling indefinitely.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_consent_decline_deletes_consent_card(
    adapter, mock_aiohttp_put
):
    """Decline path must delete the original FileConsentCard activity."""
    _register(adapter, "u1", chat_id="dm:user1", activity_id="consent-act-1")
    activity = _make_consent_activity(
        action="decline", context={"upload_id": "u1"}
    )
    ctx = _make_ctx(activity, conversation_id="dm:user1")

    await adapter._handle_file_consent_invoke(ctx)

    # The handler should have called ctx.api.conversations.activities("dm:user1").delete("consent-act-1")
    ctx.api.conversations.activities.assert_called_once_with("dm:user1")
    ctx._delete_mock.assert_awaited_once_with("consent-act-1")


@pytest.mark.asyncio
async def test_file_consent_accept_deletes_consent_card(
    adapter, mock_aiohttp_put
):
    """Accept path must also delete the original card after PUT succeeds."""
    from microsoft_teams.api.models import FileUploadInfo

    _register(adapter, "u1", chat_id="dm:user1", activity_id="consent-act-2")
    upload_info = FileUploadInfo.model_construct(
        upload_url="https://onedrive.example/upload",
        content_url="https://example/content",
        unique_id="uid",
        file_type="pdf",
        name="r.pdf",
    )
    activity = _make_consent_activity(
        action="accept",
        context={"upload_id": "u1"},
        upload_info=upload_info,
    )
    ctx = _make_ctx(activity, conversation_id="dm:user1")
    mock_aiohttp_put["status"] = 201

    # Stub _post_file_info_card so we don't go down the FileInfo path.
    adapter._post_file_info_card = AsyncMock(return_value=None)

    await adapter._handle_file_consent_invoke(ctx)

    ctx.api.conversations.activities.assert_called_once_with("dm:user1")
    ctx._delete_mock.assert_awaited_once_with("consent-act-2")


@pytest.mark.asyncio
async def test_file_consent_delete_failure_swallowed(
    adapter, mock_aiohttp_put, caplog
):
    """A failed delete must not propagate or break the invoke handler."""
    _register(adapter, "u1", chat_id="dm:user1", activity_id="consent-act-3")
    activity = _make_consent_activity(
        action="decline", context={"upload_id": "u1"}
    )
    ctx = _make_ctx(activity, conversation_id="dm:user1")
    ctx._delete_mock.side_effect = RuntimeError("graph 404")

    # Should not raise.
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.teams.adapter"):
        result = await adapter._handle_file_consent_invoke(ctx)

    assert result is None
    # Warning logged.
    assert any(
        "consent" in rec.getMessage().lower() and "delete" in rec.getMessage().lower()
        for rec in caplog.records
    )
