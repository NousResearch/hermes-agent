"""Tests for the Teams file-consent path and the rejection diagnostics.

A local non-image file is offered with a FileConsentCard in a personal chat,
uploaded to the pre-authenticated OneDrive session on Allow, and shown as a
FileInfoCard; declining or a failed upload posts a plain-text note naming the
file and reports no delivery. Group chats and channels -- no consent flow
exists there -- fail without retry instead of posting a data: URI Bot
Framework rejects. Every rejected activity logs one ERROR line carrying the
file name, MIME type, size, HTTP status and the response body, so a 400 is
diagnosable without opening the container.

The adapter is imported plain (the venv has the SDK); the instance is built
with object.__new__ so no App/auth is constructed, sends are captured by
stubbing _send_via_conv_ref / send, and the consent directory is a tmp_path.
The upload goes through an httpx.MockTransport -- nothing leaves the process.

    ./venv/bin/python -m pytest tests/plugins/test_teams_file_consent.py -q
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from plugins.platforms.teams import adapter as teams_adapter
from plugins.platforms.teams.adapter import TeamsAdapter

PERSONAL = "a:1personal-chat-id"
GROUP = "19:abc@thread.v2"
PDF = b"%PDF-1.4 hello"
BF_REJECTION = '{"error":{"code":"BadArgument","message":"Unknown attachment type"}}'
REJECT_URL = "https://smba.trafficmanager.net/ca/TENANT/v3/conversations/DM/activities"


def _run(coro):
    return asyncio.run(coro)


def _adapter(tmp_path, conv_type: str | None = None, chat_id: str = PERSONAL):
    inst = object.__new__(TeamsAdapter)
    inst._app = object()
    inst._conv_refs = {}
    if conv_type:
        inst._conv_refs[chat_id] = SimpleNamespace(conversation=SimpleNamespace(conversation_type=conv_type))
    inst._file_consent_dir = lambda: tmp_path
    inst._card_action_denied = lambda from_account: None
    inst.sent_activities = []
    inst.sent_texts = []

    async def _via(chat, activity, fallback):
        inst.sent_activities.append((chat, activity))
        return SimpleNamespace(id="m1")

    async def _send(chat, text, *a, **kw):
        inst.sent_texts.append((chat, text))

    inst._send_via_conv_ref = _via
    inst.send = _send
    return inst


def _rejecting_adapter(tmp_path, body: str = BF_REJECTION, status: int = 400):
    """An adapter whose every send raises the HTTP error Bot Framework raised on the card."""
    inst = _adapter(tmp_path, conv_type="personal")

    async def _reject(chat, activity, fallback):
        raise httpx.HTTPStatusError(
            f"Client error '{status} Bad Request' for url {REJECT_URL}",
            request=httpx.Request("POST", REJECT_URL), response=httpx.Response(status, text=body))

    inst._send_via_conv_ref = _reject
    return inst


def _attachments(activity):
    return list(getattr(activity, "attachments", None) or [])


def _errors(caplog):
    return [r for r in caplog.records if r.levelname == "ERROR"]


def _offer(inst, tmp_path, name="report.pdf", body=PDF):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    f = src / name
    f.write_bytes(body)
    result = _run(inst.send_document(PERSONAL, str(f), caption="Q3 report"))
    assert result.success, result.error
    card = _attachments(inst.sent_activities[-1][1])[0]
    return card, card.content["acceptContext"]["hermes_file_token"]


def _invoke(token, action="accept", chat_id=PERSONAL, upload_url="https://tenant-my.sharepoint.com/upload/s1"):
    return SimpleNamespace(
        conversation_ref=SimpleNamespace(conversation=SimpleNamespace(conversation_type="personal")),
        activity=SimpleNamespace(
            from_=SimpleNamespace(id="u-lucas", aad_object_id="u-lucas"),
            conversation=SimpleNamespace(id=chat_id),
            value=SimpleNamespace(
                action=action,
                context={"hermes_file_token": token},
                upload_info=SimpleNamespace(
                    name="report.pdf", upload_url=upload_url,
                    content_url="https://tenant-my.sharepoint.com/personal/x/report.pdf",
                    unique_id="UID-1", file_type="pdf"),
            ),
        ),
    )


async def _answer(inst, ctx):
    await inst._on_file_consent(ctx)
    for task in list(getattr(inst, "_file_consent_tasks", ())):
        await task


class TestOffer:
    def test_personal_chat_sends_consent_card_and_snapshots(self, tmp_path) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        card, token = _offer(inst, tmp_path)
        assert card.content_type == teams_adapter._FILE_CONSENT_CARD_TYPE
        assert card.content["sizeInBytes"] == len(PDF)
        assert card.content["description"] == "Q3 report"
        assert card.content["declineContext"] == {"hermes_file_token": token}
        assert teams_adapter._FILE_CONSENT_TOKEN_RE.match(token)
        assert (tmp_path / f"{token}.bin").read_bytes() == PDF
        meta = json.loads((tmp_path / f"{token}.json").read_text())
        assert meta["chat_id"] == PERSONAL and meta["name"] == "report.pdf"

    def test_file_name_overrides_path_basename(self, tmp_path) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        f = tmp_path / "tmp123.bin"
        f.write_bytes(b"x")
        result = _run(inst.send_document(PERSONAL, str(f), file_name="Budget.xlsx"))
        assert result.success
        assert _attachments(inst.sent_activities[-1][1])[0].name == "Budget.xlsx"

    def test_group_chat_fails_without_retry_and_sends_nothing(self, tmp_path) -> None:
        inst = _adapter(tmp_path, conv_type="groupChat", chat_id=GROUP)
        f = tmp_path / "report.pdf"
        f.write_bytes(b"data")
        result = _run(inst.send_document(GROUP, str(f)))
        assert not result.success and result.retryable is False
        assert inst.sent_activities == []

    def test_uncached_group_id_is_not_personal(self, tmp_path) -> None:
        inst = _adapter(tmp_path)
        assert inst._is_personal_chat(GROUP) is False
        assert inst._is_personal_chat(PERSONAL) is True
        assert inst._is_personal_chat("19:u1_u2@unq.gbl.spaces") is True

    def test_empty_file_refused_and_cleaned(self, tmp_path) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        result = _run(inst.send_document(PERSONAL, str(f)))
        assert not result.success and result.retryable is False
        assert list(tmp_path.glob("*.bin")) == [] and inst.sent_activities == []

    def test_missing_source_file_is_not_retryable(self, tmp_path) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        result = _run(inst.send_document(PERSONAL, str(tmp_path / "gone.pdf")))
        assert not result.success and result.retryable is False
        assert list(tmp_path.glob("*.bin")) == [] and inst.sent_activities == []

    def test_local_image_still_uses_data_uri(self, tmp_path) -> None:
        inst = _adapter(tmp_path, conv_type="groupChat", chat_id=GROUP)
        f = tmp_path / "chart.png"
        f.write_bytes(b"\x89PNG")
        result = _run(inst.send_image(GROUP, str(f)))
        assert result.success
        att = _attachments(inst.sent_activities[-1][1])[0]
        assert att.content_url.startswith("data:image/png;base64,")

    def test_offer_carries_no_inline_file_bytes(self, tmp_path) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        _offer(inst, tmp_path)
        assert not any((a.content_url or "").startswith("data:") for a in _attachments(inst.sent_activities[-1][1]))


class TestConsentInvoke:
    @pytest.fixture
    def uploads(self, monkeypatch):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            content_range = request.headers["Content-Range"]
            seen.append((content_range, request.content))
            span, total = content_range.removeprefix("bytes ").split("/")
            last = int(span.split("-")[1])
            return httpx.Response(201 if last + 1 == int(total) else 202)

        real = httpx.AsyncClient
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        return seen

    def test_accept_uploads_sends_file_info_and_cleans(self, tmp_path, uploads) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        _, token = _offer(inst, tmp_path)
        _run(_answer(inst, _invoke(token)))
        assert uploads == [("bytes 0-13/14", PDF)]
        info = _attachments(inst.sent_activities[-1][1])[0]
        assert info.content_type == teams_adapter._FILE_INFO_CARD_TYPE
        assert info.content == {"uniqueId": "UID-1", "fileType": "pdf"}
        assert info.content_url.endswith("/report.pdf")
        assert not list(tmp_path.glob(f"{token}*"))

    def test_chunks_carry_content_range(self, tmp_path, uploads, monkeypatch) -> None:
        monkeypatch.setattr(teams_adapter, "_FILE_UPLOAD_CHUNK_BYTES", 5)
        inst = _adapter(tmp_path, conv_type="personal")
        _, token = _offer(inst, tmp_path, body=b"0123456789ab")
        _run(_answer(inst, _invoke(token)))
        assert [r for r, _ in uploads] == ["bytes 0-4/12", "bytes 5-9/12", "bytes 10-11/12"]
        assert b"".join(c for _, c in uploads) == b"0123456789ab"

    def test_second_accept_does_not_upload_twice(self, tmp_path, uploads) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        _, token = _offer(inst, tmp_path)
        _run(_answer(inst, _invoke(token)))
        _run(_answer(inst, _invoke(token)))
        assert len(uploads) == 1
        assert "expired or was already answered" in inst.sent_texts[-1][1]

    def test_decline_drops_snapshot_without_upload_and_says_so(self, tmp_path, uploads) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        _, token = _offer(inst, tmp_path)
        _run(_answer(inst, _invoke(token, action="decline")))
        assert uploads == [] and not list(tmp_path.glob(f"{token}*"))
        assert "Could not deliver report.pdf" in inst.sent_texts[-1][1]
        assert "declined" in inst.sent_texts[-1][1]
        assert len(inst.sent_activities) == 1  # the offer card only, never a file card

    def test_bad_token_is_ignored(self, tmp_path, uploads) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        _, token = _offer(inst, tmp_path)
        _run(_answer(inst, _invoke("../../etc/passwd")))
        assert uploads == [] and (tmp_path / f"{token}.json").exists()

    def test_unauthorized_clicker_drops_offer(self, tmp_path, uploads) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        inst._card_action_denied = lambda from_account: "Not authorized."
        _, token = _offer(inst, tmp_path)
        _run(_answer(inst, _invoke(token)))
        assert uploads == [] and not list(tmp_path.glob(f"{token}*"))
        assert inst.sent_texts[-1][1] == "Not authorized."

    def test_answer_from_another_chat_is_refused(self, tmp_path, uploads) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        _, token = _offer(inst, tmp_path)
        _run(_answer(inst, _invoke(token, chat_id="a:other-chat")))
        assert uploads == [] and not list(tmp_path.glob(f"{token}*"))

    def test_non_https_upload_url_reports_failure(self, tmp_path, uploads) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        _, token = _offer(inst, tmp_path)
        _run(_answer(inst, _invoke(token, upload_url="http://evil.example/upload")))
        assert uploads == []
        assert "Could not deliver report.pdf" in inst.sent_texts[-1][1]
        assert not list(tmp_path.glob(f"{token}*"))

    def test_non_sharepoint_upload_host_is_refused(self, tmp_path, uploads) -> None:
        inst = _adapter(tmp_path, conv_type="personal")
        _, token = _offer(inst, tmp_path)
        _run(_answer(inst, _invoke(token, upload_url="https://sharepoint.com.evil.example/upload")))
        assert uploads == []
        assert "Could not deliver report.pdf" in inst.sent_texts[-1][1]
        assert not list(tmp_path.glob(f"{token}*"))

    def test_rejected_chunk_reports_failure(self, tmp_path, monkeypatch) -> None:
        real = httpx.AsyncClient
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(
            transport=httpx.MockTransport(lambda r: httpx.Response(403, text="quota")), **kw))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        inst = _adapter(tmp_path, conv_type="personal")
        _, token = _offer(inst, tmp_path)
        _run(_answer(inst, _invoke(token)))
        text = inst.sent_texts[-1][1]
        assert "Could not deliver report.pdf" in text and "HTTP 403" in text
        assert "quota" not in text and "sharepoint.com" not in text
        assert not list(tmp_path.glob(f"{token}*"))


class TestRejectionDiagnostics:
    """A rejected activity must leave one line naming what was sent and what came back."""

    def test_inline_rejection_logs_one_line_with_the_facts(self, tmp_path, caplog) -> None:
        inst = _rejecting_adapter(tmp_path)
        chart = tmp_path / "chart.png"
        chart.write_bytes(b"\x89PNG" + b"p" * 900)
        with caplog.at_level(logging.ERROR, logger=teams_adapter.logger.name):
            result = _run(inst.send_image(PERSONAL, str(chart)))
        assert not result.success
        errors = _errors(caplog)
        assert len(errors) == 1
        line = errors[0].getMessage()
        assert "file=chart.png" in line and "mime=image/png" in line
        assert "size=904" in line and "status=400" in line and "BadArgument" in line
        assert "p" * 64 not in line  # never the attachment's own bytes
        assert not errors[0].exc_info  # an HTTP rejection is not a traceback

    def test_offer_rejection_names_the_member_file(self, tmp_path, caplog) -> None:
        inst = _rejecting_adapter(tmp_path)
        f = tmp_path / "tmp1234.bin"
        f.write_bytes(PDF)
        with caplog.at_level(logging.ERROR, logger=teams_adapter.logger.name):
            result = _run(inst.send_document(PERSONAL, str(f), file_name="Budget.xlsx"))
        assert not result.success
        errors = _errors(caplog)
        assert len(errors) == 1
        line = errors[0].getMessage()
        assert "file=Budget.xlsx" in line
        assert "mime=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in line
        assert f"size={len(PDF)}" in line and "status=400" in line
        assert "Unknown attachment type" in line

    def test_receipt_falls_back_to_the_mime_we_sent(self, tmp_path, caplog) -> None:
        inst = _rejecting_adapter(tmp_path)
        nameless = tmp_path / "October2026"
        nameless.write_bytes(b"\x89PNG")
        with caplog.at_level(logging.ERROR, logger=teams_adapter.logger.name):
            _run(inst.send_image(PERSONAL, str(nameless)))
        assert "file=October2026 mime=image/png" in _errors(caplog)[0].getMessage()

    def test_failed_upload_logs_status_and_body(self, tmp_path, caplog, monkeypatch) -> None:
        real = httpx.AsyncClient
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(
            transport=httpx.MockTransport(lambda r: httpx.Response(403, text="upload session expired")), **kw))
        monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
        inst = _adapter(tmp_path, conv_type="personal")
        _, token = _offer(inst, tmp_path)
        with caplog.at_level(logging.ERROR, logger=teams_adapter.logger.name):
            _run(_answer(inst, _invoke(token)))
        errors = _errors(caplog)
        assert len(errors) == 1
        line = errors[0].getMessage()
        assert "file=report.pdf" in line and "mime=application/pdf" in line
        assert f"size={len(PDF)}" in line and "status=403" in line
        assert "upload session expired" in line
