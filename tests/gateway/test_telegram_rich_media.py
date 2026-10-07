"""Bot API 10.2/10.3 rich media tests: embedded documents/photos/videos, expandable quotes,
compact tables, and the permanent-vs-transient send policy.

PTB serialization is proven where the real library is importable: a subprocess-level check
runs the REAL ``RequestParameter``/``RequestData`` pipeline over the mixin's payload (the
repo's shared ``tests/gateway/conftest.py`` installs a MagicMock ``telegram`` in-process,
which makes in-process "real PTB" assertions meaningless). The in-process tests run under
that mock — exactly the environment CI uses — and the mock-agnostic contract (pure JSON
entries, attach:// URIs, hoisted top-level file params) is asserted on the payload dicts.
"""

import io
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from plugins.platforms.telegram import telegram_rich_media as rm
from plugins.platforms.telegram.telegram_rich_media import (
    RichMediaError, TelegramRichMediaMixin, build_rich_message, content_needs_rich_media,
    hoist_rich_media_uploads, media_ref_html, new_media_id, rich_media_document, rich_media_spec)

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[2]

_REAL_PTB_PROBE = r"""
import json, io, sys
sys.path.insert(0, r"{repo}")
from plugins.platforms.telegram.telegram_rich_media import rich_media_spec, TelegramRichMediaMixin
from telegram.request._requestparameter import RequestParameter
from telegram.request._requestdata import RequestData
from telegram import InputFile

class Host(TelegramRichMediaMixin):
    def _coerce_bool_extra(self, key, default=False):
        return True
    def _metadata_thread_id(self, metadata):
        return None
    def _reply_to_message_id_for_send(self, reply_to, metadata=None, reply_to_mode=None):
        return None
    def _thread_kwargs_for_send(self, *a, **k):
        return {{}}
    def _notification_kwargs(self, metadata):
        return {{}}
    def _is_rich_fallback_error(self, exc):
        return False
    def _is_rich_capability_error(self, exc):
        return False

host = Host()
specs = [
    dict(rich_media_spec("document", io.BytesIO(b"PDFBYTES"), media_id="doc1"), filename="report.pdf"),
    rich_media_spec("photo", "PHOTOFILEID", media_id="ph1"),
]
payload = host._rich_media_payload(42, "Report: tg://document?id=doc1", specs, None, None)
params = [RequestParameter.from_input(k, v) for k, v in payload.items()]
data = RequestData(parameters=params)
out = {{
    "json": data.json_parameters,
    "files": {{k: v[0] for k, v in data.multipart_data.items()}},
}}
print("___RESULT___")
print(json.dumps(out))
"""


def _probe_real_ptb():
    """Run the serialization probe in a clean interpreter (no conftest mock)."""
    code = _REAL_PTB_PROBE.format(repo=str(_REPO))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f"real python-telegram-bot unavailable: {proc.stderr.strip()[-300:]}")
    marker = proc.stdout.rfind("___RESULT___")
    return json.loads(proc.stdout[marker + len("___RESULT___"):].strip())


# --- host stub ---------------------------------------------------------------------------
class Host(TelegramRichMediaMixin):
    """Bare mixin host with the adapter semantics the transport composes."""

    def __init__(self, rich_messages=True, rich_send_disabled=False):
        self.config = SimpleNamespace(extra={"rich_messages": rich_messages})
        self._rich_send_disabled = rich_send_disabled
        self._reply_to_mode = "on"
        self._bot = SimpleNamespace(do_api_request=AsyncMock(return_value={"message_id": 777}))

    def _coerce_bool_extra(self, key, default=False):
        value = self.config.extra.get(key, default)
        return value if not isinstance(value, str) else value.lower() in {"1", "true", "yes", "on"}

    def _is_rich_fallback_error(self, exc):
        s = str(exc).lower()
        return "bad request" in s or "unsupported" in s or "not implemented" in s or "method not found" in s

    def _is_rich_capability_error(self, exc):
        return "method not found" in str(exc).lower()

    def _metadata_thread_id(self, metadata):
        return (metadata or {}).get("thread_id")

    def _reply_to_message_id_for_send(self, reply_to, metadata=None, reply_to_mode=None):
        if reply_to:
            return int(reply_to)
        if (metadata or {}).get("reply_to_message_id") and (reply_to_mode or self._reply_to_mode) != "off":
            return int(metadata["reply_to_message_id"])
        return None

    def _thread_kwargs_for_send(self, chat_id, thread_id, metadata=None, reply_to_message_id=None, reply_to_mode=None):
        return {"message_thread_id": int(thread_id)} if thread_id else {}

    def _notification_kwargs(self, metadata):
        if (metadata or {}).get("notify") is False:
            return {"disable_notification": True}
        return {}


def _payload(host, media_specs, content="Report **ready**", chat_id=42, reply_to=None, metadata=None):
    return host._rich_media_payload(chat_id, content, media_specs, reply_to, metadata)


def _entry(payload, media_id):
    for entry in payload["rich_message"]["media"]:
        if entry["id"] == media_id:
            return entry
    raise AssertionError(f"no media entry {media_id}")


# --- payload purity: media ids, refs, specs ---------------------------------------------

def test_new_media_id_matches_official_charset():
    for _ in range(20):
        assert rm._MEDIA_ID_RE.match(new_media_id("doc"))
    assert len(new_media_id()) <= 64


def test_rich_media_spec_rejects_invalid_id_and_kind():
    with pytest.raises(RichMediaError):
        rich_media_spec("document", "x", media_id="bad id!")
    with pytest.raises(RichMediaError):
        rich_media_spec("document", "x", media_id="x" * 65)
    with pytest.raises(RichMediaError):
        rich_media_spec("sticker", "x")  # unsupported kind fails loudly, never silently dropped


def test_rich_media_document_requires_exactly_one_source():
    with pytest.raises(RichMediaError):
        rich_media_document()  # none
    with pytest.raises(RichMediaError):
        rich_media_document(pathlib.Path("a.pdf"), file_id="FID")  # both


def test_rich_media_document_file_id_is_payload_pure():
    spec = rich_media_document(file_id="FILEID123", caption="Q3 report")
    assert spec["content"] == "Q3 report"
    assert spec["media"]["kind"] == "document"
    assert spec["media"]["media"] == "FILEID123"  # string stays a reference, never wrapped


def test_rich_media_document_path_is_upload_not_file_id():
    spec = rich_media_document(pathlib.Path("report.pdf"), caption="Q3")
    assert isinstance(spec["media"]["media"], pathlib.Path)  # path-like ⇒ upload path


def test_media_ref_html_references_id_and_carries_caption():
    ref = media_ref_html("document", "doc1", caption="Q3 report")
    assert 'src="tg://document?id=doc1"' in ref
    assert ref.startswith("<figure>") and "<figcaption>Q3 report</figcaption>" in ref
    assert media_ref_html("photo", "ph1") == '<img src="tg://photo?id=ph1"/>'


def test_media_ref_html_rejects_unknown_kind():
    with pytest.raises(RichMediaError):
        media_ref_html("sticker", "s1")


# --- raw-content detection: 10.3 grammar -------------------------------------------------

@pytest.mark.parametrize("content,expected", [
    ("<blockquote expandable>Long note<br>…<cite>Author</cite></blockquote>", True),
    ("<blockquote>plain quote</blockquote>", False),
    ("<table compact><tr><td>1</td></tr></table>", True),
    ('<table bordered striped><tr><td>1</td></tr></table>', True),
    ("<table><tr><td>1</td></tr></table>", False),
    ('<tg-document src="tg://document?id=doc1"></tg-document>', True),
    ('<img src="tg://photo?id=ph1"/>', True),
    ('<img src=""/>', False),
    ("see tg://document?id=doc1 inline", True),
    ("see tg://user?id=5 (not media)", False),
    ("plain **markdown** reply", False),
    ("", False),
])
def test_content_needs_rich_media_grammar(content, expected):
    assert content_needs_rich_media(content) is expected


def test_expandable_quote_alone_triggers_rich_rendering():
    assert content_needs_rich_media('<blockquote expandable>x</blockquote>')


def test_compact_table_alone_triggers_rich_rendering():
    assert content_needs_rich_media("<table compact><tr><td>only</td></tr></table>")


# --- build_rich_message: exactly one format ---------------------------------------------

def test_build_rich_message_none_without_metadata_key():
    assert build_rich_message("hello", None) is None
    assert build_rich_message("hello", {"other": 1}) is None


def test_build_rich_message_markdown_with_normalizer():
    meta = {"telegram_rich_message": {"markdown": "line1\nline2"}}
    payload = build_rich_message("ignored", meta, normalizer=lambda t: t.replace("\n", "  \n"))
    assert payload == {"markdown": "line1  \nline2"}


def test_build_rich_message_rejects_multiple_formats():
    meta = {"telegram_rich_message": {"html": "<b>x</b>", "markdown": "x"}}
    with pytest.raises(RichMediaError):
        build_rich_message("x", meta)


def test_build_rich_message_rejects_zero_formats():
    with pytest.raises(RichMediaError):
        build_rich_message("x", {"telegram_rich_message": {}})


def test_build_rich_message_blocks_passthrough():
    blocks = [{"type": "expandable_blockquote", "text": {"type": "plain", "text": "q"}}]
    assert build_rich_message("x", {"telegram_rich_message": {"blocks": blocks}}) == {"blocks": blocks}


def test_build_rich_message_file_id_media_list_is_pure_json():
    meta = {"telegram_rich_message": {
        "markdown": "doc: tg://document?id=doc1",
        "media": [{"id": "doc1", "kind": "document", "media": "FILEID9"}]}}
    payload = build_rich_message("x", meta)
    # Reuse path: entry is a plain JSON dict — wire shape {"type","media"}, no PTB object.
    assert payload["media"] == [{"id": "doc1", "media": {"type": "document", "media": "FILEID9"}}]
    assert json.dumps(payload)  # fully JSON-serializable as-is


def test_build_rich_message_rejects_over_limit_media():
    meta = {"telegram_rich_message": {
        "markdown": "x", "media": [{"kind": "photo", "media": "F%d" % i} for i in range(51)]}}
    with pytest.raises(RichMediaError):
        build_rich_message("x", meta)


def test_hoist_rich_media_uploads_moves_input_file_markers():
    # Upload entry shape (InputFile mocked away): marker key is what the hoist consumes.
    fake_file = object()
    payload = {"rich_message": {"markdown": "x", "media": [
        {"id": "doc1", "media": {"type": "document", "media": "attach://doc1"}, "_input_file": fake_file},
        {"id": "ph1", "media": {"type": "photo", "media": "PHOTOFILEID"}},
    ]}}
    hoist_rich_media_uploads(payload)
    assert payload["doc1"] is fake_file  # top-level param named by media id
    assert "_input_file" not in payload["rich_message"]["media"][0]
    assert "ph1" not in payload  # reused media contributes no parameter


def test_real_ptb_new_upload_and_file_id_one_request():
    """The transport's core promise, proven against REAL PTB 22.8: caption + media in ONE
    sendRichMessage request — multipart bytes for new uploads, pure JSON for file_id reuse."""
    out = _probe_real_ptb()
    jp, files = out["json"], out["files"]
    # RequestData.json_parameters maps name → JSON-encoded string; decode the object params.
    rich = json.loads(jp["rich_message"])
    assert rich["markdown"] == "Report: tg://document?id=doc1"
    # New upload: InputFile hoisted as top-level param + multipart part named by the media id.
    assert jp["doc1"] == "attach://doc1"  # string params pass through unencoded
    media = {m["id"]: m["media"] for m in rich["media"]}
    assert media["doc1"] == {"type": "document", "media": "attach://doc1"}
    assert files == {"doc1": "report.pdf"}
    # Reused file_id: pure JSON entry, no upload part, no top-level param.
    assert media["ph1"] == {"type": "photo", "media": "PHOTOFILEID"}
    assert set(files) == {"doc1"}  # exactly one upload — one request carries everything
    assert "ph1" not in jp


# --- send policy: routing preserved, permanent vs transient ------------------------------

@pytest.mark.asyncio
async def test_send_rich_media_success_one_call():
    host = Host()
    spec = rich_media_spec("document", "FILEID9", media_id="doc1")
    msg = await host._send_rich_media(42, "Report: tg://document?id=doc1", [spec], None, None)
    assert msg == {"message_id": 777}
    host._bot.do_api_request.assert_awaited_once()
    call = host._bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    kwargs = call.kwargs["api_kwargs"]
    assert kwargs["chat_id"] == 42
    assert kwargs["rich_message"]["markdown"] == "Report: tg://document?id=doc1"


@pytest.mark.asyncio
async def test_send_rich_media_flag_off_returns_none():
    host = Host(rich_messages=False)
    assert await host._send_rich_media(42, "x", [], None, None) is None
    host._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_rich_media_permanent_rejection_returns_none_for_fallback():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Bad Request: media invalid"))
    assert await host._send_rich_media(42, "x", [], None, None) is None  # caller falls back


@pytest.mark.asyncio
async def test_send_rich_media_capability_error_latches_off():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Endpoint sendRichMessage method not found"))
    assert await host._send_rich_media(42, "x", [], None, None) is None
    assert host._rich_send_disabled is True
    assert await host._send_rich_media(42, "x", [], None, None) is None  # latched: no retry roundtrip
    assert host._bot.do_api_request.await_count == 1


@pytest.mark.asyncio
async def test_send_rich_media_transient_is_ambiguous_not_fallback():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("timed out"))
    with pytest.raises(rm._RichMediaTransient):
        await host._send_rich_media(42, "x", [], None, None)
    result = host._rich_media_transient_result(rm._RichMediaTransient(RuntimeError("timed out"), "timed out"))
    assert isinstance(result, SendResult)
    assert result.success is False
    assert result.retryable is False  # ambiguous: may have landed — never re-send
    assert result.raw_response["ambiguous"] is True


@pytest.mark.asyncio
async def test_send_rich_media_response_without_message_id_is_ambiguous():
    host = Host()
    host._bot.do_api_request = AsyncMock(return_value={"ok": True})
    with pytest.raises(rm._RichMediaTransient):
        await host._send_rich_media(42, "x", [], None, None)


@pytest.mark.asyncio
async def test_send_rich_media_preserves_routing_and_notifications():
    host = Host()
    metadata = {"thread_id": "8", "reply_to_message_id": "555", "notify": False}
    await host._send_rich_media(-100123, "x", [], None, metadata)
    kwargs = host._bot.do_api_request.call_args.kwargs["api_kwargs"]
    assert kwargs["chat_id"] == -100123
    assert kwargs["message_thread_id"] == 8
    assert kwargs["reply_parameters"] == {"message_id": 555}
    assert kwargs["disable_notification"] is True


@pytest.mark.asyncio
async def test_send_rich_media_reply_anchor_from_reply_to():
    host = Host()
    await host._send_rich_media(42, "x", [], "99", None)
    kwargs = host._bot.do_api_request.call_args.kwargs["api_kwargs"]
    assert kwargs["reply_parameters"] == {"message_id": 99}


@pytest.mark.asyncio
async def test_caption_and_document_single_rich_message_no_legacy_call():
    """Acceptance core: caption + document in ONE sendRichMessage — the payload carries BOTH
    the caption (markdown body) and the media entry; exactly one transport call happens."""
    host = Host()
    spec = rich_media_spec("document", io.BytesIO(b"PDF"), media_id="doc1")
    spec["filename"] = "r.pdf"
    await host._send_rich_media(42, "Q3 report: tg://document?id=doc1", [spec], None, None)
    host._bot.do_api_request.assert_awaited_once()
    payload = host._bot.do_api_request.call_args.kwargs["api_kwargs"]
    rich = payload["rich_message"]
    assert "tg://document?id=doc1" in rich["markdown"]  # caption body
    assert rich["media"][0]["id"] == "doc1"  # same request carries the media
    assert "doc1" in payload  # the upload rides the same request as a top-level param


@pytest.mark.asyncio
async def test_explicit_rich_body_without_plain_fallback_is_delivered():
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter
    adapter = TelegramAdapter(PlatformConfig(extra={"rich_messages": True}, typing_indicator=False))
    adapter._bot = SimpleNamespace(do_api_request=AsyncMock(return_value={"message_id": 9}))
    adapter._record_rich_sent = AsyncMock()
    result = await adapter.send("42", "", metadata={"telegram_rich_message": {"html": "<b>answer</b>"}})
    assert result.success and result.message_id == "9"


@pytest.mark.asyncio
async def test_invalid_explicit_rich_body_is_nonretryable_without_network():
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter
    adapter = TelegramAdapter(PlatformConfig(extra={"rich_messages": True}, typing_indicator=False))
    adapter._bot = SimpleNamespace(do_api_request=AsyncMock())
    result = await adapter.send("42", "fallback", metadata={"telegram_rich_message": {}})
    assert not result.success and not result.retryable
    adapter._bot.do_api_request.assert_not_awaited()
