"""Interim commentary MEDIA directives must survive to post-turn delivery (#99409).

Interim assistant commentary is display-cleaned before send
(``GatewayStreamConsumer._clean_for_display`` strips ``MEDIA:<path>``), while every
delivery rail — the post-stream rescan (``_deliver_media_from_response``) and the
normal send (``_extract_response_content``) — scans only the turn's final response.
An interim-only directive was therefore silently dropped: the tag was hidden from
the user but the file was never uploaded.

The turn now retains raw interim payloads that contain ``MEDIA:``, merges their
parsed tags into the final response after a successful completion, and every
downstream rail delivers them. ``extract_media`` dedupes repeated paths inside the
merged payload, so a path echoed in interim + final phases uploads exactly once.
"""

from urllib.parse import unquote

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import (
    ProgressCaptureAdapter,
    _make_runner,
    _run_with_agent,
)

_SESSION_KEY = "agent:main:telegram:group:-1001:17585"


class _MediaCaptureAdapter(ProgressCaptureAdapter):
    """Capture attachment uploads triggered by the completion seam."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.documents = []
        self.image_batches = []

    async def send_document(
        self,
        chat_id,
        file_path,
        caption=None,
        file_name=None,
        reply_to=None,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        self.documents.append(file_path)
        return SendResult(success=True, message_id="doc-1")

    async def send_multiple_images(
        self,
        chat_id,
        images,
        metadata=None,
        human_delay=0.0,
    ) -> SendResult:
        self.image_batches.append({"chat_id": chat_id, "images": images})
        return SendResult(success=True, message_id="imgs")


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


def _agent_with_interim(payload, final="done"):
    """Agent that emits one interim commentary payload, then a plain-text final."""

    class _Agent:
        def __init__(self, **kwargs):
            self.tools = []
            self.interim_assistant_callback = kwargs.get("interim_assistant_callback")

        def run_conversation(
            self, message, conversation_history=None, task_id=None, **kwargs
        ):
            assert self.interim_assistant_callback is not None
            self.interim_assistant_callback(payload)
            return {"final_response": final, "messages": [], "api_calls": 1}

    return _Agent


async def _run_turn(monkeypatch, tmp_path, agent_cls, session_id):
    """One gateway turn; returns the media-capturing adapter and the agent result."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        agent_cls,
        session_id=session_id,
        adapter_cls=_MediaCaptureAdapter,
    )
    assert isinstance(adapter, _MediaCaptureAdapter)
    return adapter, result


async def _completion_seam(adapter, agent_result, response):
    """Drive ``_hmwa_deliver_turn_response`` — the seam every delivery rail reads.

    Returns the text the caller would send (``None`` = already delivered upstream)."""
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=adapter.platform,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    event = MessageEvent(
        text="hi", message_type=MessageType.TEXT, source=source, message_id="1"
    )

    class _Entry:
        session_id = "s1"

    return await runner._hmwa_deliver_turn_response(
        event,
        source,
        _Entry(),
        _SESSION_KEY,
        None,
        agent_result,
        [],
        response,
        None,
        False,
    )


@pytest.mark.asyncio
async def test_interim_media_directive_reaches_post_stream_delivery(
    monkeypatch, tmp_path
):
    """Streamed-final lifecycle: the retained interim tag is uploaded by the rescan."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    agent_cls = _agent_with_interim(f"Rendering the report\nMEDIA:{media_file}")

    adapter, result = await _run_turn(
        monkeypatch, tmp_path, agent_cls, "sess-interim-media"
    )
    # The streamed final text already reached the chat; only the media rescan remains.
    result["already_sent"] = True

    delivered = await _completion_seam(adapter, result, result["final_response"])

    assert delivered is None
    assert adapter.documents == [str(media_file)]


@pytest.mark.asyncio
async def test_interim_media_tag_rides_returned_response_for_normal_send(
    monkeypatch, tmp_path
):
    """Normal-send lifecycle: the merged tag is on the returned response, which the
    adapter pipeline strips for display while extracting the attachment."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    agent_cls = _agent_with_interim(f"Rendering the report\nMEDIA:{media_file}")

    adapter, result = await _run_turn(
        monkeypatch,
        tmp_path,
        agent_cls,
        "sess-interim-media-normal",
    )

    returned = await _completion_seam(adapter, result, result["final_response"])

    assert returned is not None
    assert f"MEDIA:{media_file}" in returned


@pytest.mark.asyncio
async def test_interim_commentary_without_media_changes_nothing(monkeypatch, tmp_path):
    """No MEDIA directive in interim commentary -> result and rails are untouched."""
    agent_cls = _agent_with_interim("just thinking out loud, nothing to attach")

    adapter, result = await _run_turn(
        monkeypatch, tmp_path, agent_cls, "sess-interim-plain"
    )
    assert result["final_response"] == "done"

    result["already_sent"] = True
    await _completion_seam(adapter, result, result["final_response"])

    assert adapter.documents == []
    assert adapter.image_batches == []


@pytest.mark.asyncio
async def test_failed_turn_does_not_publish_interim_attachments(monkeypatch, tmp_path):
    """A failed turn keeps the interim tag off the response and off the rails."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")

    class _FailingAgent:
        def __init__(self, **kwargs):
            self.tools = []
            self.interim_assistant_callback = kwargs.get("interim_assistant_callback")

        def run_conversation(
            self, message, conversation_history=None, task_id=None, **kwargs
        ):
            assert self.interim_assistant_callback is not None
            self.interim_assistant_callback(f"halfway there\nMEDIA:{media_file}")
            return {
                "final_response": "boom",
                "messages": [],
                "api_calls": 1,
                "failed": True,
            }

    adapter, result = await _run_turn(
        monkeypatch,
        tmp_path,
        _FailingAgent,
        "sess-interim-failed",
    )
    assert "MEDIA:" not in result["final_response"]

    result["already_sent"] = True
    await _completion_seam(adapter, result, result["final_response"])

    assert adapter.documents == []


@pytest.mark.asyncio
async def test_path_echoed_in_interim_and_final_uploads_once(monkeypatch, tmp_path):
    """Interim + final repeating the same path -> exactly one upload (dedup in extract)."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    final = f"All done.\nMEDIA:{media_file}"
    agent_cls = _agent_with_interim(
        f"Uploaded the report\nMEDIA:{media_file}", final=final
    )

    adapter, result = await _run_turn(
        monkeypatch, tmp_path, agent_cls, "sess-interim-dedup"
    )
    result["already_sent"] = True

    delivered = await _completion_seam(adapter, result, result["final_response"])

    assert delivered is None
    assert adapter.documents == [str(media_file)]
    uploaded = [unquote(str(p)) for p in adapter.documents]
    assert uploaded.count(str(media_file)) == 1
