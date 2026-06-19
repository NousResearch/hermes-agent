"""End-to-end regression for issue #48912 — queued native-image drain path.

Native vision mode buffers image paths per session at the SET site
(``_prepare_inbound_message_text``) and pops them at the CONSUME site
(``_run_agent`` → ``_consume_pending_native_image_paths``).  Issue #48912:
when the two sites resolve the session key by different means, the keys
diverge on long-lived / threaded / compression-healed sessions, ``pop()``
returns ``[]``, and the image is silently dropped.

The canonical fix (#48922, already on this branch) threads the
caller-resolved ``session_key`` into ``_prepare_inbound_message_text`` in the
queued drain branch and keeps the recursive ``_run_agent`` on the *original*
active ``session_key`` + ``run_generation``, so SET == CONSUME == the original
active key.

This test drives the *real* recursive ``_run_agent`` drain with a non-null,
current ``run_generation`` and proves:

  1. the queued follow-up actually runs a second ``run_conversation`` turn;
  2. the native image ``image_url`` reaches that second ``run_conversation``;
  3. the image is consumed under the **original** active session key, never
     under the thread-derived key; and
  4. the per-session run-generation invariant is preserved (the original
     key's generation stays current and no generation leaks onto the thread
     key).

It is the guard against reintroducing the rejected #48919 approach, which
mutated the recursive call's ``session_key`` to a key derived from
``pending_event.source``.  Under that mutation the image is either consumed
under the thread key (assertion #3 fails) or never delivered to the second
turn (assertion #2 fails).
"""

import base64
import importlib
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key


# Smallest valid PNG (1x1) so build_native_content_parts can read real bytes
# off disk and emit an ``image_url`` data URI.
_ONE_BY_ONE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6L2ioAAAAASUVORK5CYII="
)


class CaptureAdapter(BasePlatformAdapter):
    """Minimal adapter that records sends and inherits the real pending-message
    store so ``_dequeue_pending_event`` finds the seeded follow-up event."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []
        self.typing = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="sent-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class CaptureQueuedNativeImageAgent:
    """Stub agent that records each ``run_conversation`` message verbatim.

    The first turn runs on the plain ``"hello"`` text; the queued follow-up
    turn runs on the multimodal content list built from the buffered native
    image.  Recording the raw ``message`` lets the test assert the second turn
    actually received an ``image_url`` part.
    """

    calls = []

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).calls.append(message)
        return {
            "final_response": f"done-{len(type(self).calls)}",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner(adapter):
    """Build a bare GatewayRunner with only the state the drain path touches.

    All mutable state is seeded as fresh per-instance dicts (never via class
    defaults) so cross-test bleed is impossible and the run-generation /
    running-agent bookkeeping is observable after the drain.
    """
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._fallback_model = None
    runner._provider_routing = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._voice_mode = {}
    runner._decide_image_input_mode = lambda: "native"
    runner._draining = False
    # Resolution helpers read these directly (no getattr fallback).
    runner.session_store = None
    runner._session_db = None
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._last_resolved_model = {}
    runner._agent_cache = None
    runner._agent_cache_lock = None
    runner._pending_model_notes = {}
    runner._pending_skills_reload_notes = {}
    runner._queued_events = {}
    # Running-agent + run-generation bookkeeping (generation-guarded cleanup
    # in _release_running_agent_state touches all three of these).
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._session_run_generation = {}
    return runner


@pytest.mark.asyncio
async def test_queued_followup_delivers_native_image_on_original_session_key(
    monkeypatch, tmp_path
):
    """Queued native image reaches the 2nd run_conversation while the follow-up
    stays on the original active session key (regression for #48912)."""
    CaptureQueuedNativeImageAgent.calls = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)

    image_path = tmp_path / "queued-image.png"
    image_path.write_bytes(_ONE_BY_ONE_PNG)

    # The active session: a Telegram group with no thread_id.  This is the key
    # the top-level turn owns and under which run_generation was minted.
    active_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )
    original_key = build_session_key(active_source, group_sessions_per_user=False)

    # The queued event carries a thread_id, so deriving a key from ITS source
    # (the rejected #48919 behavior) yields a *different* key.  This divergence
    # is what makes the test distinguish the two PRs.
    pending_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    thread_key = build_session_key(pending_source, group_sessions_per_user=False)
    assert original_key != thread_key, (
        "test precondition: the thread-derived key must differ from the "
        f"original active key (got {original_key!r} == {thread_key!r})"
    )

    # Non-null, *current* run generation — model a top-level turn that already
    # minted its token for the original key (gateway/run.py mints at the
    # _handle_message_with_agent entry).
    RUN_GEN = 7
    runner._session_run_generation[original_key] = RUN_GEN

    # Record every consume call so we can prove which key actually returned the
    # buffered image.  This is the strongest guard against reintroducing the
    # #48919 key mutation.
    consume_calls = []
    _real_consume = runner._consume_pending_native_image_paths

    def _recording_consume(session_key):
        result = _real_consume(session_key)
        consume_calls.append((session_key, list(result)))
        return result

    runner._consume_pending_native_image_paths = _recording_consume

    # Seed the queued follow-up under the ORIGINAL active key — that is where
    # the drain (_dequeue_pending_event) looks.  Its source still carries the
    # thread_id so a source-derived key would diverge.
    adapter._pending_messages[original_key] = MessageEvent(
        text="describe this",
        message_type=MessageType.PHOTO,
        source=pending_source,
        media_urls=[str(image_path)],
        media_types=["image/png"],
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=active_source,
        session_id="sess-native-image-followup",
        session_key=original_key,
        run_generation=RUN_GEN,
    )

    # 1. The drain actually ran a second turn (real recursive _run_agent path,
    #    not a stub shortcut).
    assert len(CaptureQueuedNativeImageAgent.calls) == 2
    assert result["final_response"] == "done-2"

    # 2. The native image image_url reached the SECOND run_conversation.
    queued_message = CaptureQueuedNativeImageAgent.calls[1]
    assert isinstance(queued_message, list)
    assert queued_message[0]["type"] == "text"
    assert queued_message[0]["text"].startswith("describe this")
    assert any(part.get("type") == "image_url" for part in queued_message), (
        "native image was dropped — no image_url part reached the second "
        f"run_conversation call: {queued_message!r}"
    )

    # 3. The image was consumed under the ORIGINAL active key, never under the
    #    thread-derived key (fails under #48919's recursive key mutation).
    image_consumes = [(k, v) for (k, v) in consume_calls if v]
    assert image_consumes, (
        "expected at least one consume call to return the buffered image; "
        f"recorded: {consume_calls!r}"
    )
    assert all(k == original_key for (k, _v) in image_consumes), (
        "native image consumed under a non-original session key: "
        f"{image_consumes!r}"
    )
    assert all(k != thread_key for (k, _v) in image_consumes), (
        f"native image consumed under the thread-derived key {thread_key!r}"
    )

    # 4. Run-generation invariant preserved: the original key's generation is
    #    still current and no generation token leaked onto the thread key.
    assert runner._is_session_run_current(original_key, RUN_GEN)
    assert not runner._session_run_generation.get(thread_key)

    # 5. Defensive: no running-agent slot installed under the thread key.  The
    #    generation guard in track_agent normally blocks such an insert, so
    #    this is a belt-and-suspenders check rather than a guaranteed-leak
    #    assertion.
    assert thread_key not in runner._running_agents
