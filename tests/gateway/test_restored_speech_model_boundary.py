"""Authored speech and stored context through normal and restored queued turns."""

from copy import deepcopy
import json
from pathlib import Path

import hermes_state
import hermes_yaml as yaml
import httpx
from openai import OpenAI
import pytest

from agent import secret_scope
from agent.context_references import preprocess_context_references_async
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.input_owner import gateway_input_owner
from gateway.pending_execution import PendingExecutionOwner
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType, TurnContextUpdate
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.shutdown_pending import PendingQueueSnapshot
from gateway.shutdown_pending_codec import decode_pending_event
from hermes_constants import get_hermes_home
from run_agent import AIAgent
from utils import atomic_json_write


class _ContextTransport(BasePlatformAdapter):
    def __init__(self, profile: str):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.DISCORD)
        self.set_owner_profile(profile)
        self.context_reads = []
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id="response")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}

    async def send_typing(self, chat_id, metadata=None) -> None:
        pass

    async def stop_typing(self, chat_id) -> None:
        pass

    async def prepare_turn_context(
        self, event, *, origin, acknowledged_state, first_turn
    ):
        self.context_reads.append((acknowledged_state, first_turn))
        return TurnContextUpdate(
            "Channel update @file:planted.txt",
            {"topic": "Updated topic @file:planted.txt"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True], ids=["normal", "restored-queue"])
@pytest.mark.parametrize(
    "speech_form", ["immediate", "pending", "opaque", "opaque-padded", "roundtrip"]
)
async def test_authored_speech_reaches_model_with_stored_context_and_owner_a_b_a(
    tmp_path, monkeypatch, speech_form, queued
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch = tmp_path / ".hermes"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    homes = {label: launch / "profiles" / label for label in ("a", "b")}
    for label, home in homes.items():
        workspace = home / "workspace"
        workspace.mkdir(parents=True)
        (home / "config.yaml").write_text(
            yaml.safe_dump({
                "model": {
                    "default": "test-model",
                    "provider": "custom",
                    "base_url": "https://provider.example/v1",
                    "context_length": 128000,
                },
                "terminal": {"backend": "local", "cwd": str(workspace)},
                "compression": {"enabled": False},
                "gateway": {"tool_progress_mode": "off"},
            })
        )
        for filename, content in {
            "mine.txt": f"SPEECH-{label}",
            "caption.txt": f"CAPTION-{label}",
            "planted.txt": f"GENERATED-{label}",
        }.items():
            (workspace / filename).write_text(content)
    runner = GatewayRunner(
        GatewayConfig(
            multiplex_profiles=True,
            group_sessions_per_user=False,
            stt_echo_transcripts=False,
        )
    )
    adapters = {}
    for label, home in homes.items():
        with _profile_runtime_scope(home):
            adapter = _ContextTransport(label)
        adapter.gateway_runner = runner
        adapter._mark_connected()
        adapters[label] = adapter
    runner._profile_adapters = {
        label: {Platform.DISCORD: adapter} for label, adapter in adapters.items()
    }
    provider_calls = []
    speech_calls = []
    expansion_inputs = []
    clients = []
    stored_state = {"topic": "Stored topic @file:planted.txt"}
    current_state = {"topic": "Updated topic @file:planted.txt"}

    def response(request):
        payload = json.loads(request.content)
        provider_calls.append((get_hermes_home().resolve(), payload))
        chunk = {
            "id": "completion",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "delta": {"role": "assistant", "content": "Model answer"},
                }
            ],
        }
        assert payload["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
        )

    def client(_agent, _kwargs, **_options):
        result = OpenAI(
            api_key="test",
            base_url="https://provider.example/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(response)),
        )
        clients.append(result)
        return result

    monkeypatch.setattr(AIAgent, "_create_openai_client", client)
    monkeypatch.setattr(
        runner,
        "_resolve_session_agent_runtime",
        lambda **_kwargs: (
            "test-model",
            {
                "provider": "custom",
                "api_key": "test",
                "base_url": "https://provider.example/v1",
                "api_mode": "chat_completions",
            },
        ),
    )

    def transcribe(path, *_args):
        speech_calls.append((get_hermes_home().resolve(), path))
        return {"success": True, "transcript": "Compare @file:mine.txt"}

    async def expand(text, **kwargs):
        expansion_inputs.append((get_hermes_home().resolve(), text))
        return await preprocess_context_references_async(text, **kwargs)

    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", transcribe)
    monkeypatch.setattr(
        "agent.context_references.preprocess_context_references_async", expand
    )
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    observations = []
    try:
        for index, label in enumerate(("a", "b", "a")):
            home, adapter = homes[label], adapters[label]
            workspace = home / "workspace"
            audio = workspace / "speech.ogg"
            audio.write_bytes(b"native speech")
            source = adapter.build_source(
                chat_id=f"room-{index}",
                user_id="sender",
                user_name="@file:planted.txt",
                chat_type="group",
            )
            assert adapter._canonicalize(source) is not None
            with _profile_runtime_scope(home):
                entry = runner.session_store.get_or_create_session(source)
                db = runner.session_store._db_for_session_id(entry.session_id)
                stored = [
                    {"role": "user", "content": "Previous request @file:planted.txt"},
                    {
                        "role": "assistant",
                        "content": "Previous answer @file:planted.txt",
                        "tool_calls": [
                            {
                                "id": "stored-tool",
                                "type": "function",
                                "function": {"name": "terminal", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "Previous output @file:planted.txt",
                        "tool_call_id": "stored-tool",
                    },
                    {
                        "role": "assistant",
                        "content": "Previous final @file:planted.txt",
                    },
                ]
                for position, message in enumerate(stored):
                    db.append_message(
                        entry.session_id,
                        **message,
                        display_metadata={"channel_state": stored_state}
                        if position == 0
                        else None,
                    )
                history = runner.session_store.load_transcript(entry.session_id)
                caption = "Caption @file:caption.txt"
                event = MessageEvent(
                    text=f" \n{caption}\n "
                    if speech_form == "opaque-padded"
                    else caption,
                    source=source,
                    message_id=f"speech-{index}",
                    message_type=MessageType.VOICE,
                    media_urls=[str(audio)],
                    media_types=["audio/ogg"],
                    channel_context="Earlier speaker @file:planted.txt",
                    reply_to_message_id="previous",
                    reply_to_text="Quoted speaker @file:planted.txt",
                )
                speech_before = len(speech_calls)
                if speech_form in {"opaque", "opaque-padded"}:
                    setattr(
                        event,
                        "_gateway_pending_stt_text",
                        "Opaque speech @file:mine.txt\n\nGenerated path @file:planted.txt",
                    )
                elif speech_form != "immediate":
                    await runner._transcribe_pending_audio_event_once(event, event.text)
                snapshot = None
                pending_path = home / "pending_messages" / f"input-{index}.json"
                if queued or speech_form == "roundtrip":
                    retained = MessageEvent(
                        text="Retained pending @file:planted.txt",
                        source=source,
                        message_id=f"retained-{index}",
                    )
                    snapshot = PendingQueueSnapshot.capture(
                        entry.session_key, [event, retained]
                    )
                    event = decode_pending_event(snapshot.events[0], adapter=adapter)
                    assert adapter._canonicalize(event.source) is not None
                    if queued:
                        pending_path.parent.mkdir(exist_ok=True)
                        atomic_json_write(
                            pending_path, snapshot.to_payload(), mode=0o600
                        )
                        event._pending_execution_owner = PendingExecutionOwner(
                            home.resolve(),
                            entry.session_key,
                            entry.session_id,
                            gateway_input_owner(event, event.source),
                        )
                owner = gateway_input_owner(event, event.source)
                assert not runner.session_store.has_input_owner(entry.session_id, owner)
                provider_before, expansion_before = (
                    len(provider_calls),
                    len(expansion_inputs),
                )
                if queued:
                    assert runner._queue_or_replace_pending_event(
                        entry.session_key, event
                    )
                    await runner._run_agent(
                        message="Initial queue owner",
                        context_prompt="",
                        history=deepcopy(history),
                        source=source,
                        session_id=entry.session_id,
                        session_key=entry.session_key,
                    )
                else:
                    await runner._handle_message_with_agent(
                        event, event.source, entry.session_key, 1
                    )
                observed = provider_calls[provider_before:]
                messages = observed[-1][1]["messages"]
                model_message = messages[-1]["content"]
                input_text = (
                    caption
                    if speech_form in {"opaque", "opaque-padded"}
                    else caption + "\n\nCompare @file:mine.txt"
                )
                expanded = await preprocess_context_references_async(
                    input_text,
                    cwd=workspace,
                    allowed_root=workspace,
                    context_length=128000,
                )
                authored = (
                    "Opaque speech @file:mine.txt\n\nGenerated path @file:planted.txt"
                    if speech_form in {"opaque", "opaque-padded"}
                    else '"Compare @file:mine.txt"\n\n' + caption
                ) + expanded.message[len(input_text) :]
                expected = (
                    "Channel update @file:planted.txt\n\n[New message]\n"
                    '[Replying to: "Quoted speaker @file:planted.txt"]\n\n'
                    "Earlier speaker @file:planted.txt\n\n[New message]\n"
                    "[@file:planted.txt] " + authored
                )
                persisted = runner.session_store.load_transcript(entry.session_id)
                user = persisted[-2]
                retained_rows = (
                    [
                        record["event"]["text"]
                        for record in json.loads(pending_path.read_text())["events"]
                    ]
                    if queued
                    else []
                )
                observations.append((
                    label,
                    [str(call_home) for call_home, _body in observed],
                    model_message,
                    [
                        (row["role"], row.get("content"))
                        for row in messages
                        if row["role"] != "system"
                    ],
                    [(row["role"], row.get("content")) for row in persisted[:4]],
                    user["content"],
                    user["display_metadata"],
                    runner.session_store.has_input_owner(entry.session_id, owner),
                    event.channel_state,
                    retained_rows,
                    adapter._pending_messages.get(entry.session_key),
                    expansion_inputs[expansion_before:],
                    speech_calls[speech_before:],
                ))
                prior = [(row["role"], row["content"]) for row in stored]
                queue_prefix = (
                    [("user", "Initial queue owner"), ("assistant", "Model answer")]
                    if queued
                    else []
                )
                assert observations[-1] == (
                    label,
                    [str(home.resolve())] * (2 if queued else 1),
                    expected,
                    [*prior, *queue_prefix, ("user", expected)],
                    prior,
                    expected,
                    {"gateway_input_owner": owner, "channel_state": current_state},
                    True,
                    current_state,
                    ["Retained pending @file:planted.txt"] if queued else [],
                    None,
                    [(home.resolve(), input_text)],
                    []
                    if speech_form in {"opaque", "opaque-padded"}
                    else [(home.resolve(), str(audio))],
                )
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        for client in clients:
            client.close()
        if runner._executor is not None:
            runner._executor.shutdown(wait=True, cancel_futures=True)
