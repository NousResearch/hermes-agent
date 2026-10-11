"""Tests for the audio-rejection fallback in run_agent (native ``input_audio`` routing).

When a server 4xxs on an ``input_audio`` part (text-only endpoint, a wire with no audio
part, a schema gate), the agent records the (provider, model), strips audio from the
retry payload and goes text-only for the rest of the session. Mirrors
``tests/agent/test_image_rejection_fallback.py``:

* the strip happens on the send path only — canonical history keeps the clip, so a later
  audio-capable backend hears it again;
* the ledger is keyed per (provider, model), so each model in a fallback chain is judged
  on its own and re-entry from a known model falls through;
* the audio branch runs BEFORE the image branch, because both phrase lists share the
  generic "only text content is supported" wordings — an audio-carrying turn must be
  recorded as audio-rejecting, not image-rejecting;
* ``build_api_request`` applies the strip before the request kwargs are built.
"""

from __future__ import annotations

import copy
import inspect
from types import SimpleNamespace

from agent.message_sanitization import (
    _looks_like_audio_content_rejection,
    _messages_carry_audio,
    _strip_audio_from_messages,
)


class _AudioErr(Exception):
    """Provider 4xx carrying an error body (the shape turn_recovery reads)."""

    def __init__(self, body: str, status: int = 400):
        super().__init__(body)
        self.body = body
        self.status_code = status


#: Error wording naming audio specifically — never trips the image phrase list, so the
#: assertions below cannot be satisfied by the wrong branch.
_AUDIO_ONLY_ERR = "This model does not support audio input."
#: Wording shared by both phrase lists — proves the branch ORDER (audio wins).
_GENERIC_TEXT_ONLY_ERR = "Only 'text' content type is supported."


def _audio_part(data: str = "AAAA") -> dict:
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


def _agent(provider="text-only-provider", model="text-model") -> SimpleNamespace:
    return SimpleNamespace(
        provider=provider, model=model, api_mode="chat_completions",
        _force_ascii_payload=False,
        _image_rejecting_models=set(), _audio_rejecting_models=set(),
        _db_flush_scan_prefix=7, log_prefix="",
        _vprint=lambda *a, **k: None,
    )


def _history() -> list:
    """Canonical history: a text+audio row and an audio-only injected row."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what did I send you?"},
                _audio_part("AAAA"),
            ],
        },
        {"role": "user", "content": [_audio_part("BBBB")]},
    ]


def _recover(agent, messages, api_messages, body=_AUDIO_ONLY_ERR, status=400):
    from agent.turn_recovery import recover_before_classification

    return recover_before_classification(
        agent, _AudioErr(body, status), messages=messages, api_messages=api_messages,
        api_kwargs={}, active_system_prompt="sys",
    )


class TestRejectionNeverReachesPersistedHistory:
    """A rejection says what the CURRENT model accepts, not what the conversation holds.

    Same failure class as the image strip (issue #117802): stripping canonical ``messages``
    and forcing a flush would delete every clip from state.db for good.
    """

    def test_retry_triggers_and_canonical_history_keeps_its_audio(self):
        agent, history = _agent(), _history()
        before = copy.deepcopy(history)
        wire = copy.deepcopy(history)

        retry, _ = _recover(agent, history, wire)

        assert retry is True
        assert history == before, "the recovery rewrote persisted history"
        assert agent._db_flush_scan_prefix == 7, "the recovery forced a history rewrite"
        assert agent._audio_rejecting_models == {("text-only-provider", "text-model")}
        # The send path is text-only for this model...
        assert "input_audio" not in str(wire)
        # ...while a clone built from history still carries the clip.
        assert "input_audio" in str(copy.deepcopy(history))

    def test_wire_is_text_only_but_keeps_the_caption(self):
        agent = _agent()
        wire = copy.deepcopy(_history())
        retry, _ = _recover(agent, _history(), wire)

        assert retry is True
        # Text parts survive; only the audio parts go. The audio-only row has nothing
        # left, so it is dropped (no tool_call_id linkage to orphan).
        assert len(wire) == 1
        assert wire[0]["content"] == [{"type": "text", "text": "what did I send you?"}]

    def test_build_api_request_strips_audio_before_building_kwargs(self):
        """The retry re-enters build_api_request; the strip must run before the kwargs
        are built from api_messages, or the rejected part rides the retry."""
        from agent.turn_api_request import build_api_request

        src = inspect.getsource(build_api_request)
        strip_at = src.find("strip_unsupported_audio_parts(agent, api_messages)")
        assert strip_at != -1, "build_api_request no longer strips unsupported audio"
        kwargs_at = src.find("_build_api_kwargs(api_messages")
        assert kwargs_at != -1
        assert strip_at < kwargs_at, "strip must precede _build_api_kwargs"

    def test_iteration_summary_strips_audio_for_rejecting_model(self, tmp_path, monkeypatch):
        """The max-iterations summary hand-builds api_messages and bypasses
        build_api_request; it applies the same per-model strip. History keeps its clip."""
        from agent.chat_completion_helpers import _iteration_summary_api_messages
        from agent.vision_message_prep import _provider_model_key
        from run_agent import AIAgent

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        agent = AIAgent(
            api_key="k", base_url="https://api.groq.com/openai/v1", provider="custom",
            model="m", quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
        agent._cached_system_prompt = "SYS"
        agent._audio_rejecting_models.add(_provider_model_key(agent))
        history = _history() + [{"role": "assistant", "content": "ok"}]
        before = copy.deepcopy(history)

        out = _iteration_summary_api_messages(agent, history)

        assert "input_audio" not in str(out), "summary request must be text-only for a rejecting model"
        assert any("what did I send you?" in str(m.get("content"))
                   for m in out if m.get("role") == "user")
        assert history == before, "the per-call strip must not leak into canonical history"


class TestFallbackChainTracksEachModel:
    """Two models reject audio in the same turn (fallback A -> B): each is judged and
    remembered on its own, and a known model's second rejection falls through."""

    def test_each_model_recorded_and_guarded(self, monkeypatch):
        from agent import audio_routing

        monkeypatch.setattr(audio_routing, "_load_cfg_readonly", lambda: {"media": {"native_audio": "on"}})

        agent = _agent(provider="p", model="model-a")
        retry_a, _ = _recover(agent, _history(), copy.deepcopy(_history()))
        assert retry_a is True
        assert agent._audio_rejecting_models == {("p", "model-a")}

        # The fallback restart rebuilds api_messages from history, audio included, for B —
        # and B is not on the ledger yet, so nothing strips its request.
        agent.model = "model-b"
        rebuilt = copy.deepcopy(_history())
        assert audio_routing.strip_unsupported_audio_parts(agent, rebuilt) == 0
        assert "input_audio" in str(rebuilt)

        # B rejects too, in the same turn: its recovery must still run.
        retry_b, _ = _recover(agent, _history(), copy.deepcopy(_history()))
        assert retry_b is True
        assert agent._audio_rejecting_models == {("p", "model-a"), ("p", "model-b")}

        # Re-entry guard: a third rejection from a known model falls through to normal
        # error handling instead of looping.
        assert _recover(agent, _history(), copy.deepcopy(_history()))[0] is False

        # From then on every attempt to either model goes out text-only...
        for model in ("model-a", "model-b"):
            agent.model = model
            api_messages = copy.deepcopy(_history())
            # Both rows carry audio → both rewritten (the audio-only row gets a placeholder).
            assert audio_routing.strip_unsupported_audio_parts(agent, api_messages) == 2, model
            assert "input_audio" not in str(api_messages)
        # ...and an unrecorded model of the same provider still hears the clip.
        agent.model = "model-c"
        api_messages = copy.deepcopy(_history())
        assert audio_routing.strip_unsupported_audio_parts(agent, api_messages) == 0
        carried = [
            p for m in api_messages if isinstance(m.get("content"), list)
            for p in m["content"] if audio_routing.is_audio_part(p)
        ]
        assert len(carried) == 2


class TestAudioBranchBeatsImageBranch:
    """The phrase lists overlap on the generic text-only wordings; the audio branch must
    win whenever the turn actually carries audio."""

    def test_generic_body_with_audio_records_audio_not_image(self):
        agent = _agent()
        retry, _ = _recover(agent, _history(), copy.deepcopy(_history()), body=_GENERIC_TEXT_ONLY_ERR)

        assert retry is True
        assert agent._audio_rejecting_models == {("text-only-provider", "text-model")}
        assert agent._image_rejecting_models == set(), (
            "an audio-carrying turn was logged as image-rejecting"
        )

    def test_generic_body_without_audio_records_image_not_audio(self):
        """No audio in the turn → the same wording is an image rejection, as before."""
        history = [
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]},
        ]
        agent = _agent()
        retry, _ = _recover(agent, history, copy.deepcopy(history), body=_GENERIC_TEXT_ONLY_ERR)

        assert retry is True
        assert agent._image_rejecting_models == {("text-only-provider", "text-model")}
        assert agent._audio_rejecting_models == set()

    def test_transient_5xx_does_not_record_the_model(self):
        """5xx/timeouts are transient; they take the retry path, not the capability ledger."""
        agent = _agent()
        retry, _ = _recover(agent, _history(), copy.deepcopy(_history()), status=503)

        assert retry is False
        assert agent._audio_rejecting_models == set()

    def test_audio_free_turn_never_trips_the_audio_branch(self):
        """``_messages_carry_audio`` gates the branch: text-only history with an audio
        error body records nothing (the provider's complaint is about a part we never sent)."""
        history = [{"role": "user", "content": "hello"}]
        agent = _agent()
        retry, _ = _recover(agent, history, copy.deepcopy(history))

        assert retry is False
        assert agent._audio_rejecting_models == set()


class TestAudioRejectionPhraseIsolation:
    def test_real_audio_rejection_bodies_trip(self):
        bodies = [
            "This model does not support audio input.",
            "Invalid request: input_audio is not supported by this model",
            "Only 'text' content type is supported.",
            "Bad request: multimodal is not supported by this model",
            "The provided messages input is invalid. The error info is [Unexpected item type in content].",
            # DeepSeek-style Rust schema gateway, rejecting the unknown variant by name.
            'unknown variant `input_audio`, expected `text`',
        ]
        for body in bodies:
            assert _looks_like_audio_content_rejection(body) is True, f"false negative on: {body}"

    def test_image_and_corrupt_payload_bodies_do_not_trip(self):
        """Image capability refusals and bad-payload wordings must keep routing to the
        image handlers (``_try_shrink_image_parts`` etc.), not the audio ledger."""
        bodies = [
            "This model does not support images.",
            "vision is not supported on this endpoint",
            "image_url is not supported",
            "model does not support image input",
            # corrupt payload → strip-and-retry, but never a capability ledger
            "Invalid request: prepare image failed: failed to decode image: invalid or unsupported image format",
            # size errors → image_too_large handler
            "messages.0.content.1.image.source.base64: image exceeds 5 MB maximum",
        ]
        for body in bodies:
            assert _looks_like_audio_content_rejection(body) is False, f"false positive on: {body}"

    def test_messages_carry_audio_detection(self):
        assert _messages_carry_audio(_history()) is True
        assert _messages_carry_audio([{"role": "user", "content": "hi"}]) is False
        assert _messages_carry_audio([{"role": "user", "content": [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]}]) is False
        assert _messages_carry_audio([]) is False
        assert _messages_carry_audio(None) is False
        # generic spelling
        assert _messages_carry_audio([{"role": "user", "content": [{"type": "audio", "audio": {}}]}]) is True


class TestStripAudioPreservesAlternation:
    """``_strip_audio_from_messages`` must not break the invariants providers enforce."""

    def test_noop_when_no_audio(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert _strip_audio_from_messages(msgs) is False
        assert msgs == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_tool_message_with_all_audio_replaced_not_deleted(self):
        """CRITICAL: tool messages must NEVER be deleted — their tool_call_id pairs with
        an assistant tool_call and providers reject unmatched IDs."""
        msgs = [
            {"role": "user", "content": "transcribe this"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc", "type": "function",
                    "function": {"name": "record_audio", "arguments": "{}"},
                }],
            },
            {
                "role": "tool", "tool_call_id": "call_abc",
                "content": [_audio_part()],
            },
        ]
        assert _strip_audio_from_messages(msgs) is True
        assert len(msgs) == 3
        assert msgs[2]["tool_call_id"] == "call_abc"
        assert isinstance(msgs[2]["content"], str)
        assert "audio content removed" in msgs[2]["content"].lower()

    def test_tool_message_with_mixed_content_keeps_text_parts(self):
        msgs = [
            {
                "role": "tool", "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "clipped 3.2s"},
                    _audio_part(),
                ],
            },
        ]
        assert _strip_audio_from_messages(msgs) is True
        assert msgs[0]["content"] == [{"type": "text", "text": "clipped 3.2s"}]
        assert msgs[0]["tool_call_id"] == "call_1"

    def test_assistant_with_tool_calls_and_audio_only_content_preserved(self):
        msgs = [
            {
                "role": "assistant",
                "content": [_audio_part()],
                "tool_calls": [{"id": "call_xyz", "type": "function",
                                "function": {"name": "speak", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_xyz", "content": "done"},
        ]
        assert _strip_audio_from_messages(msgs) is True
        assert len(msgs) == 2
        assert msgs[0]["tool_calls"][0]["id"] == "call_xyz"
        assert isinstance(msgs[0]["content"], str)
        assert "audio content removed" in msgs[0]["content"].lower()
        assert msgs[1]["tool_call_id"] == "call_xyz"

    def test_audio_only_user_message_dropped(self):
        """Synthetic audio-only user rows (gateway injection pattern) are safe to drop —
        no tool_call_id linkage to preserve."""
        msgs = [
            {"role": "user", "content": "listen to this"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [_audio_part()]},
        ]
        assert _strip_audio_from_messages(msgs) is True
        assert len(msgs) == 2
        assert msgs[-1] == {"role": "assistant", "content": "ok"}

    def test_multiple_tool_messages_all_preserved(self):
        msgs = [
            {
                "role": "assistant", "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": [_audio_part()]},
            {"role": "tool", "tool_call_id": "c2", "content": [_audio_part("BBBB")]},
        ]
        assert _strip_audio_from_messages(msgs) is True
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert {m["tool_call_id"] for m in tool_msgs} == {"c1", "c2"}


class TestStripAudioDropsStaleApiContent:
    """A rewritten row loses its ``api_content`` sidecar so the removed bytes cannot
    replay next turn (same contract as the other content-rewrite paths)."""

    @staticmethod
    def _wire(msg):
        from agent.turn_context import substitute_api_content

        api_msg = msg.copy()
        substitute_api_content(api_msg)
        return api_msg["content"]

    def test_stripped_message_loses_its_sidecar(self):
        msgs = [{
            "role": "user",
            "content": [{"type": "text", "text": "listen"}, _audio_part()],
            "api_content": "listen<AUDIO BYTES SENT LAST TURN>",
        }]
        assert _strip_audio_from_messages(msgs) is True
        assert "api_content" not in msgs[0]
        assert "AUDIO BYTES" not in str(self._wire(msgs[0]))

    def test_untouched_messages_keep_their_sidecar(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "no clips here"}],
             "api_content": "no clips here<injected ctx>"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}, _audio_part()],
             "api_content": "ok<AUDIO BYTES>"},
        ]
        _strip_audio_from_messages(msgs)
        assert msgs[0]["api_content"] == "no clips here<injected ctx>"
        assert "api_content" not in msgs[1]
