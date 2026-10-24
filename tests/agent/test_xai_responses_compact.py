"""xAI grok-4.7 checkpoints via POST /v1/responses/compact.

The durable transcript is not truncated. The compaction item is stamped onto the
newest assistant message and replayed as ``{type, encrypted_content}`` only.
"""

from types import SimpleNamespace

import agent.conversation_compression as cc


def _agent(**overrides):
    captured = {}

    def compact(**kwargs):
        captured["kwargs"] = kwargs
        return overrides.pop("response", SimpleNamespace(
            output=[SimpleNamespace(type="compaction", id="cmp_1", encrypted_content="blob-1")],
        ))

    fields = dict(
        api_mode="codex_responses",
        provider="xai",
        model="grok-4.7",
        base_url="https://api.x.ai/v1",
        codex_responses_native_compaction=True,
        compression_enabled=True,
        compression_checkpoint_required=False,
        _cached_system_prompt="system prompt",
        _codex_reasoning_replay_enabled=True,
        client=SimpleNamespace(responses=SimpleNamespace(compact=compact)),
        context_compressor=SimpleNamespace(
            note_native_compaction_checkpoint=lambda: captured.__setitem__("noted", True),
        ),
        _resolved_api_call_timeout=lambda: 30.0,
        session_id=None,
    )
    fields.update(overrides)
    agent = SimpleNamespace(**fields)
    agent._captured = captured
    return agent


def test_compact_stamps_checkpoint_without_truncating_transcript():
    agent = _agent()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]

    result = cc._compress_context_via_xai_responses(agent, messages, "fallback")

    assert result == (messages, "system prompt")
    assert len(messages) == 2
    stamped = messages[1]["codex_reasoning_items"]
    assert stamped == [{
        "type": "compaction",
        "encrypted_content": "blob-1",
        "_issuer_kind": "xai_responses",
        "_issuer_model": "grok-4.7",
    }]
    sent = agent._captured["kwargs"]
    assert sent["model"] == "grok-4.7"
    assert sent["instructions"] == "system prompt"
    assert sent["timeout"] == 30.0
    assert any(item.get("role") == "user" for item in sent["input"])
    assert agent._captured["noted"] is True
    assert agent._usage_anchor is None


def test_dict_compaction_item_is_coerced_and_id_is_dropped():
    agent = _agent(response={"output": [{
        "type": "compaction", "id": "cmp_dict", "encrypted_content": "blob-2",
    }]})
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "codex_reasoning_items": [
            {"type": "reasoning", "encrypted_content": "keep-me", "id": "rs_1"},
        ]},
    ]

    result = cc._compress_context_via_xai_responses(agent, messages, "fallback")

    assert result is not None
    items = messages[1]["codex_reasoning_items"]
    assert items[0]["encrypted_content"] == "keep-me"
    assert items[1]["type"] == "compaction"
    assert items[1]["encrypted_content"] == "blob-2"
    assert "id" not in items[1]


def test_compact_failure_falls_through_without_stamping():
    def boom(**_kwargs):
        raise RuntimeError("upstream")

    agent = _agent()
    agent.client.responses.compact = boom
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    assert cc._compress_context_via_xai_responses(agent, messages, "fallback") is None
    assert "codex_reasoning_items" not in messages[1]


class _BlockingCompressor:
    def _automatic_compression_blocked(self):
        return True


def test_compress_context_returns_xai_checkpoint_before_local_breaker(monkeypatch):
    """A same-length checkpoint must win even when the local breaker would no-op."""
    sentinel = ([{"role": "user", "content": "kept"}], "prompt")
    monkeypatch.setattr(cc, "_route_xai_responses_compaction", lambda *args, **kwargs: sentinel)
    agent = _agent(context_compressor=_BlockingCompressor())

    assert cc.compress_context(agent, [{"role": "user", "content": "hi"}], "sys") == sentinel
