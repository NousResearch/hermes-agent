"""Tests for the post-response grounding-enforcement hook in conversation_loop.

A context engine may advertise an ``enforce_response`` capability (via a
``capabilities()`` method) to audit the final answer and, if ungrounded, return
a replacement. The hook is duck-typed: engines lacking the API are skipped, and
the built-in ContextCompressor/LCM (no ``capabilities``) are entirely unaffected.

These tests exercise the hook's contract directly against the documented
behavior, since the conversation_loop integration point is internal.
"""
import types


def _apply_enforcement(agent, assistant_message, messages):
    """Mirror of the conversation_loop grounding-enforcement block, so the
    contract can be unit-tested without driving a full turn."""
    try:
        _eng = getattr(agent, "context_compressor", None)
        _content = assistant_message.content
        _has_tools = bool(getattr(assistant_message, "tool_calls", None))
        _caps = getattr(_eng, "capabilities", None)
        if (_eng is not None and not _has_tools
                and isinstance(_content, str) and _content.strip()
                and callable(_caps) and _caps().get("enforce_response")):
            _verdict = _eng.enforce_response(
                _content, messages, model=getattr(agent, "model", ""), final=True)
            if isinstance(_verdict, dict) and _verdict.get("action") == "replace":
                assistant_message.content = _verdict.get("text", _content)
    except Exception:
        pass
    return assistant_message


class _GroundingEngine:
    def capabilities(self):
        return {"enforce_response": True}

    def enforce_response(self, content, messages, model="", final=True):
        if "unsupported" in content:
            return {"action": "replace", "text": "I can't verify that from the record."}
        return {"action": "keep"}


class _PlainEngine:
    """No capabilities() — like the built-in ContextCompressor/LCM."""
    pass


def _msg(content, tool_calls=None):
    return types.SimpleNamespace(content=content, tool_calls=tool_calls)


def test_ungrounded_answer_is_replaced():
    agent = types.SimpleNamespace(context_compressor=_GroundingEngine(), model="m")
    m = _apply_enforcement(agent, _msg("here is an unsupported claim"), [])
    assert m.content == "I can't verify that from the record."


def test_grounded_answer_is_kept():
    agent = types.SimpleNamespace(context_compressor=_GroundingEngine(), model="m")
    m = _apply_enforcement(agent, _msg("a well-grounded answer"), [])
    assert m.content == "a well-grounded answer"


def test_plain_engine_without_capabilities_is_noop():
    agent = types.SimpleNamespace(context_compressor=_PlainEngine(), model="m")
    m = _apply_enforcement(agent, _msg("unsupported but engine has no caps"), [])
    assert m.content == "unsupported but engine has no caps"


def test_no_engine_is_noop():
    agent = types.SimpleNamespace(context_compressor=None, model="m")
    m = _apply_enforcement(agent, _msg("anything"), [])
    assert m.content == "anything"


def test_tool_call_turn_is_skipped():
    # enforcement only applies to final text answers, not tool-call turns.
    agent = types.SimpleNamespace(context_compressor=_GroundingEngine(), model="m")
    m = _apply_enforcement(agent, _msg("unsupported", tool_calls=[{"id": "x"}]), [])
    assert m.content == "unsupported"


def test_empty_content_is_skipped():
    agent = types.SimpleNamespace(context_compressor=_GroundingEngine(), model="m")
    m = _apply_enforcement(agent, _msg("   "), [])
    assert m.content == "   "
