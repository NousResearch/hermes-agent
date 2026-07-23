"""Regression test for PR #15157: non-dict ``model_extra`` must not crash
tool-call normalization.

Some OpenAI-compatible providers (observed with NVIDIA NIM + qwen3.5) return a
**string** for ``model_extra`` on a tool call instead of the dict that pydantic
``BaseModel.model_extra`` produces.  The original extraction used a falsy
fallback::

    extra = (tc.model_extra or {}).get("extra_content")

A non-empty string is truthy, so ``(string or {})`` evaluates to the string and
``.get(...)`` raises ``AttributeError: 'str' object has no attribute 'get'`` —
which propagates out of ``normalize_response`` and turns every tool call into an
``[error]``.  The fix replaces the falsy fallback with an explicit
``isinstance(.., dict)`` guard, so any non-dict ``model_extra`` is treated as
"no extra_content".

These tests exercise the real ``ChatCompletionsTransport.normalize_response``
path (the non-streaming site at chat_completions.py), not a local replica of
the expression.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.transports.chat_completions import ChatCompletionsTransport


def _make_response(*, model_extra, extra_content_missing=True):
    """Build a fake OpenAI ChatCompletion with one tool call.

    ``extra_content`` is left unset so ``getattr(tc, "extra_content", None)``
    returns None and the code falls through to the ``model_extra`` branch —
    the path that crashed.
    """
    func = SimpleNamespace(name="get_weather", arguments='{"city": "SF"}')
    tc = SimpleNamespace(id="call_1", function=func, model_extra=model_extra)
    # Deliberately do NOT set tc.extra_content — getattr default None triggers
    # the model_extra branch.
    message = SimpleNamespace(
        tool_calls=[tc],
        content=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None, model="qwen3.5-397b")


class TestModelExtraTypeGuard:
    """The guard must tolerate every shape of ``model_extra``."""

    def _normalize(self, model_extra):
        transport = ChatCompletionsTransport()
        return transport.normalize_response(_make_response(model_extra=model_extra))

    def test_string_model_extra_does_not_crash(self):
        """A truthy string used to raise AttributeError — must be ignored now."""
        result = self._normalize("unexpected_string")
        assert len(result.tool_calls) == 1
        # No extra_content recovered from a non-dict — provider_data stays empty.
        assert result.tool_calls[0].provider_data is None

    def test_none_model_extra(self):
        result = self._normalize(None)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].provider_data is None

    def test_list_model_extra_does_not_crash(self):
        """Any non-dict (list) is tolerated, not just strings."""
        result = self._normalize([1, 2, 3])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].provider_data is None

    def test_dict_model_extra_still_extracts_extra_content(self):
        """The valid dict path must keep working — extra_content preserved."""
        result = self._normalize({"extra_content": {"thought_signature": "sig"}})
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].provider_data == {
            "extra_content": {"thought_signature": "sig"}
        }

    def test_dict_without_extra_content_key(self):
        """A dict that has no extra_content key yields no provider_data."""
        result = self._normalize({"other_key": "value"})
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].provider_data is None
