"""Regression tests for issue #32729.

User-configured `auxiliary.title_generation.timeout` was being ignored because
`generate_title()` hardcoded `timeout: float = 30.0` and its caller
`auto_title_session()` never passed an explicit timeout. `call_llm()` already
supports `timeout=None` and resolves the configured value when None is
forwarded, so the fix is purely on the title_generator side: change the
hardcoded 30.0 default to None so the configured timeout flows through.

These tests cover every layer listed in LAYERS.md:

  Layer 1: generate_title() default signature is Optional[float]=None (was float=30.0).
  Layer 1: when generate_title() is invoked without an explicit timeout, it
            forwards timeout=None (not 30.0) to call_llm(), letting call_llm
            resolve the configured timeout.
  Layer 1: an explicit float timeout from the caller is forwarded unchanged.
  Layer 2: auto_title_session() does NOT inject a timeout, so generate_title's
            default (None) reaches call_llm and the user's config wins.
  Layer 5: type-safety — the signature default is Optional[float], consistent
            with call_llm().
"""

from unittest.mock import MagicMock, patch

from agent.title_generator import (
    auto_title_session,
    generate_title,
)


def _ok_response(text: str = "Title") -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = text
    return r


class TestGenerateTitleTimeoutDefault:
    """Layer 1: timeout default is None, not 30.0."""

    def test_default_signature_is_optional_float_none(self):
        """generate_title() must default timeout to None so call_llm can
        consult auxiliary.title_generation.timeout."""
        import inspect
        import typing as _t
        from typing import get_type_hints, Union, get_origin, get_args

        # `from __future__ import annotations` is on, so get_type_hints
        # resolves the string "Optional[float]" — handle both pre- and
        # post-evaluation forms.
        raw_hints = generate_title.__annotations__
        timeout_anno = raw_hints.get("timeout")
        assert timeout_anno is not None, "timeout parameter must be annotated"

        # `Optional[float]` should normalize to Union[float, None].
        if timeout_anno in ("Optional[float]", "typing.Optional[float]"):
            pass  # PEP 563 string form; acceptable
        else:
            origin = get_origin(timeout_anno)
            args = get_args(timeout_anno)
            assert origin in (Union, _t.Union), (
                f"expected Union/Optional origin, got {origin!r}"
            )
            assert set(args) == {float, type(None)}, (
                f"expected Union[float, None], got args={args!r}"
            )

        sig = inspect.signature(generate_title)
        assert sig.parameters["timeout"].default is None, (
            f"expected default None, got {sig.parameters['timeout'].default!r}"
        )

    def test_no_explicit_timeout_forwards_none_to_call_llm(self):
        """When generate_title() is called without timeout=, call_llm must
        receive timeout=None, NOT 30.0."""
        captured = {}

        def mock_call_llm(**kwargs):
            captured.update(kwargs)
            return _ok_response()

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            generate_title("hi", "hello")

        assert "timeout" in captured
        assert captured["timeout"] is None, (
            f"call_llm received timeout={captured['timeout']!r}; "
            "should be None so call_llm can read auxiliary.title_generation.timeout"
        )

    def test_explicit_timeout_from_caller_is_forwarded_unchanged(self):
        captured = {}

        def mock_call_llm(**kwargs):
            captured.update(kwargs)
            return _ok_response()

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            generate_title("hi", "hello", timeout=120.0)

        assert captured["timeout"] == 120.0


class TestAutoTitleSessionPreservesConfigLayer:
    """Layer 2: auto_title_session() must NOT inject a 30s timeout,
    so the configured value can flow through to call_llm."""

    def test_auto_title_session_does_not_pass_timeout_to_generate_title(self):
        db = MagicMock()
        db.get_session_title.return_value = None

        with patch(
            "agent.title_generator.generate_title", return_value="New Title"
        ) as gen:
            auto_title_session(db, "sess-1", "hi", "hello")
            # The fix must ensure generate_title is called without
            # an injected timeout kwarg.
            call_kwargs = gen.call_args.kwargs
            assert "timeout" not in call_kwargs, (
                f"auto_title_session injected timeout={call_kwargs.get('timeout')!r}; "
                "this is the bug — generate_title's default must reach call_llm"
            )

    def test_auto_title_session_full_call_chain_preserves_none_timeout(self):
        """End-to-end: auto_title_session -> generate_title -> call_llm
        must surface timeout=None to call_llm (NOT 30.0)."""
        db = MagicMock()
        db.get_session_title.return_value = None
        captured = {}

        def mock_call_llm(**kwargs):
            captured.update(kwargs)
            return _ok_response()

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            auto_title_session(db, "sess-1", "hi", "hello")

        assert "timeout" in captured
        assert captured["timeout"] is None, (
            f"end-to-end timeout leaked: {captured['timeout']!r}; "
            "configured auxiliary.title_generation.timeout will be ignored"
        )
