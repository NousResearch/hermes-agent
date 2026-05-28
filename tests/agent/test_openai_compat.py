"""Tests for the openai parse_response NoneType guard installed in agent/_openai_compat.py."""

from __future__ import annotations

import pytest


def test_compat_module_importable():
    import agent._openai_compat as compat  # noqa: F401

    assert hasattr(compat, "_install")


def test_parse_response_guarded_on_null_output():
    pytest.importorskip("openai")
    import agent  # noqa: F401  (triggers compat install)
    from openai.lib._parsing._responses import parse_response

    guard_flag = getattr(parse_response, "_hermes_or_empty_guard", False)
    assert guard_flag is True, "compat shim did not install on agent import"

    class _FakeResponse:
        output = None

    try:
        parse_response(
            text_format=None,
            input_tools=None,
            response=_FakeResponse(),
        )
    except TypeError as exc:
        if "'NoneType' object is not iterable" in str(exc):
            pytest.fail(
                "guard ineffective — parse_response still iterates a None response.output"
            )
    except Exception:
        # Downstream pydantic / construct_type validation may still raise once
        # iteration succeeds; that's outside the bug's scope. We only assert
        # the specific NoneType-iteration regression is fixed.
        pass


def test_streaming_module_uses_guarded_reference():
    pytest.importorskip("openai")
    import agent  # noqa: F401
    from openai.lib._parsing._responses import parse_response as canonical

    streaming = pytest.importorskip("openai.lib.streaming.responses._responses")
    resource = pytest.importorskip("openai.resources.responses.responses")

    assert streaming.parse_response is canonical, (
        "openai.lib.streaming.responses._responses.parse_response not rebound to guarded copy"
    )
    assert resource.parse_response is canonical, (
        "openai.resources.responses.responses.parse_response not rebound to guarded copy"
    )
