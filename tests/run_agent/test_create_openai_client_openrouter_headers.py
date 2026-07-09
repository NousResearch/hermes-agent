"""
Regression tests for issue #61099 - OpenRouter HTTP-Referer + X-Title
headers must be sent on EVERY request, not just specific code paths.

The bug: ``build_or_headers()`` defines the right attribution headers
(HTTP-Referer, X-Title, X-OpenRouter-Categories), and the headers are
set at several specific call sites (run_agent.py:4393, auxiliary_client.py
:1930/1941/4321, agent_init.py:919). But the central ``_create_openai_client``
in auxiliary_client.py does NOT inject them, so any client constructed
via that path without an explicit ``default_headers=build_or_headers()``
argument ships without attribution. OpenRouter's Logs page then shows
"Unknown" for those requests.

The fix: in ``_create_openai_client`` (the central factory), detect when
``base_url`` matches OpenRouter and merge ``build_or_headers()`` into
``default_headers`` automatically, so every new client picks up the
headers without each call site having to remember.

These tests construct an AIAgent and drive ``_create_openai_client`` with
an OpenRouter base_url, then assert the constructed OpenAI client was
called with the right default_headers in its kwargs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _build_agent(base_url: str = "https://openrouter.ai/api/v1") -> AIAgent:
    """Build an AIAgent configured for a given base_url. We override
    quiet_mode + skip flags to keep construction fast and free of
    unrelated side effects (no network, no memory, no skill loading).
    """
    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _matching_openai_calls(mock_openai, base_url: str) -> list:
    """Return OpenAI(...) calls that match the given base_url."""
    return [
        c for c in mock_openai.call_args_list
        if c.kwargs.get("base_url") == base_url
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("run_agent.OpenAI")
def test_create_openai_client_injects_openrouter_headers_on_openrouter_url(
    mock_openai,
):
    """The bug: a client constructed for an OpenRouter base_url does NOT
    carry HTTP-Referer / X-Title / X-OpenRouter-Categories in its
    default_headers unless the caller remembered to pass them explicitly.

    After the fix: ``_create_openai_client`` auto-detects OpenRouter and
    merges ``build_or_headers()`` into ``default_headers`` so attribution
    is consistent regardless of which call site built the client.
    """
    mock_openai.return_value = MagicMock()
    agent = _build_agent()

    agent._create_openai_client(
        {"api_key": "***", "base_url": "https://openrouter.ai/api/v1"},
        reason="test",
        shared=False,
    )

    matching = _matching_openai_calls(mock_openai, "https://openrouter.ai/api/v1")
    assert matching, "OpenAI was never constructed with the OpenRouter base_url"
    headers = matching[-1].kwargs.get("default_headers", {})
    assert "HTTP-Referer" in headers, (
        f"OpenRouter client missing HTTP-Referer header; got {headers!r}. "
        f"Issue #61099: attribution headers must be set on every OpenRouter "
        f"request for the OpenRouter Logs page to identify the app."
    )
    assert "X-Title" in headers, (
        f"OpenRouter client missing X-Title header; got {headers!r}"
    )
    assert headers.get("X-Title") == "Hermes Agent", (
        f"X-Title header value should be 'Hermes Agent'; got {headers!r}"
    )
    assert "X-OpenRouter-Categories" in headers, (
        f"OpenRouter client missing X-OpenRouter-Categories; got {headers!r}"
    )


@patch("run_agent.OpenAI")
def test_create_openai_client_does_not_inject_openrouter_headers_on_other_urls(
    mock_openai,
):
    """Regression guard: a client built for a non-OpenRouter URL (e.g.
    a local Ollama) must NOT receive OpenRouter attribution headers. The
    auto-injection is OpenRouter-specific.
    """
    mock_openai.return_value = MagicMock()
    agent = _build_agent(base_url="http://localhost:11434/v1")

    agent._create_openai_client(
        {"api_key": "***", "base_url": "http://localhost:11434/v1"},
        reason="test",
        shared=False,
    )

    matching = _matching_openai_calls(mock_openai, "http://localhost:11434/v1")
    assert matching
    headers = matching[-1].kwargs.get("default_headers", {})
    # If default_headers is empty/missing entirely, the client has no
    # OpenRouter attribution. We do NOT pin default_headers to {}; we only
    # require the OpenRouter-specific keys to be absent.
    assert "X-OpenRouter-Categories" not in headers, (
        f"non-OpenRouter client was given OpenRouter attribution headers; "
        f"got {headers!r}"
    )
    assert "X-Title" not in headers or headers.get("X-Title") != "Hermes Agent", (
        f"non-OpenRouter client has X-Title='Hermes Agent'; should not. "
        f"got {headers!r}"
    )


@patch("run_agent.OpenAI")
def test_caller_explicit_default_headers_are_preserved(mock_openai):
    """Regression guard: if a caller passes default_headers explicitly,
    the auto-injection must NOT clobber them. The fix should merge, not
    replace. The user's explicit X-Title (or any other header) wins over
    the auto-attached one.
    """
    mock_openai.return_value = MagicMock()
    agent = _build_agent()

    explicit = {
        "HTTP-Referer": "https://my-custom-referer.example.com",
        "User-Agent": "my-custom-agent/1.0",
    }
    agent._create_openai_client(
        {
            "api_key": "***",
            "base_url": "https://openrouter.ai/api/v1",
            "default_headers": explicit,
        },
        reason="test",
        shared=False,
    )

    matching = _matching_openai_calls(mock_openai, "https://openrouter.ai/api/v1")
    headers = matching[-1].kwargs.get("default_headers", {})
    # The user's explicit HTTP-Referer wins over the auto-attached one.
    assert headers.get("HTTP-Referer") == "https://my-custom-referer.example.com", (
        f"explicit user HTTP-Referer was clobbered by auto-injection; got {headers!r}"
    )
    # The user's other explicit headers survive.
    assert headers.get("User-Agent") == "my-custom-agent/1.0", (
        f"user's User-Agent dropped: {headers!r}"
    )


@patch("run_agent.OpenAI")
def test_create_openai_client_still_disables_sdk_retries_alongside_headers(
    mock_openai,
):
    """Regression guard: the #26293 fix (max_retries=0) must continue to
    apply even when the new header-injection logic runs. Both
    normalizations live in the same code path.
    """
    mock_openai.return_value = MagicMock()
    agent = _build_agent()

    agent._create_openai_client(
        {"api_key": "***", "base_url": "https://openrouter.ai/api/v1"},
        reason="test",
        shared=False,
    )

    matching = _matching_openai_calls(mock_openai, "https://openrouter.ai/api/v1")
    for call in matching:
        assert call.kwargs.get("max_retries") == 0, (
            f"max_retries should remain 0 (issue #26293); got "
            f"{call.kwargs.get('max_retries')!r}"
        )
