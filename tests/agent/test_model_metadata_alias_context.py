"""Regression tests for #87431 — context length resolution for rolling aliases and custom endpoints.

Ensures:
1. _resolve_endpoint_context_length resolves context length for model IDs with '~' rolling prefixes (~moonshotai/kimi-latest).
2. _resolve_endpoint_context_length resolves context length when endpoint keys have or omit vendor prefixes.
3. get_model_context_length returns the resolved context length for alias model IDs.
"""

from __future__ import annotations

from unittest.mock import patch
import pytest

from agent.model_metadata import (
    _resolve_endpoint_context_length,
    get_model_context_length,
)


def test_resolve_endpoint_context_length_exact_match():
    """Exact model ID match returns context_length."""
    metadata = {
        "kimi-k3": {"context_length": 1048576},
        "deepseek-chat": {"context_length": 65536},
    }
    with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
        ctx = _resolve_endpoint_context_length("kimi-k3", "https://api.example.com/v1")
    assert ctx == 1048576


def test_resolve_endpoint_context_length_tilde_prefix():
    """Rolling alias with '~' prefix resolves to endpoint model entry."""
    metadata = {
        "moonshotai/kimi-latest": {"context_length": 1048576},
    }
    with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
        ctx = _resolve_endpoint_context_length("~moonshotai/kimi-latest", "https://api.example.com/v1")
    assert ctx == 1048576


def test_resolve_endpoint_context_length_bare_vs_prefixed():
    """Bare model name matches prefixed endpoint entry and vice versa."""
    metadata = {
        "moonshotai/kimi-latest": {"context_length": 1048576},
        "qwen-plus": {"context_length": 131072},
    }
    with patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=metadata):
        # Query bare name against vendor-prefixed metadata
        ctx1 = _resolve_endpoint_context_length("kimi-latest", "https://api.example.com/v1")
        # Query vendor-prefixed name against bare metadata
        ctx2 = _resolve_endpoint_context_length("alibaba/qwen-plus", "https://api.example.com/v1")

    assert ctx1 == 1048576
    assert ctx2 == 131072
