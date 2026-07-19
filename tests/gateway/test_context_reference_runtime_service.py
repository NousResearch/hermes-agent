"""DEAD path: not imported by gateway/run.py — contract-only unit tests.
See gateway/RUNTIME_SERVICES.md.
"""
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.dead_runtime_service

from gateway.context_reference_runtime_service import (
    expand_gateway_context_references,
    resolve_gateway_context_reference_cwd,
    should_expand_gateway_context_references,
)


def test_should_expand_gateway_context_references_detects_at_sign():
    assert should_expand_gateway_context_references("check @file:foo.py") is True
    assert should_expand_gateway_context_references("plain text") is False


def test_resolve_gateway_context_reference_cwd_prefers_messaging_cwd():
    assert (
        resolve_gateway_context_reference_cwd("/tmp/workspace", home_dir="/home/demo")
        == "/tmp/workspace"
    )
    assert (
        resolve_gateway_context_reference_cwd(None, home_dir="/home/demo")
        == "/home/demo"
    )


@pytest.mark.asyncio
async def test_expand_gateway_context_references_returns_expanded_message():
    async def fake_preprocessor(message, **kwargs):
        assert kwargs["cwd"] == "/tmp/workspace"
        assert kwargs["allowed_root"] == "/tmp/workspace"
        assert kwargs["context_length"] == 12345
        return SimpleNamespace(
            blocked=False,
            expanded=True,
            message="expanded message",
        )

    result = await expand_gateway_context_references(
        "check @file:foo.py",
        model="gpt-test",
        base_url="https://example.com/v1",
        messaging_cwd="/tmp/workspace",
        preprocessor=fake_preprocessor,
        context_length_loader=lambda model, base_url="": 12345,
    )

    assert result.message_text == "expanded message"
    assert result.blocked_warning is None


@pytest.mark.asyncio
async def test_expand_gateway_context_references_returns_blocked_warning():
    async def fake_preprocessor(message, **kwargs):
        return SimpleNamespace(
            blocked=True,
            expanded=False,
            warnings=["blocked 1", "blocked 2"],
        )

    result = await expand_gateway_context_references(
        "check @file:.ssh/id_rsa",
        model="gpt-test",
        preprocessor=fake_preprocessor,
        context_length_loader=lambda model, base_url="": 12345,
    )

    assert result.message_text == "check @file:.ssh/id_rsa"
    assert result.blocked_warning == "blocked 1\nblocked 2"


@pytest.mark.asyncio
async def test_expand_gateway_context_references_swallows_errors():
    class FakeLogger:
        def __init__(self):
            self.calls = []

        def debug(self, fmt, exc):
            self.calls.append((fmt, str(exc)))

    async def fake_preprocessor(message, **kwargs):
        raise RuntimeError("boom")

    logger = FakeLogger()
    result = await expand_gateway_context_references(
        "check @file:foo.py",
        model="gpt-test",
        preprocessor=fake_preprocessor,
        context_length_loader=lambda model, base_url="": 12345,
        logger=logger,
    )

    assert result.message_text == "check @file:foo.py"
    assert result.blocked_warning is None
    assert logger.calls
