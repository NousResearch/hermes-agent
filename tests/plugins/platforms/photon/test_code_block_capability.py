"""Behavior tests: Photon never receives gateway-emitted fenced code blocks.

Root cause (#photon-terminal-fences): the gateway's tool-progress path emits
terminal commands as fenced blocks on any adapter whose
``supports_code_blocks`` is True. The Photon adapter set that flag from
``PHOTON_MARKDOWN`` — but the sidecar's own outbound router
(``sidecar/send-format.mjs``) silently routes every URL-bearing message
through the plain-text builder, where fences survive as literal `` ``` ``.
Even on the markdown path, iMessage renders a fence as inline monospace
text, not a block. Either way the user saw raw commands in the bubble.

The fix: the adapter stops advertising fence support, so the gateway emits
the compact one-line ``terminal: "cmd..."`` preview instead. These tests pin
that contract.
"""
from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def test_never_claims_code_block_support_even_with_markdown_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Markdown passthrough is for prose styling only; fenced blocks degrade
    (URL fallback -> raw text with literal fences; markdown path -> inline
    monospace). The adapter must not tell the gateway it can render blocks."""
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    assert adapter.supports_code_blocks is False


def test_never_claims_code_block_support_with_markdown_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = _make_adapter(monkeypatch)
    assert adapter.supports_code_blocks is False


def test_format_message_still_passthrough_with_markdown_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix narrows the capability flag only — prose markdown passthrough
    (bold/italic/headings) is unchanged."""
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    assert adapter.format_message("**bold** and _ital_") == "**bold** and _ital_"
