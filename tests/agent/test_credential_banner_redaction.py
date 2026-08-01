"""Credential init banners must not preview secret material (issue #60319).

``agent/agent_init.py`` prints status banners on non-quiet agent startup.
Those lines are routinely captured into orchestrator logs and transcripts,
so partial head/tail previews of tokens or API keys are an exposure surface
with no operational upside over a fixed ``[configured]`` marker.

This file pins:
  * both banner sites emit the fully-redacted form
  * neither site slices credential material for display
  * Entra ID callable providers still get the static label
  * the invalid/missing-key warning still fires
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

_AGENT_INIT = (
    Path(__file__).resolve().parent.parent.parent / "agent" / "agent_init.py"
)
_SRC = _AGENT_INIT.read_text(encoding="utf-8")


class TestCredentialBannerSourcePin:
    """Source-level guards so the two print sites cannot regress silently."""

    def test_token_banner_uses_configured_marker(self):
        assert 'print("🔑 Using token: [configured]")' in _SRC

    def test_api_key_banner_uses_configured_marker(self):
        assert 'print("🔑 Using API key: [configured]")' in _SRC

    def test_no_head_tail_token_preview(self):
        # Historical leak shape from issue #60319.
        assert "effective_key[:8]" not in _SRC
        assert "key_used[:8]" not in _SRC
        assert "effective_key[-4:]" not in _SRC
        assert "key_used[-4:]" not in _SRC

    def test_entra_and_invalid_warnings_preserved(self):
        assert _SRC.count('"🔑 Using credentials: Microsoft Entra ID"') >= 2
        assert "⚠️  Warning: API key appears invalid or missing" in _SRC


def _print_openai_banner(key_used, *, is_token_provider, capsys):
    """Mirror the OpenAI-path banner block (post-#60319) for unit exercise."""
    if is_token_provider(key_used):
        print("🔑 Using credentials: Microsoft Entra ID")
    elif (
        isinstance(key_used, str)
        and key_used
        and key_used != "dummy-key"
        and len(key_used) > 12
    ):
        print("🔑 Using API key: [configured]")
    else:
        print("⚠️  Warning: API key appears invalid or missing")
    return capsys.readouterr().out


def _print_anthropic_banner(effective_key, *, is_token_provider, capsys):
    """Mirror the Anthropic-path banner block (post-#60319) for unit exercise."""
    if is_token_provider(effective_key):
        print("🔑 Using credentials: Microsoft Entra ID")
    elif isinstance(effective_key, str) and len(effective_key) > 12:
        print("🔑 Using token: [configured]")
    return capsys.readouterr().out


class TestCredentialBannerBehaviour:
    def test_api_key_banner_never_leaks_secret_material(self, capsys):
        secret = "sk-proj-SUPERSECRETVALUE1234567890"
        out = _print_openai_banner(
            secret, is_token_provider=lambda _k: False, capsys=capsys
        )
        assert "Using API key: [configured]" in out
        assert secret not in out
        assert secret[:8] not in out
        assert secret[-4:] not in out

    def test_token_banner_never_leaks_secret_material(self, capsys):
        secret = "sk-ant-api03-REALLYSECRETTOKENVALUE99"
        out = _print_anthropic_banner(
            secret, is_token_provider=lambda _k: False, capsys=capsys
        )
        assert "Using token: [configured]" in out
        assert secret not in out
        assert secret[:8] not in out
        assert secret[-4:] not in out

    def test_entra_provider_prints_static_label_without_invoking(self, capsys):
        called = {"n": 0}

        def provider():
            called["n"] += 1
            return "should-never-be-read"

        out = _print_openai_banner(
            provider, is_token_provider=lambda k: callable(k), capsys=capsys
        )
        assert "Microsoft Entra ID" in out
        assert called["n"] == 0
        assert "should-never-be-read" not in out

    def test_invalid_or_missing_key_still_warns(self, capsys):
        out = _print_openai_banner(
            "short", is_token_provider=lambda _k: False, capsys=capsys
        )
        assert "API key appears invalid or missing" in out

    def test_dummy_key_still_warns(self, capsys):
        out = _print_openai_banner(
            "dummy-key", is_token_provider=lambda _k: False, capsys=capsys
        )
        assert "API key appears invalid or missing" in out
