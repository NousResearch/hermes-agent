"""Tests for API key display redaction in CLI configuration.

Verifies that show_config does not expose any API key fragment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_minimal_hermes_cli(monkeypatch):
    """Create a minimally-initialized HermesCLI for testing.

    Avoids the full constructor complexity by directly setting the
    attributes that show_config reads.
    """
    # Prevent any real env-based config loading
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", "/tmp")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "60")

    # Create a bare object and set just the attributes show_config reads
    import cli
    obj = cli.HermesCLI.__new__(cli.HermesCLI)
    # Minimal attributes needed by show_config
    obj.api_key = "sk-test-api-key-1234567890abcdef"
    obj.model = "test-model"
    obj.base_url = "https://test.example.com"
    obj.max_turns = 90
    obj.enabled_toolsets = ["web", "terminal"]
    obj.verbose = False
    obj.session_start = __import__("datetime").datetime.now()
    # Compression setting
    obj.compact = False
    # The real constructor always seeds self.agent = None (cli.py:2822); this
    # helper builds the instance via __new__, so it must set the attribute
    # explicitly — show_config reads self.agent unguarded.
    obj.agent = None
    return obj


def test_api_key_config_display_is_masked(capsys, monkeypatch):
    """show_config must display a MASK, never the configured key body.

    The real contract is a short prefix/suffix mask — upstream's
    test_show_config_credential.py asserts a visible fragment on purpose so a
    mis-sourced credential stays diagnosable. This test guards the other half:
    the body of the key must never survive into the output.
    """
    import cli

    obj = _make_minimal_hermes_cli(monkeypatch)
    obj.show_config()
    captured = capsys.readouterr()

    assert "sk-test-" in captured.out, "the masked prefix should be visible for diagnosis"
    assert "...cdef" in captured.out, "the mask keeps the last four characters"
    assert "1234567890abcdef" not in captured.out, "the key body must never appear"
    assert "Not set!" not in captured.out, "a configured key must not display as Not set!"


def test_api_key_config_not_set_shows_not_set(capsys, monkeypatch):
    """When api_key is empty, show_config should display 'Not set!'."""
    obj = _make_minimal_hermes_cli(monkeypatch)
    obj.api_key = ""
    obj.show_config()
    captured = capsys.readouterr()
    assert "Not set!" in captured.out


def test_api_key_config_short_key_is_never_echoed_whole(capsys, monkeypatch):
    """Documented boundary: upstream masks only keys longer than 12 chars.

    A shorter key therefore reads as 'Not set!'. That is upstream's deliberate
    trade (cli.py: ``len(display_key) > 12``); what must never happen is the
    short secret being echoed whole. Pinned here so a future move to
    presence-only masking has to update this test consciously.
    """
    obj = _make_minimal_hermes_cli(monkeypatch)
    obj.api_key = "sk-short"
    obj.show_config()
    captured = capsys.readouterr()

    assert "sk-short" not in captured.out, "a short key must never be echoed whole"
    assert "Not set!" in captured.out, "upstream masks only keys longer than 12 chars"


def test_api_key_config_microsoft_entra_display(capsys, monkeypatch):
    """When api_key is a callable (Entra ID provider), show Microsoft Entra ID."""
    from cli import HermesCLI
    obj = _make_minimal_hermes_cli(monkeypatch)

    # Set api_key to a callable, simulating the Entra ID provider pattern
    def entra_provider():
        return "fake-token"

    # But is_token_provider detects callables... we need to mock that
    # The code checks `is_token_provider(self.api_key)` first
    with patch("agent.azure_identity_adapter.is_token_provider", return_value=True):
        obj.api_key = entra_provider
        obj.show_config()
    captured = capsys.readouterr()
    assert "Microsoft Entra ID" in captured.out
    assert "[set]" not in captured.out
