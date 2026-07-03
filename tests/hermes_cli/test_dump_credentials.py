"""Tests for ``hermes dump`` API-keys section, focused on OAuth detection.

Regression coverage for the bug where providers authenticated via OAuth
(Nous Portal, OpenAI Codex, Anthropic via claude_code) reported as
"not set" in the debug dump even when their tokens were live in
``auth.json::credential_pool``.
"""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _run_dump(show_keys: bool = False) -> str:
    """Invoke ``run_dump`` and capture its stdout."""
    from hermes_cli.dump import run_dump

    args = SimpleNamespace(show_keys=show_keys)
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        run_dump(args)
    finally:
        sys.stdout = old_stdout
    return captured.getvalue()


def _write_credential_pool(hermes_home, pool: dict) -> None:
    """Persist a credential_pool payload into ``auth.json`` under HERMES_HOME."""
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(
        json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": pool,
        })
    )


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME for dump tests."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class TestDumpApiKeysOAuthDetection:
    """``hermes dump`` should report OAuth-backed providers as set."""

    def test_nous_oauth_token_reports_set_oauth(self, hermes_home):
        """Nous logged in via OAuth → reported as ``set (oauth)``."""
        _write_credential_pool(
            hermes_home,
            {
                "nous": [
                    {
                        "id": "abc123",
                        "source": "device_code",
                        "auth_type": "oauth",
                        "access_token": "eyJhbGciOiJSUzI1NiJ9.fake.token",
                    }
                ],
            },
        )

        out = _run_dump()

        # Find the nous row and assert it shows oauth, not "not set".
        nous_lines = [l for l in out.splitlines() if l.strip().startswith("nous ")]
        assert nous_lines, f"no nous row in dump output:\n{out}"
        assert "set (oauth)" in nous_lines[0], (
            f"nous OAuth login should show 'set (oauth)', got: {nous_lines[0]}"
        )

    def test_anthropic_oauth_via_claude_code_reports_set_oauth(self, hermes_home):
        """Anthropic claude_code OAuth seed → ``anthropic`` shows ``set (oauth)``."""
        _write_credential_pool(
            hermes_home,
            {
                "anthropic": [
                    {
                        "id": "cc1",
                        "source": "claude_code",
                        "auth_type": "oauth",
                        "access_token": "sk-ant-oat01-fake-claude-code-token",
                    }
                ],
            },
        )

        out = _run_dump()
        anthropic_lines = [
            l
            for l in out.splitlines()
            if l.strip().startswith("anthropic ")
            or l.strip().startswith("anthropic_token ")
        ]
        assert any("set (oauth)" in l for l in anthropic_lines), (
            f"anthropic should show 'set (oauth)' when claude_code is in pool, "
            f"got: {anthropic_lines}"
        )

    def test_codex_oauth_only_provider_appears(self, hermes_home):
        """OpenAI Codex (OAuth-only, no env var) shows up when logged in."""
        _write_credential_pool(
            hermes_home,
            {
                "openai-codex": [
                    {
                        "id": "cdx1",
                        "source": "device_code",
                        "auth_type": "oauth",
                        "access_token": "ChatGPT-token",
                    }
                ],
            },
        )

        out = _run_dump()
        codex_lines = [
            l for l in out.splitlines() if l.strip().startswith("openai_codex ")
        ]
        assert codex_lines, f"openai_codex row missing from dump:\n{out}"
        assert "set (oauth)" in codex_lines[0]

    def test_codex_not_logged_in_shows_not_set(self, hermes_home):
        """OAuth-only providers without a pool entry render as ``not set``."""
        _write_credential_pool(hermes_home, {})

        out = _run_dump()
        codex_lines = [
            l for l in out.splitlines() if l.strip().startswith("openai_codex ")
        ]
        assert codex_lines
        assert "not set" in codex_lines[0]

    def test_env_set_takes_precedence_over_pool(self, hermes_home, monkeypatch):
        """Env var present → ``set (env)``, even if pool also has an entry."""
        monkeypatch.setenv("NOUS_API_KEY", "manual-api-key")
        _write_credential_pool(
            hermes_home,
            {
                "nous": [
                    {
                        "id": "abc",
                        "source": "device_code",
                        "auth_type": "oauth",
                        "access_token": "oauth-token",
                    }
                ],
            },
        )

        out = _run_dump()
        nous_lines = [l for l in out.splitlines() if l.strip().startswith("nous ")]
        assert nous_lines
        assert "set (env)" in nous_lines[0], f"env var should win, got: {nous_lines[0]}"

    def test_empty_pool_entry_does_not_falsely_report_set(self, hermes_home):
        """Pool entries with empty ``access_token`` must not count as set."""
        _write_credential_pool(
            hermes_home,
            {
                "nous": [
                    {
                        "id": "abc",
                        "source": "device_code",
                        "auth_type": "oauth",
                        "access_token": "",
                    }
                ],
            },
        )

        out = _run_dump()
        nous_lines = [l for l in out.splitlines() if l.strip().startswith("nous ")]
        assert nous_lines
        assert "not set" in nous_lines[0]

    def test_no_auth_file_falls_back_to_not_set(self, tmp_path, monkeypatch):
        """Missing auth.json → providers report ``not set`` cleanly (no crash)."""
        home = tmp_path / ".hermes-empty"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))

        out = _run_dump()
        # Spot-check: nous should still appear, marked as not set.
        nous_lines = [l for l in out.splitlines() if l.strip().startswith("nous ")]
        assert nous_lines
        assert "not set" in nous_lines[0]

    def test_show_keys_redacts_env_value(self, hermes_home, monkeypatch):
        """``--show-keys`` continues to render redacted env values."""
        monkeypatch.setenv("NOUS_API_KEY", "abcd-very-long-secret-1234")

        out = _run_dump(show_keys=True)
        nous_lines = [l for l in out.splitlines() if l.strip().startswith("nous ")]
        assert nous_lines
        # Redaction shows first/last chars but not the full string.
        assert "abcd-very-long-secret-1234" not in nous_lines[0]
        # And it does NOT show "set (env)" because show_keys path emits the
        # redacted value instead.
        assert "set (env)" not in nous_lines[0]

    def test_oauth_only_label_unchanged_when_show_keys(self, hermes_home):
        """``--show-keys`` does not leak OAuth tokens — still emits ``set (oauth)``."""
        _write_credential_pool(
            hermes_home,
            {
                "openai-codex": [
                    {
                        "id": "cdx",
                        "source": "device_code",
                        "auth_type": "oauth",
                        "access_token": "very-secret-codex-token",
                    }
                ],
            },
        )

        out = _run_dump(show_keys=True)
        codex_lines = [
            l for l in out.splitlines() if l.strip().startswith("openai_codex ")
        ]
        assert codex_lines
        assert "set (oauth)" in codex_lines[0]
        assert "very-secret-codex-token" not in codex_lines[0]
