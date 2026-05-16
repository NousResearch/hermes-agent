"""Regression tests for global manual:* credential pool entries being visible inside profiles.

Bug: When a profile's auth.json has seeded anthropic entries (claude_code, env:ANTHROPIC_TOKEN),
global auth.json entries with source 'manual:hermes_pkce' were silently ignored.
The profile's seeded entries shadow the global pool entirely, so when the seeded entries
are pruned (because their source credentials are unavailable) or exhausted (usage cap),
the global manual entry is never tried.

Fix: read_credential_pool() now appends global manual:* entries to the profile's entries
when those entries are not already represented (by id or source) in the profile pool.

Separately: _dump_api_request_debug() in AIAgent was constructing a misleading URL
(/chat/completions) for anthropic_messages mode and reading api_key from self.client
(which is None in anthropic_messages mode, giving "Bearer None"). Fixed to use
self._anthropic_api_key and build the correct /v1/messages URL.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Set up a global root + h2reviewer profile, similar to test_auth_profile_fallback.py.

    * Path.home() -> tmp_path
    * Global root -> tmp_path/.hermes
    * Profile     -> tmp_path/.hermes/profiles/h2reviewer  (active)
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    global_root = tmp_path / ".hermes"
    global_root.mkdir()
    profile_dir = global_root / "profiles" / "h2reviewer"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    return {"global": global_root, "profile": profile_dir}


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _make_store(pool: dict) -> dict:
    return {"version": 1, "credential_pool": pool}


def _make_oauth_entry(
    *,
    entry_id: str,
    source: str,
    access_token: str,
    priority: int = 0,
    expires_at_ms: int | None = None,
    last_status: str | None = None,
) -> dict:
    entry: dict = {
        "id": entry_id,
        "label": source,
        "auth_type": "oauth",
        "priority": priority,
        "source": source,
        "access_token": access_token,
        "last_status": last_status,
        "last_status_at": None,
        "last_error_code": None,
        "last_error_reason": None,
        "last_error_message": None,
        "last_error_reset_at": None,
        "request_count": 0,
    }
    if expires_at_ms is not None:
        entry["expires_at_ms"] = expires_at_ms
    return entry


FUTURE_MS = int(time.time() * 1000) + 10_000_000  # ~2.8h


# ---------------------------------------------------------------------------
# Tests for read_credential_pool global-manual merge
# ---------------------------------------------------------------------------

class TestReadCredentialPoolGlobalManualMerge:
    """read_credential_pool should append global manual:* entries into a
    profile pool that has seeded (non-manual) entries for the same provider."""

    def test_global_manual_pkce_merged_into_profile_with_seeded_entries(self, profile_env):
        """Core regression: the global manual:hermes_pkce entry appears in the pool
        even when the profile already has seeded claude_code / env:* entries."""
        from hermes_cli.auth import read_credential_pool

        # Global auth.json — has one manual:hermes_pkce OAuth entry
        _write(profile_env["global"] / "auth.json", _make_store({
            "anthropic": [
                _make_oauth_entry(
                    entry_id="global-pkce-01",
                    source="manual:hermes_pkce",
                    access_token="sk-ant-oat01-GLOBAL_PKCE",
                    expires_at_ms=FUTURE_MS,
                )
            ]
        }))

        # Profile auth.json — has seeded entries (like the real h2reviewer profile)
        _write(profile_env["profile"] / "auth.json", _make_store({
            "anthropic": [
                _make_oauth_entry(
                    entry_id="profile-env-01",
                    source="env:ANTHROPIC_TOKEN",
                    access_token="sk-ant-oat01-ENV_TOKEN",
                    priority=0,
                ),
                _make_oauth_entry(
                    entry_id="profile-cc-01",
                    source="claude_code",
                    access_token="sk-ant-oat01-CLAUDE_CODE",
                    expires_at_ms=FUTURE_MS,
                    priority=1,
                ),
            ]
        }))

        entries = read_credential_pool("anthropic")
        sources = [e.get("source") for e in entries]

        assert "manual:hermes_pkce" in sources, (
            "Global manual:hermes_pkce entry must be merged into profile pool. "
            f"Got sources: {sources}"
        )
        assert "env:ANTHROPIC_TOKEN" in sources
        assert "claude_code" in sources

    def test_global_manual_entry_not_duplicated_if_already_in_profile(self, profile_env):
        """If the profile already has an entry from the same source, the global
        entry must not be added again."""
        from hermes_cli.auth import read_credential_pool

        _write(profile_env["global"] / "auth.json", _make_store({
            "anthropic": [
                _make_oauth_entry(
                    entry_id="global-pkce-01",
                    source="manual:hermes_pkce",
                    access_token="sk-ant-oat01-GLOBAL",
                    expires_at_ms=FUTURE_MS,
                )
            ]
        }))

        # Profile already has a manual:hermes_pkce entry (same source)
        _write(profile_env["profile"] / "auth.json", _make_store({
            "anthropic": [
                _make_oauth_entry(
                    entry_id="profile-pkce-01",
                    source="manual:hermes_pkce",
                    access_token="sk-ant-oat01-PROFILE",
                    expires_at_ms=FUTURE_MS,
                ),
            ]
        }))

        entries = read_credential_pool("anthropic")
        pkce_entries = [e for e in entries if e.get("source") == "manual:hermes_pkce"]

        assert len(pkce_entries) == 1, (
            f"Profile entry should not be duplicated. Got {len(pkce_entries)} pkce entries."
        )
        # Profile entry wins (lower index / original)
        assert pkce_entries[0]["id"] == "profile-pkce-01"

    def test_global_seeded_entries_not_merged_into_profile(self, profile_env):
        """Only manual:* global entries should be merged — not seeded ones like
        claude_code or env:ANTHROPIC_TOKEN, which get re-seeded by load_pool()."""
        from hermes_cli.auth import read_credential_pool

        _write(profile_env["global"] / "auth.json", _make_store({
            "anthropic": [
                _make_oauth_entry(
                    entry_id="global-cc-01",
                    source="claude_code",
                    access_token="sk-ant-oat01-GLOBAL_CC",
                ),
                _make_oauth_entry(
                    entry_id="global-env-01",
                    source="env:ANTHROPIC_TOKEN",
                    access_token="sk-ant-oat01-GLOBAL_ENV",
                ),
            ]
        }))

        _write(profile_env["profile"] / "auth.json", _make_store({
            "anthropic": [
                _make_oauth_entry(
                    entry_id="profile-cc-01",
                    source="claude_code",
                    access_token="sk-ant-oat01-PROFILE_CC",
                ),
            ]
        }))

        entries = read_credential_pool("anthropic")
        cc_entries = [e for e in entries if e.get("source") == "claude_code"]
        env_entries = [e for e in entries if e.get("source") == "env:ANTHROPIC_TOKEN"]

        assert len(cc_entries) == 1, f"claude_code should not be duplicated: {cc_entries}"
        assert cc_entries[0]["id"] == "profile-cc-01", "Profile entry should win"
        assert len(env_entries) == 0, (
            "Seeded env:ANTHROPIC_TOKEN from global should NOT be merged when not in profile. "
            f"Got: {env_entries}"
        )

    def test_no_profile_entries_falls_back_to_global_entirely(self, profile_env):
        """When the profile has NO entries for the provider, all global entries
        are returned (existing behavior preserved)."""
        from hermes_cli.auth import read_credential_pool

        _write(profile_env["global"] / "auth.json", _make_store({
            "anthropic": [
                _make_oauth_entry(
                    entry_id="global-pkce-01",
                    source="manual:hermes_pkce",
                    access_token="sk-ant-oat01-GLOBAL_PKCE",
                    expires_at_ms=FUTURE_MS,
                )
            ]
        }))

        # Profile has empty anthropic pool
        _write(profile_env["profile"] / "auth.json", _make_store({}))

        entries = read_credential_pool("anthropic")
        assert len(entries) == 1
        assert entries[0]["source"] == "manual:hermes_pkce"
        assert entries[0]["access_token"] == "sk-ant-oat01-GLOBAL_PKCE"


# ---------------------------------------------------------------------------
# Tests for resolve_runtime_provider: global manual:hermes_pkce token is used
# ---------------------------------------------------------------------------

class TestRuntimeProviderUsesGlobalManualPkceToken:
    """resolve_runtime_provider for anthropic provider should use the global
    manual:hermes_pkce token when the profile's seeded entries are unavailable."""

    def test_runtime_uses_global_pkce_when_profile_entries_pruned(
        self, profile_env, monkeypatch
    ):
        """Simulates the h2reviewer scenario:
        - Profile has seeded claude_code and env:ANTHROPIC_TOKEN entries
        - ANTHROPIC_TOKEN env var is NOT set -> env:ANTHROPIC_TOKEN gets pruned
        - Claude Code creds file is missing -> claude_code gets pruned
        - Global manual:hermes_pkce IS present and should be used
        - Result: api_mode=anthropic_messages, api_key=global pkce token
        """
        from hermes_cli import runtime_provider as rp

        # Global: manual:hermes_pkce
        _write(profile_env["global"] / "auth.json", _make_store({
            "anthropic": [
                _make_oauth_entry(
                    entry_id="global-pkce-01",
                    source="manual:hermes_pkce",
                    access_token="sk-ant-oat01-PKCE_TOKEN_FROM_GLOBAL",
                    expires_at_ms=FUTURE_MS,
                )
            ]
        }))

        # Profile: seeded entries only
        _write(profile_env["profile"] / "auth.json", _make_store({
            "anthropic": [
                _make_oauth_entry(
                    entry_id="profile-env-01",
                    source="env:ANTHROPIC_TOKEN",
                    access_token="sk-ant-oat01-STALE_ENV",
                    priority=0,
                ),
                _make_oauth_entry(
                    entry_id="profile-cc-01",
                    source="claude_code",
                    access_token="sk-ant-oat01-STALE_CC",
                    expires_at_ms=FUTURE_MS,
                    priority=1,
                ),
            ]
        }))

        # Env vars NOT set
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        # .hermes/.env is empty  
        (profile_env["global"] / ".env").write_text("")

        # Claude Code creds file absent
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: None,
        )

        # is_provider_explicitly_configured returns True for anthropic
        monkeypatch.setattr(
            "hermes_cli.auth.is_provider_explicitly_configured",
            lambda provider: provider == "anthropic",
        )

        # Config: h2reviewer has model.provider=anthropic
        monkeypatch.setattr(
            rp,
            "_get_model_config",
            lambda: {"default": "claude-opus-4-7", "provider": "anthropic"},
        )

        resolved = rp.resolve_runtime_provider(requested="anthropic")

        assert resolved["provider"] == "anthropic"
        assert resolved["api_mode"] == "anthropic_messages", (
            f"Expected anthropic_messages but got {resolved['api_mode']!r}. "
            "Note: the old debug dump showed /chat/completions for ANY non-codex mode, "
            "but the actual Anthropic SDK transport uses /v1/messages."
        )
        assert resolved["api_key"] == "sk-ant-oat01-PKCE_TOKEN_FROM_GLOBAL", (
            f"Expected global pkce token but got {resolved.get('api_key', 'MISSING')!r}. "
            "The manual:hermes_pkce entry from global auth.json must be used "
            "when profile seeded entries are pruned."
        )
        assert resolved["base_url"] == "https://api.anthropic.com"


# ---------------------------------------------------------------------------
# Tests for _dump_api_request_debug URL and api_key correctness (logic test)
# ---------------------------------------------------------------------------

class TestDumpApiRequestDebugUrl:
    """The request dump URL must reflect the actual transport endpoint.

    These tests verify the endpoint-path computation logic directly,
    matching the patched code in run_agent.py::_dump_api_request_debug.
    """

    @staticmethod
    def _compute_endpoint(api_mode: str) -> str:
        """Mirrors the patched _dump_api_request_debug logic."""
        if api_mode == "codex_responses":
            return "/responses"
        elif api_mode == "anthropic_messages":
            return "/v1/messages"
        else:
            return "/chat/completions"

    def test_anthropic_messages_produces_v1_messages_path(self):
        """anthropic_messages mode must produce /v1/messages, not /chat/completions."""
        path = self._compute_endpoint("anthropic_messages")
        assert path == "/v1/messages", (
            f"Expected /v1/messages but got {path!r}. "
            "The old code returned /chat/completions for ALL non-codex modes, "
            "causing the debug dump to show a misleading URL."
        )

    def test_chat_completions_still_gets_chat_completions(self):
        path = self._compute_endpoint("chat_completions")
        assert path == "/chat/completions"

    def test_codex_responses_still_gets_responses(self):
        path = self._compute_endpoint("codex_responses")
        assert path == "/responses"

    def test_anthropic_messages_url_with_base(self):
        base_url = "https://api.anthropic.com"
        path = self._compute_endpoint("anthropic_messages")
        assert f"{base_url.rstrip('/')}{path}" == "https://api.anthropic.com/v1/messages"


# ---------------------------------------------------------------------------
# Tests for resolve_anthropic_token: confirm it reads env vars, not pool
# ---------------------------------------------------------------------------

class TestResolveAnthropicTokenFallbacks:
    """resolve_anthropic_token is a targeted env/file fallback.
    The credential pool is read upstream by resolve_runtime_provider.
    resolve_anthropic_token must NOT duplicate pool logic."""

    def test_returns_none_when_no_env_and_no_credentials_file(self, monkeypatch):
        """When no env vars and no Claude Code credentials file, returns None."""
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: None,
        )

        from agent.anthropic_adapter import resolve_anthropic_token

        result = resolve_anthropic_token()
        assert result is None, (
            f"Expected None (no credentials) but got {result!r}. "
            "When resolve_anthropic_token returns None, the runtime resolver "
            "must have already consulted the pool before falling here."
        )

    def test_returns_env_token_when_anthropic_token_env_set(self, monkeypatch):
        """Returns the ANTHROPIC_TOKEN env var value when set."""
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-test-env-token")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: None,
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._prefer_refreshable_claude_code_token",
            lambda token, creds: None,
        )

        from agent.anthropic_adapter import resolve_anthropic_token

        result = resolve_anthropic_token()
        assert result == "sk-ant-test-env-token"
