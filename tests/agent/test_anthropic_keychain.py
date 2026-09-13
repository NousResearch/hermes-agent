"""Tests for Bug #12905 fixes in agent/anthropic_adapter.py — macOS Keychain support."""

import json
import platform
import subprocess
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from agent.anthropic_credentials import (
    _read_claude_code_credentials_from_keychain,
    read_claude_code_credentials,
    _refresh_oauth_token,
    _merge_keychain_credential_payload,
    _mirror_claude_code_credentials_to_keychain,
)


# This module exercises the reader itself with explicit platform and subprocess
# mocks, so it opts out of the suite-wide guard without touching a real Keychain.
pytestmark = pytest.mark.allow_macos_keychain


@pytest.mark.macos_only
class TestReadClaudeCodeCredentialsFromKeychain:
    """Bug 4: macOS Keychain support for Claude Code >=2.1.114.

    ``macos_only``: the reader is gated on ``platform.system() == "Darwin"``
    and shells out to the ``security`` CLI. Faking Darwin on Linux selected
    the branch but proved nothing about the host it exists for; on the real
    macOS runner only ``subprocess.run`` is mocked (via the
    ``allow_macos_keychain`` opt-out of the suite-wide guard), so no real
    Keychain is ever touched.
    """



    def test_returns_none_when_security_command_not_found(self):
        """OSError from missing security binary must be handled gracefully."""
        with patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=OSError("security not found")):
            assert _read_claude_code_credentials_from_keychain() is None

    def test_returns_none_on_nonzero_exit_code(self):
        """security returns non-zero when the Keychain entry doesn't exist."""
        with patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            assert _read_claude_code_credentials_from_keychain() is None







@pytest.mark.macos_only
class TestReadClaudeCodeCredentialsPriority:
    """Bug 4: Keychain must be checked before the JSON file."""

    def test_keychain_takes_priority_over_json_file(self, tmp_path, monkeypatch):
        """When both Keychain and JSON file have credentials, Keychain wins."""
        # Set up JSON file with "older" token
        json_cred_file = tmp_path / ".claude" / ".credentials.json"
        json_cred_file.parent.mkdir(parents=True)
        json_cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "json-token",
                "refreshToken": "json-refresh",
                "expiresAt": 9999999999999,
            }
        }))
        monkeypatch.setattr("agent.anthropic_credentials.Path.home", lambda: tmp_path)

        # Mock Keychain to return a "newer" token
        with patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "claudeAiOauth": {
                        "accessToken": "keychain-token",
                        "refreshToken": "keychain-refresh",
                        "expiresAt": 9999999999999,
                    }
                }),
                stderr="",
            )
            creds = read_claude_code_credentials()

        # Keychain token should be returned, not JSON file token
        assert creds is not None
        assert creds["accessToken"] == "keychain-token"
        assert creds["source"] == "macos_keychain"

    def test_falls_back_to_json_when_keychain_returns_none(self, tmp_path, monkeypatch):
        """When Keychain has no entry, JSON file is used as fallback."""
        json_cred_file = tmp_path / ".claude" / ".credentials.json"
        json_cred_file.parent.mkdir(parents=True)
        json_cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "json-fallback-token",
                "refreshToken": "json-refresh",
                "expiresAt": 9999999999999,
            }
        }))
        monkeypatch.setattr("agent.anthropic_credentials.Path.home", lambda: tmp_path)

        with patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            # Simulate Keychain entry not found
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            creds = read_claude_code_credentials()

        assert creds is not None
        assert creds["accessToken"] == "json-fallback-token"
        assert creds["source"] == "claude_code_credentials_file"

    def test_returns_none_when_neither_keychain_nor_json_has_creds(self, tmp_path, monkeypatch):
        """No credentials anywhere — must return None cleanly."""
        monkeypatch.setattr("agent.anthropic_credentials.Path.home", lambda: tmp_path)

        with patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            creds = read_claude_code_credentials()

        assert creds is None


@pytest.mark.macos_only
class TestReadClaudeCodeCredentialsDesync:
    """Reconciliation when Keychain and JSON file disagree.

    Observed in the wild on Claude Code 2.1.x: a refresh updates one source
    (commonly the JSON file) but leaves the other holding an expired token.
    The reader must not blindly return whichever source it consulted first;
    it must prefer the non-expired credential.
    """

    # Far-future ms-epoch — comfortably valid under is_claude_code_token_valid.
    _FRESH = 9_999_999_999_999
    # Past ms-epoch — comfortably expired (with the 60s buffer).
    _EXPIRED = 1

    def _setup(self, tmp_path, monkeypatch, *, file_expires_at, file_token="json-token"):
        json_cred_file = tmp_path / ".claude" / ".credentials.json"
        json_cred_file.parent.mkdir(parents=True)
        json_cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": file_token,
                "refreshToken": "json-refresh",
                "expiresAt": file_expires_at,
            }
        }))
        monkeypatch.setattr("agent.anthropic_credentials.Path.home", lambda: tmp_path)

    def _keychain_payload(self, *, access_token, expires_at, refresh_token="kc-refresh"):
        return MagicMock(
            returncode=0,
            stdout=json.dumps({
                "claudeAiOauth": {
                    "accessToken": access_token,
                    "refreshToken": refresh_token,
                    "expiresAt": expires_at,
                }
            }),
            stderr="",
        )

    def test_keychain_expired_file_fresh_returns_file(self, tmp_path, monkeypatch):
        """Regression: when the Keychain holds an expired token but the JSON
        file has a valid one, callers must receive the valid file token rather
        than None. (Pre-fix behavior returned the expired Keychain token, and
        downstream validity checks then yielded None — surfacing the misleading
        ``No Anthropic credentials found`` error.)
        """
        self._setup(tmp_path, monkeypatch, file_expires_at=self._FRESH, file_token="fresh-file-token")
        with patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = self._keychain_payload(
                access_token="stale-keychain-token", expires_at=self._EXPIRED,
            )
            creds = read_claude_code_credentials()

        assert creds is not None
        assert creds["accessToken"] == "fresh-file-token"
        assert creds["source"] == "claude_code_credentials_file"



    def test_both_expired_prefers_later_expiry(self, tmp_path, monkeypatch):
        """When both are expired, return the one with the later ``expiresAt``;
        its ``refresh_token`` is the most recently issued and most likely to
        succeed at the OAuth refresh endpoint.
        """
        self._setup(tmp_path, monkeypatch, file_expires_at=self._EXPIRED + 5, file_token="newer-expired-file")
        with patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = self._keychain_payload(
                access_token="older-expired-keychain", expires_at=self._EXPIRED,
            )
            creds = read_claude_code_credentials()

        assert creds is not None
        assert creds["accessToken"] == "newer-expired-file"


class TestRefreshOAuthTokenAdoptsFreshCredential:
    """``_refresh_oauth_token`` should adopt a credential Claude Code has
    already refreshed rather than POSTing a (possibly already-rotated)
    single-use refresh token and racing Claude Code into ``invalid_grant``.
    """

    _FRESH = 9_999_999_999_999

    def test_adopts_already_refreshed_token_without_posting(self, tmp_path, monkeypatch):
        """When a live source already holds a valid token, return it and skip
        the network refresh entirely.
        """
        monkeypatch.setattr(
            "agent.anthropic_credentials.claude_code_credentials_path",
            lambda: tmp_path / ".claude" / ".credentials.json",
        )
        fresh = {
            "accessToken": "already-refreshed-token",
            "refreshToken": "live-refresh",
            "expiresAt": self._FRESH,
        }
        monkeypatch.setattr(
            "agent.anthropic_credentials.read_claude_code_credentials",
            lambda: fresh,
        )

        def _should_not_be_called(*args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("refresh_anthropic_oauth_pure must not be called")

        monkeypatch.setattr(
            "agent.anthropic_credentials.refresh_anthropic_oauth_pure",
            _should_not_be_called,
        )

        # Stale creds passed in by the caller — should be ignored in favor
        # of the live, already-refreshed token.
        result = _refresh_oauth_token({"refreshToken": "stale", "expiresAt": 1})
        assert result == "already-refreshed-token"

    def test_falls_back_to_network_refresh_when_no_fresh_credential(self, tmp_path, monkeypatch):
        """When no live source has a valid token, fall back to refreshing
        ourselves using the freshest available refresh token.
        """
        monkeypatch.setattr(
            "agent.anthropic_credentials.claude_code_credentials_path",
            lambda: tmp_path / ".claude" / ".credentials.json",
        )
        # Live read returns an expired credential carrying a refresh token.
        monkeypatch.setattr(
            "agent.anthropic_credentials.read_claude_code_credentials",
            lambda: {"accessToken": "expired", "refreshToken": "live-refresh", "expiresAt": 1},
        )
        captured = {}

        def _fake_refresh(refresh_token, **kwargs):
            captured["refresh_token"] = refresh_token
            return {
                "access_token": "newly-minted",
                "refresh_token": "rotated",
                "expires_at_ms": self._FRESH,
            }

        monkeypatch.setattr(
            "agent.anthropic_credentials.refresh_anthropic_oauth_pure", _fake_refresh
        )
        monkeypatch.setattr(
            "agent.anthropic_credentials._write_claude_code_credentials",
            lambda *a, **k: None,
        )

        result = _refresh_oauth_token({"refreshToken": "caller-refresh", "expiresAt": 1})
        assert result == "newly-minted"
        # Prefers the live source's refresh token over the caller's stale copy.
        assert captured["refresh_token"] == "live-refresh"

    def test_concurrent_refreshes_use_one_shared_credentials_lock(self, tmp_path, monkeypatch):
        """Direct resolver refreshes must not spend one Claude token twice."""
        shared_credentials_path = tmp_path / ".claude" / ".credentials.json"
        monkeypatch.setattr(
            "agent.anthropic_credentials.claude_code_credentials_path",
            lambda: shared_credentials_path,
        )

        state = {
            "accessToken": "stale-access",
            "refreshToken": "stale-refresh",
            "expiresAt": 1,
        }
        state_lock = threading.Lock()
        calls = []

        def read_credentials():
            with state_lock:
                return dict(state)

        def write_credentials(access_token, refresh_token, expires_at_ms, **_kwargs):
            with state_lock:
                state.update(
                    accessToken=access_token,
                    refreshToken=refresh_token,
                    expiresAt=expires_at_ms,
                )

        def refresh(refresh_token, **_kwargs):
            calls.append(refresh_token)
            # Without the production shared lock, both callers read the stale
            # pair before either fake network request commits its rotation.
            time.sleep(0.05)
            with state_lock:
                if state["refreshToken"] != refresh_token:
                    raise ValueError("invalid_grant: refresh token already used")
                return {
                    "access_token": "fresh-access",
                    "refresh_token": "fresh-refresh",
                    "expires_at_ms": self._FRESH,
                }

        monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", read_credentials)
        monkeypatch.setattr("agent.anthropic_credentials._write_claude_code_credentials", write_credentials)
        monkeypatch.setattr("agent.anthropic_credentials.refresh_anthropic_oauth_pure", refresh)

        results = {}
        errors = {}
        start = threading.Barrier(2)

        def run(name):
            try:
                start.wait(timeout=5)
                results[name] = _refresh_oauth_token(
                    {
                        "accessToken": "stale-access",
                        "refreshToken": "stale-refresh",
                        "expiresAt": 1,
                    }
                )
            except BaseException as exc:  # pragma: no cover - failure diagnostics
                errors[name] = exc

        threads = [threading.Thread(target=run, args=(name,)) for name in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert not [thread for thread in threads if thread.is_alive()]
        assert not errors, errors
        assert results == {"a": "fresh-access", "b": "fresh-access"}
        assert calls == ["stale-refresh"], calls


class TestMergeKeychainCredentialPayload:
    """``_merge_keychain_credential_payload`` — the pure merge that a Keychain
    refresh write performs over the existing entry. Host-agnostic, so it runs
    on every lane and pins the #98334 invariant: rotate the token triple while
    preserving the metadata Claude Code gates on."""

    _EXISTING = {
        "claudeAiOauth": {
            "accessToken": "old-access",
            "refreshToken": "old-refresh",
            "expiresAt": 1,
            "scopes": ["user:inference", "user:profile"],
            "subscriptionType": "max",
        },
        "rateLimitTier": "tier-1",
    }

    def test_rotates_triple_preserves_metadata(self):
        merged = _merge_keychain_credential_payload(self._EXISTING, "new-access", "new-refresh", 42)
        oauth = merged["claudeAiOauth"]
        assert oauth["accessToken"] == "new-access"
        assert oauth["refreshToken"] == "new-refresh"
        assert oauth["expiresAt"] == 42
        # The fields Claude Code >=2.1.81 gates on survive the merge.
        assert oauth["scopes"] == ["user:inference", "user:profile"]
        assert oauth["subscriptionType"] == "max"
        assert merged["rateLimitTier"] == "tier-1"
        # Input payload is not mutated (no aliasing surprise).
        assert self._EXISTING["claudeAiOauth"]["refreshToken"] == "old-refresh"

    def test_tolerates_missing_oauth_block(self):
        merged = _merge_keychain_credential_payload({"other": 1}, "a", "b", 7)
        assert merged["claudeAiOauth"] == {"accessToken": "a", "refreshToken": "b", "expiresAt": 7}
        assert merged["other"] == 1


class TestMirrorClaudeCodeCredentialsToKeychain:
    """``_mirror_claude_code_credentials_to_keychain`` — the #98334 write mirror.

    The write path is gated on ``platform.system() == "Darwin"`` and shells out
    to ``security``. We force the gate to "Darwin" so the logic runs on every
    lane, and mock only the raw reader and ``subprocess.run`` — no real Keychain
    is ever touched and no real ``security`` binary is required.
    """

    @pytest.fixture(autouse=True)
    def _darwin_gate(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")

    def test_writes_via_stdin_not_argv_when_entry_exists(self, monkeypatch):
        """The rotated pair must reach the existing Keychain item via
        ``add-generic-password -U`` with the payload on stdin — never as a
        ``-w`` argv token (a live secret would be visible in the process table)."""
        existing = {
            "claudeAiOauth": {"accessToken": "old", "refreshToken": "old-ref",
                              "expiresAt": 1, "scopes": ["user:inference"],
                              "subscriptionType": "max"},
        }
        monkeypatch.setattr(
            "agent.anthropic_credentials._read_claude_code_keychain_payload", lambda: existing)
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((list(argv), kwargs))
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        _mirror_claude_code_credentials_to_keychain("new-access", "new-ref", 99)

        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert "add-generic-password" in argv
        assert "-U" in argv
        assert "-s" in argv and "Claude Code-credentials" in argv
        # The bare -w flag reads the password from stdin, so the secret must NOT
        # appear anywhere on the command line.
        assert "-w" in argv
        joined = " ".join(argv)
        assert "new-access" not in joined
        assert "new-ref" not in joined
        # The payload goes to stdin and carries the rotated triple + preserved metadata.
        payload = json.loads(kwargs["input"])
        assert payload["claudeAiOauth"]["refreshToken"] == "new-ref"
        assert payload["claudeAiOauth"]["accessToken"] == "new-access"
        assert payload["claudeAiOauth"]["subscriptionType"] == "max"

    def test_no_write_when_no_entry_exists(self, monkeypatch):
        """Never create a Keychain item the user has not."""
        monkeypatch.setattr(
            "agent.anthropic_credentials._read_claude_code_keychain_payload", lambda: None)
        called = []
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a))

        _mirror_claude_code_credentials_to_keychain("a", "b", 1)

        assert called == []

    def test_fail_soft_when_security_raises(self, monkeypatch):
        """A mirror failure is logged, never raised: the file commit already
        succeeded and the resolver still resolves from it."""
        monkeypatch.setattr(
            "agent.anthropic_credentials._read_claude_code_keychain_payload",
            lambda: {"claudeAiOauth": {"accessToken": "x"}})

        def boom(*a, **k):
            raise OSError("security not available")

        monkeypatch.setattr(subprocess, "run", boom)

        # Must not raise.
        _mirror_claude_code_credentials_to_keychain("a", "b", 1)

    def test_fail_soft_on_nonzero_exit(self, monkeypatch):
        monkeypatch.setattr(
            "agent.anthropic_credentials._read_claude_code_keychain_payload",
            lambda: {"claudeAiOauth": {"accessToken": "x"}})
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: MagicMock(returncode=1, stdout="", stderr="duplicate item"))

        # Must not raise.
        _mirror_claude_code_credentials_to_keychain("a", "b", 1)

