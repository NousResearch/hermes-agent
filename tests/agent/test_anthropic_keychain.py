"""Tests for Bug #12905 fixes in agent/anthropic_adapter.py — macOS Keychain support."""

import json
from unittest.mock import patch, MagicMock


from agent.anthropic_adapter import (
    _read_claude_code_credentials_from_keychain,
    read_claude_code_credentials,
    _refresh_oauth_token,
)


class TestReadClaudeCodeCredentialsFromKeychain:
    """Bug 4: macOS Keychain support for Claude Code >=2.1.114."""

    def test_returns_none_on_linux(self):
        """Keychain reading is Darwin-only; must return None on other platforms."""
        with patch("agent.anthropic_adapter.platform.system", return_value="Linux"):
            assert _read_claude_code_credentials_from_keychain() is None

    def test_returns_none_on_windows(self):
        with patch("agent.anthropic_adapter.platform.system", return_value="Windows"):
            assert _read_claude_code_credentials_from_keychain() is None

    def test_returns_none_when_security_command_not_found(self):
        """OSError from missing security binary must be handled gracefully."""
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=OSError("security not found")):
            assert _read_claude_code_credentials_from_keychain() is None

    def test_returns_none_on_nonzero_exit_code(self):
        """security returns non-zero when the Keychain entry doesn't exist."""
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            assert _read_claude_code_credentials_from_keychain() is None

    def test_returns_none_for_empty_stdout(self):
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            assert _read_claude_code_credentials_from_keychain() is None

    def test_returns_none_for_non_json_payload(self):
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="not valid json", stderr="")
            assert _read_claude_code_credentials_from_keychain() is None

    def test_returns_none_when_password_field_is_missing_claude_ai_oauth(self):
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({"someOtherService": {"accessToken": "tok"}}),
                stderr="",
            )
            assert _read_claude_code_credentials_from_keychain() is None

    def test_returns_none_when_access_token_is_empty(self):
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({"claudeAiOauth": {"accessToken": "", "refreshToken": "x"}}),
                stderr="",
            )
            assert _read_claude_code_credentials_from_keychain() is None

    def test_parses_valid_keychain_entry(self):
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "claudeAiOauth": {
                        "accessToken": "kc-access-token-abc",
                        "refreshToken": "kc-refresh-token-xyz",
                        "expiresAt": 9999999999999,
                    }
                }),
                stderr="",
            )
            creds = _read_claude_code_credentials_from_keychain()
            assert creds is not None
            assert creds["accessToken"] == "kc-access-token-abc"
            assert creds["refreshToken"] == "kc-refresh-token-xyz"
            assert creds["expiresAt"] == 9999999999999
            assert creds["source"] == "macos_keychain"


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
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        # Mock Keychain to return a "newer" token
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
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
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            # Simulate Keychain entry not found
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            creds = read_claude_code_credentials()

        assert creds is not None
        assert creds["accessToken"] == "json-fallback-token"
        assert creds["source"] == "claude_code_credentials_file"

    def test_returns_none_when_neither_keychain_nor_json_has_creds(self, tmp_path, monkeypatch):
        """No credentials anywhere — must return None cleanly."""
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            creds = read_claude_code_credentials()

        assert creds is None


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
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

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
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = self._keychain_payload(
                access_token="stale-keychain-token", expires_at=self._EXPIRED,
            )
            creds = read_claude_code_credentials()

        assert creds is not None
        assert creds["accessToken"] == "fresh-file-token"
        assert creds["source"] == "claude_code_credentials_file"

    def test_keychain_fresh_file_expired_returns_keychain(self, tmp_path, monkeypatch):
        """Mirror case: file is the stale source; Keychain wins on validity."""
        self._setup(tmp_path, monkeypatch, file_expires_at=self._EXPIRED, file_token="stale-file-token")
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = self._keychain_payload(
                access_token="fresh-keychain-token", expires_at=self._FRESH,
            )
            creds = read_claude_code_credentials()

        assert creds is not None
        assert creds["accessToken"] == "fresh-keychain-token"
        assert creds["source"] == "macos_keychain"

    def test_both_valid_prefers_later_expiry_when_file_is_fresher(self, tmp_path, monkeypatch):
        """When both are valid, the one with the later ``expiresAt`` wins so
        that any subsequent refresh uses the freshest ``refresh_token``.
        """
        self._setup(tmp_path, monkeypatch, file_expires_at=self._FRESH, file_token="newer-file-token")
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = self._keychain_payload(
                access_token="older-keychain-token", expires_at=self._FRESH - 1_000_000,
            )
            creds = read_claude_code_credentials()

        assert creds is not None
        assert creds["accessToken"] == "newer-file-token"

    def test_both_expired_prefers_later_expiry(self, tmp_path, monkeypatch):
        """When both are expired, return the one with the later ``expiresAt``;
        its ``refresh_token`` is the most recently issued and most likely to
        succeed at the OAuth refresh endpoint.
        """
        self._setup(tmp_path, monkeypatch, file_expires_at=self._EXPIRED + 5, file_token="newer-expired-file")
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
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

    def test_adopts_already_refreshed_token_without_posting(self, monkeypatch):
        """When a live source already holds a valid token, return it and skip
        the network refresh entirely.
        """
        fresh = {
            "accessToken": "already-refreshed-token",
            "refreshToken": "live-refresh",
            "expiresAt": self._FRESH,
        }
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: fresh,
        )

        def _should_not_be_called(*args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("refresh_anthropic_oauth_pure must not be called")

        monkeypatch.setattr(
            "agent.anthropic_adapter.refresh_anthropic_oauth_pure",
            _should_not_be_called,
        )

        # Stale creds passed in by the caller — should be ignored in favor
        # of the live, already-refreshed token.
        result = _refresh_oauth_token({"refreshToken": "stale", "expiresAt": 1})
        assert result == "already-refreshed-token"

    def test_falls_back_to_network_refresh_when_no_fresh_credential(self, monkeypatch):
        """When no live source has a valid token, fall back to refreshing
        ourselves using the freshest available refresh token.
        """
        # Live read returns an expired credential carrying a refresh token.
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
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
            "agent.anthropic_adapter.refresh_anthropic_oauth_pure", _fake_refresh
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._write_claude_code_credentials",
            lambda *a, **k: None,
        )

        result = _refresh_oauth_token({"refreshToken": "caller-refresh", "expiresAt": 1})
        assert result == "newly-minted"
        # Prefers the live source's refresh token over the caller's stale copy.
        assert captured["refresh_token"] == "live-refresh"

