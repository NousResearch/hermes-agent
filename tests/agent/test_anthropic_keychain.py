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
    is_claude_code_token_valid,
    read_claude_code_credentials,
    _refresh_oauth_token,
    _find_claude_code_keychain_item,
    _keychain_mirror_command,
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
    """``_merge_keychain_credential_payload`` — the pure merge a Keychain refresh write
    performs over the existing entry (#98334): rotate the token triple, keep everything
    Claude Code stores beside it."""

    def test_rotates_triple_and_keeps_every_sibling(self):
        existing = {
            "claudeAiOauth": {
                "accessToken": "old-access", "refreshToken": "old-refresh", "expiresAt": 1,
                "scopes": ["user:inference", "user:profile"], "subscriptionType": "max",
            },
            "rateLimitTier": "tier-1",
            # Claude Code keeps its MCP server OAuth tokens in the same item; a refresh that
            # dropped them would log the user out of every MCP server at once.
            "mcpOAuth": {f"srv{i}": {"accessToken": f"t{i}"} for i in range(24)},
        }
        merged = _merge_keychain_credential_payload(existing, "new-access", "new-refresh", 42)
        oauth = merged["claudeAiOauth"]
        assert (oauth["accessToken"], oauth["refreshToken"], oauth["expiresAt"]) == ("new-access", "new-refresh", 42)
        # Everything except the rotated triple is byte-identical to the input.
        assert {k: v for k, v in oauth.items() if k not in ("accessToken", "refreshToken", "expiresAt")} == {
            "scopes": ["user:inference", "user:profile"], "subscriptionType": "max"}
        assert {k: v for k, v in merged.items() if k != "claudeAiOauth"} == {
            k: v for k, v in existing.items() if k != "claudeAiOauth"}
        assert existing["claudeAiOauth"]["refreshToken"] == "old-refresh"  # input not mutated


class TestKeychainMirrorCommand:
    """``_keychain_mirror_command`` — host-agnostic: it builds the ``security`` invocation
    without running it."""

    def test_secret_travels_hex_encoded_on_stdin_under_the_items_own_account(self):
        payload = {"claudeAiOauth": {"accessToken": "sk-ant-oat01-new", "refreshToken": 'r"q\\x'}, "mcpOAuth": {"a": 1}}
        argv, line = _keychain_mirror_command("alice smith", payload)

        # ``security -i`` reads the command from stdin: nothing secret on argv.
        assert argv == ["security", "-i"]
        assert "sk-ant-oat01-new" not in line and "-w" not in line.split()
        # -U updates the item Claude Code reads (matched on account + service), never a second one.
        assert line.startswith('add-generic-password -U -a "alice smith" -s "Claude Code-credentials" -X ')
        hex_blob = line.split(" -X ", 1)[1].strip()
        assert json.loads(bytes.fromhex(hex_blob)) == payload


class TestFindClaudeCodeKeychainItem:
    """``_find_claude_code_keychain_item`` parses one ``find-generic-password -g`` call. ``security``
    prints plain-ASCII attributes quoted but UNescaped, anything else as ``0x<HEX>  "<echo>"``."""

    @pytest.mark.macos_only
    @pytest.mark.parametrize(
        "acct_line, expected_account",
        [
            ('    "acct"<blob>="bob"', "bob"),
            ('    "acct"<blob>="my "user" name"', 'my "user" name'),  # embedded quote, printed raw
            ('    "acct"<blob>=0x616C2269636520C3BC  "al\\"ice \\303\\274"', 'al"ice \u00fc'),  # non-ASCII → hex form
        ],
    )
    def test_reads_account_and_payload_in_both_output_encodings(self, monkeypatch, acct_line, expected_account):
        payload = {"claudeAiOauth": {"refreshToken": "r"}, "mcpOAuth": {"a": 1}}
        fake = MagicMock(returncode=0, stdout=f'keychain: "/Users/x/Library/Keychains/login.keychain-db"\n{acct_line}\n',
                         stderr=f"password: 0x{json.dumps(payload).encode().hex()}  \"...\"\n")
        monkeypatch.setattr(subprocess, "run", lambda argv, **k: fake)

        assert _find_claude_code_keychain_item() == (expected_account, payload)


class TestMirrorClaudeCodeCredentialsToKeychain:
    """The #98334 write mirror shells out to ``security``; ``subprocess.run`` is mocked so no
    real Keychain is touched."""

    @pytest.mark.macos_only
    def test_no_write_when_no_entry_exists(self, monkeypatch):
        """Never create a Keychain item the user has not."""
        monkeypatch.setattr("agent.anthropic_credentials._find_claude_code_keychain_item", lambda: None)
        run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.setattr(subprocess, "run", run)

        _mirror_claude_code_credentials_to_keychain("a", "b", 1, spent_refresh_token="old")

        run.assert_not_called()

    @pytest.mark.macos_only
    def test_only_the_item_holding_the_spent_pair_is_updated(self, monkeypatch):
        """A different pair in the Keychain is another login or a rotation Claude Code already made;
        overwriting it would be the bug in the other direction."""
        item = ("bob", {"claudeAiOauth": {"accessToken": "A0", "refreshToken": "R0"}})
        monkeypatch.setattr("agent.anthropic_credentials._find_claude_code_keychain_item", lambda: item)
        run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.setattr(subprocess, "run", run)

        _mirror_claude_code_credentials_to_keychain("A1", "R1", 1, spent_refresh_token="not-R0")
        run.assert_not_called()

        _mirror_claude_code_credentials_to_keychain("A1", "R1", 1, spent_refresh_token="R0")
        (argv,), kwargs = run.call_args
        assert argv == ["security", "-i"]
        assert json.loads(bytes.fromhex(kwargs["input"].split(" -X ", 1)[1].strip()))["claudeAiOauth"] == {
            "accessToken": "A1", "refreshToken": "R1", "expiresAt": 1}


class TestClaudeCodeCorruptExpiresAt:
    """A non-numeric ``expiresAt`` in a Claude Code credentials payload must not
    crash the merge: ``is_claude_code_token_valid`` did ``expiresAt - 60_000``
    unguarded, and the both-invalid freshness compare did ``>=`` on the raw
    values. Unreadable expiry fails closed (treated as expired)."""

    @pytest.mark.parametrize("expires_at", ["soon", {"x": 1}, [1]])
    def test_non_numeric_expires_at_is_expired_not_crash(self, expires_at):
        creds = {"accessToken": "a", "refreshToken": "r", "expiresAt": expires_at}
        assert is_claude_code_token_valid(creds) is False

    def test_missing_expires_at_still_uses_access_token_presence(self):
        assert is_claude_code_token_valid({"accessToken": "a"}) is True
        assert is_claude_code_token_valid({}) is False

    def test_numeric_expires_at_unchanged(self):
        fresh = {"accessToken": "a", "expiresAt": int(time.time() * 1000) + 600_000}
        stale = {"accessToken": "a", "expiresAt": 1}
        assert is_claude_code_token_valid(fresh) is True
        assert is_claude_code_token_valid(stale) is False

    def _patch_sources(self, monkeypatch, kc_creds, file_creds):
        monkeypatch.setattr(
            "agent.anthropic_credentials._read_claude_code_credentials_from_keychain",
            lambda: kc_creds,
        )
        monkeypatch.setattr(
            "agent.anthropic_credentials._read_claude_code_credentials_from_file",
            lambda: file_creds,
        )

    def test_merge_corrupt_keychain_fresh_file_returns_file(self, monkeypatch):
        """Corrupt expiresAt sorts invalid; the valid file credential wins."""
        self._patch_sources(
            monkeypatch,
            {"accessToken": "kc", "expiresAt": "soon"},
            {"accessToken": "file", "expiresAt": int(time.time() * 1000) + 600_000},
        )
        creds = read_claude_code_credentials()
        assert creds is not None
        assert creds["accessToken"] == "file"

    def test_merge_both_corrupt_does_not_crash(self, monkeypatch):
        """Both-invalid falls to the freshness compare, which must not raise on
        non-numeric expiresAt values."""
        self._patch_sources(
            monkeypatch,
            {"accessToken": "kc", "expiresAt": "soon"},
            {"accessToken": "file", "expiresAt": {"x": 1}},
        )
        creds = read_claude_code_credentials()
        assert creds is not None
        assert creds["accessToken"] == "kc"  # both sort to 0 -> first wins


class TestTokenEndpointResponseShapes:
    """``_post_oauth_token``/``_oauth_token_state`` read a network response:
    a non-object body or non-numeric ``expires_in`` must not crash the
    refresh path."""

    def test_non_object_token_response_raises_not_crash(self):
        from agent.anthropic_credentials import _post_oauth_token

        payload = json.dumps([1, 2]).encode()
        with patch("urllib.request.urlopen") as mock_open:
            resp = MagicMock()
            resp.read.return_value = payload
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = resp
            with pytest.raises(Exception):
                _post_oauth_token(b"data", content_type="application/json",
                                  timeout=5, what="refresh")

    @pytest.mark.parametrize("expires_in", ["x", {"a": 1}, None])
    def test_oauth_token_state_bad_expires_in_uses_default(self, expires_in):
        from agent.anthropic_credentials import _oauth_token_state

        before = int(time.time() * 1000)
        state = _oauth_token_state({"access_token": "a", "expires_in": expires_in})
        after = int(time.time() * 1000)
        assert before + 3_600_000 <= state["expires_at_ms"] <= after + 3_600_000

    def test_refresh_adoption_gate_survives_corrupt_expires_at(
        self, tmp_path, monkeypatch
    ):
        """``_refresh_oauth_token``'s "adopt a fresher token" compare must not
        raise on a non-numeric ``expiresAt``."""
        monkeypatch.setattr(
            "agent.anthropic_credentials.read_claude_code_credentials",
            lambda: {"accessToken": "other", "refreshToken": "r2",
                     "expiresAt": "soon"},
        )
        monkeypatch.setattr(
            "agent.anthropic_credentials.claude_code_credentials_path",
            lambda: tmp_path / "creds.json",
        )
        refresh = MagicMock(side_effect=ValueError("offline"))
        monkeypatch.setattr(
            "agent.anthropic_credentials.refresh_anthropic_oauth_pure", refresh)
        from agent.anthropic_credentials import _refresh_oauth_token

        # Corrupt expiresAt must not kill the compare before the refresh attempt:
        # the refresh path itself is stubbed, so reaching it is the assertion.
        result = _refresh_oauth_token({"accessToken": "old", "refreshToken": ""})
        assert result is None
        assert refresh.called, (
            "corrupt expiresAt crashed the adoption gate before the refresh attempt"
        )
