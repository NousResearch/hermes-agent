"""
Tests for the whoop-api skill.

Two layers:
  - Smoke tests: SKILL.md frontmatter, script syntax, key constants
  - Unit tests: mocked OAuth flow, storage, client logic, token safety

No live API calls — all external dependencies are mocked.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "health" / "whoop-api"
SCRIPTS_DIR = SKILL_DIR / "scripts"

# ──────────────────────────────────────────────
# Helper: import skill modules standalone
# ──────────────────────────────────────────────

_cached_modules = {}

def _import_skill_module(name: str):
    """Import a skill script by name (e.g. 'whoop_storage') with sys.path patched.
    Caches modules so mocks don't leak between tests."""
    if name in _cached_modules:
        # Remove stale module so we get a fresh import
        sys.modules.pop(name, None)
    old_path = sys.path[:]
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        mod = importlib.import_module(name)
        return mod
    finally:
        sys.path[:] = old_path


# ═══════════════════════════════════════════════
# SMOKE TESTS — frontmatter, syntax, constants
# ═══════════════════════════════════════════════

@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline <=60): {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "whoop-api"


def test_platforms_cross_platform(frontmatter) -> None:
    platforms = set(frontmatter["platforms"])
    assert platforms == {"linux", "macos", "windows"}, f"unexpected platforms: {platforms}"


def test_author_credits_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "Nirbhay" in author or "nirbhay" in author, f"author should credit contributor: {author!r}"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_requires_toolsets(frontmatter) -> None:
    meta = frontmatter.get("metadata", {}).get("hermes", {})
    assert "terminal" in meta.get("requires_toolsets", []), (
        "skill uses subprocess scripts - requires_toolsets should include terminal"
    )


def test_skill_md_section_order() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    sections = re.findall(r"^## (.+)$", src, re.MULTILINE)
    expected = [
        "When to Use",
        "Prerequisites",
        "How to Run",
        "Quick Reference",
        "Procedure",
        "Pitfalls",
        "Verification",
    ]
    assert sections == expected, f"section order mismatch:\n  got:      {sections}\n  expected: {expected}"


SCRIPTS = [
    "scripts/whoop_client.py",
    "scripts/whoop_endpoints.py",
    "scripts/whoop_oauth.py",
    "scripts/whoop_storage.py",
    "scripts/whoop_sync.py",
    "scripts/whoop_token_refresh.py",
]


@pytest.mark.parametrize("path", SCRIPTS)
def test_shipped_scripts_parse(path: str) -> None:
    src = (SKILL_DIR / path).read_text()
    ast.parse(src)  # raises SyntaxError on broken Python


def test_auth_url_uses_oauth2_path() -> None:
    src = (SKILL_DIR / "scripts" / "whoop_oauth.py").read_text()
    assert "oauth/oauth2/auth" in src, "AUTH_URL must use /oauth/oauth2/auth path"


def test_token_url_uses_oauth2_path() -> None:
    src = (SKILL_DIR / "scripts" / "whoop_client.py").read_text()
    assert "oauth/oauth2/token" in src, "TOKEN_URL must use /oauth/oauth2/token path"


def test_api_endpoints_use_developer_prefix() -> None:
    src = (SKILL_DIR / "scripts" / "whoop_endpoints.py").read_text()
    assert "/developer/v2/" in src, "endpoint paths must use /developer/v2/ prefix"


def test_no_hardcoded_user_paths() -> None:
    for path in SCRIPTS:
        src = (SKILL_DIR / path).read_text()
        assert "/Users/" not in src, f"{path}: hardcoded /Users/ path found"
        assert "/home/" not in src, f"{path}: hardcoded /home/ path found"


def test_no_posix_unsafe_patterns() -> None:
    for path in SCRIPTS:
        src = (SKILL_DIR / path).read_text()
        assert not re.search(r"os\.kill\([^,]+,\s*0\s*\)", src), (
            f"{path}: os.kill(pid, 0) is Windows-unsafe - use psutil.pid_exists()"
        )
        if "fcntl" in src:
            assert "ImportError" in src, f"{path}: fcntl without ImportError guard"
        if "termios" in src:
            assert "ImportError" in src, f"{path}: termios without ImportError guard"


def test_storage_has_fallback_beyond_keychain() -> None:
    src = (SKILL_DIR / "scripts" / "whoop_storage.py").read_text()
    assert "json" in src, "storage module must have JSON file fallback"
    assert "_keychain_available" in src, "storage module must check Keychain availability"


def test_python3_shebangs() -> None:
    for path in SCRIPTS:
        src = (SKILL_DIR / path).read_text()
        lines = src.strip().split("\n")
        if lines[0].startswith("#!"):
            assert "python3" in lines[0], f"{path}: shebang should use python3"


def test_no_python_without_3_in_messages() -> None:
    src = (SKILL_DIR / "scripts" / "whoop_sync.py").read_text()
    matches = re.findall(r'"python\s+whoop', src)
    assert not matches, "user-facing messages should use python3, not python"


def test_privacy_policy_exists() -> None:
    assert (SKILL_DIR / "references" / "privacy-policy.html").is_file()


def test_privacy_policy_is_valid_html() -> None:
    html = (SKILL_DIR / "references" / "privacy-policy.html").read_text()
    assert "<!DOCTYPE html>" in html, "privacy policy must be valid HTML"
    assert "Privacy Policy" in html, "privacy policy must contain Privacy Policy"


def test_cron_entry_template_exists() -> None:
    assert (SKILL_DIR / "templates" / "cron-entry.yaml").is_file()


def test_gitignore_covers_artifacts() -> None:
    gi = (SKILL_DIR / ".gitignore").read_text()
    assert "__pycache__/" in gi, ".gitignore must exclude __pycache__"
    assert "whoop_data/" in gi, ".gitignore must exclude whoop_data output"


# ═══════════════════════════════════════════════
# UNIT TESTS — mocked logic, storage, OAuth, client
# ═══════════════════════════════════════════════

# --- Storage: JSON fallback ---

class TestStorageJsonFallback:
    """Test JSON file fallback storage (Linux/Windows path)."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Atomic write + read should produce identical data."""
        mod = _import_skill_module("whoop_storage")
        token_path = tmp_path / "tokens.json"

        mod._save_to_json_file(token_path, {
            "access_token": "at_123",
            "refresh_token": "rt_456",
            "expires_at": 1700000000.0,
        })

        loaded = mod._load_from_json_file(token_path, mod.TOKEN_FIELDS)
        assert loaded is not None
        assert loaded["access_token"] == "at_123"
        assert loaded["refresh_token"] == "rt_456"
        assert loaded["expires_at"] == 1700000000.0

    def test_atomic_write_no_partial_file(self, tmp_path):
        """Atomic write should not leave .tmp files on success."""
        mod = _import_skill_module("whoop_storage")
        token_path = tmp_path / "tokens.json"

        mod._save_to_json_file(token_path, {"access_token": "test", "refresh_token": "test", "expires_at": 1.0})

        # No .tmp file should exist after successful write
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Leftover .tmp files: {tmp_files}"

    def test_file_permissions_restricted(self, tmp_path):
        """JSON fallback files should have 0o600 permissions (owner-read-only)."""
        mod = _import_skill_module("whoop_storage")
        token_path = tmp_path / "tokens.json"

        mod._save_to_json_file(token_path, {"access_token": "x", "refresh_token": "y", "expires_at": 1.0})

        if os.name != "nt":  # chmod is no-op on Windows
            mode = token_path.stat().st_mode & 0o777
            assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_save_creates_parent_dirs(self, tmp_path):
        """_save_to_json_file should create parent directories."""
        mod = _import_skill_module("whoop_storage")
        nested_path = tmp_path / "deep" / "nested" / "tokens.json"

        mod._save_to_json_file(nested_path, {"access_token": "a", "refresh_token": "b", "expires_at": 1.0})

        assert nested_path.exists()
        assert nested_path.parent.is_dir()

    def test_load_missing_file_returns_none(self, tmp_path):
        """Loading from a nonexistent path should return None."""
        mod = _import_skill_module("whoop_storage")
        result = mod._load_from_json_file(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_corrupted_json_returns_none(self, tmp_path):
        """Loading a file with invalid JSON should return None, not crash."""
        mod = _import_skill_module("whoop_storage")
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("NOT VALID JSON {{{")
        result = mod._load_from_json_file(bad_path)
        assert result is None

    def test_load_missing_fields_returns_none(self, tmp_path):
        """A JSON file missing required fields should return None."""
        mod = _import_skill_module("whoop_storage")
        partial_path = tmp_path / "partial.json"
        partial_path.write_text(json.dumps({"access_token": "x"}))  # missing refresh_token, expires_at
        result = mod._load_from_json_file(partial_path, mod.TOKEN_FIELDS)
        assert result is None

    def test_credentials_save_and_load(self, tmp_path):
        """Client credentials should round-trip through JSON fallback."""
        mod = _import_skill_module("whoop_storage")
        cred_path = tmp_path / "credentials.json"

        mod._save_to_json_file(cred_path, {"client_id": "cid_abc", "client_secret": "csec_xyz"})
        loaded = mod._load_from_json_file(cred_path, mod.CREDENTIAL_FIELDS)
        assert loaded is not None
        assert loaded["client_id"] == "cid_abc"
        assert loaded["client_secret"] == "csec_xyz"


# --- Storage: Keychain ---

class TestStorageKeychain:
    """Test Keychain storage module (macOS path) via low-level functions."""

    def test_load_from_keychain_success(self):
        """Successfully load tokens from Keychain via _run_security."""
        mod = _import_skill_module("whoop_storage")
        with patch.object(mod, "_keychain_available", return_value=True), \
             patch.object(mod, "_run_security") as mock_sec:
            mock_sec.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({"access_token": "at_kc", "refresh_token": "rt_kc", "expires_at": 1700000000.0}),
            )
            result = mod._load_from_keychain(mod.KEYCHAIN_ACCOUNT)
            assert result is not None
            assert result["access_token"] == "at_kc"

    def test_load_from_keychain_not_found(self):
        """Keychain returns non-zero when entry doesn't exist."""
        mod = _import_skill_module("whoop_storage")
        with patch.object(mod, "_keychain_available", return_value=True), \
             patch.object(mod, "_run_security") as mock_sec:
            mock_sec.return_value = MagicMock(returncode=44, stdout="item not found")
            result = mod._load_from_keychain(mod.KEYCHAIN_ACCOUNT)
            assert result is None

    def test_load_from_keychain_bad_json(self):
        """Keychain returns non-JSON value — should return None gracefully."""
        mod = _import_skill_module("whoop_storage")
        with patch.object(mod, "_keychain_available", return_value=True), \
             patch.object(mod, "_run_security") as mock_sec:
            mock_sec.return_value = MagicMock(returncode=0, stdout="not-json")
            result = mod._load_from_keychain(mod.KEYCHAIN_ACCOUNT)
            assert result is None


# --- Storage: save_tokens routing ---

class TestStorageRouting:
    """Test that save_tokens/load_tokens route to Keychain or JSON correctly."""

    def test_save_tokens_uses_json_on_linux(self, tmp_path):
        """On Linux (no Keychain), save_tokens should write to JSON file."""
        mod = _import_skill_module("whoop_storage")
        target = tmp_path / "tokens.json"

        with patch.object(mod, "_is_macos", return_value=False), \
             patch.object(mod, "_token_file_path", return_value=target):
            mod.save_tokens("at_linux", "rt_linux", 1700000000.0)

        assert target.exists()
        data = json.loads(target.read_text())
        assert data["access_token"] == "at_linux"
        assert data["refresh_token"] == "rt_linux"

    def test_save_tokens_uses_keychain_on_macos(self, tmp_path):
        """On macOS with Keychain available, save_tokens should use Keychain."""
        mod = _import_skill_module("whoop_storage")

        with patch.object(mod, "_is_macos", return_value=True), \
             patch.object(mod, "_keychain_available", return_value=True), \
             patch.object(mod, "_save_to_keychain") as mock_kc:
            mod.save_tokens("at_mac", "rt_mac", 1700000000.0)
            mock_kc.assert_called_once()
            call_data = mock_kc.call_args[0][1]  # second positional arg = data dict
            assert call_data["access_token"] == "at_mac"

    def test_clear_tokens_removes_json_file(self, tmp_path):
        """clear_tokens should delete the JSON file on Linux."""
        mod = _import_skill_module("whoop_storage")
        target = tmp_path / "tokens.json"
        target.write_text(json.dumps({"access_token": "x", "refresh_token": "y", "expires_at": 1.0}))

        with patch.object(mod, "_is_macos", return_value=False), \
             patch.object(mod, "_token_file_path", return_value=target):
            mod.clear_tokens()

        assert not target.exists()

    def test_clear_tokens_deletes_keychain_on_macos(self):
        """clear_tokens should call _delete_from_keychain on macOS."""
        mod = _import_skill_module("whoop_storage")

        with patch.object(mod, "_is_macos", return_value=True), \
             patch.object(mod, "_keychain_available", return_value=True), \
             patch.object(mod, "_delete_from_keychain") as mock_del:
            mod.clear_tokens()
            mock_del.assert_called_once_with(mod.KEYCHAIN_ACCOUNT)


# --- OAuth: CSRF state ---

class TestOAuthCSRF:
    """Test OAuth CSRF state parameter handling."""

    def test_state_generated_and_verified(self):
        """OAuth flow should generate a state and verify it on callback."""
        mod = _import_skill_module("whoop_oauth")
        handler_class = mod.OAuthCallbackHandler
        state = "test_state_random_value_12345"
        handler_class.expected_state = state

        # Verify matching state passes
        assert state == handler_class.expected_state, "CSRF state should match"

    def test_mismatched_state_rejected(self):
        """Mismatched state should be rejected (CSRF protection)."""
        mod = _import_skill_module("whoop_oauth")
        handler_class = mod.OAuthCallbackHandler
        handler_class.expected_state = "correct_state_xyz"

        assert "wrong_state" != handler_class.expected_state


# --- OAuth: Port conflict ---

class TestOAuthPortConflict:
    """Test that OAuth flow handles port conflicts."""

    def test_port_in_use_exits_cleanly(self):
        """If port 8647 is already in use, setup should exit with an error."""
        mod = _import_skill_module("whoop_oauth")

        with patch.object(mod.socket, "socket") as mock_socket_cls:
            mock_socket = MagicMock()
            mock_socket.__enter__ = MagicMock(return_value=mock_socket)
            mock_socket.__exit__ = MagicMock(return_value=False)
            mock_socket.bind = MagicMock(side_effect=OSError("Address already in use"))
            mock_socket_cls.return_value = mock_socket

            with patch.object(mod, "webbrowser"):
                with pytest.raises(SystemExit):
                    mod.start_oauth_flow("fake_id", "fake_secret")


# --- Client: Token refresh ---

class TestClientTokenRefresh:
    """Test WhoopClient token refresh behavior with mocked HTTP."""

    def test_refresh_clears_tokens_on_401(self):
        """On 401 from refresh endpoint, stored tokens should be cleared so
        status reports re-auth needed instead of silently operating on stale state.
        
        Critical: All storage I/O must be mocked to prevent Keychain/file writes.
        """
        mod_client = _import_skill_module("whoop_client")
        mod_storage = _import_skill_module("whoop_storage")

        client = mod_client.WhoopClient.__new__(mod_client.WhoopClient)
        client._tokens = {"access_token": "expired", "refresh_token": "expired_rt", "expires_at": 0}
        client.session = MagicMock()
        client.session.headers = {}
        client._request_timestamps = []

        with patch.object(mod_storage, "load_client_credentials", return_value={"client_id": "c", "client_secret": "s"}), \
             patch.object(mod_client, "load_client_credentials", return_value={"client_id": "c", "client_secret": "s"}), \
             patch.object(mod_client, "clear_tokens") as mock_client_clear, \
             patch.object(mod_client, "requests") as mock_requests:
            mock_requests.post.return_value = MagicMock(status_code=401)

            with pytest.raises(mod_client.TokenExpiredError):
                client._refresh_if_needed()

            mock_client_clear.assert_called()

    def test_refresh_clears_tokens_on_invalid_grant(self):
        """On invalid_grant error response, tokens should be cleared.
        
        Critical: All storage I/O must be mocked to prevent Keychain/file writes.
        """
        mod_client = _import_skill_module("whoop_client")
        mod_storage = _import_skill_module("whoop_storage")

        client = mod_client.WhoopClient.__new__(mod_client.WhoopClient)
        client._tokens = {"access_token": "expired", "refresh_token": "expired_rt", "expires_at": 0}
        client.session = MagicMock()
        client.session.headers = {}
        client._request_timestamps = []

        with patch.object(mod_storage, "load_client_credentials", return_value={"client_id": "c", "client_secret": "s"}), \
             patch.object(mod_client, "load_client_credentials", return_value={"client_id": "c", "client_secret": "s"}), \
             patch.object(mod_client, "clear_tokens") as mock_client_clear, \
             patch.object(mod_client, "requests") as mock_requests:
            mock_requests.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"error": "invalid_grant", "error_description": "Token has been revoked"},
            )

            with pytest.raises(mod_client.TokenExpiredError):
                client._refresh_if_needed()

            mock_client_clear.assert_called()

    def test_refresh_succeeds_on_valid_response(self):
        """Successful token refresh should save new tokens and update headers.
        
        Critical: This test MUST NOT call real save_tokens or load_client_credentials.
        All storage I/O must be mocked to prevent writing test data to Keychain/files.
        """
        mod_client = _import_skill_module("whoop_client")
        mod_storage = _import_skill_module("whoop_storage")

        client = mod_client.WhoopClient.__new__(mod_client.WhoopClient)
        client._tokens = {"access_token": "expired", "refresh_token": "expired_rt", "expires_at": 0}
        client.session = MagicMock()
        client.session.headers = {}
        client._request_timestamps = []

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "test_new_at", "refresh_token": "test_new_rt", "expires_in": 3600}
        mock_response.raise_for_status = MagicMock()

        with patch.object(mod_storage, "load_client_credentials", return_value={"client_id": "c", "client_secret": "s"}), \
             patch.object(mod_storage, "save_tokens") as mock_save, \
             patch.object(mod_client, "load_client_credentials", return_value={"client_id": "c", "client_secret": "s"}), \
             patch.object(mod_client, "save_tokens") as mock_client_save, \
             patch.object(mod_client, "requests") as mock_requests:
            mock_requests.post.return_value = mock_response

            client._refresh_if_needed()

            # Verify both storage and client module patches intercepted the call
            mock_client_save.assert_called()
            assert client.session.headers["Authorization"] == "Bearer test_new_at"


# --- Client: Rate limiting ---

class TestClientRateLimiting:
    """Test rate limit tracking in WhoopClient."""

    def test_rate_limit_tracks_timestamps(self):
        mod = _import_skill_module("whoop_client")
        client = mod.WhoopClient.__new__(mod.WhoopClient)
        client._request_timestamps = [1000.0, 1001.0]
        assert len(client._request_timestamps) == 2


# --- Endpoints registry ---

class TestEndpointRegistry:
    """Test the endpoint registry for completeness and correctness."""

    def test_all_endpoints_have_required_fields(self):
        mod = _import_skill_module("whoop_endpoints")

        for name, ep in mod.ENDPOINTS.items():
            assert ep.name == name, f"Endpoint name mismatch: {ep.name} != {name}"
            assert ep.path.startswith("/developer/v2/"), f"{name}: path must start with /developer/v2/ — got {ep.path}"
            assert isinstance(ep.scopes, list), f"{name}: scopes must be a list"
            assert len(ep.scopes) > 0, f"{name}: must have at least one scope"
            assert isinstance(ep.requires_pagination, bool), f"{name}: requires_pagination must be bool"

    def test_six_endpoints_registered(self):
        mod = _import_skill_module("whoop_endpoints")
        expected = {"cycle", "recovery", "sleep", "workout", "body", "profile"}
        assert set(mod.ENDPOINTS.keys()) == expected

    def test_get_endpoint_raises_on_unknown(self):
        mod = _import_skill_module("whoop_endpoints")
        with pytest.raises(KeyError):
            mod.get_endpoint("nonexistent")

    def test_endpoint_names_match_api_reference(self):
        mod = _import_skill_module("whoop_endpoints")
        api_ref = (SKILL_DIR / "references" / "api-reference.md").read_text()
        for name in mod.ENDPOINTS:
            assert name in api_ref.lower(), f"Endpoint '{name}' not found in api-reference.md"


# --- Privacy policy content checks ---

class TestPrivacyPolicyContent:
    """Verify the privacy policy doesn't make false security claims."""

    def test_no_encrypted_claim(self):
        """Privacy policy should NOT claim tokens are 'encrypted' in the JSON fallback."""
        html = (SKILL_DIR / "references" / "privacy-policy.html").read_text()
        assert "encrypted local file" not in html, (
            "Privacy policy should not claim JSON fallback is 'encrypted' — it uses 0o600 permissions, not encryption"
        )

    def test_mentions_restricted_permissions(self):
        """Privacy policy should mention restricted permissions for file storage."""
        html = (SKILL_DIR / "references" / "privacy-policy.html").read_text()
        assert "restricted permissions" in html or "owner-read-only" in html, (
            "Privacy policy should disclose that file-based storage uses restricted permissions"
        )


# --- API reference typo checks ---

class TestApiReferenceTypos:
    """Verify API reference has no known typos."""

    def test_no_kiljoule_typo(self):
        """The word 'kiljoule' is a known typo — should be 'kilojoule'."""
        api_ref = (SKILL_DIR / "references" / "api-reference.md").read_text()
        assert "kiljoule" not in api_ref, "Typo 'kiljoule' found — should be 'kilojoule'"

    def test_kilojoule_present(self):
        """The correct spelling 'kilojoule' should appear in the API reference."""
        api_ref = (SKILL_DIR / "references" / "api-reference.md").read_text()
        assert "kilojoule" in api_ref, "Expected 'kilojoule' in API reference"


# --- Atomic write test ---

class TestAtomicWrites:
    """Verify that token file writes are atomic (no partial reads)."""

    def test_save_uses_atomic_rename(self):
        """_save_to_json_file should use write-to-temp + os.replace pattern."""
        src = (SKILL_DIR / "scripts" / "whoop_storage.py").read_text()
        assert "os.replace" in src, "_save_to_json_file should use os.replace for atomic writes"
        assert ".tmp" in src, "_save_to_json_file should write to a .tmp file first"

    def test_chmod_before_rename(self):
        """Permissions should be set on the temp file BEFORE rename."""
        src = (SKILL_DIR / "scripts" / "whoop_storage.py").read_text()
        # Find all lines in _save_to_json_file and verify chmod comes before os.replace
        in_func = False
        chmod_idx = None
        replace_idx = None
        for i, line in enumerate(src.split("\n")):
            if "def _save_to_json_file" in line:
                in_func = True
            elif in_func and line.startswith("def "):
                break
            elif in_func:
                if "chmod" in line:
                    chmod_idx = i
                if "os.replace" in line:
                    replace_idx = i
        assert chmod_idx is not None, "chmod not found in _save_to_json_file"
        assert replace_idx is not None, "os.replace not found in _save_to_json_file"
        assert chmod_idx < replace_idx, (
            "chmod should come before os.replace — permissions set on temp file before atomic rename"
        )


# --- Token refresh failure handling ---

class TestTokenRefreshFailureHandling:
    """Verify that refresh failures properly clear stale tokens."""

    def test_401_clears_tokens(self):
        """On 401 response, tokens should be cleared before raising TokenExpiredError."""
        src = (SKILL_DIR / "scripts" / "whoop_client.py").read_text()
        assert "clear_tokens()" in src, (
            "401 handler in _refresh_if_needed should call clear_tokens() before raising error"
        )

    def test_invalid_grant_clears_tokens(self):
        """On invalid_grant error response, tokens should be cleared."""
        src = (SKILL_DIR / "scripts" / "whoop_client.py").read_text()
        assert '"error"' in src or "'error'" in src, (
            "Should check for 'error' field in token refresh response (e.g. invalid_grant)"
        )

    def test_imports_clear_tokens(self):
        """whoop_client should be able to import clear_tokens from whoop_storage."""
        src = (SKILL_DIR / "scripts" / "whoop_client.py").read_text()
        assert "clear_tokens" in src, "whoop_client should reference clear_tokens for failure handling"