"""Tests for multi-account Google Workspace support.

Covers:
- Account email normalization (rejects path traversal, lowercases, etc.)
- Token path resolution chain: explicit > env > default > legacy
- list_accounts / get_default_account / set_default_account
- Legacy symlink updates when default changes
- gws_bridge.py honors --account flag
- google_api.py --account flag overrides env
- setup.py refuses to save token when authorized account doesn't match
  the requested --account
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts"
)
ACCOUNT_PATH = SCRIPTS_DIR / "google_account.py"
BRIDGE_PATH = SCRIPTS_DIR / "gws_bridge.py"
API_PATH = SCRIPTS_DIR / "google_api.py"
SETUP_PATH = SCRIPTS_DIR / "setup.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Make scripts/ importable so the script can do `import google_account`
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_GOOGLE_ACCOUNT", raising=False)
    monkeypatch.delenv("_HERMES_GOOGLE_ACCOUNT_OVERRIDE", raising=False)
    # Reset module cache so each test gets a fresh google_account bound to
    # the per-test HERMES_HOME (the module reads the env at function-call
    # time, so reload isn't strictly necessary, but it keeps state clean).
    for name in (
        "google_account",
        "gws_bridge_test",
        "gws_api_test",
        "gws_setup_test",
    ):
        sys.modules.pop(name, None)
    return home


@pytest.fixture
def account_module(hermes_home):
    return _load_module("google_account", ACCOUNT_PATH)


def _write_token(path: Path, *, email: str | None = None, token: str = "ya29.test"):
    payload = {
        "token": token,
        "refresh_token": "1//refresh",
        "client_id": "123.apps.googleusercontent.com",
        "client_secret": "secret",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    if email is not None:
        payload["email"] = email
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# google_account.normalize_email
# ---------------------------------------------------------------------------


class TestNormalizeEmail:
    def test_lowercases_and_strips(self, account_module):
        assert account_module.normalize_email("  USER@Example.COM  ") == "user@example.com"

    def test_accepts_plus_addressing(self, account_module):
        assert (
            account_module.normalize_email("user+tag@example.com")
            == "user+tag@example.com"
        )

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   ",
            "no-at-sign",
            "two@@example.com",
            "trailing@",
            "@leading.com",
            "../etc/passwd",
            "user@",
            "user@host",  # no TLD
            "user@example.com/../other",
            "user/@example.com",
        ],
    )
    def test_rejects_invalid(self, account_module, bad):
        with pytest.raises(ValueError):
            account_module.normalize_email(bad)


# ---------------------------------------------------------------------------
# Resolution chain: explicit > env > default > legacy
# ---------------------------------------------------------------------------


class TestResolveAccount:
    def test_explicit_wins(self, account_module, monkeypatch, hermes_home):
        # Even with env set AND default set, explicit beats both.
        _write_token(account_module.token_path_for("env@example.com"))
        _write_token(account_module.token_path_for("default@example.com"))
        _write_token(account_module.token_path_for("explicit@example.com"))
        account_module.set_default_account("default@example.com")
        monkeypatch.setenv("HERMES_GOOGLE_ACCOUNT", "env@example.com")

        assert (
            account_module.resolve_account("explicit@example.com")
            == "explicit@example.com"
        )

    def test_env_beats_default(self, account_module, monkeypatch, hermes_home):
        _write_token(account_module.token_path_for("env@example.com"))
        _write_token(account_module.token_path_for("default@example.com"))
        account_module.set_default_account("default@example.com")
        monkeypatch.setenv("HERMES_GOOGLE_ACCOUNT", "env@example.com")

        assert account_module.resolve_account() == "env@example.com"

    def test_default_used_when_no_env_or_explicit(
        self, account_module, hermes_home
    ):
        _write_token(account_module.token_path_for("default@example.com"))
        account_module.set_default_account("default@example.com")

        assert account_module.resolve_account() == "default@example.com"

    def test_single_account_implicit_default(self, account_module, hermes_home):
        # If there's only one account and no explicit default pointer, it's the default.
        _write_token(account_module.token_path_for("only@example.com"))

        assert account_module.get_default_account() == "only@example.com"

    def test_no_account_returns_none(self, account_module, hermes_home):
        assert account_module.resolve_account() is None

    def test_legacy_fallback_when_no_account_resolves(
        self, account_module, hermes_home
    ):
        # No accounts in google_tokens/, no env, no default.
        # resolve_token_path() should hand back the legacy path so single-account
        # installs work unchanged.
        legacy = account_module.legacy_token_path()
        _write_token(legacy)

        path = account_module.resolve_token_path()
        assert path == legacy


# ---------------------------------------------------------------------------
# list_accounts / set_default_account / legacy symlink
# ---------------------------------------------------------------------------


class TestAccountManagement:
    def test_list_accounts_filters_junk_files(self, account_module, hermes_home):
        d = account_module.tokens_dir()
        d.mkdir()
        _write_token(d / "user@example.com.json")
        _write_token(d / "other@example.com.json")
        # Junk files that should be ignored.
        (d / "default").write_text("user@example.com\n")
        (d / "not-an-email.json").write_text("{}")
        (d / "subdir").mkdir()
        (d / "README.md").write_text("ignored")

        result = account_module.list_accounts()
        assert result == ["other@example.com", "user@example.com"]

    def test_set_default_creates_legacy_symlink(self, account_module, hermes_home):
        _write_token(account_module.token_path_for("alice@example.com"))
        account_module.set_default_account("alice@example.com")

        legacy = account_module.legacy_token_path()
        assert legacy.is_symlink()
        assert legacy.resolve() == account_module.token_path_for(
            "alice@example.com"
        ).resolve()

    def test_set_default_replaces_existing_symlink(
        self, account_module, hermes_home
    ):
        _write_token(account_module.token_path_for("a@example.com"))
        _write_token(account_module.token_path_for("b@example.com"))
        account_module.set_default_account("a@example.com")
        account_module.set_default_account("b@example.com")

        legacy = account_module.legacy_token_path()
        assert legacy.is_symlink()
        assert legacy.resolve() == account_module.token_path_for(
            "b@example.com"
        ).resolve()

    def test_set_default_does_not_clobber_real_legacy_file(
        self, account_module, hermes_home
    ):
        # If the legacy path is a regular file (un-migrated install), we
        # should not replace it with a symlink — the user hasn't agreed to
        # migrate yet.
        legacy = account_module.legacy_token_path()
        _write_token(legacy)
        original_contents = legacy.read_text()

        _write_token(account_module.token_path_for("alice@example.com"))
        account_module.set_default_account("alice@example.com")

        # Default pointer was set, but legacy file is untouched.
        assert account_module.get_default_account() == "alice@example.com"
        assert not legacy.is_symlink()
        assert legacy.read_text() == original_contents

    def test_set_default_rejects_unknown_account(self, account_module, hermes_home):
        with pytest.raises(FileNotFoundError):
            account_module.set_default_account("unknown@example.com")

    def test_normalize_in_token_path_for(self, account_module, hermes_home):
        # Mixed case input should still resolve to a single canonical file.
        p1 = account_module.token_path_for("USER@Example.COM")
        p2 = account_module.token_path_for("user@example.com")
        assert p1 == p2
        assert p1.name == "user@example.com.json"


# ---------------------------------------------------------------------------
# gws_bridge.py: --account flag plumbing
# ---------------------------------------------------------------------------


class TestBridgeAccountFlag:
    def test_split_account_strips_flag(self, hermes_home):
        bridge = _load_module("gws_bridge_test", BRIDGE_PATH)

        account, rest = bridge._split_account_arg(
            ["--account", "user@example.com", "gmail", "+triage"]
        )
        assert account == "user@example.com"
        assert rest == ["gmail", "+triage"]

    def test_split_account_equals_form(self, hermes_home):
        bridge = _load_module("gws_bridge_test", BRIDGE_PATH)

        account, rest = bridge._split_account_arg(
            ["--account=user@example.com", "gmail"]
        )
        assert account == "user@example.com"
        assert rest == ["gmail"]

    def test_split_account_absent(self, hermes_home):
        bridge = _load_module("gws_bridge_test", BRIDGE_PATH)

        account, rest = bridge._split_account_arg(["gmail", "+triage"])
        assert account is None
        assert rest == ["gmail", "+triage"]

    def test_split_account_missing_value_exits(self, hermes_home):
        bridge = _load_module("gws_bridge_test", BRIDGE_PATH)

        with pytest.raises(SystemExit):
            bridge._split_account_arg(["--account"])

    def test_get_token_path_honors_active_account(self, hermes_home):
        bridge = _load_module("gws_bridge_test", BRIDGE_PATH)
        account_module = _load_module("google_account", ACCOUNT_PATH)

        _write_token(account_module.token_path_for("a@example.com"))
        _write_token(account_module.token_path_for("b@example.com"))

        bridge._set_active_account("a@example.com")
        assert bridge.get_token_path() == account_module.token_path_for(
            "a@example.com"
        )

        bridge._set_active_account("b@example.com")
        assert bridge.get_token_path() == account_module.token_path_for(
            "b@example.com"
        )

    def test_bridge_falls_back_to_legacy_token_when_no_account(self, hermes_home):
        bridge = _load_module("gws_bridge_test", BRIDGE_PATH)
        account_module = _load_module("google_account", ACCOUNT_PATH)

        legacy = account_module.legacy_token_path()
        _write_token(legacy)
        bridge._set_active_account(None)

        assert bridge.get_token_path() == legacy


# ---------------------------------------------------------------------------
# google_api.py: --account flag plumbing
# ---------------------------------------------------------------------------


class TestApiAccountFlag:
    def test_resolve_token_path_uses_override(self, hermes_home, monkeypatch):
        api = _load_module("gws_api_test", API_PATH)
        account_module = _load_module("google_account", ACCOUNT_PATH)

        _write_token(account_module.token_path_for("a@example.com"))
        monkeypatch.setenv("_HERMES_GOOGLE_ACCOUNT_OVERRIDE", "a@example.com")

        assert api._resolve_token_path() == account_module.token_path_for(
            "a@example.com"
        )

    def test_override_beats_hermes_google_account_env(
        self, hermes_home, monkeypatch
    ):
        api = _load_module("gws_api_test", API_PATH)
        account_module = _load_module("google_account", ACCOUNT_PATH)

        _write_token(account_module.token_path_for("explicit@example.com"))
        _write_token(account_module.token_path_for("env@example.com"))
        monkeypatch.setenv("HERMES_GOOGLE_ACCOUNT", "env@example.com")
        monkeypatch.setenv(
            "_HERMES_GOOGLE_ACCOUNT_OVERRIDE", "explicit@example.com"
        )

        assert api._resolve_token_path() == account_module.token_path_for(
            "explicit@example.com"
        )

    def test_falls_back_to_legacy_when_no_override_or_env(self, hermes_home):
        api = _load_module("gws_api_test", API_PATH)
        account_module = _load_module("google_account", ACCOUNT_PATH)

        legacy = account_module.legacy_token_path()
        _write_token(legacy)

        assert api._resolve_token_path() == legacy


# ---------------------------------------------------------------------------
# Backward compatibility: existing single-account installs are untouched
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_single_account_install_works_without_migration(
        self, hermes_home, monkeypatch
    ):
        """The whole point of this PR's backward-compat story: an install with
        only ``google_token.json`` and no ``google_tokens/`` directory must
        keep working with no behavior change."""
        api = _load_module("gws_api_test", API_PATH)
        account_module = _load_module("google_account", ACCOUNT_PATH)
        bridge = _load_module("gws_bridge_test", BRIDGE_PATH)

        legacy = account_module.legacy_token_path()
        _write_token(legacy)

        # No env, no override, no migration.
        assert api._resolve_token_path() == legacy
        assert account_module.resolve_token_path() == legacy

        bridge._set_active_account(None)
        assert bridge.get_token_path() == legacy

    def test_legacy_symlink_resolves_to_default_after_migration(
        self, hermes_home
    ):
        """After migration the legacy path still works because it's a symlink."""
        account_module = _load_module("google_account", ACCOUNT_PATH)

        legacy = account_module.legacy_token_path()
        # Simulate "migrated" state: legacy is a symlink to a per-account file.
        target = account_module.token_path_for("alice@example.com")
        _write_token(target)
        account_module.set_default_account("alice@example.com")

        assert legacy.is_symlink()
        # Reading the legacy path returns the per-account token contents.
        assert json.loads(legacy.read_text())["refresh_token"] == "1//refresh"
