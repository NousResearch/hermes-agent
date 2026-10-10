"""Fingerprint stability for external-process providers (#129094).

An external-process provider's catalog comes from the launched program's
handshake, not from any token file — so a routine OAuth rotation (Claude Code
rewrites ``~/.claude/.credentials.json`` on every refresh) must not discard
the cached row and drop the picker to ``fallback_models``. copilot-acp is the
exception: the ACP session is the only source reflecting the account's
enablement, so its hosts.json login signal stays fingerprinted and an account
switch (or logout) invalidates the cached row.
"""

import json
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_cli.models as models_mod
from hermes_cli.models import (
    _credential_fingerprint,
    cached_provider_model_ids,
    update_provider_cache_entry,
)

# Runs on every host AND is imported by every OS lane (the Windows/macOS lanes
# only import files carrying a platforms() spec): the fixture pins ~ resolution
# via USERPROFILE/HOMEDRIVE/HOMEPATH, which only matters on native Windows.
pytestmark = pytest.mark.platforms("any")

PROVIDER = "fp-external-proc"
CONTROL = "fp-control-never-registered"
COPILOT_ACP = "copilot-acp"


def _fake_profile(**overrides):
    base = {
        "auth_type": "external_process",
        "process_command_env_vars": ("TEST_FP_EXT_CMD",),
        "process_args_env_var": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def isolated_creds(tmp_path, monkeypatch):
    """HOME with a Claude credentials file; HERMES_HOME isolated for the cache."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # os.path.expanduser reads USERPROFILE (then HOMEDRIVE/HOMEPATH), not HOME,
    # on Windows — isolate all three so the fingerprinted ~/... token files
    # resolve under the tmp home on every host (cf. test_image_routing.py).
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("TEST_FP_EXT_CMD", raising=False)
    creds = home / ".claude" / ".credentials.json"
    creds.parent.mkdir(parents=True)
    creds.write_text(json.dumps({"token": "token-a"}), encoding="utf-8")
    return creds


@pytest.fixture
def ext_profile(monkeypatch):
    """Route only PROVIDER to a fake external-process profile."""
    real = __import__("providers", fromlist=["get_provider_profile"]).get_provider_profile

    def _fake(name):
        if str(name).lower() == PROVIDER:
            return _fake_profile()
        return real(name)

    monkeypatch.setattr("providers.get_provider_profile", _fake)


def _rotate(creds):
    creds.write_text(json.dumps({"token": "token-b"}), encoding="utf-8")
    # mtime granularity: ext4 is ns, but be explicit for coarse filesystems.
    stamp = time.time() + 5
    os.utime(creds, (stamp, stamp))


class TestExternalProcessFingerprint:
    def test_token_rotation_keeps_fingerprint(self, isolated_creds, ext_profile):
        before = _credential_fingerprint(PROVIDER)
        _rotate(isolated_creds)
        assert _credential_fingerprint(PROVIDER) == before

    def test_command_override_still_invalidates(self, isolated_creds, ext_profile, monkeypatch):
        before = _credential_fingerprint(PROVIDER)
        monkeypatch.setenv("TEST_FP_EXT_CMD", "/other/cli")
        assert _credential_fingerprint(PROVIDER) != before

    def test_hermes_credential_files_ignored(self, isolated_creds, ext_profile):
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        home.mkdir(parents=True, exist_ok=True)
        before = _credential_fingerprint(PROVIDER)
        (home / "auth.json").write_text("{}", encoding="utf-8")
        assert _credential_fingerprint(PROVIDER) == before

    def test_other_providers_still_track_token_files(self, isolated_creds):
        before = _credential_fingerprint(CONTROL)
        _rotate(isolated_creds)
        assert _credential_fingerprint(CONTROL) != before


class TestPickerSurvivesRotation:
    def test_cached_row_served_after_rotation(self, isolated_creds, ext_profile):
        update_provider_cache_entry(PROVIDER, ["claude-sonnet-5-5", "claude-opus-4-6"])
        _rotate(isolated_creds)
        with patch.object(models_mod, "_spawn_swr_refresh") as swr:
            rows = cached_provider_model_ids(PROVIDER, non_blocking=True)
        assert "claude-sonnet-5-5" in rows
        swr.assert_not_called()


def _write_hosts(home, user, token):
    """Write a github-copilot hosts.json login for *user* with a fresh mtime."""
    hosts = home / ".config" / "github-copilot" / "hosts.json"
    hosts.parent.mkdir(parents=True, exist_ok=True)
    hosts.write_text(
        json.dumps({"github.com": {"user": user, "oauth_token": token}}), encoding="utf-8")
    # mtime granularity: ext4 is ns, but be explicit for coarse filesystems.
    stamp = time.time() + 5
    os.utime(hosts, (stamp, stamp))
    return hosts


class TestCopilotAcpLoginSignal:
    """copilot-acp keeps tracking hosts.json: the ACP session is the only source
    reflecting the account's enablement, so an account switch (or logout) must
    invalidate the cached row instead of serving the previous account's catalog."""

    def test_account_switch_invalidates(self, isolated_creds):
        home = isolated_creds.parent.parent
        _write_hosts(home, "alice", "tok-a")
        before = _credential_fingerprint(COPILOT_ACP)
        _write_hosts(home, "bob", "tok-b")
        assert _credential_fingerprint(COPILOT_ACP) != before

    def test_stale_row_not_served_after_switch(self, isolated_creds):
        home = isolated_creds.parent.parent
        _write_hosts(home, "alice", "tok-a")
        update_provider_cache_entry(COPILOT_ACP, ["model-alice-1"])
        _write_hosts(home, "bob", "tok-b")
        with patch.object(models_mod, "_spawn_swr_refresh") as swr:
            assert cached_provider_model_ids(COPILOT_ACP, non_blocking=True) == []
        swr.assert_called_once_with(COPILOT_ACP)

    def test_absent_login_invalidates(self, isolated_creds):
        home = isolated_creds.parent.parent
        hosts = _write_hosts(home, "alice", "tok-a")
        before = _credential_fingerprint(COPILOT_ACP)
        hosts.unlink()
        assert _credential_fingerprint(COPILOT_ACP) != before

    def test_claude_rotation_keeps_copilot_fingerprint(self, isolated_creds):
        home = isolated_creds.parent.parent
        _write_hosts(home, "alice", "tok-a")
        before = _credential_fingerprint(COPILOT_ACP)
        _rotate(isolated_creds)
        assert _credential_fingerprint(COPILOT_ACP) == before
