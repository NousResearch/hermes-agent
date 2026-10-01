"""Regression tests for #76414: `hermes honcho peers` showed "(not set)"
for every non-default profile.

_all_profile_host_configs() built the per-profile host key inline as
f"{HOST}.{profile}" ("hermes.work") while every other reader/writer —
profile_host_key(), resolve_active_host(), honcho status/enable/sync and
the runtime plugin — uses the underscore form ("hermes_work"). The lookup
always missed, so cmd_peers fell back to "(not set)" and leaked the raw
malformed key into the AI-peer column.

These tests drive the real cmd_peers / _all_profile_host_configs against
a real honcho.json (temp HERMES_HOME, no network).
"""
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import plugins.memory.honcho.cli as honcho_cli


@pytest.fixture
def honcho_home(tmp_path, monkeypatch):
    cfg = {
        "peerName": "alice",
        "hosts": {
            "hermes": {"peerName": "alice", "aiPeer": "hermes"},
            "hermes_work": {"peerName": "alice", "aiPeer": "hermes"},
            "hermes_my_profile": {"peerName": "bob", "aiPeer": "hermes"},
        },
    }
    path = tmp_path / "honcho.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(honcho_cli, "_config_path", lambda: path)
    return tmp_path


def _peers_output(profiles):
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        honcho_cli.cmd_peers(SimpleNamespace())
    finally:
        sys.stdout = old
    return buf.getvalue()


class TestAllProfileHostConfigs:
    def test_profile_host_keys_match_writer_form(self, honcho_home, monkeypatch):
        """The lookup key must be profile_host_key()'s underscore form —
        the same one honcho sync/enable/status and the runtime write to."""
        monkeypatch.setattr(
            "hermes_cli.profiles.list_profiles",
            lambda: [SimpleNamespace(name="default"), SimpleNamespace(name="work")],
        )
        rows = honcho_cli._all_profile_host_configs()
        by_name = {name: (host, block) for name, host, block in rows}
        host, block = by_name["work"]
        assert host == "hermes_work"  # not "hermes.work"
        assert block.get("peerName") == "alice"  # the populated block was found

    def test_sanitized_profile_names_resolve(self, honcho_home, monkeypatch):
        """Profiles needing sanitization (dots/spaces in the name) also
        resolve — profile_host_key maps 'my.profile' -> 'hermes_my_profile';
        the inline dot form never could."""
        monkeypatch.setattr(
            "hermes_cli.profiles.list_profiles",
            lambda: [SimpleNamespace(name="default"),
                     SimpleNamespace(name="my.profile")],
        )
        rows = honcho_cli._all_profile_host_configs()
        by_name = {name: block for name, _, block in rows}
        assert by_name["my.profile"].get("peerName") == "bob"

    def test_legacy_dot_form_host_key_still_readable(self, honcho_home, monkeypatch):
        """Back-compat: honcho.json files with LEGACY dot-form host keys
        ("hermes.work") must keep working — the README promises those keys
        stay readable, and _host_block() exists precisely for that fallback.
        A bare hosts.get(profile_host_key(...)) would regress them."""
        path = honcho_home / "honcho.json"
        cfg = json.loads(path.read_text())
        del cfg["hosts"]["hermes_work"]
        cfg["hosts"]["hermes.work"] = {"peerName": "carol", "aiPeer": "hermes"}
        path.write_text(json.dumps(cfg))
        monkeypatch.setattr(
            "hermes_cli.profiles.list_profiles",
            lambda: [SimpleNamespace(name="default"), SimpleNamespace(name="work")],
        )
        rows = honcho_cli._all_profile_host_configs()
        by_name = {name: block for name, _, block in rows}
        assert by_name["work"].get("peerName") == "carol"


class TestCmdPeers:
    def test_peers_shows_populated_identity_not_host_key_leak(
            self, honcho_home, monkeypatch):
        """Issue #76414's visible symptom: the AI-peer column showed the
        raw malformed key 'hermes.work' (or '(not set)')."""
        monkeypatch.setattr(
            "hermes_cli.profiles.list_profiles",
            lambda: [SimpleNamespace(name="default"), SimpleNamespace(name="work")],
        )
        out = _peers_output(SimpleNamespace())
        assert "hermes.work" not in out
        assert "(not set)" not in out
        # work row shows the populated block's values
        work_line = [l for l in out.splitlines() if l.strip().startswith("work")][0]
        assert "alice" in work_line and "hermes" in work_line

    def test_peers_falls_back_cleanly_when_block_missing(
            self, honcho_home, monkeypatch):
        """A profile with no host block still falls back to the top-level
        peerName and the (well-formed) host key — not a crash or a leak."""
        monkeypatch.setattr(
            "hermes_cli.profiles.list_profiles",
            lambda: [SimpleNamespace(name="default"), SimpleNamespace(name="new")],
        )
        out = _peers_output(SimpleNamespace())
        assert "hermes.new" not in out  # well-formed key, no dot-form leak
        new_line = [l for l in out.splitlines() if l.strip().startswith("new")][0]
        assert "alice" in new_line  # top-level peerName fallback
