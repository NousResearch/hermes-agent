"""Tests for the macOS 26 launchd plist provenance auto-repair.

These tests use ``monkeypatch`` to stand in for the actual ``xattr`` and
``launchctl`` binaries — no macOS 26 host is required to run them.

The repair path has three layers:
1. ``_plist_has_provenance_xattr`` — fast detection via ``xattr`` (names only).
2. ``_maybe_clear_plist_provenance_xattr`` — best-effort sudo + cache reset.
3. ``_atomic_label_migration_to_clear_provenance`` — clones the plist with a
   timestamp-suffixed Label so launchd's stale cache cannot match.

The single entry point that ties them together is
``_ensure_clean_plist_for_bootstrap`` — the only one invoked from the
launchd_install / launchd_start / launchd_restart call sites and from
``_cmd_update_impl``'s post-update pre-flight.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# The provenance-xattr repair path is macOS-specific; the helpers are
# always importable (they're guarded by sys.platform at call sites) but
# the tests below only exercise them when sys.platform == "darwin".
requires_macos = pytest.mark.skipif(
    sys.platform != "darwin", reason="macOS-only repair path"
)

import hermes_cli.gateway as gateway_cli


# ---------------------------------------------------------------------------
# Detection: _plist_has_provenance_xattr
# ---------------------------------------------------------------------------


class TestPlistHasProvenanceXattr:
    """Subtle bug history: v1 of this helper used ``xattr -l`` (which prints
    ``name: value`` per line) and looked for the literal ``"com.apple.provenance"``
    — that always returned False because the actual output is
    ``"com.apple.provenance: \x01\x02"``. v2 uses ``xattr`` (names only) and
    splits on newlines. These tests pin the v2 behaviour."""

    def test_returns_false_when_xattr_command_missing(self, monkeypatch, tmp_path):
        plist = tmp_path / "ai.hermes.gateway.plist"
        plist.write_text("<plist/>", encoding="utf-8")
        monkeypatch.setattr(
            gateway_cli.subprocess, "run",
            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError),
        )
        assert gateway_cli._plist_has_provenance_xattr(plist) is False

    def test_returns_false_when_xattr_lists_other_attrs(self, monkeypatch, tmp_path):
        plist = tmp_path / "ai.hermes.gateway.plist"
        plist.write_text("<plist/>", encoding="utf-8")
        result = SimpleNamespace(
            returncode=0,
            stdout="com.apple.metadata:_kMDItemUserTags\n",
        )
        monkeypatch.setattr(gateway_cli.subprocess, "run",
                            lambda *a, **kw: result)
        assert gateway_cli._plist_has_provenance_xattr(plist) is False

    def test_returns_true_when_provenance_listed(self, monkeypatch, tmp_path):
        plist = tmp_path / "ai.hermes.gateway.plist"
        plist.write_text("<plist/>", encoding="utf-8")
        result = SimpleNamespace(
            returncode=0,
            stdout="com.apple.provenance\ncom.apple.quarantine\n",
        )
        monkeypatch.setattr(gateway_cli.subprocess, "run",
                            lambda *a, **kw: result)
        assert gateway_cli._plist_has_provenance_xattr(plist) is True

    def test_returns_false_when_xattr_nonzero_exit(self, monkeypatch, tmp_path):
        plist = tmp_path / "ai.hermes.gateway.plist"
        plist.write_text("<plist/>", encoding="utf-8")
        result = SimpleNamespace(returncode=1, stdout="")
        monkeypatch.setattr(gateway_cli.subprocess, "run",
                            lambda *a, **kw: result)
        assert gateway_cli._plist_has_provenance_xattr(plist) is False

    def test_returns_false_for_missing_file(self, tmp_path):
        ghost = tmp_path / "does-not-exist.plist"
        assert gateway_cli._plist_has_provenance_xattr(ghost) is False


# ---------------------------------------------------------------------------
# Migration: _atomic_label_migration_to_clear_provenance
# ---------------------------------------------------------------------------


class TestAtomicLabelMigration:
    def test_noop_when_provenance_clean(self, monkeypatch, tmp_path):
        plist = tmp_path / "ai.hermes.gateway.plist"
        plist.write_text(
            '<?xml version="1.0"?>\n<plist><dict>'
            '<key>Label</key><string>ai.hermes.gateway</string>'
            '</dict></plist>',
            encoding="utf-8",
        )
        # Plist has no xattr → early return.
        monkeypatch.setattr(
            gateway_cli, "_plist_has_provenance_xattr", lambda p: False
        )
        result = gateway_cli._atomic_label_migration_to_clear_provenance(plist)
        assert result is None
        # No new plist file should be created.
        siblings = list(tmp_path.glob("ai.hermes.gateway*.plist"))
        assert siblings == [plist]

    def test_creates_timestamped_clone_with_new_label(
        self, monkeypatch, tmp_path
    ):
        plist = tmp_path / "ai.hermes.gateway.plist"
        plist.write_text(
            '<?xml version="1.0"?>\n<plist><dict>'
            '<key>Label</key><string>ai.hermes.gateway</string>'
            '<key>ProgramArguments</key><array>'
            '<string>/usr/bin/python</string>'
            '</array>'
            '</dict></plist>',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            gateway_cli, "_plist_has_provenance_xattr", lambda p: True
        )
        # The migration calls subprocess.run only for launchctl disable/bootout.
        # Capture those calls without actually invoking launchd.
        launchctl_calls = []
        def fake_run(args, **kw):
            launchctl_calls.append(args)
            return SimpleNamespace(returncode=3, stdout="", stderr="")
        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        # Don't actually persist the active label.
        monkeypatch.setattr(gateway_cli, "_write_active_label", lambda l: None)
        # Pin the timestamp to 1_780_938_650.0 // 10 = 178093865.
        import time as _real_time
        monkeypatch.setattr(_real_time, "time", lambda: 1_780_938_650.0)

        result = gateway_cli._atomic_label_migration_to_clear_provenance(plist)
        assert result is not None
        assert result.stem == "ai.hermes.gateway.178093865"
        # The new plist should exist on disk with the new label.
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        # The Label key in the clone must be the timestamped one, not the base.
        assert "<string>ai.hermes.gateway.178093865</string>" in content
        # launchctl was invoked to disable + bootout the base label.
        assert any(
            "ai.hermes.gateway" in str(args)
            for args in launchctl_calls
        )


# ---------------------------------------------------------------------------
# End-to-end orchestrator: _ensure_clean_plist_for_bootstrap
# ---------------------------------------------------------------------------


class TestEnsureCleanPlistForBootstrap:
    def test_returns_unchanged_when_already_clean(self, monkeypatch, tmp_path):
        plist = tmp_path / "ai.hermes.gateway.plist"
        plist.write_text("<plist/>", encoding="utf-8")
        monkeypatch.setattr(
            gateway_cli, "_plist_has_provenance_xattr", lambda p: False
        )
        # The expensive repair helpers must NOT be invoked on a clean plist.
        monkeypatch.setattr(
            gateway_cli, "_maybe_clear_plist_provenance_xattr",
            lambda p: pytest.fail("clear should not be called on clean plist"),
        )
        assert gateway_cli._ensure_clean_plist_for_bootstrap(plist) == plist

    def test_delegates_to_xattr_clear_when_dirty(
        self, monkeypatch, tmp_path
    ):
        plist = tmp_path / "ai.hermes.gateway.plist"
        plist.write_text("<plist/>", encoding="utf-8")
        # Plist is dirty before Tier 2, clean after — simulates the
        # happy path where sudo successfully removed the xattr.
        state = {"dirty": True}
        monkeypatch.setattr(
            gateway_cli, "_plist_has_provenance_xattr",
            lambda p: state["dirty"],
        )
        def fake_clear(p):
            state["dirty"] = False
            return True
        monkeypatch.setattr(
            gateway_cli, "_maybe_clear_plist_provenance_xattr", fake_clear,
        )
        monkeypatch.setattr(
            gateway_cli, "_atomic_label_migration_to_clear_provenance",
            lambda p: pytest.fail("migration should not run if xattr clear works"),
        )
        assert gateway_cli._ensure_clean_plist_for_bootstrap(plist) == plist

    def test_falls_back_to_migration_when_xattr_clear_fails(
        self, monkeypatch, tmp_path
    ):
        plist = tmp_path / "ai.hermes.gateway.plist"
        plist.write_text("<plist/>", encoding="utf-8")
        monkeypatch.setattr(
            gateway_cli, "_plist_has_provenance_xattr", lambda p: True
        )
        # Tier 2 fails (no sudo) → fall to Tier 3.
        monkeypatch.setattr(
            gateway_cli, "_maybe_clear_plist_provenance_xattr",
            lambda p: False,
        )
        new_path = tmp_path / "ai.hermes.gateway.1234567890.plist"
        new_path.write_text("<plist/>", encoding="utf-8")
        monkeypatch.setattr(
            gateway_cli, "_atomic_label_migration_to_clear_provenance",
            lambda p: new_path,
        )
        result = gateway_cli._ensure_clean_plist_for_bootstrap(plist)
        assert result == new_path
        assert result != plist

    def test_returns_original_when_all_tiers_fail(
        self, monkeypatch, tmp_path
    ):
        plist = tmp_path / "ai.hermes.gateway.plist"
        plist.write_text("<plist/>", encoding="utf-8")
        monkeypatch.setattr(
            gateway_cli, "_plist_has_provenance_xattr", lambda p: True
        )
        monkeypatch.setattr(
            gateway_cli, "_maybe_clear_plist_provenance_xattr",
            lambda p: False,
        )
        monkeypatch.setattr(
            gateway_cli, "_atomic_label_migration_to_clear_provenance",
            lambda p: None,  # migration also gave up
        )
        # Caller gets the original plist back — bootstrap will fail with
        # errno 5 and the CLI will degrade to the detached launcher.
        assert gateway_cli._ensure_clean_plist_for_bootstrap(plist) == plist


# ---------------------------------------------------------------------------
# Active label indirection
# ---------------------------------------------------------------------------


class TestActiveLabelIndirection:
    def test_falls_back_to_canonical_label_when_state_unset(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(
            gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway"
        )
        assert gateway_cli.get_active_launchd_label() == "ai.hermes.gateway"

    def test_reads_active_label_from_state_file(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: tmp_path)
        (tmp_path / "_state").mkdir()
        (tmp_path / "_state" / "launchd-active-label").write_text(
            "ai.hermes.gateway.1780938650\n", encoding="utf-8"
        )
        assert (
            gateway_cli.get_active_launchd_label() == "ai.hermes.gateway.1780938650"
        )

    def test_write_active_label_is_atomic(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: tmp_path)
        gateway_cli._write_active_label("ai.hermes.gateway.99")
        path = tmp_path / "_state" / "launchd-active-label"
        assert path.read_text(encoding="utf-8") == "ai.hermes.gateway.99\n"
        # No leftover .tmp file from the atomic write.
        assert not (tmp_path / "_state" / "launchd-active-label.tmp").exists()
