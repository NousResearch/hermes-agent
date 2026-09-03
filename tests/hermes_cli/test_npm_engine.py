"""Tests for npm ``EBADENGINE`` recovery (``hermes_cli/npm_engine.py``).

The behaviour under test is a contract about *reacting* to npm's own engine
check: npm states the range it wants in the failure, Hermes upgrades only an
npm it owns, and every other case leaves the original failure alone.
"""

import json
import subprocess
from pathlib import Path

import pytest

import hermes_cli.npm_engine as npm_engine
from hermes_cli.npm_engine import (
    actual_npm_version,
    is_ebadengine,
    managed_npm_prefix,
    maybe_repair_npm_engine,
    required_npm_range,
)


# Verbatim npm 10 output shape (`npm error`), and the npm 9 shape (`npm ERR!`).
EBADENGINE_OUTPUT = """
npm error code EBADENGINE
npm error engine Unsupported engine
npm error engine Not compatible with your version of node/npm: hermes-agent@1.0.0
npm error notsup Not compatible with your version of node/npm: hermes-agent@1.0.0
npm error notsup Required: {"node":">=20.0.0","npm":"<11.10.0 || >=12.0.0"}
npm error notsup Actual:   {"npm":"11.10.0","node":"v22.23.1"}
"""

LEGACY_EBADENGINE_OUTPUT = """
npm ERR! code EBADENGINE
npm ERR! engine Unsupported engine
npm ERR! notsup Required: {"node":">=20.0.0","npm":">=12.0.0"}
npm ERR! notsup Actual:   {"npm":"9.6.7","node":"v20.1.0"}
"""

# A lockfile mismatch — the other common `npm ci` failure. Must NOT be treated
# as an engine problem, or every out-of-sync lockfile would trigger an upgrade.
ELOCK_OUTPUT = """
npm error code EUSAGE
npm error `npm ci` can only install packages when your package.json and
npm error package-lock.json are in sync.
"""


class TestDetection:
    def test_recognises_modern_and_legacy_engine_failures(self):
        assert is_ebadengine(EBADENGINE_OUTPUT)
        assert is_ebadengine(LEGACY_EBADENGINE_OUTPUT)

    def test_unrelated_failures_are_not_engine_failures(self):
        assert not is_ebadengine(ELOCK_OUTPUT)
        assert not is_ebadengine("")
        assert not is_ebadengine("npm error code E404")

    def test_range_comes_from_the_error_not_a_hardcoded_list(self):
        assert required_npm_range(EBADENGINE_OUTPUT) == "<11.10.0 || >=12.0.0"
        assert required_npm_range(LEGACY_EBADENGINE_OUTPUT) == ">=12.0.0"

    def test_actual_version_is_reported_back(self):
        assert actual_npm_version(EBADENGINE_OUTPUT) == "11.10.0"

    def test_no_range_for_non_engine_output(self):
        assert required_npm_range(ELOCK_OUTPUT) is None
        assert required_npm_range("") is None

    def test_node_only_mismatch_yields_no_npm_range(self):
        """Upgrading npm cannot fix a Node version mismatch, so don't try."""
        node_only = (
            'npm error code EBADENGINE\n'
            'npm error notsup Required: {"node":">=20.0.0"}\n'
            'npm error notsup Actual:   {"npm":"10.9.8","node":"v18.0.0"}\n'
        )
        assert required_npm_range(node_only) is None

    def test_malformed_required_block_is_ignored(self):
        broken = (
            "npm error code EBADENGINE\n"
            "npm error notsup Required: {not json}\n"
        )
        assert required_npm_range(broken) is None


class TestManagedDetection:
    """The upgrade must fire for every spelling of the managed npm, and for
    no other npm — this is the boundary between "Hermes fixes it" and "the
    user's own toolchain is left alone"."""

    @pytest.fixture
    def managed_tree(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        node = home / "node"
        (node / "bin").mkdir(parents=True)
        (node / "lib" / "node_modules" / "npm" / "bin").mkdir(parents=True)
        cli = node / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"
        cli.write_text("#!/usr/bin/env node\n", encoding="utf-8")
        (node / "bin" / "npm").symlink_to(cli)
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    def test_direct_managed_bin_is_managed(self, managed_tree):
        npm = managed_tree / "node" / "bin" / "npm"
        assert managed_npm_prefix(npm) == managed_tree / "node"

    def test_symlink_from_local_bin_resolves_to_managed(self, managed_tree, tmp_path):
        """An install links ~/.local/bin/npm at the managed tree; that link is
        the npm a user's PATH actually resolves, so it must count as managed."""
        local_bin = tmp_path / "local-bin"
        local_bin.mkdir()
        link = local_bin / "npm"
        link.symlink_to(managed_tree / "node" / "bin" / "npm")
        assert managed_npm_prefix(link) == managed_tree / "node"

    def test_system_npm_is_not_managed(self, managed_tree, tmp_path):
        system_npm = tmp_path / "usr" / "bin" / "npm"
        system_npm.parent.mkdir(parents=True)
        system_npm.write_text("#!/bin/sh\n", encoding="utf-8")
        assert managed_npm_prefix(system_npm) is None

    def test_no_npm_is_not_managed(self, managed_tree):
        assert managed_npm_prefix(None) is None
        assert managed_npm_prefix("") is None


class TestInUseDeferral:
    """The managed tree cannot be written while a running app executes from
    it (WinError 5 on npm.cmd, #80926) — the npm upgrade defers instead."""

    @pytest.fixture
    def managed_npm(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        bin_dir = home / "node" / "bin"
        bin_dir.mkdir(parents=True)
        npm = bin_dir / "npm"
        npm.write_text("#!/bin/sh\n", encoding="utf-8")
        npm.chmod(0o755)
        monkeypatch.setenv("HERMES_HOME", str(home))
        return npm

    def test_in_use_managed_tree_defers_upgrade_without_running_npm(
        self, managed_npm, monkeypatch
    ):
        monkeypatch.setattr(npm_engine, "managed_node_tree_in_use", lambda: True)

        def forbidden_run(cmd, **kwargs):
            raise AssertionError(f"npm must not run while the tree is in use: {cmd}")

        monkeypatch.setattr(subprocess, "run", forbidden_run)

        result = npm_engine.upgrade_managed_npm(
            str(managed_npm),
            ">=11.0.0",
            prefix=managed_npm.parent,
            quiet=True,
        )
        assert result is False

    def test_in_use_deferral_blocks_repair_retry(self, managed_npm, monkeypatch):
        """End-to-end: an in-use tree means no npm subprocess runs and no
        retry is offered — the original EBADENGINE failure stands with the
        deferral notice."""
        monkeypatch.setattr(npm_engine, "managed_node_tree_in_use", lambda: True)

        def forbidden_run(cmd, **kwargs):
            raise AssertionError(f"npm must not run while the tree is in use: {cmd}")

        monkeypatch.setattr(subprocess, "run", forbidden_run)

        assert (
            maybe_repair_npm_engine(str(managed_npm), EBADENGINE_OUTPUT, quiet=True)
            is None
        )


class TestRepairDecision:
    """`maybe_repair_npm_engine` returns the npm to retry with (truthy) only
    when a repair actually happened, because its return value is what gates
    the caller's single retry."""

    @pytest.fixture
    def managed_npm(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        bin_dir = home / "node" / "bin"
        bin_dir.mkdir(parents=True)
        npm = bin_dir / "npm"
        npm.write_text("#!/bin/sh\n", encoding="utf-8")
        npm.chmod(0o755)
        monkeypatch.setenv("HERMES_HOME", str(home))
        return npm

    def test_upgrades_managed_npm_with_the_range_npm_asked_for(
        self, managed_npm, monkeypatch
    ):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        repaired = maybe_repair_npm_engine(
            str(managed_npm), EBADENGINE_OUTPUT, quiet=True
        )
        assert repaired == str(managed_npm)

        upgrade_cmd = calls[0][0]
        assert upgrade_cmd[1:3] == ["install", "--global"]
        # The range must come from npm's error, and target the managed prefix
        # explicitly (the managed etc/npmrc points `prefix` elsewhere).
        assert "npm@<11.10.0 || >=12.0.0" in upgrade_cmd
        prefix_index = upgrade_cmd.index("--prefix")
        assert Path(upgrade_cmd[prefix_index + 1]) == managed_npm.parent.parent

    def test_upgrade_runs_outside_the_checkout(self, managed_npm, monkeypatch):
        """The repo .npmrc sets min-release-age, which would gate the very npm
        release we need; the upgrade must not run under it."""
        seen = {}

        def fake_run(cmd, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        maybe_repair_npm_engine(str(managed_npm), EBADENGINE_OUTPUT, quiet=True)

        cwd = Path(seen["cwd"])
        assert not (cwd / ".npmrc").exists()
        assert seen["env"]["npm_config_min_release_age"] == "0"

    def test_failed_upgrade_reports_no_retry(self, managed_npm, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"),
        )
        assert not maybe_repair_npm_engine(
            str(managed_npm), EBADENGINE_OUTPUT, quiet=True
        )

    def test_foreign_npm_provisions_managed_runtime_instead(
        self, tmp_path, monkeypatch
    ):
        """A system/nvm/brew/Nix npm is never modified — Hermes provisions its
        own managed tree, upgrades THAT npm into range, and returns it."""
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(home))
        system_npm = tmp_path / "usr-bin-npm"
        system_npm.write_text("#!/bin/sh\n", encoding="utf-8")

        managed = home / "node" / "bin" / "npm"

        import hermes_cli.npm_engine as npm_engine

        def fake_bootstrap():
            managed.parent.mkdir(parents=True, exist_ok=True)
            managed.write_text("#!/bin/sh\n", encoding="utf-8")
            managed.chmod(0o755)
            return str(managed)

        upgrades = []
        monkeypatch.setattr(
            npm_engine, "bootstrap_hermes_managed_node", fake_bootstrap
        )
        monkeypatch.setattr(
            npm_engine,
            "upgrade_managed_npm",
            lambda npm, rng, *, prefix, quiet=False: upgrades.append((npm, rng))
            or True,
        )

        repaired = maybe_repair_npm_engine(
            str(system_npm), EBADENGINE_OUTPUT, quiet=True
        )
        assert repaired == str(managed)
        # The upgrade targeted the MANAGED npm with npm's own stated range —
        # the system npm was never the target of anything.
        assert upgrades == [(str(managed), "<11.10.0 || >=12.0.0")]

    def test_foreign_npm_failed_bootstrap_prints_manual_fix(
        self, tmp_path, monkeypatch, capsys
    ):
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(home))
        system_npm = tmp_path / "usr-bin-npm"
        system_npm.write_text("#!/bin/sh\n", encoding="utf-8")

        import hermes_cli.npm_engine as npm_engine

        monkeypatch.setattr(
            npm_engine, "bootstrap_hermes_managed_node", lambda: None
        )
        assert not maybe_repair_npm_engine(str(system_npm), EBADENGINE_OUTPUT)

        # The user gets the exact command to run, since we refuse to run it.
        err = capsys.readouterr().err
        assert 'npm install -g npm@"<11.10.0 || >=12.0.0"' in err

    def test_non_engine_failure_never_repairs(self, managed_npm, monkeypatch):
        def explode(cmd, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("a lockfile mismatch must not trigger a repair")

        monkeypatch.setattr(subprocess, "run", explode)
        assert not maybe_repair_npm_engine(str(managed_npm), ELOCK_OUTPUT, quiet=True)

    def test_node_only_mismatch_on_foreign_npm_still_provisions(
        self, tmp_path, monkeypatch
    ):
        """A too-old system NODE can't be fixed by any npm upgrade, but the
        managed tree ships a supported Node — provisioning covers it. The
        managed npm is still upgraded to the repo's own engines.npm range."""
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(home))
        system_npm = tmp_path / "usr-bin-npm"
        system_npm.write_text("#!/bin/sh\n", encoding="utf-8")

        node_only = (
            "npm error code EBADENGINE\n"
            'npm error notsup Required: {"node":">=20.0.0"}\n'
            'npm error notsup Actual:   {"npm":"10.9.8","node":"v18.0.0"}\n'
        )

        managed = home / "node" / "bin" / "npm"

        import hermes_cli.npm_engine as npm_engine

        def fake_bootstrap():
            managed.parent.mkdir(parents=True, exist_ok=True)
            managed.write_text("#!/bin/sh\n", encoding="utf-8")
            managed.chmod(0o755)
            return str(managed)

        upgrades = []
        monkeypatch.setattr(
            npm_engine, "bootstrap_hermes_managed_node", fake_bootstrap
        )
        monkeypatch.setattr(
            npm_engine,
            "upgrade_managed_npm",
            lambda npm, rng, *, prefix, quiet=False: upgrades.append(rng) or True,
        )

        repaired = maybe_repair_npm_engine(str(system_npm), node_only, quiet=True)
        assert repaired == str(managed)
        # No range in npm's error → fall back to the repo's own engines.npm
        # so the fresh tree's bundled npm doesn't fail the retry identically.
        repo_range = npm_engine._repo_npm_range()
        assert upgrades == ([repo_range] if repo_range else [])

    def test_node_only_mismatch_on_managed_npm_does_not_upgrade(
        self, managed_npm, monkeypatch
    ):
        """Upgrading a managed npm cannot fix a managed-Node mismatch."""
        node_only = (
            "npm error code EBADENGINE\n"
            'npm error notsup Required: {"node":">=20.0.0"}\n'
            'npm error notsup Actual:   {"npm":"10.9.8","node":"v18.0.0"}\n'
        )

        def explode(cmd, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("npm upgrade cannot fix a Node mismatch")

        monkeypatch.setattr(subprocess, "run", explode)
        assert not maybe_repair_npm_engine(str(managed_npm), node_only, quiet=True)


class TestRepoRangeIsSatisfiable:
    """Invariant: whatever the root package.json demands, the recovery can
    parse and act on it — a malformed range would make the repair a no-op."""

    def test_root_engines_npm_range_is_a_usable_constraint(self):
        repo_root = Path(__file__).resolve().parents[2]
        package_json = repo_root / "package.json"
        engines = json.loads(package_json.read_text(encoding="utf-8")).get("engines", {})
        npm_range = engines.get("npm")
        if not npm_range:
            pytest.skip("root package.json does not pin engines.npm")

        synthetic = (
            "npm error code EBADENGINE\n"
            'npm error notsup Required: '
            + json.dumps({"node": ">=20.0.0", "npm": npm_range})
            + "\n"
        )
        assert required_npm_range(synthetic) == npm_range
