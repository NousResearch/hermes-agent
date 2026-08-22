"""Tests for npm ``EBADENGINE`` recovery (``hermes_cli/npm_engine.py``).

The behaviour under test is a contract about *reacting* to npm's own engine
check: npm states the range it wants in the failure, Hermes upgrades only an
npm it owns, and every other case leaves the original failure alone.
"""

import json
import os
import subprocess
import sys
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


def _write_fake_npm(directory: Path, *, version: str) -> Path:
    """A real npm-shaped executable that logs argv and answers ``--version``.

    ``install --global`` succeeds (managed upgrade); any other ``install``
    exits 1 with empty output (the historical ``--silent`` EBADENGINE).
    """
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "_hermes_fake_npm.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"VERSION = {version!r}\n"
        "log = Path(__file__).with_name('fake-npm-calls.json')\n"
        "calls = json.loads(log.read_text()) if log.exists() else []\n"
        "calls.append(sys.argv[1:])\n"
        "log.write_text(json.dumps(calls))\n"
        "if '--version' in sys.argv:\n"
        "    print(VERSION)\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(0 if '--global' in sys.argv else 1)\n",
        encoding="utf-8",
    )
    if sys.platform == "win32":
        wrapper = directory / "npm.cmd"
        wrapper.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n'
            "exit /b %ERRORLEVEL%\r\n",
            encoding="utf-8",
        )
        return wrapper
    wrapper = directory / "npm"
    wrapper.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def _fake_npm_calls(npm: Path) -> list[list[str]]:
    log = npm.parent / "fake-npm-calls.json"
    if not log.is_file():
        return []
    return json.loads(log.read_text(encoding="utf-8"))


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


class TestNpmSatisfiesRange:
    """Opaque-failure repair reuses the engines.npm matcher the repo already
    authors in tests/test_engines_satisfiable.py — not a second parser."""

    def test_or_alternatives_and_and_clauses(self):
        fn = getattr(npm_engine, "npm_satisfies_range", None)
        assert fn is not None, "opaque repair needs npm_satisfies_range"
        spec = "<11.10.0 || >=11.17.0"
        assert fn("11.9.0", spec)
        assert fn("11.17.0", spec)
        assert fn("12.0.0", spec)
        assert not fn("11.16.0", spec)
        assert not fn("11.10.0", spec)

    def test_empty_or_garbage_is_not_a_match(self):
        fn = getattr(npm_engine, "npm_satisfies_range", None)
        assert fn is not None, "opaque repair needs npm_satisfies_range"
        assert not fn("", ">=12.0.0")
        assert not fn("11.0.0", "")
        assert not fn("not-a-version", ">=12.0.0")


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


class TestOpaqueEngineRepair:
    """#78826 / #78878: empty install output still repairs when npm is
    outside engines.npm, and does not false-repair when it is in range.

    These go through a fake npm executable (``--version`` / ``install
    --global``) rather than stubbing ``_probe_version``.
    """

    @pytest.fixture
    def managed_tree(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home / "node" / "bin"

    def test_opaque_failure_repairs_when_npm_outside_repo_range(self, managed_tree):
        npm = _write_fake_npm(managed_tree, version="11.16.0")
        repaired = maybe_repair_npm_engine(str(npm), "", quiet=True)
        assert repaired == str(npm)
        calls = _fake_npm_calls(npm)
        assert any("--version" in c for c in calls), calls
        assert any("--global" in c for c in calls), calls

    def test_opaque_failure_with_in_range_npm_never_repairs(self, managed_tree):
        npm = _write_fake_npm(managed_tree, version="11.17.0")
        repaired = maybe_repair_npm_engine(str(npm), "\n", quiet=True)
        assert not repaired
        calls = _fake_npm_calls(npm)
        assert any("--version" in c for c in calls), calls
        assert not any("--global" in c for c in calls), calls

    def test_non_engine_failure_never_probes_or_repairs(self, managed_tree):
        """A lockfile mismatch with diagnostic text is not opaque — even an
        out-of-range npm must leave the original error alone."""
        npm = _write_fake_npm(managed_tree, version="11.16.0")
        repaired = maybe_repair_npm_engine(str(npm), ELOCK_OUTPUT, quiet=True)
        assert not repaired
        assert _fake_npm_calls(npm) == []

    def test_opaque_failure_on_foreign_nvm_npm_provisions_managed_runtime(
        self, tmp_path, monkeypatch
    ):
        """#78826: nvm/system npm outside engines.npm + empty install output.

        The failing npm is a PATH shim that looks like nvm (not under
        ``$HERMES_HOME/node``). Repair must probe ``npm --version`` on that
        shim, provision a Hermes-managed runtime, return the managed npm,
        and never run ``install --global`` against the foreign toolchain.
        """
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(home))

        nvm_bin = (
            tmp_path / "nvm" / "versions" / "node" / "v24.18.1" / "bin"
        )
        foreign_npm = _write_fake_npm(nvm_bin, version="11.16.0")
        monkeypatch.setenv(
            "PATH",
            os.pathsep.join([str(nvm_bin), os.environ.get("PATH", "")]),
        )

        assert managed_npm_prefix(foreign_npm) is None

        provisioned: list[str] = []

        def fake_bootstrap() -> str:
            managed = _write_fake_npm(home / "node" / "bin", version="11.17.0")
            provisioned.append(str(managed))
            return str(managed)

        monkeypatch.setattr(
            npm_engine, "bootstrap_hermes_managed_node", fake_bootstrap
        )

        repaired = maybe_repair_npm_engine(str(foreign_npm), "", quiet=True)

        assert provisioned, "foreign opaque failure must provision managed Node"
        assert repaired == provisioned[0]
        assert repaired != str(foreign_npm)
        assert Path(repaired).resolve() != foreign_npm.resolve()

        foreign_calls = _fake_npm_calls(foreign_npm)
        assert any("--version" in c for c in foreign_calls), foreign_calls
        assert not any("--global" in c for c in foreign_calls), (
            "foreign/nvm npm must never be the target of install --global: "
            f"{foreign_calls}"
        )

        managed_calls = _fake_npm_calls(Path(repaired))
        assert any("--global" in c for c in managed_calls), managed_calls


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
