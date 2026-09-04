"""Tests for npm ``EBADENGINE`` recovery (``hermes_cli/npm_engine.py``).

The behaviour under test is a contract about *reacting* to npm's own engine
check: npm states the range it wants in the failure, Hermes provisions its
pm-pinned npm instead of touching a foreign one, and every other case leaves
the original failure alone.
"""

import json
from pathlib import Path

import pytest

import hermes_cli.npm_engine as npm_engine
from hermes_cli.npm_engine import (
    actual_npm_version,
    is_ebadengine,
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
# as an engine problem, or every out-of-sync lockfile would trigger a repair.
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


class TestRepairDecision:
    """`maybe_repair_npm_engine` returns the npm to retry with (truthy) only
    when a repair actually happened, because its return value is what gates
    the caller's single retry."""

    def test_foreign_npm_provisions_pm_npm_instead(self, tmp_path, monkeypatch):
        """A system/nvm/brew/Nix npm is never modified — Hermes ensures its
        own pm-pinned npm and returns it."""
        system_npm = tmp_path / "usr-bin-npm"
        system_npm.write_text("#!/bin/sh\n", encoding="utf-8")
        managed = tmp_path / "store" / "npm-x" / "npm"
        managed.parent.mkdir(parents=True)
        managed.write_text("#!/bin/sh\n", encoding="utf-8")

        monkeypatch.setattr(
            npm_engine, "_pm_npm", lambda quiet=False: str(managed)
        )
        repaired = maybe_repair_npm_engine(
            str(system_npm), EBADENGINE_OUTPUT, quiet=True
        )
        assert repaired == str(managed)

    def test_failing_pm_npm_reports_no_retry(self, tmp_path, monkeypatch, capsys):
        """When the failing npm already IS pm's npm, a re-ensure cannot change
        anything — no retry, manual guidance instead."""
        managed = tmp_path / "store" / "npm-x" / "npm"
        managed.parent.mkdir(parents=True)
        managed.write_text("#!/bin/sh\n", encoding="utf-8")

        monkeypatch.setattr(
            npm_engine, "_pm_npm", lambda quiet=False: str(managed)
        )
        assert maybe_repair_npm_engine(str(managed), EBADENGINE_OUTPUT) is None
        err = capsys.readouterr().err
        assert 'npm install -g npm@"<11.10.0 || >=12.0.0"' in err

    def test_failed_provisioning_prints_manual_fix(self, tmp_path, monkeypatch, capsys):
        system_npm = tmp_path / "usr-bin-npm"
        system_npm.write_text("#!/bin/sh\n", encoding="utf-8")

        monkeypatch.setattr(npm_engine, "_pm_npm", lambda quiet=False: None)
        assert not maybe_repair_npm_engine(str(system_npm), EBADENGINE_OUTPUT)

        # The user gets the exact command to run, since we refuse to run it.
        err = capsys.readouterr().err
        assert 'npm install -g npm@"<11.10.0 || >=12.0.0"' in err

    def test_non_engine_failure_never_repairs(self, tmp_path, monkeypatch):
        def explode(quiet=False):  # pragma: no cover - must not be reached
            raise AssertionError("a lockfile mismatch must not trigger a repair")

        monkeypatch.setattr(npm_engine, "_pm_npm", explode)
        npm = tmp_path / "npm"
        npm.write_text("#!/bin/sh\n", encoding="utf-8")
        assert not maybe_repair_npm_engine(str(npm), ELOCK_OUTPUT, quiet=True)

    def test_node_only_mismatch_on_foreign_npm_still_provisions(
        self, tmp_path, monkeypatch
    ):
        """A too-old system NODE can't be fixed by any npm upgrade, but the
        pm store ships a supported Node — provisioning covers it."""
        system_npm = tmp_path / "usr-bin-npm"
        system_npm.write_text("#!/bin/sh\n", encoding="utf-8")
        node_only = (
            "npm error code EBADENGINE\n"
            'npm error notsup Required: {"node":">=20.0.0"}\n'
            'npm error notsup Actual:   {"npm":"10.9.8","node":"v18.0.0"}\n'
        )
        managed = tmp_path / "store" / "npm-x" / "npm"
        managed.parent.mkdir(parents=True)
        managed.write_text("#!/bin/sh\n", encoding="utf-8")

        monkeypatch.setattr(
            npm_engine, "_pm_npm", lambda quiet=False: str(managed)
        )
        repaired = maybe_repair_npm_engine(str(system_npm), node_only, quiet=True)
        assert repaired == str(managed)


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
