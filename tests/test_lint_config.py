"""Tests for ruff lint config — guards against accidental rule removal.

PLW1514 (unspecified-encoding) was enabled after a debug session on
Windows turned up three separate UTF-8 regressions in execute_code.
The rule catches bare ``open()`` / ``read_text()`` / ``write_text()``
calls that default to locale encoding — cp1252 on Windows — which
silently corrupts non-ASCII content.

These tests ensure:
  1. PLW1514 stays in ``[tool.ruff.lint.select]``
  2. The CI workflow's blocking step still invokes ``ruff check .``
  3. pyproject.toml has ``preview = true`` (required — PLW1514 is a
     preview rule in ruff 0.15.x)

If someone removes any of these, CI stops enforcing UTF-8-explicit
opens and we're back to the original Windows-regression trap.
"""

from __future__ import annotations

import pathlib

import pytest

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover — 3.10 and earlier
    import tomli as tomllib  # type: ignore

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


class TestRuffConfig:
    def test_plw1514_is_in_select_list(self):
        """pyproject.toml must keep PLW1514 in [tool.ruff.lint.select]."""
        cfg = _load_pyproject()
        selected = (
            cfg.get("tool", {})
            .get("ruff", {})
            .get("lint", {})
            .get("select", [])
        )
        assert "PLW1514" in selected, (
            "PLW1514 (unspecified-encoding) was removed from "
            "[tool.ruff.lint.select].  This rule blocks bare open() calls "
            "that default to locale encoding on Windows — removing it "
            "re-opens a class of UTF-8 bugs we already paid to close.  "
            "If you genuinely want to remove it, delete this test in the "
            "same commit so the intent is deliberate."
        )

    def test_preview_mode_enabled(self):
        """PLW1514 is a preview rule in ruff 0.15.x — preview=true is
        required for it to actually run."""
        cfg = _load_pyproject()
        ruff_cfg = cfg.get("tool", {}).get("ruff", {})
        assert ruff_cfg.get("preview") is True, (
            "[tool.ruff] preview=true is required — PLW1514 is a preview "
            "rule and silently becomes a no-op without it.  If this ever "
            "becomes a stable rule, you can drop preview=true but must "
            "verify PLW1514 still fires in a sample test run first."
        )


class TestLintWorkflow:
    WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "lint.yml"

    def test_workflow_exists(self):
        assert self.WORKFLOW_PATH.exists(), (
            f"CI workflow missing: {self.WORKFLOW_PATH}"
        )

    def test_workflow_has_blocking_ruff_step(self):
        """The workflow must run a blocking ``ruff check .`` step
        (one without --exit-zero) so violations fail the job."""
        content = self.WORKFLOW_PATH.read_text(encoding="utf-8")
        # Look for the blocking step's named line + its command.  We want
        # at least one ``ruff check .`` that does NOT have ``--exit-zero``
        # nearby.
        import re
        # Split into lines and find ruff check invocations
        lines = content.splitlines()
        found_blocking = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("ruff check") and "--exit-zero" not in stripped:
                # Also check it's not piped to `|| true` which would mask
                # the exit code.
                window = " ".join(lines[i:i + 3])
                if "|| true" not in window:
                    found_blocking = True
                    break
        assert found_blocking, (
            "lint.yml no longer contains a blocking ``ruff check .`` step "
            "(one without --exit-zero and not masked by || true).  "
            "Restore it — the PLW1514 rule is only useful if CI actually "
            "fails on violation."
        )

    def test_workflow_yaml_is_valid(self):
        """Workflow file must parse as valid YAML (can't ship a broken
        CI config to main)."""
        import yaml
        content = self.WORKFLOW_PATH.read_text(encoding="utf-8")
        try:
            parsed = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            pytest.fail(f"lint.yml is not valid YAML: {exc}")
        assert isinstance(parsed, dict)
        assert "jobs" in parsed


class TestDevDependencies:
    """Guard against dev-dependency omissions that break the canonical test runner.

    pytest-split was missing from the dev extra, causing scripts/run_tests.sh
    to attempt a runtime `pip install` that fails in uv-created venvs without
    pip (issue #22401).
    """

    def test_pytest_split_declared_in_dev_extra(self):
        """pytest-split must appear in [project.optional-dependencies].dev."""
        cfg = _load_pyproject()
        dev_deps = (
            cfg.get("project", {})
            .get("optional-dependencies", {})
            .get("dev", [])
        )
        assert any("pytest-split" in dep for dep in dev_deps), (
            "pytest-split is missing from [project.optional-dependencies].dev. "
            "scripts/run_tests.sh requires it; without the declaration a fresh "
            "`uv sync --extra dev` will not install it and the runtime bootstrap "
            "in the script will attempt `pip install` which fails in uv-managed "
            "venvs that do not include pip (issue #22401)."
        )


class TestRunTestsShScript:
    """Guard against environment contamination in scripts/run_tests.sh.

    HERMES_CRON_SESSION was not being unset, causing approval tests to see
    cron-deny behavior when the test runner was invoked from a cron job
    (issue #22400).
    """

    SCRIPT_PATH = REPO_ROOT / "scripts" / "run_tests.sh"

    def test_hermes_cron_session_is_unset(self):
        """HERMES_CRON_SESSION must appear in the unset list so cron-invoked
        runs do not contaminate approval-gate tests."""
        content = self.SCRIPT_PATH.read_text(encoding="utf-8")
        assert "HERMES_CRON_SESSION" in content, (
            "HERMES_CRON_SESSION is not unset in scripts/run_tests.sh. "
            "When the test runner is invoked from a Hermes cron job this var "
            "leaks into pytest and switches approval logic to cron-deny mode, "
            "making approval tests fail (issue #22400)."
        )

    def test_bootstrap_supports_uv_pip(self):
        """The pytest-split bootstrap must try `uv pip install` before falling
        back to `python -m pip` so that uv-created venvs without pip work."""
        content = self.SCRIPT_PATH.read_text(encoding="utf-8")
        assert "uv pip install" in content, (
            "scripts/run_tests.sh pytest-split bootstrap does not attempt "
            "`uv pip install`. In uv-created venvs without pip the fallback "
            "`python -m pip install` fails immediately (issue #22401). "
            "Add a `uv pip install` path guarded by `command -v uv`."
        )
