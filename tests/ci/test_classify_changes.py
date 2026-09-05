"""Tests for scripts/ci/classify_changes.py.

Check some common patterns of file modifications and the CI lanes they should run.
We should always fail open. We may run a lane we didn't need, never skip one a
change could have broken.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "classify_changes.py"
_spec = importlib.util.spec_from_file_location("classify_changes", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load classify_changes.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
classify = _mod.classify
ci_review_files = _mod.ci_review_files
pull_request_changed_files = _mod.pull_request_changed_files
main = _mod.main

DEFAULT = {
    "python": True,
    "python_prod": True,
    "frontend": True,
    "docker": True,
    "docker_meta": True,
    "nix": True,
    "site": True,
    "scan": True,
    "deps": True,
    "uv_lock": True,
    "npm_lock": True,
    "installer": True,
    "desktop_updater": True,
    "rust": True,
    "mcp_catalog": False,
    "ci_review": True,
}


def _lanes(python=False, frontend=False, site=False, scan=False, deps=False, uv_lock=False, npm_lock=False, installer=False, desktop_updater=False, rust=False, mcp_catalog=False, docker_meta=False, ci_review=False, python_prod=None, nix=None, docker=None) -> dict[str, bool]:
    # python_prod tracks python except for tests-only diffs; default it to
    # python so the majority of cases don't need to spell it out.
    #
    # docker and nix are derived: both build the product, so both ride on
    # python_prod and frontend. The image ships the built web assets, and the
    # flake bundles the compiled ui-tui. Pass either explicitly to override.
    _python_prod = python if python_prod is None else python_prod
    _product = _python_prod or frontend
    return {
        "python": python,
        "python_prod": _python_prod,
        "docker": (docker_meta or _product) if docker is None else docker,
        "nix": _product if nix is None else nix,
        "frontend": frontend,
        "docker_meta": docker_meta,
        "site": site,
        "scan": scan,
        "deps": deps,
        "uv_lock": uv_lock,
        "npm_lock": npm_lock,
        "installer": installer,
        "desktop_updater": desktop_updater,
        "rust": rust,
        "mcp_catalog": mcp_catalog,
        "ci_review": ci_review,
    }


CASES = {
    "docs-only → nothing heavy": (["README.md", "docs/guide.md"], _lanes()),
    "python source → python": (["run_agent.py"], _lanes(python=True, scan=True)),
    # pyproject.toml declares the pytest markers the OS lanes select on, so it
    # also re-arms the desktop_updater integration tests (fail-open).
    "dep manifest → python": (["pyproject.toml"], _lanes(python=True, scan=True, deps=True, uv_lock=True, desktop_updater=True)),
    "uv.lock → python": (["uv.lock"], _lanes(python=True, uv_lock=True)),
    "ts package → frontend": (["apps/desktop/src/app.tsx"], _lanes(frontend=True)),
    "ui-tui → frontend": (["ui-tui/src/entry.ts"], _lanes(frontend=True)),
    # Lockfile bump shifts every TS package's tree, but not the Python suite.
    "root lockfile → frontend, not python": (["package-lock.json"], _lanes(frontend=True, npm_lock=True)),
    "nested lockfile → npm_lock": (["website/package-lock.json"], _lanes(site=True, npm_lock=True)),
    # A website file the Python suite cannot read stays site-only.
    "website config → site": (["website/docusaurus.config.ts"], _lanes(site=True)),
    # uv lock --check re-resolves against PyPI, so it must stay off for any
    # diff that can't desync the lockfile — a registry blip on a docs PR
    # otherwise shows up as a blocking "uv.lock out of sync" red X.
    "docs → no uv_lock": (
        ["website/docs/developer-guide/plugins/index.md"],
        _lanes(python=True, site=True),
    ),
    "frontend → no uv_lock": (["apps/desktop/src/store/profile.ts"], _lanes(frontend=True)),
    # The published CIMD document is asserted about by the Python suite, so a
    # lone edit there must not skip the lane that would catch a bad edit.
    "cimd document → python + site": (
        ["website/static/oauth/client-metadata.json"],
        _lanes(python=True, site=True),
    ),
    # A new docs page must reach llms.txt, and the generator that puts it there
    # has its own tests. Skipping Python on either is how the index drifted to
    # 53% coverage while every PR stayed green.
    "docs page → python + site": (
        ["website/docs/user-guide/bot-mode.md"],
        _lanes(python=True, site=True),
    ),
    "docs generator → python + site": (
        ["website/scripts/generate-llms-txt.py"],
        _lanes(python=True, scan=True, site=True),
    ),
    # SKILL.md reads like docs, but the skill-doc tests read skills/, so a
    # skill edit must still run Python.
    "skill md → python + site": (["skills/github/SKILL.md"], _lanes(python=True, site=True)),
    "dockerfile → docker meta": (["Dockerfile"], _lanes(docker_meta=True)),
    # Only the flake reads these, so they run nix alone. No Python test opens
    # them, unlike pyproject.toml and uv.lock below.
    "nix module → nix only": (["nix/homeManagerModules.nix"], _lanes(nix=True)),
    "flake.nix → nix only": (["flake.nix"], _lanes(nix=True)),
    "flake.lock → nix only": (["flake.lock"], _lanes(nix=True)),
    # A flake-only file must not mask a Python change beside it.
    "nix + python → both": (["nix/checks.nix", "agent/x.py"], _lanes(python=True, scan=True)),
    # Nine checks run the built binary, so product Python is a nix input even
    # when the diff touches no file under nix/.
    "product python → nix": (["hermes_cli/config.py"], _lanes(python=True, scan=True)),
    # tests/ is not packaged, so the built binary cannot change.
    "tests-only → no nix": (
        ["tests/agent/test_foo.py"],
        _lanes(python=True, python_prod=False, scan=True),
    ),
    # Prose cannot change the closure or the binary.
    "docs-only → no nix": (["README.md"], _lanes()),
    # install.ps1 is a shell script Python never imports, but it's also not
    # provably prose, so python stays on (fail-open) alongside the Windows lane.
    "install.ps1 → installer": (["scripts/install.ps1"], _lanes(python=True, installer=True)),
    "installer test → installer": (
        ["scripts/tests/test-install-ps1-longpath.ps1"],
        _lanes(python=True, installer=True),
    ),
    "python source alone → no installer lane": (["run_agent.py"], _lanes(python=True, scan=True)),
    # The Windows desktop-update hand-off is a PowerShell integration surface:
    # its tests spawn the real script and poll its loopback server. They run
    # when the script, the Electron side that launches it, or their own test
    # files change — not on every hermes_state.py PR.
    "windows.ps1 → desktop_updater": (
        ["scripts/desktop-update/windows.ps1"],
        _lanes(python=True, desktop_updater=True),
    ),
    "desktop-update test → desktop_updater": (
        ["tests/test_desktop_update_windows_progress.py"],
        _lanes(python=True, python_prod=False, scan=True, desktop_updater=True),
    ),
    "updater-process.ts → desktop_updater": (
        ["apps/desktop/electron/updater-process.ts"],
        _lanes(frontend=True, desktop_updater=True),
    ),
    "python source alone → no desktop_updater lane": (["hermes_state.py"], _lanes(python=True, scan=True)),
    # `.rs` lives under apps/, so it matches `frontend` too. That lane builds
    # TypeScript and cannot notice a Rust error — before `rust` existed it was
    # the ONLY lane a Rust change ran, and the crate's tests never executed.
    "rust source → rust": (
        ["apps/bootstrap-installer/src-tauri/src/powershell.rs"],
        _lanes(frontend=True, rust=True),
    ),
    "cargo lockfile → rust": (
        ["apps/bootstrap-installer/src-tauri/Cargo.lock"],
        _lanes(frontend=True, rust=True),
    ),
    # Non-.rs files in the crate still change what cargo builds.
    "tauri config → rust": (
        ["apps/bootstrap-installer/src-tauri/tauri.conf.json"],
        _lanes(frontend=True, rust=True),
    ),
    "ts source alone → no rust lane": (
        ["apps/bootstrap-installer/src/main.tsx"],
        _lanes(frontend=True),
    ),
    # Unknown top-level file keeps Python on rather than risk a silent skip.
    "unknown toplevel → python": (["Makefile"], _lanes(python=True)),
    "mixed docs+python → python": (["README.md", "agent/x.py"], _lanes(python=True, scan=True)),
    "mixed docs+frontend → frontend": (["README.md", "apps/x.tsx"], _lanes(frontend=True)),
    # tests-only diffs: pytest lanes stay ON, product jobs (Desktop E2E,
    # Docker) gate on python_prod and skip.
    "tests-only → python without python_prod": (
        ["tests/agent/test_foo.py"],
        _lanes(python=True, python_prod=False, scan=True),
    ),
    # conftest.py owns the _OS_MARKS skip logic, so it re-arms the
    # desktop_updater integration tests too (fail-open).
    "conftest → python + desktop_updater": (
        ["tests/conftest.py"],
        _lanes(python=True, python_prod=False, scan=True, desktop_updater=True),
    ),
    "tests + prod source → both lanes": (
        ["tests/agent/test_foo.py", "agent/x.py"],
        _lanes(python=True, scan=True),
    ),
    # Runner infrastructure is NOT tests-only — a bad runner edit can mask
    # real failures, so it keeps the conservative full lane set.
    "test runner script → python_prod stays on": (
        ["scripts/run_tests_parallel.py"],
        _lanes(python=True, scan=True),
    ),
    # Supply-chain lanes
    ".pth file → scan": (["evil.pth"], _lanes(python=True, scan=True)),
    "setup.py → scan": (["setup.py"], _lanes(python=True, scan=True)),
    "mcp catalog manifest → mcp_catalog": (
        ["optional-mcps/foo/manifest.yaml"],
        _lanes(python=True, mcp_catalog=True),
    ),
    "mcp_catalog.py → mcp_catalog": (
        ["hermes_cli/mcp_catalog.py"],
        _lanes(python=True, scan=True, mcp_catalog=True),
    ),
    # CI-sensitive files require explicit review label.
    "eslint config → ci_review": (
        ["apps/desktop/eslint.config.mjs"],
        _lanes(frontend=True, ci_review=True),
    ),
    "shared eslint config → ci_review": (
        ["eslint.config.shared.mjs"],
        _lanes(python=True, ci_review=True),
    ),
    "ui-tui eslint config → ci_review": (
        ["ui-tui/eslint.config.mjs"],
        _lanes(frontend=True, ci_review=True),
    ),
    "web eslint config → ci_review": (
        ["web/eslint.config.js"],
        _lanes(frontend=True, ci_review=True),
    ),
    "shared package eslint config → ci_review": (
        ["apps/shared/eslint.config.mjs"],
        _lanes(frontend=True, ci_review=True),
    ),
    "bootstrap-installer eslint config → ci_review": (
        ["apps/bootstrap-installer/eslint.config.mjs"],
        _lanes(frontend=True, ci_review=True),
    ),
    "prettier config → ci_review": (
        [".prettierrc"],
        _lanes(python=True, ci_review=True),
    ),
    "workflow yml → ci_review (also fail-open all)": (
        [".github/workflows/typecheck.yml"],
        DEFAULT,
    ),
    "composite action → ci_review (also fail-open all)": (
        [".github/actions/retry/action.yml"],
        DEFAULT,
    ),
    # Normal desktop source doesn't trigger ci_review.
    "desktop src → no ci_review": (
        ["apps/desktop/src/app.tsx"],
        _lanes(frontend=True),
    ),
    # tests-js is a declared npm workspace and js-tests.yml checks it (it even
    # caches tests-js/node_modules), so a broken spec there must not ride in
    # under a skipped lane.
    "tests-js spec → frontend": (
        ["tests-js/node-engine-alignment.test.ts"],
        _lanes(frontend=True),
    ),
    # detect-changes runs this script from the PR's own checkout, so it decides
    # whether the ci_review gate fires at all.
    "classifier → ci_review": (
        ["scripts/ci/classify_changes.py"],
        _lanes(python=True, scan=True, ci_review=True),
    ),
    # Fail open: CI-config / empty / blank diffs run everything.
    ".github change → all": ([".github/workflows/tests.yml"], DEFAULT),
    "action change → all": ([".github/actions/detect-changes/action.yml"], DEFAULT),
    "empty diff → all": ([], DEFAULT),
    "blank lines → all": (["", "  "], DEFAULT),
}


@pytest.mark.parametrize("files,expected", CASES.values(), ids=CASES.keys())
def test_classify(files, expected):
    assert classify(files) == expected


_REPO = Path(__file__).resolve().parents[2]


def _yaml(rel: str) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((_REPO / rel).read_text(encoding="utf-8"))


def test_every_lane_reaches_the_composite_action():
    """The action is the one surface every consumer reads, so it must carry all
    of them — ci.yaml, nix.yml and docker.yml each re-export a different subset.
    """
    lanes = set(classify(["run_agent.py"]))
    action_outputs = set(_yaml(".github/actions/detect-changes/action.yml")["outputs"])
    assert lanes - action_outputs == set(), "lane(s) missing from the composite action's outputs"


def test_ci_jobs_only_gate_on_detect_outputs_that_detect_actually_declares():
    """An ``if`` that reads an undeclared output resolves to the empty string.

    The lane then reports "skipping" on every PR, forever, and nothing goes red
    — there is no error for referencing an output a job never declared. That is
    exactly how the ``rust`` lane shipped dead: the classifier emitted it and
    the composite action re-exported it, but ci.yaml's ``detect`` job did not,
    so ``needs.detect.outputs.rust`` was never anything but "".
    """
    ci = _yaml(".github/workflows/ci.yaml")
    declared = set(ci["jobs"]["detect"]["outputs"])

    referenced: set[str] = set()
    for job in ci["jobs"].values():
        for expr in _iter_if_expressions(job):
            referenced.update(re.findall(r"needs\.detect\.outputs\.(\w+)", expr))

    assert referenced, "found no detect-gated jobs — the walk is broken, not the wiring"
    assert referenced - declared == set(), "job(s) gate on an output detect never declares"


def _iter_if_expressions(job: object):
    """Yield every ``if:`` string in a job, including inside its steps."""
    if not isinstance(job, dict):
        return
    if isinstance(cond := job.get("if"), str):
        yield cond
    for step in job.get("steps", []) or []:
        if isinstance(step, dict) and isinstance(cond := step.get("if"), str):
            yield cond


def _npm_workspace_dirs() -> list[str]:
    """Every workspace the root manifest declares, resolved against the tree."""
    manifest = json.loads((_REPO / "package.json").read_text(encoding="utf-8"))
    return [
        path.relative_to(_REPO).as_posix()
        for pattern in manifest["workspaces"]
        for path in sorted(_REPO.glob(pattern))
        if (path / "package.json").is_file()
    ]


def test_every_npm_workspace_selects_the_frontend_lane():
    """js-tests.yml discovers its work with ``npm query .workspace``, so every
    workspace the root manifest declares is checked there. A workspace the
    classifier does not map to ``frontend`` has its vitest/tsc/eslint run
    skipped, and a skipped job counts as success in the merge gate.
    """
    workspaces = _npm_workspace_dirs()
    assert workspaces, "root package.json declares no npm workspaces"
    missed = [d for d in workspaces if not classify([f"{d}/src/probe.ts"])["frontend"]]
    assert not missed, f"npm workspaces whose changes skip the js-tests lane: {missed}"


def test_the_classifier_the_gate_runs_is_itself_ci_sensitive():
    """``detect-changes`` executes this script out of the PR's own checkout,
    so it decides every lane — ``ci_review`` included. The composite action
    around it already requires the review label; the script it runs must too,
    or the gate is editable by the diff it is meant to gate.
    """
    assert classify([".github/actions/detect-changes/action.yml"])["ci_review"]
    rel = _PATH.relative_to(_REPO).as_posix()
    assert classify([rel])["ci_review"], f"{rel} does not require CI review"
    assert ci_review_files([rel]) == [rel]


def test_ci_review_files_returns_only_sensitive_paths_sorted_and_unique():
    assert ci_review_files([
        "apps/desktop/src/app.tsx",
        ".github/workflows/ci.yml",
        "apps/desktop/eslint.config.mjs",
        ".github/workflows/ci.yml",
    ]) == [
        ".github/workflows/ci.yml",
        "apps/desktop/eslint.config.mjs",
    ]


def _write_event(tmp_path, number: int | None = 88442) -> Path:
    payload = {"pull_request": {"number": number}} if number is not None else {}
    path = tmp_path / "event.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_pull_request_changed_files_skips_non_pr_events(monkeypatch):
    monkeypatch.setenv("EVENT_NAME", "push")
    monkeypatch.setenv("REPO", "NousResearch/hermes-agent")
    assert pull_request_changed_files() == []


def test_pull_request_changed_files_skips_without_pr_number(tmp_path, monkeypatch):
    monkeypatch.setenv("EVENT_NAME", "pull_request")
    monkeypatch.setenv("REPO", "NousResearch/hermes-agent")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(_write_event(tmp_path, number=None)))
    assert pull_request_changed_files() == []


def test_pull_request_changed_files_parses_gh_output(tmp_path, monkeypatch):
    monkeypatch.setenv("EVENT_NAME", "pull_request")
    monkeypatch.setenv("REPO", "NousResearch/hermes-agent")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(_write_event(tmp_path)))

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout="scripts/install.sh\ntests/test_install_sh_node_deps_workspaces.py\n",
            stderr="",
        )

    monkeypatch.setattr(_mod.subprocess, "run", fake_run)
    assert pull_request_changed_files() == [
        "scripts/install.sh",
        "tests/test_install_sh_node_deps_workspaces.py",
    ]


def test_pull_request_changed_files_returns_empty_when_gh_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("EVENT_NAME", "pull_request")
    monkeypatch.setenv("REPO", "NousResearch/hermes-agent")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(_write_event(tmp_path)))

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="gh: Not Found")

    monkeypatch.setattr(_mod.subprocess, "run", fake_run)
    assert pull_request_changed_files() == []


def test_main_recovers_pr_files_instead_of_fail_open_ci_review(monkeypatch, capsys):
    """A fork compare 404 must not demand ci-reviewed for a CLI-only install."""
    monkeypatch.setattr(
        _mod,
        "pull_request_changed_files",
        lambda: ["scripts/install.sh", "tests/test_install_sh_node_deps_workspaces.py"],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    assert main() == 0
    out = capsys.readouterr().out
    assert "ci_review=false" in out
    assert "python=true" in out
    assert "python_prod=true" in out


def test_main_still_fail_opens_when_recovery_is_empty(monkeypatch, capsys):
    monkeypatch.setattr(_mod, "pull_request_changed_files", lambda: [])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    assert main() == 0
    out = capsys.readouterr().out
    assert "ci_review=true" in out
