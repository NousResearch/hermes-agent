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
pull_request_labels = _mod.pull_request_labels
main = _mod.main

DEFAULT = {
    "python": True,
    "python_prod": True,
    "frontend": True,
    "docker": True,
    "docker_meta": True,
    "nix": True,
    "e2e": True,
    "e2e_upgrade": True,
    "e2e_desktop_core": True,
    "e2e_desktop_update": True,
    "site": True,
    "scan": True,
    "deps": True,
    "uv_lock": True,
    "npm_lock": True,
    "bootstrap": True,
    "desktop_updater": True,
    "rust": True,
    "mcp_catalog": False,
    "ci_review": True,
}

SLOW_LANES = {"docker", "nix", "e2e", "e2e_upgrade", "e2e_desktop_core", "e2e_desktop_update"}


def _lanes(python=False, frontend=False, site=False, scan=False, deps=False, uv_lock=False, npm_lock=False, bootstrap=False, desktop_updater=False, rust=False, mcp_catalog=False, docker_meta=False, ci_review=False, python_prod=None, nix=False, docker=None, e2e=False, e2e_upgrade=False, e2e_desktop_core=False, e2e_desktop_update=False) -> dict[str, bool]:
    # python_prod tracks python except for tests-only diffs; default it to
    # python so the majority of cases don't need to spell it out.
    #
    # The slow lanes (docker, nix, the E2E suites) are off unless the case
    # names them: an ordinary product change reaches them on main, not on
    # the PR. Docker meta files are docker-lane paths, so docker follows
    # docker_meta unless the case says otherwise.
    _python_prod = python if python_prod is None else python_prod
    return {
        "python": python,
        "python_prod": _python_prod,
        "docker": docker_meta if docker is None else docker,
        "nix": nix,
        "e2e": e2e,
        "e2e_upgrade": e2e_upgrade,
        "e2e_desktop_core": e2e_desktop_core,
        "e2e_desktop_update": e2e_desktop_update,
        "frontend": frontend,
        "docker_meta": docker_meta,
        "site": site,
        "scan": scan,
        "deps": deps,
        "uv_lock": uv_lock,
        "npm_lock": npm_lock,
        "bootstrap": bootstrap,
        "desktop_updater": desktop_updater,
        "rust": rust,
        "mcp_catalog": mcp_catalog,
        "ci_review": ci_review,
    }


CASES = {
    "shared JS builder → frontend": (["scripts/build/web.mjs"], _lanes(python=True, frontend=True)),
    "root JS tests → frontend": (["tests-js/product-builders.test.mjs"], _lanes(python=True, frontend=True)),
    "docs-only → nothing heavy": (["README.md", "docs/guide.md"], _lanes()),
    "python source → python": (["run_agent.py"], _lanes(python=True, scan=True)),
    # pyproject.toml declares the pytest markers the OS lanes select on, so it
    # also re-arms the desktop_updater integration tests (fail-open).
    # The dependency manifests are inputs to the image, the flake and every
    # install, so they start those slow lanes on the PR itself.
    "dep manifest → python": (
        ["pyproject.toml"],
        _lanes(python=True, scan=True, deps=True, uv_lock=True, desktop_updater=True, docker=True, nix=True, e2e_upgrade=True),
    ),
    "uv.lock → python": (["uv.lock"], _lanes(python=True, uv_lock=True, docker=True, nix=True, e2e_upgrade=True)),
    "ts package → frontend": (["apps/desktop/src/app.tsx"], _lanes(frontend=True)),
    "ui-tui → frontend": (["ui-tui/src/entry.ts"], _lanes(frontend=True)),
    # Lockfile bump shifts every TS package's tree, but not the Python suite.
    "root lockfile → frontend, not python": (["package-lock.json"], _lanes(frontend=True, npm_lock=True, docker=True, nix=True)),
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
    # Cross-language contract JSON under apps/: the pytest that pins it against
    # the Python side must run even when nothing else in the PR is Python.
    "generated gateway contract → python + frontend": (
        ["apps/shared/src/gateway-contract.generated.ts"],
        _lanes(python=True, frontend=True, e2e_desktop_core=True),
    ),
    "gateway OpenRPC document → python + frontend": (
        ["apps/shared/src/gateway-contract.openrpc.json"],
        _lanes(python=True, frontend=True, e2e_desktop_core=True),
    ),
    "desktop slash-registry JSON → python + frontend": (
        ["apps/desktop/src/lib/desktop-slash-registry.json"],
        _lanes(python=True, frontend=True),
    ),
    "desktop card-tool names → python + frontend": (
        ["apps/desktop/src/lib/tool-render-class.ts"],
        _lanes(python=True, frontend=True),
    ),
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
    "nix + python → both": (["nix/checks.nix", "agent/x.py"], _lanes(python=True, scan=True, nix=True)),
    # Product Python can still break the flake (nine checks run the built
    # binary), but the flake meets it on main, not on the PR.
    "product python → no nix on the PR": (["hermes_cli/config.py"], _lanes(python=True, scan=True)),
    # tests/ is not packaged, so the built binary cannot change.
    "tests-only → no nix": (
        ["tests/agent/test_foo.py"],
        _lanes(python=True, python_prod=False, scan=True),
    ),
    # Prose cannot change the closure or the binary.
    "docs-only → no nix": (["README.md"], _lanes()),
    # install.ps1 and its PowerShell suites are exercised by platforms("windows")
    # pytest files, so they must turn on python (which gates tests-os).
    "install.ps1 → python + e2e_upgrade": (["scripts/install.ps1"], _lanes(python=True, e2e_upgrade=True)),
    "installer suite → python": (["scripts/tests/test-install-ps1-longpath.ps1"], _lanes(python=True)),
    # The Windows desktop-update hand-off is a PowerShell integration surface:
    # its tests spawn the real script and poll its loopback server. They run
    # when the script, the Electron side that launches it, or their own test
    # files change — not on every hermes_state.py PR.
    "windows.ps1 → desktop_updater": (
        ["scripts/desktop-update/windows.ps1"],
        _lanes(python=True, desktop_updater=True, e2e_desktop_update=True),
    ),
    # The shipped updater page is exercised by the desktop Electron suite;
    # a page-only change must run that suite as well as the server tests.
    "updater ui.html → frontend + desktop_updater": (
        ["scripts/desktop-update/ui.html"],
        _lanes(python=True, frontend=True, desktop_updater=True, e2e_desktop_update=True),
    ),
    "desktop-update test → desktop_updater": (
        ["tests/scripts/desktop_update/test_desktop_update_windows_progress.py"],
        _lanes(python=True, python_prod=False, scan=True, desktop_updater=True),
    ),
    "updater-process.ts → desktop_updater": (
        ["apps/desktop/electron/updater-process.ts"],
        _lanes(frontend=True, desktop_updater=True, e2e_desktop_update=True),
    ),
    "python source alone → no desktop_updater lane": (["gateway/run.py"], _lanes(python=True, scan=True)),
    # `.rs` lives under apps/, so it matches `frontend` too. That lane builds
    # TypeScript and cannot notice a Rust error — before `rust` existed it was
    # the ONLY lane a Rust change ran, and the crate's tests never executed.
    "rust source → rust": (
        ["apps/bootstrap-installer/src-tauri/src/powershell.rs"],
        _lanes(frontend=True, bootstrap=True, rust=True),
    ),
    "cargo lockfile → rust": (
        ["apps/bootstrap-installer/src-tauri/Cargo.lock"],
        _lanes(frontend=True, bootstrap=True, rust=True),
    ),
    # Non-.rs files in the crate still change what cargo builds.
    "tauri config → rust": (
        ["apps/bootstrap-installer/src-tauri/tauri.conf.json"],
        _lanes(frontend=True, bootstrap=True, rust=True),
    ),
    "ts source alone → no rust lane": (
        ["apps/bootstrap-installer/src/main.tsx"],
        _lanes(frontend=True, bootstrap=True),
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
    # The shared harness can break every Python E2E suite.
    "conftest → python + desktop_updater + python e2e": (
        ["tests/conftest.py"],
        _lanes(python=True, python_prod=False, scan=True, desktop_updater=True, e2e=True, e2e_upgrade=True),
    ),
    "conftest fixture module → python + desktop_updater + python e2e": (
        ["tests/_fixtures/platform_gating.py"],
        _lanes(python=True, python_prod=False, scan=True, desktop_updater=True, e2e=True, e2e_upgrade=True),
    ),
    "tests + prod source → both lanes": (
        ["tests/agent/test_foo.py", "agent/x.py"],
        _lanes(python=True, scan=True),
    ),
    # Runner infrastructure is NOT tests-only — a bad runner edit can mask
    # real failures, so it keeps the conservative full lane set. The .py
    # runner additionally trips the supply-chain scan lane (executable
    # .py/.pth payloads are what it scans for).
    "test runner script → python_prod stays on": (
        ["scripts/run_tests_parallel.py"],
        _lanes(python=True, scan=True, e2e=True, e2e_upgrade=True),
    ),
    # Supply-chain lanes
    ".pth file → scan": (["evil.pth"], _lanes(python=True, scan=True)),
    "setup.py → scan": (["setup.py"], _lanes(python=True, scan=True, docker=True, nix=True, e2e_upgrade=True)),
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
        _lanes(frontend=True, bootstrap=True, ci_review=True),
    ),
    "prettier config → ci_review": (
        [".prettierrc"],
        _lanes(python=True, ci_review=True),
    ),
    "workflow yml → ci_review (also fail-open all)": (
        [".github/workflows/typecheck.yml"],
        DEFAULT,
    ),
    # The bootstrap installer lane: shell installer, dev-checkout wrapper,
    # and the Tauri app's non-Rust sources.
    "install.sh → bootstrap lane": (
        ["scripts/install.sh"],
        _lanes(python=True, bootstrap=True, python_prod=True, e2e_upgrade=True, e2e_desktop_update=True),
    ),
    "setup-hermes.sh → bootstrap lane": (
        ["setup-hermes.sh"],
        _lanes(python=True, bootstrap=True, python_prod=True, e2e_upgrade=True),
    ),
    "tauri installer source → bootstrap + rust": (
        ["apps/bootstrap-installer/src-tauri/src/lib.rs"],
        _lanes(frontend=True, bootstrap=True, rust=True),
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
    # Slow lanes on a pull request: an ordinary product change starts none of
    # them; the suite, its harness, or the code it guards starts its own.
    "gateway python → no slow lane": (["gateway/run.py", "agent/x.py"], _lanes(python=True, scan=True)),
    "desktop renderer → no slow lane": (["apps/desktop/src/app/chat/composer.tsx"], _lanes(frontend=True)),
    "python e2e suite → e2e only": (
        ["tests/e2e/core/sqlite/test_torture.py"],
        _lanes(python=True, python_prod=False, scan=True, e2e=True),
    ),
    "state db → e2e": (["hermes_state_wal.py"], _lanes(python=True, scan=True, e2e=True)),
    "upgrade suite → e2e_upgrade, not e2e": (
        ["tests/e2e/core/upgrade/pm/test_pm_lifecycle.py"],
        _lanes(python=True, python_prod=False, scan=True, e2e_upgrade=True),
    ),
    "updater → e2e_upgrade + desktop update": (
        ["hermes_cli/update_cmd_git.py"],
        _lanes(python=True, scan=True, e2e_upgrade=True, e2e_desktop_update=True),
    ),
    "PM → e2e_upgrade + docker": (["pm/environments.py"], _lanes(python=True, scan=True, e2e_upgrade=True, docker=True)),
    "desktop backend spawn → desktop core": (
        ["apps/desktop/electron/backend-child.ts"],
        _lanes(frontend=True, e2e_desktop_core=True),
    ),
    "desktop core spec → desktop core": (
        ["apps/desktop/e2e/core/transcript-integrity.spec.ts"],
        _lanes(frontend=True, e2e_desktop_core=True),
    ),
    "desktop update spec → desktop update, not core": (
        ["apps/desktop/e2e/update/app-update.spec.ts"],
        _lanes(frontend=True, e2e_desktop_update=True),
    ),
    # The update suite runs on the core suite's harness.
    "desktop core harness → both desktop suites": (
        ["apps/desktop/e2e/core/harness.ts"],
        _lanes(frontend=True, e2e_desktop_core=True, e2e_desktop_update=True),
    ),
    "image tests → docker": (
        ["tests/docker/test_image_smoke.py"],
        _lanes(python=True, python_prod=False, scan=True, docker=True),
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


@pytest.mark.parametrize("files", [["README.md"], ["gateway/run.py"], ["apps/desktop/src/app.tsx"]])
def test_run_e2e_label_turns_every_slow_lane_on_and_nothing_else(files):
    labelled = classify(files, run_e2e=True)
    assert {lane for lane in SLOW_LANES if labelled[lane]} == SLOW_LANES
    unlabelled = classify(files)
    assert {k: v for k, v in labelled.items() if k not in SLOW_LANES} == \
        {k: v for k, v in unlabelled.items() if k not in SLOW_LANES}


def test_every_slow_lane_path_matches_a_tracked_file():
    """A renamed file silently stops starting its lane on pull requests.

    Every prefix in the slow-lane tables must still match something in the
    tree, or the lane only ever runs on main.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=_REPO, capture_output=True, text=True, check=False,
    )
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    files = tracked.stdout.splitlines()
    tables = {**_mod._E2E_LANES, "docker": _mod._DOCKER_PATHS, "nix": _mod._NIX_LANE_PATHS}
    dead = {
        (lane, prefix)
        for lane, prefixes in tables.items()
        for prefix in prefixes
        if not any(f.startswith(prefix) for f in files)
    }
    assert dead == set()


_REPO = Path(__file__).resolve().parents[2]


def _yaml(rel: str) -> dict:
    yaml = pytest.importorskip("hermes_yaml")
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
    """Yield every ``if:`` string in a job, including inside its steps, plus
    every reusable-workflow ``with:`` value (an undeclared output passed as an
    input is the same silent "")."""
    if not isinstance(job, dict):
        return
    if isinstance(cond := job.get("if"), str):
        yield cond
    for value in (job.get("with") or {}).values():
        if isinstance(value, str):
            yield value
    for step in job.get("steps", []) or []:
        if isinstance(step, dict) and isinstance(cond := step.get("if"), str):
            yield cond


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
            stdout="scripts/install.sh\ntests/scripts/install/test_install_sh_node_deps_workspaces.py\n",
            stderr="",
        )

    monkeypatch.setattr(_mod.subprocess, "run", fake_run)
    assert pull_request_changed_files() == [
        "scripts/install.sh",
        "tests/scripts/install/test_install_sh_node_deps_workspaces.py",
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
        lambda: ["scripts/install.sh", "tests/scripts/install/test_install_sh_node_deps_workspaces.py"],
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


def test_main_reads_the_run_e2e_label(monkeypatch, capsys):
    monkeypatch.setattr(_mod, "pull_request_labels", lambda: ["ci-reviewed", _mod.RUN_E2E_LABEL])
    monkeypatch.setattr(sys, "stdin", io.StringIO("gateway/run.py\n"))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    assert main() == 0
    out = capsys.readouterr().out
    assert all(f"{lane}=true" in out for lane in SLOW_LANES)


def test_pull_request_labels_prefers_the_live_labels_over_the_replayed_event(tmp_path, monkeypatch):
    """A rerun replays the push's payload; a label added since must still count."""
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 7, "labels": []}}), encoding="utf-8")
    monkeypatch.setenv("EVENT_NAME", "pull_request")
    monkeypatch.setenv("REPO", "NousResearch/hermes-agent")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))

    live = subprocess.CompletedProcess([], 0, stdout="run-e2e\n", stderr="")
    monkeypatch.setattr(_mod.subprocess, "run", lambda *a, **k: live)
    assert pull_request_labels() == ["run-e2e"]

    failed = subprocess.CompletedProcess([], 1, stdout="", stderr="gh: Not Found")
    monkeypatch.setattr(_mod.subprocess, "run", lambda *a, **k: failed)
    event.write_text(json.dumps({"pull_request": {"number": 7, "labels": [{"name": "run-e2e"}]}}), encoding="utf-8")
    assert pull_request_labels() == ["run-e2e"]
