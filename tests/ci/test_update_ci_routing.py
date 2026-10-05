"""Update-path CI routing: ownership-derived selectors and the full dispatch chain.

Two failure modes this file pins (independent review R15):

* **Selector drift.** A hand list of "update files" misses whatever the update
  modules import next (``_subprocess_compat``, ``desktop_build_lock``,
  ``memory_provider_migration``, the ``scripts/build`` compilers). Here the
  update entry points' real imports are read from the source (Python AST, ESM
  ``import``/``step()`` literals) and every module they reach must start the
  suite that runs it, or be one of the classifier's declared shared hubs.
* **Dead dispatch.** A lane the classifier sets can still never reach its job:
  ``desktop_updater=true`` with ``python=false`` used to skip ``tests-os``
  entirely. Here the REAL classifier's output is replayed through the REAL
  workflow files (composite action -> ``ci.yaml`` detect outputs -> job
  ``if:``/``with:`` -> the called workflow's job ``if:`` -> the rendered step),
  so a gate that does not consume a lane fails here, not on a PR.
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from tests.ci import _gha_expr as gha
from tests.ci import workflow_steps

_REPO = Path(__file__).resolve().parents[2]
_CLASSIFIER = _REPO / "scripts" / "ci" / "classify_changes.py"
_spec = importlib.util.spec_from_file_location("classify_changes_routing", _CLASSIFIER)
assert _spec is not None and _spec.loader is not None
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


# -- the real classifier, as CI runs it ---------------------------------------------


def _real_classifier(paths: list[str]) -> dict[str, bool]:
    """``scripts/ci/classify_changes.py`` as a process, stdin -> ``key=value`` lines."""
    env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_OUTPUT", "GITHUB_EVENT_PATH")}
    env["EVENT_NAME"] = "pull_request_replay"  # never `pull_request`: no gh calls from a test
    out = subprocess.run([sys.executable, str(_CLASSIFIER)], input="\n".join(paths) + "\n",
                         capture_output=True, text=True, env=env, check=True, timeout=60).stdout
    pairs = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return {k: v == "true" for k, v in pairs.items()}


# -- workflow replay ---------------------------------------------------------------------


@lru_cache(maxsize=None)
def _yaml(rel: str) -> dict:
    yaml = pytest.importorskip("hermes_yaml")
    return yaml.safe_load((_REPO / rel).read_text(encoding="utf-8-sig"))


def _on(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True)) or {}


def _detect_outputs(lanes: dict[str, bool]) -> dict[str, Any]:
    """The classifier's lines -> the composite action's outputs -> ci.yaml ``detect`` outputs."""
    raw = {k: gha.to_string(v) for k, v in lanes.items()}
    action = _yaml(".github/actions/detect-changes/action.yml")
    action_out = {k: gha.render(v["value"], {"steps": {"classify": {"outputs": raw}}})
                  for k, v in action["outputs"].items()}
    detect = _yaml(".github/workflows/ci.yaml")["jobs"]["detect"]
    gate = next(s for s in detect["steps"] if s.get("id") == "gate-lanes")
    classify = next(s for s in detect["steps"] if s.get("id") == "classify")
    assert classify["uses"] == "./.github/actions/detect-changes"
    steps = {"classify": {"outputs": action_out}}
    ctx = {"steps": steps, "github": {"event_name": "pull_request"}, "inputs": {}}
    steps["gate-lanes"] = {"outputs": workflow_steps.outputs(gate, ctx)}
    return {k: gha.render(v, ctx) for k, v in detect["outputs"].items()}


def _inputs_for(called: str, given: dict[str, Any]) -> dict[str, Any]:
    declared = (_on(_yaml(called)).get("workflow_call") or {}).get("inputs") or {}
    unknown = set(given) - set(declared)
    assert not unknown, f"{called} is passed input(s) it never declares: {sorted(unknown)}"
    return {name: given.get(name, spec.get("default")) for name, spec in declared.items()}


def _run_workflow(rel: str, *, inputs: dict[str, Any], detect: dict[str, Any] | None = None) -> dict:
    """Which jobs of ``rel`` run on a pull request, recursing into called workflows.

    Returns ``{job: {"inputs": ..., "jobs": <child result>}}`` for every job that runs.
    ``detect`` (ci.yaml only) is the replayed ``detect`` job's outputs.
    """
    jobs = _yaml(rel)["jobs"]
    ran: dict[str, dict] = {}
    done: set[str] = set()
    pending = list(jobs)
    while pending:
        progressed = False
        for name in list(pending):
            body = jobs[name]
            needs = body.get("needs") or []
            needs = [needs] if isinstance(needs, str) else list(needs)
            if any(n not in done for n in needs):
                continue
            pending.remove(name)
            done.add(name)
            progressed = True
            if name == "detect" and detect is not None:
                ran[name] = {"outputs": detect}
                continue
            all_ran = all(n in ran for n in needs)
            ctx = {
                "inputs": inputs,
                "github": {"event_name": "pull_request", "ref_type": "branch"},
                "needs": {n: {"outputs": (ran.get(n) or {}).get("outputs", {}),
                              "result": "success" if n in ran else "skipped"} for n in needs},
                "__status__": {"always": True, "success": all_ran, "failure": False, "cancelled": False},
            }
            cond = body.get("if")
            uses_status = isinstance(cond, str) and re.search(r"\b(always|success|failure|cancelled)\(", cond)
            if not uses_status and not all_ran:
                continue  # implicit success(): a skipped dependency skips this job
            if not gha.condition(cond, ctx):
                continue
            entry: dict[str, Any] = {"ctx": ctx, "body": body}
            if name == "e2e-upgrade-plan":
                plan = next(s for s in body["steps"] if s.get("id") == "plan")
                entry["outputs"] = workflow_steps.outputs(plan, ctx, _REPO)
                assert json.loads(entry["outputs"]["shards"]), "upgrade plan selected no shards"
            uses = body.get("uses")
            if isinstance(uses, str) and uses.startswith("./.github/workflows/"):
                called = uses[2:]
                given = {k: gha.render(v, ctx) for k, v in (body.get("with") or {}).items()}
                entry["inputs"] = _inputs_for(called, given)
                entry["jobs"] = _run_workflow(called, inputs=entry["inputs"])
            ran[name] = entry
        assert progressed, f"{rel}: unresolvable needs among {pending}"
    return ran


def _ci_run(lanes: dict[str, bool]) -> dict:
    return _run_workflow(".github/workflows/ci.yaml", inputs={}, detect=_detect_outputs(lanes))


def _reached(run: dict, *path: str) -> dict | None:
    node: dict | None = {"jobs": run}
    for job in path:
        node = (node or {}).get("jobs", {}).get(job)
        if node is None:
            return None
    return node


def _selected_test_files(node: dict, name: str, *, windows_only: bool = False) -> set[str]:
    body, ctx = node["body"], node["ctx"]
    steps = [s for s in body["steps"] if s.get("name", "").startswith(name)]
    assert len(steps) == 1, f"missing/ambiguous required step: {name}"
    matrix = body.get("strategy", {}).get("matrix", {})
    if "shard" in matrix:
        cells = [{"shard": s} for s in gha.render(matrix["shard"], ctx)]
    else:
        cells = matrix.get("include", [{}])
    if windows_only:
        cells = [cell for cell in cells if cell.get("marker") == "windows"]
    assert cells, f"{name}: empty matrix"
    selected = set()
    for cell in cells:
        runner = gha.render(body["runs-on"], {**ctx, "matrix": cell})
        context = {**ctx, "matrix": cell, "runner": {
            "os": "Windows" if "windows" in runner else "Linux",
            "arch": "ARM64" if "arm" in runner else "X64",
        }}
        workflow_steps.required(body, context)
        selected |= workflow_steps.selected_files(steps[0], context, _REPO)
    return selected


def _windows_desktop_updater_tests_selected(run: dict) -> bool:
    """tests-os os-tests runs, and its Windows step keeps the desktop-update hand-off files."""
    os_tests = _reached(run, "tests-os", "os-tests")
    if os_tests is None:
        return False
    selected = _selected_test_files(os_tests, "Run ${{ matrix.marker }} tests", windows_only=True)
    expected = {p.relative_to(_REPO).as_posix() for p in
                (_REPO / "tests/scripts/desktop_update").glob("test_desktop_update_windows_*.py")}
    assert expected, "no Windows hand-off consumers exist"
    return expected <= selected


# Each real consumer of a lane, as a path of job names from ci.yaml down.
_CONSUMERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "e2e_upgrade": (
        ("tests", "e2e-upgrade-plan"),
        ("tests", "e2e-upgrade"),
        ("tests-os", "install-update-e2e", "install-update"),
    ),
    "e2e": (("tests", "e2e"), ("tests-os", "e2e-windows")),
    "e2e_desktop_update": (("e2e-desktop-update", "update"),),
}


def _consumers_reached(run: dict, lane: str) -> dict[str, bool]:
    if lane == "desktop_updater":
        return {"tests-os/os-tests[windows] keeps test_desktop_update_windows_*":
                _windows_desktop_updater_tests_selected(run)}
    reached = {}
    for path in _CONSUMERS[lane]:
        node = _reached(run, *path)
        if node is None or path[-1] == "e2e-upgrade-plan":
            reached["/".join(path)] = node is not None
            continue
        name = next(name for _, _, job, name in _REQUIRED_STEP_CASES if job == path[-1])
        reached["/".join(path)] = bool(_selected_test_files(node, name))
    return reached


_GATED = ("python", "desktop_updater", "e2e", "e2e_upgrade", "e2e_desktop_update")


def test_replay_evaluator_follows_actions_semantics():
    ctx = {"needs": {"detect": {"outputs": {"python": "false", "e2e": "true"}}}, "inputs": {"flag": True}}
    assert gha.condition("needs.detect.outputs.python == 'true' || needs.detect.outputs.e2e == 'true'", ctx)
    assert not gha.condition("needs.detect.outputs.python == 'true'", ctx)
    assert not gha.evaluate("'true' == true", ctx)  # a string output is not the boolean
    assert gha.render('x "${{ inputs.flag }}" y', ctx) == 'x "true" y'
    assert gha.render("${{ inputs.flag && '1' || '' }}", ctx) == "1"
    assert gha.render("${{ inputs.missing && '1' || '' }}", ctx) == ""
    with pytest.raises(gha.Unsupported):
        gha.condition("hashFiles('x') != ''", ctx)  # unmodelled: fail, never guess


@pytest.mark.parametrize("combo", list(itertools.product((False, True), repeat=len(_GATED))),
                         ids=lambda c: "-".join(n for n, on in zip(_GATED, c) if on) or "none")
def test_every_set_lane_reaches_its_consumer_whatever_the_other_lanes_say(combo):
    """The R15 hole was a parent ``if:`` keyed on ``python`` alone: a TypeScript-only
    change to the hand-off's marker reader set ``desktop_updater`` and nothing ran it."""
    lanes = {k: False for k in cc.classify(["README.md"])}
    lanes.update(zip(_GATED, combo))
    run = _ci_run(lanes)
    for lane in ("desktop_updater", "e2e", "e2e_upgrade", "e2e_desktop_update"):
        reached = _consumers_reached(run, lane)
        if lanes[lane]:
            assert all(reached.values()), f"{lane}=true but not dispatched: {reached}"
        elif lane != "desktop_updater":
            assert not any(reached.values()), f"{lane}=false but ran anyway: {reached}"


# One-file changes from the review's direct classifier run (acceptance/classifier.json),
# and the suites each must dispatch on a pull request.
_R15_CASES = {
    "scripts/build/desktop.mjs": ("e2e_upgrade", "e2e_desktop_update"),
    "scripts/build/web.mjs": ("e2e_upgrade", "e2e_desktop_update"),
    "hermes_cli/_subprocess_compat.py": ("e2e_upgrade", "e2e_desktop_update"),
    "hermes_cli/desktop_build_lock.py": ("e2e_upgrade", "e2e_desktop_update"),
    "hermes_cli/memory_provider_migration.py": ("e2e_upgrade", "e2e_desktop_update"),
    "hermes_cli/source_build.py": ("e2e_upgrade", "e2e_desktop_update"),
    "hermes_cli/source_completion.py": ("e2e_upgrade", "e2e_desktop_update"),
    "apps/desktop/electron/update-marker.ts": ("desktop_updater", "e2e_desktop_update"),
    "apps/desktop/electron/handoff-result.ts": ("desktop_updater", "e2e_desktop_update"),
}


@pytest.mark.parametrize("path,lanes", list(_R15_CASES.items()))
def test_update_owner_change_dispatches_its_suites_end_to_end(path, lanes):
    assert (_REPO / path).is_file(), f"{path} moved: update this table"
    classified = _real_classifier([path])
    run = _ci_run(classified)
    for lane in lanes:
        assert classified[lane], f"{path}: classifier leaves {lane} off"
        reached = _consumers_reached(run, lane)
        assert all(reached.values()), f"{path}: {lane}=true never reaches {reached}"


# Review R6 m11: update-path files whose suites the classifier left off. Some land in a sibling
# PR of the update stack (update-marker-gate.ts), so the table routes paths, not files on disk.
_R6_CASES = {
    # The Desktop gate's marker reader: the hand-off script's contract (desktop_updater) and the
    # Desktop update E2E that drives the gate.
    "apps/desktop/electron/update-marker-gate.ts": ("desktop_updater", "e2e_desktop_update"),
    # Stops the backend before a remote update: typecheck/vitest and the hand-off script tests.
    "apps/desktop/electron/remote-lifecycle.ts": ("frontend", "desktop_updater"),
    # The Tauri updater's marker claim: cargo, the only CI that builds it (review CI3).
    "apps/bootstrap-installer/src-tauri/src/marker.rs": ("rust",),
    # The launchers reach the launch repair: the Desktop update relaunches through them too.
    "hermes_cli/_launchers.py": ("e2e_upgrade", "e2e_desktop_update"),
}


@pytest.mark.parametrize("path,lanes", list(_R6_CASES.items()))
def test_update_path_change_selects_every_suite_that_exercises_it(path, lanes):
    classified = _real_classifier([path])
    run = _ci_run(classified)
    for lane in lanes:
        assert classified[lane], f"{path}: classifier leaves {lane} off"
        if lane.startswith("e2e"):
            reached = _consumers_reached(run, lane)
            assert all(reached.values()), f"{path}: {lane}=true never reaches {reached}"
        if lane == "desktop_updater":
            assert _windows_desktop_updater_tests_selected(run), f"{path}: desktop_updater never reaches its tests"


# Review CI2/CI3: update-path files the classifier sent to the wrong suites.
_CI3_CASES = {
    # /api/health `commit`: host-backend-attach compares it before attaching after an update.
    "hermes_cli/web_routers/status.py": ("python", "e2e_desktop_update"),
    # The SSH remote's marker judge/gate: typecheck/vitest and the marker contract's script tests.
    "apps/desktop/electron/remote-update-marker-programs.ts": ("frontend", "desktop_updater"),
    "apps/desktop/electron/remote-update-marker-programs.test.ts": ("frontend", "desktop_updater"),
}


@pytest.mark.parametrize("path,lanes", list(_CI3_CASES.items()))
def test_update_path_change_reaches_the_suites_that_consume_it(path, lanes):
    test_update_path_change_selects_every_suite_that_exercises_it(path, lanes)


_TAURI_UPDATER = tuple(f"apps/bootstrap-installer/src-tauri/src/{name}.rs" for name in ("marker", "update", "paths"))


@pytest.mark.parametrize("path", _TAURI_UPDATER)
def test_tauri_updater_routes_as_one_unit_to_the_lane_that_builds_it(path):
    """No slow lane builds the bootstrap installer: starting one for its updater runs nothing of it."""
    assert (_REPO / path).is_file(), f"{path} moved: update this table"
    classified = _real_classifier([path])
    assert classified["rust"], f"{path}: cargo test never runs"
    started = [lane for lane in ("e2e", "e2e_upgrade", "e2e_desktop_update", "e2e_desktop_core") if classified[lane]]
    assert not started, f"{path}: starts {started}, which never build the Tauri installer"


def test_gateway_status_routing_names_the_stamp_not_its_prefix_siblings():
    assert _real_classifier(["gateway/status.py"])["e2e_upgrade"]
    assert not _real_classifier(["gateway/status_phrases.py"])["e2e_upgrade"], \
        "status_phrases.py (turn-status wording) is not on the update path"


def test_every_detect_output_a_lane_sets_is_consumed_by_some_job():
    """A lane output nothing reads is a lane that can never run anything."""
    ci = _yaml(".github/workflows/ci.yaml")
    text = json.dumps({k: v for k, v in ci["jobs"].items() if k != "detect"})
    # python_prod's only reader is the deferred Desktop E2E job (`if: false`, see ci.yaml).
    unread = [k for k in ci["jobs"]["detect"]["outputs"]
              if k not in ("event_name", "python_prod") and f"needs.detect.outputs.{k}" not in text]
    assert unread == []


# -- shared cross-language fixtures (review D14) --------------------------------------------

# The job (path from ci.yaml down) that runs each lane's consumers of a shared fixture.
_FIXTURE_LANE_JOBS: dict[str, tuple[str, ...]] = {
    "python": ("tests", "test"),  # pytest on Linux (marker.sh corpus included)
    "rust": ("rust-tests", "bootstrap-installer"),  # cargo test of the Tauri crate
    "frontend": ("js-tests", "check"),  # apps/desktop `check` -> vitest --project electron
}
_CORPUS = "tests/fixtures/update_marker_corpus.json"
_TEST_FILE = re.compile(r"(^tests/(?!ci/).*\.py$)|(_tests?\.rs$)|(\.test\.[cm]?[jt]sx?$)")


def _fixture_lanes_reached(fixture: str) -> dict[str, bool]:
    classified = _real_classifier([fixture])
    run = _ci_run(classified)
    reached = {lane: classified[lane] and _reached(run, *job) is not None
               for lane, job in _FIXTURE_LANE_JOBS.items()}
    reached["desktop_updater"] = classified["desktop_updater"] and _windows_desktop_updater_tests_selected(run)
    return reached


def test_marker_corpus_change_runs_every_language_that_reads_it():
    """The corpus is A7 rule 7's single source of truth for four readers. Classified by
    path alone it selected ``python`` only: cargo (marker_tests.rs), vitest
    (update-marker-corpus.test.ts) and the PowerShell corpus test (desktop_updater on
    the Windows lane) never ran on the PR that edited their expected answers."""
    reached = _fixture_lanes_reached(_CORPUS)
    assert all(reached.values()), f"{_CORPUS}: a corpus consumer's lane never runs: {reached}"


def _tracked_mentions(name: str) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(_REPO), "grep", "-l", "-F", name, "--", "."],
                             capture_output=True, text=True, timeout=60)
    except OSError:
        pytest.skip("git unavailable: cannot list the fixture's consumers")
    if out.returncode not in (0, 1):
        pytest.skip(f"not a git checkout: {out.stderr.strip()[:200]}")
    return out.stdout.split()


# Listed consumers that land in a named sibling PR merged BEFORE this one (the update batch
# lands LOCK, COMMIT, WIN, DESK, SCRIPTS, POST, then this CI PR). Only these may be absent,
# and only while the sibling is unmerged: an entry whose file exists fails the guard, so the
# table empties when the siblings land and a later deletion of a consumer turns it red.
_LANDS_IN_SIBLING_PR: dict[str, str] = {
    "apps/desktop/electron/update-marker-corpus.test.ts": "#132345 desktop update gate",
    "apps/desktop/electron/remote-update-marker-programs.test.ts": "#132345 desktop update gate",
    "tests/scripts/desktop_update/test_desktop_update_posix_marker_corpus.py": "#132354 hand-off scripts",
    "tests/scripts/desktop_update/test_desktop_update_windows_marker_corpus.py": "#132354 hand-off scripts",
}


def test_every_test_that_reads_a_shared_fixture_is_routed_by_it():
    """Derived, not remembered: every test file in the tree that names a shared fixture is
    a listed consumer, and the fixture selects each test lane that consumer's own edit would."""
    assert _CORPUS in cc._SHARED_FIXTURE_CONSUMERS
    every_listed = {c for listed in cc._SHARED_FIXTURE_CONSUMERS.values() for c in listed}
    stale = sorted(set(_LANDS_IN_SIBLING_PR) - every_listed)
    assert not stale, f"sibling-PR allowance names no listed consumer: {stale}"
    landed = sorted(c for c in _LANDS_IN_SIBLING_PR if (_REPO / c).is_file())
    assert not landed, f"sibling landed: delete its _LANDS_IN_SIBLING_PR allowance: {landed}"
    for fixture, listed in cc._SHARED_FIXTURE_CONSUMERS.items():
        assert (_REPO / fixture).is_file(), f"missing shared fixture: {fixture}"
        assert listed, f"{fixture}: no listed consumers"
        absent = [c for c in listed if not (_REPO / c).is_file()]
        unowned = [c for c in absent if c not in _LANDS_IN_SIBLING_PR]
        assert not unowned, f"{fixture}: missing consumer(s) no sibling PR lands: {unowned}"
        assert len(absent) < len(listed), f"{fixture}: every listed consumer is absent"
        readers = {p for p in _tracked_mentions(Path(fixture).name) if _TEST_FILE.search(p)}
        assert readers, f"{fixture}: no discovered test readers"
        assert readers <= set(listed), f"{fixture}: unlisted consumer(s) {sorted(readers - set(listed))}"
        on = cc.classify([fixture])
        for consumer in listed:
            own = cc.classify([consumer])
            lost = [lane for lane in cc._FIXTURE_CONSUMER_LANES if own[lane] and not on[lane]]
            assert not lost, f"{fixture}: editing it skips {lost}, which run {consumer}"


def test_sibling_allowance_expires_once_its_consumer_is_in_the_tree(monkeypatch):
    """An allowance that outlives its sibling's merge would excuse that consumer's later deletion."""
    present = next(c for c in cc._SHARED_FIXTURE_CONSUMERS[_CORPUS] if (_REPO / c).is_file())
    monkeypatch.setitem(_LANDS_IN_SIBLING_PR, present, "a sibling that already landed")
    with pytest.raises(AssertionError, match="sibling landed"):
        test_every_test_that_reads_a_shared_fixture_is_routed_by_it()


# -- replay guard sensitivity: mutate data, not the replay implementation ------------------

_REQUIRED_STEP_CASES = (
    ("e2e_desktop_update", ".github/workflows/e2e-desktop-update.yml", "update", "Run the update suite under xvfb"),
    ("e2e_upgrade", ".github/workflows/tests.yml", "e2e-upgrade", "Run upgrade e2e tests"),
    ("e2e_upgrade", ".github/workflows/windows-install-update-e2e.yml", "install-update", "Run Windows install + update E2E"),
    ("desktop_updater", ".github/workflows/tests-os.yml", "os-tests", "Run ${{ matrix.marker }} tests"),
    ("e2e", ".github/workflows/tests.yml", "e2e", "Run e2e tests"),
    ("e2e", ".github/workflows/tests-os.yml", "e2e-windows", "Run Windows E2E suite"),
)


def _mutated_yaml(monkeypatch, rel):
    from copy import deepcopy

    original = _yaml
    changed = deepcopy(original(rel))
    monkeypatch.setattr(sys.modules[__name__], "_yaml", lambda path: changed if path == rel else original(path))
    return changed


@pytest.mark.parametrize("lane,rel,job,name", _REQUIRED_STEP_CASES)
@pytest.mark.parametrize("mutation", ["disabled", "advisory", "no-command", "empty-selection"])
def test_replay_rejects_ineffective_required_step(monkeypatch, lane, rel, job, name, mutation):
    lanes = cc.classify([])
    assert all(_consumers_reached(_ci_run(lanes), lane).values())
    workflow = _mutated_yaml(monkeypatch, rel)
    step = next(s for s in workflow["jobs"][job]["steps"] if s.get("name", "").startswith(name))
    if mutation == "disabled":
        step["if"] = "${{ false }}"
    elif mutation == "advisory":
        step["continue-on-error"] = "${{ true }}"
    elif mutation == "no-command":
        # Keep every old substring/regex check satisfied, but execute no tests.
        step["run"] = "if false; then\n" + step["run"] + "\nfi\n"
    elif job == "os-tests":
        step["run"] = step["run"].replace('"${{ matrix.marker }}"', '"macos"')
    elif job == "update":
        for cell in workflow["jobs"][job]["strategy"]["matrix"]["include"]:
            cell["specs"] = "e2e/update/absent.spec.ts"
    else:
        step["run"] = step["run"].replace("tests/e2e", "tests/absent-e2e")
    with pytest.raises(AssertionError):
        reached = _consumers_reached(_ci_run(lanes), lane)
        assert all(reached.values()), reached


def test_selection_receipt_does_not_require_the_native_jobs_venv(tmp_path):
    workflow = _yaml(".github/workflows/tests.yml")
    step = next(s for s in workflow["jobs"]["e2e-upgrade"]["steps"]
                if s.get("name", "").startswith("Run upgrade e2e tests"))
    rel = "tests/e2e/core/upgrade/test_owned.py"
    test_file = tmp_path / rel
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_owned(): pass\n", encoding="utf-8")
    ctx = {"matrix": {"shard": "core"}, "inputs": {}}
    assert workflow_steps.selected_files(step, ctx, tmp_path) == {rel}


def test_replay_uses_executed_gate_output(monkeypatch):
    path = "apps/desktop/electron/handoff-result.ts"
    assert all(_consumers_reached(_ci_run(_real_classifier([path])), "desktop_updater").values())
    workflow = _mutated_yaml(monkeypatch, ".github/workflows/ci.yaml")
    gate = next(s for s in workflow["jobs"]["detect"]["steps"] if s.get("id") == "gate-lanes")
    needle = "    for lane, value in values.items():"
    assert needle in gate["run"]
    gate["run"] = gate["run"].replace(needle, "    values['desktop_updater'] = 'false'\n" + needle)
    with pytest.raises(AssertionError):
        test_update_owner_change_dispatches_its_suites_end_to_end(path, ("desktop_updater",))


_PROBE_STEP = {"run": r'''python3 -c 'import sys; print("py3=" + sys.executable)' >> "$GITHUB_OUTPUT"
python -c 'import sys; print("py=" + sys.prefix)' >> "$GITHUB_OUTPUT"
echo "tools=$(printf 'b\na\n' | sort | tr -d '\r' | paste -sd , -)$(find . -maxdepth 0)" >> "$GITHUB_OUTPUT"
'''}


def test_replay_starts_the_python_and_tools_hosted_steps_call():
    """Every host (native Windows included) runs python3/python and the coreutils selection steps use."""
    out = workflow_steps.outputs(_PROBE_STEP, {})
    assert (out["py"], out["tools"]) == (sys.prefix, "a,b.")
    assert subprocess.run([out["py3"], "-c", "import sys; print(sys.prefix)"], capture_output=True,
                          text=True, check=True, timeout=60).stdout.strip() == sys.prefix


@pytest.mark.platforms("posix")
def test_replay_python3_is_the_replay_interpreter_when_its_dir_has_only_python(tmp_path):
    """A Windows venv dir holds only ``python``: python3 must not fall through to a host interpreter."""
    interpreter = tmp_path / "Scripts" / "python"
    interpreter.parent.mkdir()
    interpreter.symlink_to(os.path.realpath(sys.executable))
    probe = ("import json, sys; from tests.ci import workflow_steps; "
             "print(json.dumps([sys.executable, workflow_steps.outputs(json.loads(sys.argv[1]), {})]))")
    env = {**os.environ, "PYTHONPATH": str(_REPO)}
    result = subprocess.run([str(interpreter), "-c", probe, json.dumps(_PROBE_STEP)], cwd=_REPO, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    replay_python, out = json.loads(result.stdout)
    assert out["py3"] == replay_python


@pytest.mark.parametrize("missing", ["fixture", "consumer", "readers", "listed"])
def test_shared_fixture_guard_rejects_phantom_graph(monkeypatch, tmp_path, missing):
    fixture = "tests/fixtures/owned_corpus.json"
    consumer = "tests/test_owned_corpus.py"
    for rel in (fixture, consumer):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}" if rel == fixture else "def test_reader(): pass\n", encoding="utf-8")
    if missing in ("fixture", "consumer"):
        (tmp_path / (fixture if missing == "fixture" else consumer)).unlink()
    monkeypatch.setattr(sys.modules[__name__], "_REPO", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "_CORPUS", fixture)
    monkeypatch.setattr(cc, "_SHARED_FIXTURE_CONSUMERS", {fixture: () if missing == "listed" else (consumer,)})
    monkeypatch.setattr(sys.modules[__name__], "_tracked_mentions", lambda _: [] if missing in ("readers", "listed") else [consumer])
    with pytest.raises(AssertionError):
        test_every_test_that_reads_a_shared_fixture_is_routed_by_it()


@pytest.mark.parametrize("case,error", [
    ("sibling-absent", None),  # the stacked PR: one reader here, one landing in a sibling
    ("unowned-absent", "no sibling PR lands"),
    ("only-sibling", "every listed consumer is absent"),
    ("stale-allowance", "names no listed consumer"),
])
def test_sibling_pr_allowance_admits_only_named_pending_consumers(monkeypatch, tmp_path, case, error):
    fixture, here, pending = ("tests/fixtures/owned_corpus.json", "tests/test_owned_corpus.py",
                              "tests/scripts/test_sibling_corpus.py")
    (tmp_path / fixture).parent.mkdir(parents=True)
    (tmp_path / fixture).write_text("{}", encoding="utf-8")
    if case != "only-sibling":
        (tmp_path / here).write_text("def test_reader(): pass\n", encoding="utf-8")
    allowance = {} if case == "unowned-absent" else {pending: "#1 sibling"}
    if case == "stale-allowance":
        allowance["tests/test_removed_corpus.py"] = "#2 sibling"
    monkeypatch.setattr(sys.modules[__name__], "_REPO", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "_CORPUS", fixture)
    monkeypatch.setattr(sys.modules[__name__], "_LANDS_IN_SIBLING_PR", allowance)
    monkeypatch.setattr(cc, "_SHARED_FIXTURE_CONSUMERS",
                        {fixture: (pending,) if case == "only-sibling" else (here, pending)})
    monkeypatch.setattr(sys.modules[__name__], "_tracked_mentions", lambda _: [here])
    if error is None:
        test_every_test_that_reads_a_shared_fixture_is_routed_by_it()
    else:
        with pytest.raises(AssertionError, match=error):
            test_every_test_that_reads_a_shared_fixture_is_routed_by_it()


# -- ownership: the update entry points' real imports ---------------------------------------

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", ".worktrees", "tests", "website", "__pycache__"}


def _product_python() -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(_REPO):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        out += [Path(dirpath) / f for f in filenames if f.endswith(".py")]
    return out


def _module_file(module: str) -> Path | None:
    base = _REPO / module.replace(".", "/")
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=None)
def _imports(path: Path) -> frozenset[str]:
    """Repo modules ``path`` imports anywhere (module level or lazily in a function)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (SyntaxError, UnicodeDecodeError):
        return frozenset()
    package = path.relative_to(_REPO).with_suffix("").parts[:-1]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = list(package[: len(package) - (node.level - 1)])
                base = ".".join([*parts, base] if base else parts)
            names.add(base)
            names.update(f"{base}.{a.name}" for a in node.names)
    found = set()
    for name in names:
        target = _module_file(name)
        if target is not None and target != path:
            found.add(target.relative_to(_REPO).as_posix())
    return frozenset(found)


def _entry_modules(prefixes: tuple[str, ...]) -> list[Path]:
    hits = [p for p in _product_python() if p.relative_to(_REPO).as_posix().startswith(prefixes)]
    assert hits, f"no module matches {prefixes}: the entry-point list went stale"
    return sorted(hits)


@lru_cache(maxsize=None)
def _importers() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in _product_python():
        for target in _imports(path):
            counts[target] = counts.get(target, 0) + 1
    return counts


def _unowned(entries: tuple[str, ...], lane: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for module in _entry_modules(entries):
        for target in _imports(module):
            if not cc.classify([target])[lane] and target not in cc._SHARED_HUBS:
                out.setdefault(target, []).append(module.relative_to(_REPO).as_posix())
    return out


@pytest.mark.parametrize("entries,lane", [
    ("_UPDATE_ENTRY_POINTS", "e2e_upgrade"),
    ("_DESKTOP_BUILD_ENTRY_POINTS", "e2e_desktop_update"),
])
def test_every_module_an_update_entry_point_imports_starts_its_suite(entries, lane):
    """Add the module to the classifier's owner list for this lane. Only a shared
    hub (``_SHARED_HUBS``: imported across the product, covered by the unit lanes
    and by this suite on every push to main) may stay out."""
    missing = _unowned(getattr(cc, entries), lane)
    assert missing == {}, f"{lane}: imported by the update path but not routed to it: {missing}"


def test_entry_points_themselves_start_their_suites():
    for module in _entry_modules(cc._UPDATE_ENTRY_POINTS):
        assert cc.classify([module.relative_to(_REPO).as_posix()])["e2e_upgrade"], module
    for module in _entry_modules(cc._DESKTOP_BUILD_ENTRY_POINTS):
        rel = module.relative_to(_REPO).as_posix()
        assert cc.classify([rel])["e2e_desktop_update"] and cc.classify([rel])["e2e_upgrade"], rel


def test_shared_hubs_are_real_hubs_still_on_the_update_path():
    """The hub exemption cannot hide an update-specific module: each hub must be
    imported by many product modules, and still be imported by an entry point."""
    counts = _importers()
    thin = {h: counts.get(h, 0) for h in cc._SHARED_HUBS if counts.get(h, 0) < cc.HUB_MIN_IMPORTERS}
    assert thin == {}, f"not a shared hub (fewer than {cc.HUB_MIN_IMPORTERS} importers): own it instead"
    reached = set()
    for entries in (cc._UPDATE_ENTRY_POINTS, cc._DESKTOP_BUILD_ENTRY_POINTS):
        for module in _entry_modules(entries):
            reached |= _imports(module)
    assert set(cc._SHARED_HUBS) - reached == set(), "stale hub exemption: no entry point imports it"


# -- ownership: the JavaScript compilers an update runs -------------------------------------

_SCRIPT_LITERAL = re.compile(r"""["']((?:scripts/build|apps/desktop/scripts)/[\w./-]+\.(?:mjs|js|py|ps1))["']""")
_ESM_IMPORT = re.compile(r"""(?:from|import)\s*\(?\s*['"](\.{1,2}/[^'"]+)['"]""")
_STEP = re.compile(r"""step\(\s*['"]([\w./-]+\.mjs)['"]""")


def _js_closure(seeds: set[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(seeds)
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        path = _REPO / rel
        assert path.is_file(), f"{rel} is referenced on the update path but does not exist"
        seen.add(rel)
        if not rel.endswith((".mjs", ".js")):
            continue
        text = path.read_text(encoding="utf-8-sig")
        for spec in _ESM_IMPORT.findall(text):
            target = (path.parent / spec).resolve()
            if target.is_file() and _REPO in target.parents:
                stack.append(target.relative_to(_REPO).as_posix())
        stack += _STEP.findall(text)
    return seen


def _python_script_seeds(entries: tuple[str, ...]) -> set[str]:
    seeds = set()
    for module in _entry_modules(entries):
        seeds |= set(_SCRIPT_LITERAL.findall(module.read_text(encoding="utf-8-sig")))
    return seeds


def test_build_scripts_the_update_runs_start_the_upgrade_suite():
    reached = _js_closure(_python_script_seeds(cc._UPDATE_ENTRY_POINTS))
    assert "scripts/build/web.mjs" in reached, "seed scan broke: source_build runs web.mjs"
    unowned = sorted(f for f in reached if not cc.classify([f])["e2e_upgrade"])
    assert unowned == []


def test_desktop_build_scripts_start_the_desktop_update_suite():
    """`hermes desktop --build-only` (the update's Desktop rebuild) runs
    apps/desktop's `build` script, which steps through scripts/build/desktop.mjs."""
    package = json.loads((_REPO / "apps/desktop/package.json").read_text(encoding="utf-8-sig"))
    build = re.search(r"node\s+(\S+\.mjs)", package["scripts"]["build"])
    assert build, package["scripts"]["build"]
    seeds = {("apps/desktop/" + build.group(1)).replace("/./", "/")}
    seeds |= _python_script_seeds(cc._DESKTOP_BUILD_ENTRY_POINTS)
    reached = _js_closure(seeds)
    assert "scripts/build/desktop.mjs" in reached, "seed scan broke: the Desktop build runs desktop.mjs"
    unowned = sorted(f for f in reached if not cc.classify([f])["e2e_desktop_update"])
    assert unowned == []


# -- strict acceptance: the dispatch input reaches every E2E suite's environment ------------

_STRICT_STEPS = (
    (("tests", "e2e"), ".github/workflows/tests.yml", "e2e", "Run e2e tests"),
    (("tests", "e2e-upgrade"), ".github/workflows/tests.yml", "e2e-upgrade", "Run upgrade e2e tests"),
    (("tests-os", "e2e-windows"), ".github/workflows/tests-os.yml", "e2e-windows", "Run Windows E2E suite"),
    (("tests-os", "install-update-e2e", "install-update"), ".github/workflows/windows-install-update-e2e.yml",
     "install-update", "Run Windows install + update E2E"),
)


def _strict_env(run: dict, path: tuple[str, ...], rel: str, job: str, step_name: str) -> str:
    node = _reached(run, *path)
    assert node is not None, f"{'/'.join(path)} did not run"
    step = next(s for s in _yaml(rel)["jobs"][job]["steps"] if str(s.get("name", "")).startswith(step_name))
    assert _selected_test_files(node, step_name)
    return gha.to_string(gha.render(step["env"]["HERMES_E2E_STRICT_ACCEPTANCE"], node["ctx"]))


@pytest.mark.parametrize("value", ["upd-txn", "1"])
def test_strict_acceptance_dispatch_reaches_every_e2e_suite(value):
    """`gh workflow run ci.yaml -f strict_acceptance=upd-txn` (the batch's acceptance run)."""
    lanes = cc.classify([])  # a dispatch has no diff: every lane on
    run = _run_workflow(".github/workflows/ci.yaml", inputs={"release": False, "strict_acceptance": value},
                        detect=_detect_outputs(lanes))
    for path, rel, job, step in _STRICT_STEPS:
        assert _strict_env(run, path, rel, job, step) == value, "/".join(path)


def test_pull_requests_never_run_strict():
    run = _ci_run(cc.classify([]))
    for path, rel, job, step in _STRICT_STEPS:
        assert _strict_env(run, path, rel, job, step) == "", "/".join(path)


def test_windows_install_update_dispatch_alone_sets_strict():
    jobs = _run_workflow(".github/workflows/windows-install-update-e2e.yml",
                         inputs={"strict_acceptance": "upd-txn"})
    assert _strict_env({"w": {"jobs": jobs}}, ("w", "install-update"),
                       ".github/workflows/windows-install-update-e2e.yml", "install-update",
                       "Run Windows install + update E2E") == "upd-txn"
    assert "workflow_dispatch" in _on(_yaml(".github/workflows/windows-install-update-e2e.yml"))


def test_run_tests_forwards_the_strict_switch():
    """run_tests.sh starts pytest under `env -i`: an unlisted variable never arrives."""
    text = (_REPO / "scripts/run_tests.sh").read_text(encoding="utf-8-sig")
    allow = re.search(r"for _test_var in (.*?); do", text, re.S)
    assert allow and "HERMES_E2E_STRICT_ACCEPTANCE" in allow.group(1).split()
