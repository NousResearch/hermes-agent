"""Tag admission for .github/workflows/desktop-bundled-release.yml (plan item 7).

The release build runs under the ``release-signing`` environment, so the
validate job must be more than a shape check: a correctly-shaped tag on an
unreviewed commit must never reach the signing build. Two layers are tested:

* Structure — the workflow declares the admitted SHA as a job output and
  every privileged job checks out THAT, not the (moveable) tag ref.
* Behavior — the production admission command runs inside real temp git
  repositories: an annotated claim on origin/main passes and exports the full
  SHA; a claim for a commit NOT on main is refused.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.ci.desktop_release_roles import (
    DOWNLOADABLE_DISPATCHES, DRY_DISPATCH, SHA, admitted, credential_gate, evaluate, gate,
    native_builds, needs_of,
)

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "desktop-bundled-release.yml"
_SIGNING_ENV = "release-signing"
_CONTROLLER = "c" * 40


def _workflow() -> dict:
    yaml = pytest.importorskip("hermes_yaml")
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Structure: the admitted SHA is the only build input privileged jobs see.
# ---------------------------------------------------------------------------


def test_validate_exports_the_admitted_sha_as_a_job_output():
    wf = _workflow()
    outputs = wf["jobs"]["validate"].get("outputs") or {}
    assert "sha" in outputs, "validate must export the admitted SHA"
    assert "steps.admission.outputs.sha" in outputs["sha"]


def _checkout_refs(job: dict, inputs: dict, needs: dict) -> list:
    """The revision each checkout that runs under *inputs* resolves to, in order."""
    refs = []
    for step in job.get("steps", []):
        if "checkout" not in step.get("uses", ""):
            continue
        if "if" in step and not gate(step["if"], inputs, needs, job_if=False):
            continue
        refs.append(evaluate(step.get("with", {}).get("ref", ""), inputs, needs,
                             job_if=False, github={"sha": _CONTROLLER}))
    return refs


def test_signing_jobs_pin_source_and_controller_revisions_not_mutable_tags():
    jobs = _workflow()["jobs"]
    privileged = {name: job for name, job in jobs.items()
                  if isinstance(job, dict) and job.get("environment") == _SIGNING_ENV}
    assert privileged, "walk broken: no release-signing jobs found"
    # The admission job has nothing admitted yet; it resolves the revision
    # every other signing job consumes (the one-dispatch design).
    admission = [name for name, job in privileged.items() if not needs_of(job)]
    assert admission == ["validate"], admission
    exercised = set()
    for name, job in privileged.items():
        if name == "validate":
            continue
        assert "validate" in needs_of(job), f"signing job {name!r} reads needs.validate.outputs.sha but does not need validate"
        for dispatch, inputs in DOWNLOADABLE_DISPATCHES.items():
            channel = dispatch == "channel"
            needs = admitted(needs_of(job), channel=channel)
            if "if" in job and not gate(job["if"], inputs, needs):
                continue
            refs = _checkout_refs(job, inputs, needs)
            exercised.add(name)
            # Only the admitted source or this run's own trusted controller
            # may be checked out; a tag can be moved after admission.
            assert set(refs) <= {SHA, _CONTROLLER}, (name, dispatch, refs)
            if refs and not channel:
                assert refs[0] == SHA, (
                    f"signing job {name!r} builds from {refs[0]!r} for a {dispatch} build; "
                    "outside a pinned channel build it must build the admitted source")
    assert set(native_builds(jobs).values()) <= exercised, "walk broken: a native build leg never ran"
    assert exercised, "walk broken: no signing job runs for a downloadable build"


# ---------------------------------------------------------------------------
# Behavior: run the admission script against real repositories.
# ---------------------------------------------------------------------------

bash = shutil.which("bash")
pytestmark = pytest.mark.skipif(bash is None, reason="bash is required to run the admission script")


def _native_tool(name: str) -> str:
    """Resolve *name* to an executable CreateProcess can actually start.

    run_tests.sh / conftest blank SystemRoot/ComSpec for hermeticity, and on
    this host PATH can point ``git``/``bash`` at the MSIX payload copy under
    ``C:\\Program Files\\WindowsApps\\...`` — a store-app location that
    fails CreateProcess with WinError 5 outside its package context. Prefer
    a conventional install; give children a complete environment too.
    """
    candidates = [hit for hit in (shutil.which(name),) if hit]
    if sys.platform == "win32":
        git_base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git"
        for rel in (("cmd", f"{name}.exe"), ("bin", f"{name}.exe"), ("usr", "bin", f"{name}.exe")):
            p = git_base.joinpath(*rel)
            if p.exists():
                candidates.append(str(p))
    for cand in candidates:
        if "windowsapps" not in cand.lower():
            return cand
    return candidates[0]


def _child_env(**overrides: str) -> dict:
    env = os.environ.copy()
    if sys.platform == "win32":
        env.setdefault("SystemRoot", r"C:\Windows")
        env.setdefault("ComSpec", r"C:\Windows\system32\cmd.exe")
        env.setdefault("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    env.update(overrides)
    return env


_GIT = _native_tool("git")
_BASH = _native_tool("bash")


def _git(*args: str, cwd: Path) -> str:
    out = subprocess.run(
        [_GIT, *args], cwd=cwd, capture_output=True, text=True, check=True,
        env=_child_env(),
    )
    return out.stdout.strip()


def _seed_repo(root: Path) -> tuple[Path, Path]:
    """origin (upstream) + clone (where releases are tagged from).

    origin/main holds a pyproject whose version matches the stable tag, so
    the pyproject-lockstep leg of the shape check passes for v0.1.2.
    """
    origin = root / "origin"
    origin.mkdir()
    _git("init", "-b", "main", cwd=origin)
    _git("config", "user.email", "ci@example.com", cwd=origin)
    _git("config", "user.name", "ci", cwd=origin)
    (origin / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.1.2"\n', encoding="utf-8")
    (origin / "README.md").write_text("seed\n", encoding="utf-8")
    _git("add", "-A", cwd=origin)
    _git("commit", "-m", "seed", cwd=origin)

    clone = root / "clone"
    _git("clone", str(origin), str(clone), cwd=root)
    _git("config", "user.email", "ci@example.com", cwd=clone)
    _git("config", "user.name", "ci", cwd=clone)
    return origin, clone


def _run_admission(clone: Path, tag: str, claim_tag: str) -> subprocess.CompletedProcess:
    gh_output = clone / "github_output.txt"
    gh_output.write_text("", encoding="utf-8")
    env = _child_env(
        TAG=tag,
        RELEASE_TAG=tag,
        RELEASE_CLAIM_TAG=claim_tag,
        RELEASE_CLAIM_OBJECT=_git("rev-parse", f"refs/tags/{claim_tag}", cwd=clone),
        GITHUB_OUTPUT=str(gh_output),
        RELEASE_PHASE="candidate",
        GITHUB_REF=f"refs/tags/{claim_tag}",
        GITHUB_SHA=_git("rev-parse", "HEAD", cwd=clone),
        PYTHONPATH=str(_REPO),
    )
    return subprocess.run(
        [sys.executable, "-m", "scripts.releases.stable", "verify"],
        cwd=clone,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _claim_message(clone: Path) -> str:
    return json.dumps({
        "schema": 1,
        "version": "0.1.2",
        "attempt": 1,
        "commit": _git("rev-parse", "HEAD", cwd=clone),
        "autopublish": False,
        "skipBundles": False,
        "skipTests": False,
        "claimEpoch": 1_790_000_000,
    }, sort_keys=True, separators=(",", ":"))


def test_claim_on_origin_main_is_admitted_and_exports_the_full_sha(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _origin, clone = _seed_repo(tmp_path)
    monkeypatch.setenv("GIT_COMMITTER_DATE", "@1790000000 +0000")
    _git("tag", "-a", "rc.1-v0.1.2", "-m", _claim_message(clone), cwd=clone)
    _git("push", "origin", "refs/tags/rc.1-v0.1.2", cwd=clone)

    proc = _run_admission(clone, "v0.1.2", "rc.1-v0.1.2")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    gh_output = (clone / "github_output.txt").read_text(encoding="utf-8")
    expected = _git("rev-parse", "HEAD", cwd=clone)
    assert f"sha={expected}" in gh_output


def test_claim_not_on_origin_main_is_refused(tmp_path: Path):
    _origin, clone = _seed_repo(tmp_path)
    # A commit that exists ONLY in the clone — never pushed, never reviewed.
    (clone / "rogue.txt").write_text("unreviewed\n", encoding="utf-8")
    _git("add", "-A", cwd=clone)
    _git("commit", "-m", "rogue", cwd=clone)
    _git("tag", "-a", "rc.1-v0.1.2", "-m", _claim_message(clone), cwd=clone)
    _git("push", "origin", "refs/tags/rc.1-v0.1.2", cwd=clone)

    proc = _run_admission(clone, "v0.1.2", "rc.1-v0.1.2")
    assert proc.returncode != 0, "a tag off origin/main must not be admitted"
    assert "is not on main" in proc.stdout + proc.stderr
    # And nothing was exported for the signing jobs to consume.
    assert "sha=" not in (clone / "github_output.txt").read_text(encoding="utf-8")


def test_malformed_claim_is_refused(tmp_path: Path):
    _origin, clone = _seed_repo(tmp_path)
    _git("tag", "-a", "v0.1.2-rc1", "-m", "bad claim", cwd=clone)
    _git("push", "origin", "refs/tags/v0.1.2-rc1", cwd=clone)
    proc = _run_admission(clone, "v0.1.2", "v0.1.2-rc1")
    assert proc.returncode != 0
    assert "is not a claim tag" in proc.stdout + proc.stderr


def _builds(step: dict) -> bool:
    run = step.get("run", "")
    return "scripts/bundles/desktop.py" in run and "--prepared " in run


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_downloadable_native_builds_refuse_to_ship_unsigned(tmp_path: Path):
    """The Windows signer only warns without AZURE_SIGN_*; every native build
    leg whose artifacts are downloadable must therefore fail before building,
    under one gate shared with the macOS signing-credential check."""
    jobs = _workflow()["jobs"]
    legs = native_builds(jobs)
    for name in legs.values():
        steps = jobs[name]["steps"]
        step = credential_gate(jobs[name])
        builds = [index for index, candidate in enumerate(steps) if _builds(candidate)]
        assert builds and steps.index(step) < builds[0], f"{name} checks credentials after building"
        for dispatch, inputs in DOWNLOADABLE_DISPATCHES.items():
            assert gate(step["if"], inputs, {}, job_if=False), (name, dispatch)
        assert not gate(step["if"], DRY_DISPATCH, {}, job_if=False), name

    windows = {credential_gate(jobs[name])["run"]: credential_gate(jobs[name])
               for (target, _), name in legs.items() if target.startswith("win32-")}
    assert windows, "walk broken: no Windows build legs"
    for step in windows.values():
        names = list(step["env"])
        assert {"AZURE_SIGN_ENDPOINT", "AZURE_SIGN_ACCOUNT", "AZURE_SIGN_PROFILE", "AZURE_CLIENT_ID"} <= set(names)

        def run(**values: str) -> subprocess.CompletedProcess:
            env = {"PATH": os.environ["PATH"], "RUNNER_TEMP": str(tmp_path)}
            env.update({name: "" for name in names})
            env.update(values)
            return subprocess.run(["bash", "-euo", "pipefail", "-c", step["run"]], env=env,
                                  capture_output=True, text=True, timeout=60)

        proc = run()
        assert proc.returncode != 0 and "::error::" in proc.stdout and "AZURE_SIGN_ENDPOINT" in proc.stdout
        assert run(**{name: "x" for name in names}).returncode == 0
        partial = run(**{name: "x" for name in names if name != "AZURE_CLIENT_ID"})
        assert partial.returncode != 0 and "AZURE_CLIENT_ID" in partial.stdout
