"""Tag admission for .github/workflows/desktop-bundled-release.yml (plan item 7).

The release build runs under the ``release-signing`` environment, so the
validate job must be more than a shape check: a correctly-shaped tag on an
unreviewed commit must never reach the signing build. Two layers are tested:

* Structure — the workflow declares the admitted SHA as a job output and
  every privileged job checks out THAT, not the (moveable) tag ref.
* Behavior — the admission script is executed, verbatim from the workflow
  YAML, inside real temp git repositories: a tag on origin/main passes and
  exports the full SHA; a tag-shaped ref NOT on main is refused.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "desktop-bundled-release.yml"
_SIGNING_ENV = "release-signing"


def _workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def _admission_script() -> str:
    """The validate job's run script, verbatim — the single source of truth."""
    steps = _workflow()["jobs"]["validate"]["steps"]
    scripts = [s for s in steps if isinstance(s, dict) and "run" in s]
    assert len(scripts) == 1, "expected exactly one run step in the validate job"
    return scripts[0]["run"]


def _checkout_refs(job: dict) -> list[str | None]:
    refs = []
    for step in job.get("steps", []) or []:
        if isinstance(step, dict) and "checkout" in step.get("uses", ""):
            refs.append(step.get("with", {}).get("ref"))
    return refs


# ---------------------------------------------------------------------------
# Structure: the admitted SHA is the only build input privileged jobs see.
# ---------------------------------------------------------------------------


def test_validate_exports_the_admitted_sha_as_a_job_output():
    wf = _workflow()
    outputs = wf["jobs"]["validate"].get("outputs") or {}
    assert "sha" in outputs, "validate must export the admitted SHA"
    assert "steps.admission.outputs.sha" in outputs["sha"]


def test_admission_script_requires_ancestry_on_origin_main():
    script = _admission_script()
    assert "merge-base --is-ancestor" in script
    assert "origin/main" in script
    assert "::error::" in script, "the refusal must be a loud, greppable error"
    assert "GITHUB_OUTPUT" in script, "the resolved SHA must be exported"


def test_every_signing_job_checks_out_the_admitted_sha_not_the_tag():
    wf = _workflow()
    privileged = {
        name: job
        for name, job in wf["jobs"].items()
        if (isinstance(job, dict) and job.get("environment") == _SIGNING_ENV)
    }
    assert privileged, "walk broken: no release-signing jobs found"

    for name, job in privileged.items():
        refs = _checkout_refs(job)
        for ref in refs:
            assert ref == "${{ needs.validate.outputs.sha }}", (
                f"signing job {name!r} checks out {ref!r} — it must check out "
                "the SHA validate admitted, never the tag ref"
            )
    # jobs that need the SHA must actually need validate
    for name, job in privileged.items():
        needs = job.get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        assert "validate" in needs, f"signing job {name!r} reads needs.validate.outputs.sha but does not need validate"


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


def _run_admission(clone: Path, tag: str) -> subprocess.CompletedProcess:
    gh_output = clone / "github_output.txt"
    gh_output.write_text("", encoding="utf-8")
    env = _child_env(
        TAG=tag,
        GITHUB_OUTPUT=str(gh_output),
        # checkout@v6 runs run-steps with `bash -e -o pipefail`; -e/-o are on
        # the command line below, so nothing else is needed from the env.
    )
    script = _admission_script()
    script_file = clone / "admission.sh"
    script_file.write_text(script, encoding="utf-8")
    return subprocess.run(
        [_BASH, "-e", "-o", "pipefail", str(script_file)],
        cwd=clone,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_tag_on_origin_main_is_admitted_and_exports_the_full_sha(tmp_path: Path):
    _origin, clone = _seed_repo(tmp_path)
    _git("tag", "v0.1.2", cwd=clone)

    proc = _run_admission(clone, "v0.1.2")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    gh_output = (clone / "github_output.txt").read_text(encoding="utf-8")
    expected = _git("rev-parse", "HEAD", cwd=clone)
    assert f"sha={expected}" in gh_output


def test_tag_not_on_origin_main_is_refused(tmp_path: Path):
    _origin, clone = _seed_repo(tmp_path)
    # A commit that exists ONLY in the clone — never pushed, never reviewed.
    (clone / "rogue.txt").write_text("unreviewed\n", encoding="utf-8")
    _git("add", "-A", cwd=clone)
    _git("commit", "-m", "rogue", cwd=clone)
    _git("tag", "v0.1.3-canary.20260830120000", cwd=clone)

    proc = _run_admission(clone, "v0.1.3-canary.20260830120000")
    assert proc.returncode != 0, "a tag off origin/main must not be admitted"
    assert "not an ancestor of origin/main" in proc.stdout + proc.stderr
    # And nothing was exported for the signing jobs to consume.
    assert "sha=" not in (clone / "github_output.txt").read_text(encoding="utf-8")


def test_malformed_tag_is_refused_before_any_git_work(tmp_path: Path):
    _origin, clone = _seed_repo(tmp_path)
    proc = _run_admission(clone, "v0.1.2-rc1")
    assert proc.returncode != 0
    # Refused at the shape/lockstep legs — either message is a valid refusal.
    assert (
        "not a release tag" in proc.stdout + proc.stderr
        or "does not match pyproject.toml version" in proc.stdout + proc.stderr
    )
