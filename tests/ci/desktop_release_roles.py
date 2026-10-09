"""Find desktop release workflow jobs by what they do, not by their ids.

Job ids in desktop-bundled-release.yml change whenever the build graph is
reshaped (one ``build-win32-commit`` job became four per-arch legs). Tests that
index ``jobs['build-win32-x64-commit']`` break on every such rename while the
property they protect still holds. These lookups key each role on a stable
structural fact: the action a job uses, the receipt it stages, the jobs it
needs. A lookup that finds no match or more than one fails loudly, so a real
graph change cannot silently empty a test.
"""
from __future__ import annotations

import re
from pathlib import Path

import hermes_yaml

from scripts.releases.job_groups import JOB_GROUPS

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "desktop-bundled-release.yml"
SMOKE_WORKFLOW = "./.github/workflows/desktop-bundle-smoke.yml"
BUILD_CACHE_ACTION = "./.github/actions/desktop-build-cache"
SHA = "a" * 40
CANARY_TAG = "v0.28.0+canary.20260818T101010Z"
PLATFORMS = ("darwin", "win32")
ARCHES = ("arm64", "x64")
NATIVE_TARGETS = tuple(f"{platform}-{arch}" for platform in PLATFORMS for arch in ARCHES)
# The release branch may write the shared dependency cache; commit and channel
# builds run unreviewed inputs and may only read it.
_MODE_BY_CACHE = {"write": "release", "read": "commit"}


def load(path: Path = WORKFLOW) -> dict:
    return hermes_yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def needs_of(job: dict) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def _only(matches: list[str], role: str) -> str:
    assert len(matches) == 1, f"expected exactly one {role} job, found {matches}"
    return matches[0]


def native_builds(jobs: dict) -> dict[tuple[str, str], str]:
    """``(target, "release" | "commit")`` -> id of that native build leg."""
    legs: dict[tuple[str, str], str] = {}
    for name, job in jobs.items():
        if not any(step.get("uses") == BUILD_CACHE_ACTION for step in job.get("steps", [])):
            continue
        (target,) = [row["label"] for row in job["strategy"]["matrix"]["target"]]
        key = (target, _MODE_BY_CACHE[job["cache-mode"]])
        assert key not in legs, f"{name} and {legs[key]} both build {key}"
        legs[key] = name
    assert {target for target, _ in legs} == set(NATIVE_TARGETS), legs
    assert len(legs) == 2 * len(NATIVE_TARGETS), legs
    return legs


def selection_gates(jobs: dict) -> dict[str, str]:
    """target -> the job that turns "one trust branch ran" into one result."""
    legs = native_builds(jobs)
    gates = {}
    for target in NATIVE_TARGETS:
        pair = {legs[(target, "release")], legs[(target, "commit")]}
        gates[target] = _only([name for name, job in jobs.items() if pair <= set(needs_of(job))],
                              f"{target} selection gate")
    return gates


def smoke_callers(jobs: dict) -> dict[str, str]:
    """target -> the job that calls the reusable native smoke for it."""
    callers: dict[str, str] = {}
    for name, job in jobs.items():
        if job.get("uses") == SMOKE_WORKFLOW:
            target = f"{job['with']['platform']}-{job['with']['arch']}"
            assert target not in callers, f"{name} and {callers[target]} both smoke {target}"
            callers[target] = name
    assert set(callers) == set(NATIVE_TARGETS), callers
    return callers


def _stages(job: dict, receipt: str) -> bool:
    return any("scripts.releases.handoff stage" in step.get("run", "")
               and re.search(rf"--name\s+{re.escape(receipt)}\b", step["run"])
               for step in job.get("steps", []))


def universal_assembler(jobs: dict) -> str:
    return _only([name for name, job in jobs.items() if _stages(job, "windows-universal")],
                 "universal bundle assembly")


def termux_builder(jobs: dict) -> str:
    return _only([name for name, job in jobs.items() if _stages(job, "termux")], "termux deb")


def updater_publishers(jobs: dict) -> dict[str, str]:
    """platform -> the job that publishes that platform's updater feed.

    It is the job gated on exactly that platform's smokes; channel publication
    and summaries need every platform's smoke.
    """
    smokes = smoke_callers(jobs)
    publishers = {}
    for platform in PLATFORMS:
        own = {smokes[f"{platform}-{arch}"] for arch in ARCHES}
        publishers[platform] = _only(
            [name for name, job in jobs.items() if set(needs_of(job)) & set(smokes.values()) == own],
            f"{platform} updater publisher")
    return publishers


def _renders(job: dict, flag: str) -> bool:
    return any("render-builds-table.py" in step.get("run", "") and flag in step["run"]
               for step in job.get("steps", []))


def channel_publisher(jobs: dict) -> str:
    return _only([name for name, job in jobs.items() if _renders(job, "--channel-build")],
                 "channel publisher")


def commit_summary(jobs: dict) -> str:
    return _only([name for name, job in jobs.items() if _renders(job, "--summary-commit")],
                 "commit build summary")


def tag_summary(jobs: dict) -> str:
    """The tag build table, which the canary publisher waits on."""
    canary = jobs[canary_publisher(jobs)]
    return _only([name for name in needs_of(canary) if _renders(jobs[name], "--tag")], "tag build summary")


def canary_publisher(jobs: dict) -> str:
    return _only([name for name, job in jobs.items()
                  if any("scripts.releases.channel_releases canary" in step.get("run", "")
                         for step in job.get("steps", []))], "canary publisher")


def phase_result(jobs: dict) -> str:
    """The stable-phase verdict: the job that judges every phase job's result."""
    return _only([name for name, job in jobs.items()
                  if any({"RELEASE_NEEDS", "SKIP_TESTS"} <= set(step.get("env") or {})
                         for step in job.get("steps", []))], "stable phase result")


def stage_step(job: dict, *, channel: bool = False) -> dict:
    """The step that stages this job's receipt, for a channel or tag/commit build."""
    flag = "--channel-request" if channel else "--commit-build"
    (step,) = [step for step in job["steps"]
               if "scripts.releases.handoff stage" in step.get("run", "") and flag in step["run"]]
    return step


def credential_gate(job: dict) -> dict:
    """The step that refuses to build when signing credentials are missing."""
    (step,) = [step for step in job["steps"]
               if step.get("env") and re.search(r"::error::required .*missing", step.get("run", ""))]
    return step


# Dispatch shapes that produce downloadable artifacts, keyed by trust branch.
DOWNLOADABLE_DISPATCHES = {
    "tag": {"tag": CANARY_TAG, "upload_release": True, "build_commit": "", "channel": "",
            "release-phase": "", "jobs": ",".join(JOB_GROUPS)},
    "candidate": {"tag": "v0.28.0", "upload_release": False, "build_commit": "", "channel": "",
                  "release-phase": "candidate", "jobs": ",".join(JOB_GROUPS)},
    "commit": {"tag": "", "upload_release": False, "build_commit": SHA, "channel": "",
               "release-phase": "", "jobs": ",".join(JOB_GROUPS)},
    "channel": {"tag": "", "upload_release": False, "build_commit": SHA, "channel": "preview",
                "release-phase": "", "jobs": ",".join(JOB_GROUPS)},
}
# A dry build: nothing leaves the runner.
DRY_DISPATCH = {"tag": "v0.28.0", "upload_release": False, "build_commit": "", "channel": "",
                "release-phase": "", "jobs": ",".join(JOB_GROUPS)}


def admitted(names, *, channel=False):
    """Needs rows for jobs that all succeeded after validate admitted every group."""
    outputs = {"sha": SHA, **{group: "true" for group in JOB_GROUPS}}
    if channel:
        outputs["channel-build"] = "b" * 32
    return {name: {"result": "success", "outputs": dict(outputs)} for name in names}


def evaluate(expression, inputs, needs, *, cancelled=False, failed=False, job_if=True, github=None, env=None,
             steps=None):
    """Evaluate the workflow expression subset, including Actions' implicit success.

    This is not a scheduler simulation; native Actions still owns cancellation
    and skipped-ancestor propagation. Requiring a status function prevents an
    implicit success() from suppressing consumers of the skipped trust branch.
    ``job_if`` judges implicit success by the needs' results; a step ``if``
    (``job_if=False``) judges it by whether an earlier step ``failed``.
    """
    expression = str(expression).strip().removeprefix('${{').removesuffix('}}').strip()
    if not re.search(r'\b(always|cancelled|success|failure)\(', expression):
        if cancelled or failed:
            return False
        if job_if and any(row.get('result') != 'success' for row in needs.values()):
            return False

    def value(match):
        bits = match[0].split('.')
        result = {'inputs': inputs, 'needs': needs, 'github': github or {}, 'env': env or {}, 'steps': steps or {}}
        for bit in bits:
            result = result.get(bit, '') if isinstance(result, dict) else ''
        return repr(result)

    expression = re.sub(r'\b(?:inputs|needs|github|env|steps)(?:\.[\w-]+)+', value, expression)
    expression = expression.replace('&&', ' and ').replace('||', ' or ')
    expression = re.sub(r'!(?!=)', ' not ', expression)
    expression = re.sub(r'\btrue\b', 'True', expression)
    expression = re.sub(r'\bfalse\b', 'False', expression)
    return eval('(' + expression.strip() + ')', {'__builtins__': {}}, {
        'always': lambda: True, 'cancelled': lambda: cancelled,
        'success': lambda: not (cancelled or failed), 'failure': lambda: failed,
        'contains': lambda value, needle: needle in value,
    })


def gate(expression, inputs, needs, **options) -> bool:
    return bool(evaluate(expression, inputs, needs, **options))
