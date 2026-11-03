"""Find desktop release workflow jobs by what they do, not by their ids.

Job ids in desktop-bundled-release.yml change whenever the build graph is
reshaped (the per-arch legs once had release and commit twins). Tests that
index ``jobs['build-win32-x64']`` break on every such rename while the
property they protect still holds. These lookups key each role on a stable
structural fact: the action a job uses, the receipt it stages, the jobs it
needs. A lookup that finds no match or more than one fails loudly, so a real
graph change cannot silently empty a test.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import hermes_yaml

from scripts.releases.job_groups import JOB_GROUPS

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "desktop-bundled-release.yml"
SMOKE_WORKFLOW = "./.github/workflows/desktop-bundle-smoke.yml"
BUILD_CACHE_ACTION = "./.github/actions/desktop-build-cache"
SHA = "a" * 40
CANARY_TAG = "canary.20260818T101010Z-v0.28.0"
CHANNEL_TAG = "channel.20260818T101010Z-v0.28.0"
COMMIT_TAG = "commit.20260818T101010Z-v0.28.0"
STABLE_TAG = "rc.1-v0.28.0"
PLATFORMS = ("darwin", "win32")
ARCHES = ("arm64", "x64")
NATIVE_TARGETS = tuple(f"{platform}-{arch}" for platform in PLATFORMS for arch in ARCHES)


def load(path: Path = WORKFLOW) -> dict:
    return hermes_yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def needs_of(job: dict) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def _only(matches: list[str], role: str) -> str:
    assert len(matches) == 1, f"expected exactly one {role} job, found {matches}"
    return matches[0]


def native_builds(jobs: dict) -> dict[str, str]:
    """target -> id of that native build leg (one per target: every caller is a claim)."""
    legs: dict[str, str] = {}
    for name, job in jobs.items():
        if not any(step.get("uses") == BUILD_CACHE_ACTION for step in job.get("steps", [])):
            continue
        (target,) = [row["label"] for row in job["strategy"]["matrix"]["target"]]
        assert target not in legs, f"{name} and {legs[target]} both build {target}"
        legs[target] = name
    assert set(legs) == set(NATIVE_TARGETS), legs
    return legs


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


def phase_result(jobs: dict) -> str:
    """The stable-phase verdict: the job that judges every phase job's result."""
    return _only([name for name, job in jobs.items()
                  if any({"RELEASE_NEEDS", "SKIP_TESTS"} <= set(step.get("env") or {})
                         for step in job.get("steps", []))], "stable phase result")


def stage_step(job: dict) -> dict:
    """The step that stages this job's receipt under its claim archive."""
    (step,) = [step for step in job["steps"] if "scripts.releases.handoff stage" in step.get("run", "")]
    return step


def credential_gate(job: dict) -> dict:
    """The step that refuses to build when signing credentials are missing."""
    (step,) = [step for step in job["steps"]
               if step.get("env") and re.search(r"::error::required .*missing", step.get("run", ""))]
    return step


# Stable Release's candidate calls, keyed by claim kind: every run that builds downloadable artifacts.
_CANDIDATE = {"release-phase": "candidate", "jobs": ",".join(JOB_GROUPS)}
DOWNLOADABLE_DISPATCHES = {
    "stable": {"tag": STABLE_TAG, **_CANDIDATE},
    "canary": {"tag": CANARY_TAG, **_CANDIDATE},
    "commit": {"tag": COMMIT_TAG, **_CANDIDATE},
    # A channel claim's candidate: Stable Release passes the request it admitted.
    "channel": {"tag": CHANNEL_TAG, **_CANDIDATE, "channel_build": "b" * 32, "channel_request_sha256": "c" * 64},
}


def admitted(names, *, channel=False):
    """Needs rows for jobs that all succeeded after validate admitted every group."""
    outputs = {"sha": SHA, **{group: "true" for group in JOB_GROUPS}}
    if channel:
        outputs["channel-build"] = "b" * 32
    return {name: {"result": "success", "outputs": dict(outputs)} for name in names}


def evaluate(expression, inputs, needs, *, cancelled=False, failed=False, job_if=True, github=None, env=None,
             steps=None, matrix=None, hash_files=None):
    """Evaluate the workflow expression subset, including Actions' implicit success.

    This is not a scheduler simulation; native Actions still owns cancellation
    and skipped-ancestor propagation. Requiring a status function prevents an
    implicit success() from suppressing consumers of a skipped job.
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
        result = {'inputs': inputs, 'needs': needs, 'github': github or {}, 'env': env or {},
                  'steps': steps or {}, 'matrix': matrix or {}}
        for bit in bits:
            result = result.get(bit, '') if isinstance(result, dict) else ''
        return repr(result)

    expression = re.sub(r'\b(?:inputs|needs|github|env|steps|matrix)(?:\.[\w-]+)+', value, expression)
    expression = expression.replace('&&', ' and ').replace('||', ' or ')
    expression = re.sub(r'!(?!=)', ' not ', expression)
    expression = re.sub(r'\btrue\b', 'True', expression)
    expression = re.sub(r'\bfalse\b', 'False', expression)
    return eval('(' + expression.strip() + ')', {'__builtins__': {}}, {
        'always': lambda: True, 'cancelled': lambda: cancelled,
        'success': lambda: not (cancelled or failed), 'failure': lambda: failed,
        'contains': lambda value, needle: needle in value,
        'startsWith': lambda value, prefix: str(value).startswith(prefix),
        'format': lambda template, *args: template.format(*args),
        # Hashing belongs to the runner. Require explicit results for every pattern.
        'hashFiles': lambda pattern: hash_files[pattern],
        # Expression strings reach eval already rewritten (true -> True), so read them as literals.
        'fromJSON': ast.literal_eval,
    })


def gate(expression, inputs, needs, **options) -> bool:
    return bool(evaluate(expression, inputs, needs, **options))
