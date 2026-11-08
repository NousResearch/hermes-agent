"""The manifest's ``engines`` must be satisfiable by a toolchain we can actually ship.

`engineStrict: true` in `pnpm-workspace.yaml` makes `engines` a hard gate on every
`pnpm install` — the installer's workspace step, `hermes update`'s dependency refresh,
and CI alike. So a floor nobody's toolchain can meet is not a strict-hygiene win; it
is a total install outage.

That is exactly what happened once: `engines.npm` was raised to `>=12.0.0` while
**no Node release bundled npm 12**. Every fresh install died at the first install,
and `hermes update` left installs in a mixed state. These tests encode the
invariants that would have caught it.

Deliberately behavioral, not a snapshot: nothing here pins a version we
expect to change. Each test asserts a *relationship* — between the floor we
declare and the toolchain that has to satisfy it.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def _root_manifest() -> dict:
    return json.loads((REPO_ROOT / "package.json").read_text())

def _pnpm_lock() -> dict:
    """pnpm-lock.yaml, which is one YAML document (pmOnFail: ignore writes no pin document)."""
    from ruamel.yaml import YAML

    documents = list(YAML(typ="safe").load_all((REPO_ROOT / "pnpm-lock.yaml").read_text(encoding="utf-8")))
    assert len(documents) == 1, f"pnpm-lock.yaml: expected one YAML document, found {len(documents)}"
    return documents[0]

def _pm_lock():
    from pm.lock import Lockfile

    return Lockfile(REPO_ROOT / "pm" / "lock.json")

def _parse_major_minor_patch(version: str) -> tuple[int, int, int]:
    parts = version.split("-", 1)[0].split(".")
    nums = [int(p) for p in parts[:3]]
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]

def _satisfies_clause(version: str, clause: str) -> bool:
    """Evaluate one `>=x.y.z` / `<x.y.z` / `^x.y.z` comparator against *version*."""
    clause = clause.strip()
    if clause.startswith("^"):
        bound = clause[1:].strip()
        have = _parse_major_minor_patch(version)
        want = _parse_major_minor_patch(bound)
        # ^x.y.z allows >=x.y.z within the same major (x > 0).
        return have[0] == want[0] and have >= want
    for op in (">=", "<=", "<", ">", "="):
        if clause.startswith(op):
            bound = clause[len(op) :].strip()
            break
    else:
        op, bound = "=", clause
    have = _parse_major_minor_patch(version)
    want = _parse_major_minor_patch(bound)
    if op == ">=":
        return have >= want
    if op == "<=":
        return have <= want
    if op == "<":
        return have < want
    if op == ">":
        return have > want
    return have == want

def _satisfies_range(version: str, spec: str) -> bool:
    """Evaluate the `A || B` / space-joined-AND subset of semver we author."""
    for alternative in spec.split("||"):
        clauses = [c for c in alternative.strip().split() if c]
        if clauses and all(_satisfies_clause(version, c) for c in clauses):
            return True
    return False

class TestEnginesAreSatisfiable:
    def test_node_floor_is_met_by_the_managed_runtime(self):
        """The Node PM provisions must clear engines.node."""
        node_range = _root_manifest()["engines"]["node"]
        managed_node = _pm_lock().version("node")
        assert managed_node, "pm/lock.json does not pin node"
        assert _satisfies_range(managed_node, node_range), (
            f"engines.node is {node_range!r} but PM provisions Node "
            f"{managed_node}. The runtime we ship must satisfy the floor we "
            "declare, or the install we just performed cannot install deps."
        )

def _normalize_range(spec: str) -> str:
    """Normalize the wilder styles real deps publish so our tiny evaluator
    can read them: collapse space after operators (``">= 10"``), drop ``v``
    prefixes (``">=v12.22.7"``), and rewrite ``x``/``*`` wildcards to floors.
    """
    import re

    spec = re.sub(r"(>=|<=|>|<|\^|~|=)\s+", r"\1", spec)
    spec = re.sub(r"(>=|<=|>|<|\^|~|=)v", r"\1", spec)
    # "6.x" / "10.*" -> "^6.0.0"-ish floor within the major; ">= 10.*" -> ">=10.0.0"
    spec = re.sub(r"(\d+)\.[x*](?:\.[x*])?", r"\1.0.0", spec)
    return spec

class TestDeclaredFloorsClearTheLockedTree:
    """Every Node version our own gates accept must survive `pnpm install`.

    The class of outage this pins: the installers' version gates
    (node_satisfies_build in install.sh, Test-NodeVersionOk in install.ps1)
    and `engines.node` are hand-maintained, while the *real* floor is
    whatever the strictest locked dependency demands. When they drift, a
    user's system Node clears every gate we own and then dies at
    `pnpm install` with ERR_PNPM_UNSUPPORTED_ENGINE under engineStrict.

    Aug 2026 instance: @babel/* 8.x requires `^22.18.0 || >=24.11.0`; our
    engines arm said `^24.0.0`, so Node 24.4 passed the installer and the
    manifest and failed on 28 babel packages.
    """

    def _arm_floors(self, node_range: str) -> list[str]:
        floors = []
        for arm in node_range.split("||"):
            arm = arm.strip()
            for op in ("^", ">=", "="):
                if arm.startswith(op):
                    floors.append(arm[len(op):].strip())
                    break
            else:
                floors.append(arm)
        return floors

    def _locked_node_ranges(self) -> dict[str, str]:
        ranges: dict[str, str] = {}
        for path, meta in _pnpm_lock()["packages"].items():
            engines = meta.get("engines")
            if not isinstance(engines, dict):
                continue
            node_range = engines.get("node")
            if isinstance(node_range, str) and node_range.strip() not in ("", "*"):
                ranges.setdefault(node_range, path)
        return ranges

    def test_every_engines_arm_floor_clears_every_locked_dependency(self):
        node_range = _root_manifest()["engines"]["node"]
        violations = []
        for floor in self._arm_floors(node_range):
            for dep_range, example in self._locked_node_ranges().items():
                if not _satisfies_range(floor, _normalize_range(dep_range)):
                    violations.append((floor, dep_range, example))
        assert not violations, (
            "engines.node arms admit Node versions the locked dependency "
            "tree rejects — those users pass every install gate and then "
            "die at `pnpm install` with ERR_PNPM_UNSUPPORTED_ENGINE (engineStrict). "
            "Raise the arm floor (and the installer gates: "
            "node_satisfies_build in scripts/install.sh, Test-NodeVersionOk "
            f"in scripts/install.ps1) or relax the dep. Violations: {violations}"
        )
