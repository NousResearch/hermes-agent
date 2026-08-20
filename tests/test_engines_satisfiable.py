"""The manifest's ``engines`` must be satisfiable by a toolchain we can actually ship.

`engine-strict=true` in `.npmrc` makes `engines` a hard gate on every
`npm ci` / `npm install` — the installer's workspace step, `hermes update`'s
dependency refresh, and CI alike. So a floor nobody's toolchain can meet is
not a strict-hygiene win; it is a total install outage.

That is exactly what happened: `engines.npm` was raised to `>=12.0.0` while
**no Node release bundles npm 12** (Node 26 ships 11.17.0, 24 ships 11.16.0,
22 ships 10.9.8). Every fresh install died at the first `npm ci`, and
`hermes update` left installs in a mixed state. These tests encode the
invariants that would have caught it.

Deliberately behavioral, not a snapshot: nothing here pins a version we
expect to change. Each test asserts a *relationship* — between the floor we
declare and the toolchain that has to satisfy it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# npm releases bundled with a Node major, newest-per-major. Not a catalog
# snapshot: the point is that *some* real, shipping toolchain must clear the
# floor, and these are the ones users actually arrive with.
_STOCK_NPM_BY_NODE_MAJOR = {
    20: "10.8.2",
    22: "10.9.8",
    24: "11.16.0",
    26: "11.17.0",
}


def _root_manifest() -> dict:
    return json.loads((REPO_ROOT / "package.json").read_text())


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
    def test_npm_floor_is_met_by_a_shipping_node(self):
        """Some stock Node must bundle an npm our floor accepts.

        Without this, a fresh install cannot run `npm ci` at all: the
        installer provisions a Node from nodejs.org and immediately uses the
        npm that came with it.
        """
        npm_range = _root_manifest()["engines"]["npm"]
        satisfying = {
            major: npm
            for major, npm in _STOCK_NPM_BY_NODE_MAJOR.items()
            if _satisfies_range(npm, npm_range)
        }
        assert satisfying, (
            f"engines.npm is {npm_range!r}, which no shipping Node bundles "
            f"(checked {_STOCK_NPM_BY_NODE_MAJOR}). With engine-strict=true "
            "every fresh install fails at the first `npm ci`."
        )

    def test_node_floor_is_met_by_the_managed_runtime(self):
        """The Node major the installers provision must clear engines.node."""
        node_range = _root_manifest()["engines"]["node"]
        install_sh = (REPO_ROOT / "scripts" / "install.sh").read_text()
        for line in install_sh.splitlines():
            if line.startswith("NODE_VERSION="):
                managed_major = int(line.split("=", 1)[1].strip().strip('"').strip("'"))
                break
        else:  # pragma: no cover - install.sh always defines it
            pytest.fail("install.sh does not define NODE_VERSION")

        # install.sh fetches latest-v{major}.x, not {major}.0.0. Use a high
        # representative release from that major so ranges that enumerate LTS
        # lines (rather than one continuous floor) are checked correctly.
        managed_release = f"{managed_major}.999.999"
        assert _satisfies_range(managed_release, node_range), (
            f"engines.node is {node_range!r} but install.sh provisions Node "
            f"{managed_major}.x. The runtime we ship must satisfy the floor we "
            "declare, or the install we just performed cannot install deps."
        )

    def test_managed_node_bundles_an_npm_the_engines_accept(self):
        """The Node major install.sh fetches must ship an npm that clears
        engines.npm. Node 22 bundles 11.16.0, which is in the excluded
        11.10–11.16 band — fresh Hermes-managed installs then die at
        `npm ci` with EBADENGINE (#80769).
        """
        npm_range = _root_manifest()["engines"]["npm"]
        install_sh = (REPO_ROOT / "scripts" / "install.sh").read_text()
        for line in install_sh.splitlines():
            if line.startswith("NODE_VERSION="):
                managed_major = int(line.split("=", 1)[1].strip().strip('"').strip("'"))
                break
        else:  # pragma: no cover
            pytest.fail("install.sh does not define NODE_VERSION")
        stock_npm = _STOCK_NPM_BY_NODE_MAJOR.get(managed_major)
        assert stock_npm is not None, (
            f"install.sh NODE_VERSION={managed_major} is not in the known "
            f"stock map {_STOCK_NPM_BY_NODE_MAJOR}"
        )
        assert _satisfies_range(stock_npm, npm_range), (
            f"install.sh provisions Node {managed_major}.x (stock npm "
            f"{stock_npm}), but engines.npm is {npm_range!r}. A fresh "
            "Hermes-managed install cannot run npm ci."
        )

    def test_desktop_node_floor_is_not_stricter_than_its_toolchain(self):
        """apps/desktop must not demand more Node than its own build tools do.

        Vite is the real constraint (it needs `node:util.styleText`). Raising
        the desktop floor beyond it silently force-migrates every user's
        toolchain for no dependency reason.
        """
        desktop = json.loads((REPO_ROOT / "apps" / "desktop" / "package.json").read_text())
        node_range = desktop["engines"]["node"]
        # The tightest floor any dependency actually declares (react-router
        # 8.3.0 -> >=22.22.0). If this legitimately rises, the assertion
        # documents the reason for the bump rather than blocking it.
        assert _satisfies_range("22.22.0", node_range), (
            f"apps/desktop engines.node is {node_range!r}, which rejects Node "
            "22.12 — stricter than Vite requires. A desktop floor above the "
            "build toolchain's own floor replaces working user toolchains for "
            "nothing."
        )


class TestExcludedNpmBand:
    """npm 11.10–11.16 honor `min-release-age` but ignore `min-release-age-exclude`.

    `.npmrc` sets both, so that band applies the 14-day age gate to packages
    we deliberately exempted and installs fail with ETARGET. The floor must
    keep excluding them.
    """

    @pytest.mark.parametrize("bad_npm", ["11.10.0", "11.12.1", "11.16.0"])
    def test_band_that_ignores_the_exclude_list_is_rejected(self, bad_npm):
        npm_range = _root_manifest()["engines"]["npm"]
        assert not _satisfies_range(bad_npm, npm_range), (
            f"engines.npm {npm_range!r} accepts npm {bad_npm}, which supports "
            "min-release-age but not min-release-age-exclude — it will fail "
            "ETARGET on any freshly published dependency in .npmrc's exclude list."
        )

    @pytest.mark.parametrize("good_npm", ["10.9.8", "11.17.0", "12.0.2"])
    def test_versions_handling_the_exclude_list_are_accepted(self, good_npm):
        npm_range = _root_manifest()["engines"]["npm"]
        assert _satisfies_range(good_npm, npm_range), (
            f"engines.npm {npm_range!r} rejects npm {good_npm}, which handles "
            ".npmrc correctly and should be usable."
        )


class TestManifestMirrors:
    def test_lockfile_engines_match_the_manifest(self):
        """A stale lockfile mirror re-imposes the old floor on `npm ci`."""
        manifest = _root_manifest()["engines"]
        lock = json.loads((REPO_ROOT / "package-lock.json").read_text())
        assert lock["packages"][""]["engines"] == manifest
