"""Invariant: the @assistant-ui dependency cluster agrees on one tap version.

The Hermes desktop app (``apps/desktop``) is built from source on every
install/update via ``scripts/install.ps1`` → ``npm ci``/``npm install`` →
``tsc -b && vite build``. The ``@assistant-ui`` packages share an internal
reactivity lib, ``@assistant-ui/tap``, and they only interoperate when they
all resolve the *same* tap version:

* ``@assistant-ui/react@0.12.28`` and ``@assistant-ui/core`` pin
  ``@assistant-ui/tap@^0.5.x`` (which exports ``.`` and ``./react``).
* ``@assistant-ui/store@0.2.18`` bumped its tap peer to ``^0.9.0`` and started
  importing ``@assistant-ui/tap/react-shim`` — an entry point that only exists
  in the tap ``0.9.x`` line.

Because ``react@0.12.28`` requests ``store@^0.2.9`` (a caret range), a fresh
install silently floated ``store`` up to ``0.2.18``, which then could not find
``./react-shim`` in the hoisted ``tap@0.5.x`` and crashed ``vite build`` with::

    "./react-shim" is not exported ... from package @assistant-ui/tap

i.e. the opaque "apps/desktop build failed (exit 1)" every user hit when
updating. The fix pins ``@assistant-ui/store`` (via root ``overrides``) to the
last release that targets ``tap@^0.5.x``.

This is a *contract* test, not a snapshot: it does not assert specific version
numbers, only that the cluster resolves a single shared tap (wherever npm
places it — hoisted to root, or nested under the ``apps/desktop`` workspace
since the 0.14 bump dropped the ``store`` override) and that this tap satisfies
every ``@assistant-ui/*`` package's declared requirement. It fails if any
future bump reintroduces a split tap version or requirement across the cluster.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TAP = "@assistant-ui/tap"


def _caret_satisfies(version: str, spec: str) -> bool:
    """Minimal npm semver check for the ranges this cluster actually uses.

    Supports exact versions, ``^x.y.z`` (with correct 0.x semantics), and
    ``||`` unions. Pre-release tags are ignored (none are used here).
    """

    def parse(v: str) -> tuple[int, int, int]:
        core = v.lstrip("^~>=<v ").split("-")[0].split("+")[0]
        parts = (core.split(".") + ["0", "0", "0"])[:3]
        return tuple(int(p) for p in parts)  # type: ignore[return-value]

    ver = parse(version)
    for clause in spec.split("||"):
        clause = clause.strip()
        if not clause:
            continue
        if clause.startswith("^"):
            lo = parse(clause)
            if ver < lo:
                continue
            major, minor, _ = lo
            if major > 0:
                hi = (major + 1, 0, 0)
            elif minor > 0:
                hi = (0, minor + 1, 0)
            else:
                hi = (0, 0, lo[2] + 1)
            if ver < hi:
                return True
        elif clause[0].isdigit() or clause.startswith("v"):
            if ver == parse(clause):
                return True
    return False


def _lock_packages() -> dict:
    lock_path = REPO_ROOT / "package-lock.json"
    if not lock_path.exists():
        pytest.skip("package-lock.json not materialized in this CI shard")
    with lock_path.open("r", encoding="utf-8") as fh:
        return json.load(fh).get("packages", {})


def _shared_tap_version(packages: dict) -> str:
    """The one tap version every install site resolves to.

    npm hoists tap to root ``node_modules`` when the cluster lives at the top
    level, but nests the whole cluster under a workspace
    (``apps/desktop/node_modules``) when only that workspace depends on it — as
    it does since the 0.14 bump dropped the ``@assistant-ui/store`` override.
    Either layout is fine; the invariant is that a single tap version is shared
    across every install site, so ``vite build`` never hits a split API.
    """
    versions = {
        meta["version"]
        for key, meta in packages.items()
        if key.rsplit("node_modules/", 1)[-1] == TAP
    }
    assert versions, (
        "package-lock.json has no @assistant-ui/tap entry — the "
        "@assistant-ui cluster should resolve a single shared tap version."
    )
    assert len(versions) == 1, (
        f"@assistant-ui/tap resolves to multiple versions {sorted(versions)} — "
        "the cluster must share one tap line (see this test's docstring)."
    )
    return versions.pop()


def test_assistant_ui_cluster_agrees_on_one_tap() -> None:
    """Every @assistant-ui/* package's tap requirement must be satisfiable.

    Encodes the contract that broke the desktop build: a single hoisted
    @assistant-ui/tap must satisfy the tap range declared by react, core,
    store, and any sibling — otherwise the missing ``./react-shim`` export
    (or a similar API split) breaks ``vite build``.
    """
    packages = _lock_packages()
    tap_version = _shared_tap_version(packages)

    offenders: list[str] = []
    for key, meta in packages.items():
        name = key.rsplit("node_modules/", 1)[-1]
        if not name.startswith("@assistant-ui/") or name == TAP:
            continue
        peer_meta = meta.get("peerDependenciesMeta", {}).get(TAP, {})
        if peer_meta.get("optional"):
            continue
        spec = meta.get("dependencies", {}).get(TAP) or meta.get(
            "peerDependencies", {}
        ).get(TAP)
        if not spec:
            continue
        if not _caret_satisfies(tap_version, spec):
            offenders.append(f"{name} requires {TAP}{spec!r}")

    assert not offenders, (
        f"Hoisted {TAP}@{tap_version} does not satisfy: "
        + "; ".join(offenders)
        + ". The @assistant-ui cluster has split tap requirements — pin the "
        "offending package (e.g. via root package.json `overrides`) so the "
        "whole cluster shares one tap line. See this test's module docstring."
    )


def test_caret_satisfies_helper() -> None:
    """Guard the tiny semver helper the invariant relies on."""
    assert _caret_satisfies("0.5.14", "^0.5.10")
    assert _caret_satisfies("0.5.14", "^0.5.14")
    assert not _caret_satisfies("0.5.14", "^0.9.0")
    assert not _caret_satisfies("0.5.14", "^0.6.0")
    assert _caret_satisfies("1.2.5", "^1.2.0")
    assert not _caret_satisfies("2.0.0", "^1.2.0")
    assert _caret_satisfies("0.5.14", "^0.5.0 || ^0.9.0")
