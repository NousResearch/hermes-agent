"""The managed-Node staleness gate must honor the FULL ``engines.node`` floor.

``.npmrc`` sets ``engine-strict=true``, so ``engines.node`` is a hard gate on every
``npm ci`` / ``npm install`` — the installer's workspace step, ``hermes update``'s
dependency refresh, and the desktop rebuild alike.

Hermes self-heals a stale managed tree: ``find_hermes_node_executable()`` redownloads
when ``_managed_node_tree_outdated()`` says so. That check compared only the MAJOR, so a
Hermes-managed Node 22.14.0 tree was judged *current* while npm rejected that same tree
with EBADENGINE against ``^22.22.0``. Nothing else rescued the user:
``npm_engine.required_npm_range()`` returns ``None`` for a node-side EBADENGINE by design,
and ``with_hermes_node_path()`` never triggers a heal. The comparator was the only thing
standing between those users and a working update.

These are behaviour contracts, not snapshots: each asserts a *relationship* between the
floor the manifest declares and the tree the staleness gate accepts.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import hermes_constants

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"


def _engines_node_floor() -> tuple[int, int, int]:
    """Lowest version ``engines.node`` accepts, parsed from the root manifest.

    Derived from the manifest rather than hardcoded so the tests track the floor.
    """
    spec = json.loads((REPO_ROOT / "package.json").read_text())["engines"]["node"]
    first = spec.split("||")[0].strip().lstrip("^><=~ ")
    return hermes_constants._parse_node_version(first)


def _stub_node(bin_dir: Path, version: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    node = bin_dir / "node"
    node.write_text(f"#!/bin/sh\necho '{version}'\nexit 0\n")
    node.chmod(0o755)
    return node


class TestPythonStalenessGate:
    def test_within_major_stale_tree_is_outdated(self, tmp_path, monkeypatch):
        """A managed tree below the floor but at the target major must be healed.

        This is the reported failure: managed Node v22.14.0 against ``^22.22.0``.
        """
        floor = _engines_node_floor()
        home = tmp_path / "home"
        _stub_node(home / "node" / "bin", f"v{floor[0]}.{max(floor[1] - 1, 0)}.0")
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert hermes_constants._managed_node_tree_outdated(home) is True

    def test_tree_at_the_floor_is_current(self, tmp_path, monkeypatch):
        """Exactly the floor satisfies it — the heal must not loop on a good tree."""
        floor = _engines_node_floor()
        home = tmp_path / "home"
        _stub_node(home / "node" / "bin", f"v{floor[0]}.{floor[1]}.{floor[2]}")
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert hermes_constants._managed_node_tree_outdated(home) is False

    def test_tree_above_the_floor_is_current(self, tmp_path, monkeypatch):
        floor = _engines_node_floor()
        home = tmp_path / "home"
        _stub_node(home / "node" / "bin", f"v{floor[0]}.{floor[1] + 5}.3")
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert hermes_constants._managed_node_tree_outdated(home) is False

    def test_prerelease_is_outdated_whatever_its_version(self, tmp_path, monkeypatch):
        """nodejs.org publishes headers only for final releases (node-gyp/node-pty)."""
        home = tmp_path / "home"
        _stub_node(home / "node" / "bin", "v99.0.0-alpha.1")
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert hermes_constants._managed_node_tree_outdated(home) is True

    def test_declared_floor_matches_the_manifest(self):
        """The gate's floor must not drift from the floor npm actually enforces."""
        assert hermes_constants._HERMES_NODE_TARGET_MIN == _engines_node_floor()

    def test_stale_tree_triggers_the_heal_on_resolution(self, tmp_path, monkeypatch):
        """End of the chain: a stale tree must actually reach the redownload."""
        floor = _engines_node_floor()
        home = tmp_path / "home"
        node = _stub_node(home / "node" / "bin", f"v{floor[0]}.{max(floor[1] - 1, 0)}.0")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)

        healed = {"value": False}

        def _heal():
            healed["value"] = True
            node.write_text(f"#!/bin/sh\necho 'v{floor[0]}.{floor[1] + 1}.0'\nexit 0\n")
            node.chmod(0o755)
            return True

        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", _heal)

        assert hermes_constants.find_hermes_node_executable("node") == str(node)
        assert healed["value"] is True


class TestParseNodeVersion:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("v22.22.0", (22, 22, 0)),
            ("22.22.0", (22, 22, 0)),
            ("v26.0.0-alpha.1", (26, 0, 0)),
            ("v24", (24, 0, 0)),
            ("v24.11", (24, 11, 0)),
        ],
    )
    def test_parses_the_shapes_node_emits(self, raw, expected):
        assert hermes_constants._parse_node_version(raw) == expected

    @pytest.mark.parametrize("raw", ["", "not-a-version", "v.x"])
    def test_rejects_garbage(self, raw):
        with pytest.raises(ValueError):
            hermes_constants._parse_node_version(raw)

    def test_malformed_override_degrades_instead_of_raising(self, monkeypatch):
        """This module is imported at load time from 30+ sites; it must never raise."""
        monkeypatch.setenv("HERMES_NODE_TARGET_MIN_VERSION", "garbage")
        assert hermes_constants._target_node_min() == (
            hermes_constants._HERMES_NODE_TARGET_MAJOR, 0, 0)


def _bash(snippet: str, *, node_version: str, home: Path) -> int:
    """Source node-bootstrap.sh against a stubbed managed tree and run *snippet*."""
    bin_dir = home / "node" / "bin"
    _stub_node(bin_dir, node_version)
    result = subprocess.run(
        ["bash", "-c", f'export HERMES_HOME="{home}"; . "{BOOTSTRAP}" >/dev/null 2>&1; {snippet}'],
        capture_output=True, text=True, check=False,
    )
    return result.returncode


@pytest.mark.linux_only
class TestShellStalenessMirror:
    """``_nb_managed_node_outdated`` is documented as a mirror of the Python gate.

    Both are load-bearing: POSIX heals shell out to this function. They are asserted
    together so they cannot drift apart again.
    """

    def test_within_major_stale_tree_is_outdated(self, tmp_path):
        floor = _engines_node_floor()
        stale = f"v{floor[0]}.{max(floor[1] - 1, 0)}.0"
        assert _bash("_nb_managed_node_outdated", node_version=stale, home=tmp_path) == 0

    def test_tree_at_the_floor_is_current(self, tmp_path):
        floor = _engines_node_floor()
        good = f"v{floor[0]}.{floor[1]}.{floor[2]}"
        assert _bash("_nb_managed_node_outdated", node_version=good, home=tmp_path) != 0

    def test_prerelease_is_outdated(self, tmp_path):
        assert _bash(
            "_nb_managed_node_outdated", node_version="v99.0.0-rc.1", home=tmp_path) == 0

    def test_shell_floor_matches_the_python_floor(self, tmp_path):
        """One floor, two implementations — a drift here re-opens the outage."""
        result = subprocess.run(
            ["bash", "-c",
             f'. "{BOOTSTRAP}" >/dev/null 2>&1; printf "%s" "$HERMES_NODE_TARGET_MIN_VERSION"'],
            capture_output=True, text=True, check=False,
        )
        shell_floor = hermes_constants._parse_node_version(result.stdout.strip())
        assert shell_floor == hermes_constants._HERMES_NODE_TARGET_MIN
