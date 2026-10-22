"""The bootstrap cannot read a stamped ``runtimeDir`` — and no packager may ship one.

``pm.environments.store_root()`` resolves the store from, in order: the
``HERMES_RUNTIME_DIR`` override, a sealed payload's manifest, an
``install-stamp.json`` field ``runtimeDir``, the borrowing root, and finally
``get_default_hermes_root()/tools``.

A bootstrap runs before Python exists, so it can resolve only the parts that need
nothing but the environment: the explicit override and the machine root. The
manifest and the stamp are files it cannot parse, so its answer is the machine
root's ``tools`` regardless of what the tree contains.

That is a real divergence, and it is harmless only because **no shipped packager
writes ``runtimeDir`` today**. This module makes both halves executable:

* :func:`test_a_stamped_runtime_dir_is_invisible_to_the_bootstrap` shows the split
  — same tree, bootstrap and PM naming different stores;
* :func:`test_no_shipped_packager_pins_a_runtime_dir` fails the moment a packager
  makes that split reachable.

Reading source here is the same deliberate exception the store-root suite already
takes: CI YAML and shell installers are not importable, and the property under test
is "no shipped packager does X", which only the shipping files can answer.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_SH = REPO_ROOT / "scripts" / "bootstrap" / "store_root.sh"

#: Files that can bake a stamp into a shipped artifact. A packager added later
#: belongs here; anything that writes ``runtimeDir`` outside this list is caught by
#: the tree-wide sweep in :func:`test_no_shipped_packager_pins_a_runtime_dir`.
_PACKAGER_GLOBS = (
    ".github/workflows/*.yml",
    "scripts/termux/*.sh",
    "nix/*.nix",
    "docker/**/*.sh",
)


def _packagers() -> list[Path]:
    found: list[Path] = []
    for pattern in _PACKAGER_GLOBS:
        found.extend(sorted(REPO_ROOT.glob(pattern)))
    return found


def _tree_with_a_stamped_runtime_dir(root: Path, runtime_dir: str) -> Path:
    """An install tree whose stamp names a store, as a packager would write it.

    Returns the checkout path. The stamp sits beside it, where
    ``pm.paths.install_stamp_path`` and ``store_root``'s parent walk both look.
    """
    checkout = root / "hermes-agent"
    checkout.mkdir(parents=True)
    (root / "install-stamp.json").write_text(
        json.dumps({"commit": "0" * 40, "runtimeDir": runtime_dir}), encoding="utf-8"
    )
    return checkout


@pytest.mark.platforms("posix")
def test_a_stamped_runtime_dir_is_invisible_to_the_bootstrap(real_bash, tmp_path: Path, monkeypatch) -> None:
    """Same tree: PM reads the stamp, the bootstrap cannot, so they name two stores.

    This is the gap the contract has to live with. It is asserted rather than
    described so that whoever teaches the bootstrap to read the stamp sees this
    test change, and whoever makes the gap reachable sees the packager guard fail.
    """
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "native-home"))
    root = tmp_path / "install-root"
    # Absolute, because ``store_root`` resolves the field against the caller's cwd:
    # a relative value would make this row assert a resolution base rather than the
    # stamp contract.
    stamped_store = tmp_path / "payload-store"
    checkout = _tree_with_a_stamped_runtime_dir(root, str(stamped_store))

    from pm.environments import store_root as pm_store_root

    pm_answer = pm_store_root(checkout)
    assert pm_answer == stamped_store, f"PM did not honor runtimeDir: {pm_answer}"

    driver = tmp_path / "root.sh"
    driver.write_text(
        f'. "{CANONICAL_SH.as_posix()}"\n'
        f'HERMES_HOME="{root.as_posix()}" hermes_root_of\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [real_bash, driver.as_posix()], env={k: v for k, v in os.environ.items() if not k.startswith("HERMES_")},
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    bootstrap_answer = Path(result.stdout.strip()) / "tools"

    assert bootstrap_answer != pm_answer, (
        "the bootstrap unexpectedly agreed with the stamp — if reading it became "
        "possible, this contract can be retired and the packager guard with it"
    )
    assert bootstrap_answer == root / "tools", bootstrap_answer


def test_no_shipped_packager_pins_a_runtime_dir() -> None:
    """Nothing shipped may make the split above reachable.

    A packager that writes ``runtimeDir`` gets a store the bootstrap will not find:
    the bootstrap stages a sha256-verified uv into ``<root>/tools`` while PM looks in
    the stamped directory, and every install re-downloads. Either teach the bootstrap
    to read the stamp first, or do not write one.
    """
    offenders: list[str] = []
    for path in _packagers():
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        # A stamp writer takes it as a flag or a keyword; a baked manifest uses the
        # same field name.
        for match in re.finditer(r"--runtime-dir|runtime_dir\s*[=:]|\"runtimeDir\"|runtimeDir\s*:", text):
            line = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{line}")
    assert not offenders, (
        "a shipped packager pins a runtimeDir, which a pre-Python bootstrap cannot "
        "read: bootstrap and PM would stage into different stores (#101269). Teach "
        "scripts/bootstrap/store_root.{sh,ps1} to read the stamp first, or drop the "
        "field:\n  " + "\n  ".join(offenders)
    )


def test_the_packager_guard_actually_scans_something() -> None:
    """A glob that matches nothing would make the guard above vacuously pass."""
    files = _packagers()
    assert len(files) >= 2, f"the packager globs matched {len(files)} file(s): {files}"
    assert any(p.name == "docker.yml" for p in files), files
