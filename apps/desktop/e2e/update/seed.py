"""Seed and reset the real local install the Desktop update suite drives.

The install is made by the real entry points a Linux user runs (``scripts/install.sh`` and then
``hermes desktop --build-only``) against a local bare origin, inside the upgrade suite's sandbox
machinery (``tests/e2e/core/upgrade``). Building it takes minutes, so it is built once per run into
``<root>/sb`` and snapshotted; each spec restores the snapshot back into the SAME path (the install
bakes absolute paths into launchers and PM facts, so a copy elsewhere would not be the same install).

    python seed.py install <root>     -> JSON {home, hermesHome, checkout, origin, env, headSha}
    python seed.py restore <root>     -> the snapshot back in place (install + origin)
    python seed.py publish <root> <message> <relpath> <content-file>  -> sha of the new origin/main

HERMES_E2E_UPDATE_INSTALL_REF=<sha> installs that commit instead of HEAD (local A/B against an open
fix, or a sabotage commit, without touching the checkout the suite runs from).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I


def _paths(root: Path) -> dict[str, Path]:
    return {
        "sb": root / "sb",
        "origin": root / "origin.git",
        "golden": root / "golden",
        "facts": root / "install.json",
    }


def _sandbox(root: Path) -> I.Sandbox:
    facts = json.loads(_paths(root)["facts"].read_text(encoding="utf-8"))
    return I.Sandbox(root=Path(facts["sandboxRoot"]), env=facts["env"])


def _fail(what: str, cp: subprocess.CompletedProcess) -> None:
    sys.stderr.write(f"{what} failed:\n{H.describe(cp)}\n")
    raise SystemExit(1)


def install(root: Path) -> None:
    p = _paths(root)
    root.mkdir(parents=True, exist_ok=True)
    p["facts"].unlink(missing_ok=True)
    for key in ("sb", "origin", "golden"):
        shutil.rmtree(p[key], ignore_errors=True)
    head = os.environ.get("HERMES_E2E_UPDATE_INSTALL_REF") or I.head_sha()
    origin = I.make_origin(root, head)
    # Serve full clones. When a CI runner's git honours the installer's --filter=tree:0 against a
    # local file:// origin, the clone's lazy tree fetches fan out into more than 1300 concurrent
    # upload-packs, and the runner is OOM-killed about 90 s into seeding. The Desktop cells need an
    # installed checkout, not a treeless one.
    I.git("config", "uploadpack.allowFilter", "false", cwd=origin)
    sb = I.new_sandbox(p["sb"], origin)
    # The installer's treeless clone still marks the checkout as a promisor, and on the CI
    # runner's git every missing-object probe then starts a lazy fetch whose own probe starts
    # another one (a chain of 78+ nested `fetch --filter=blob:none --stdin`). The clone is full, so
    # there is nothing to lazily fetch; stop the chain for the installer run only.
    cp = I.run_installer(I.Sandbox(root=sb.root, env={**sb.env, "GIT_NO_LAZY_FETCH": "1"}))
    if cp.returncode != 0:
        _fail("scripts/install.sh --non-interactive", cp)
    # publishUpstream() lands commits on main, never a release: follow main, not the stable default.
    I.pin_main_channel(sb)
    # The real way a CLI install gets the Desktop app on Linux: build + package into
    # apps/desktop/release/linux-unpacked, the tree the Desktop updater swaps in place.
    cp = sb.cli("desktop", "--build-only", timeout=1800)
    if cp.returncode != 0:
        _fail("hermes desktop --build-only", cp)
    facts = {
        "sandboxRoot": str(sb.root),
        "home": str(sb.home),
        "hermesHome": str(sb.hermes_home),
        "checkout": str(sb.checkout),
        "hermes": sb.hermes,
        "origin": str(origin),
        "env": sb.env,
        "headSha": head,
    }
    p["golden"].mkdir()
    for key in ("sb", "origin"):
        subprocess.run(["cp", "-a", str(p[key]), str(p["golden"] / key)], check=True)
    # Written last: its presence means the snapshot is complete (REUSE keys on it).
    p["facts"].write_text(json.dumps(facts, indent=1), encoding="utf-8")
    print(json.dumps(facts))


def restore(root: Path) -> None:
    p = _paths(root)
    for key in ("sb", "origin"):
        shutil.rmtree(p[key], ignore_errors=True)
        subprocess.run(["cp", "-a", str(p["golden"] / key), str(p[key])], check=True)
    print(p["facts"].read_text(encoding="utf-8"))


def publish(root: Path, message: str, rel: str, content_file: str) -> None:
    p = _paths(root)
    scratch = root / "publish"
    scratch.mkdir(exist_ok=True)
    text = Path(content_file).read_text(encoding="utf-8")
    print(I.publish_commit(p["origin"], scratch, message, {rel: text}))


def main(argv: list[str]) -> None:
    cmd, root = argv[0], Path(argv[1]).resolve()
    if cmd == "install":
        install(root)
    elif cmd == "restore":
        restore(root)
    elif cmd == "publish":
        publish(root, argv[2], argv[3], argv[4])
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
