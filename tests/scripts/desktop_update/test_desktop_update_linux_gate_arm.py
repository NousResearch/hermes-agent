"""ARM64 unpacked-dir coverage for the linux relaunch gate (#94703).

electron-builder names the unpacked dir ``linux-unpacked`` on x86_64 but
``linux-<arch>-unpacked`` on every other arch (``linux-arm64-unpacked`` is
what Hermes ships for ARM). The gate hardcoded the x86_64 name, so a healthy
ARM install false-gated as "skew" on EVERY update, telling the user to
reinstall an app that was already correct.

Drives the real ``--self-test-gate`` entry point of ``posix.sh`` (the same one
``scripts/desktop-update/repro.sh gate`` uses). On macOS, where BSD readlink
lacks ``-m``, a PATH shim provides a GNU-compatible ``readlink`` so the gate
logic itself is exercised everywhere (the canonicalisation behaviour is
already covered by the linux_only matrix in test_desktop_update_linux_gate.py).
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

POSIX_SH = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "desktop-update" / "posix.sh"

# NOT linux_only: unlike the symlink-matrix file, these cases only exercise
# the gate's dir-selection logic (the relaunch/skew paths return before the
# GNU-only `stat -c` sandbox probe), so a PATH shim for `readlink -m` on
# BSD-readlink hosts is enough to run the real posix.sh everywhere. On Linux
# the system readlink is used as-is.


def _readlink_shim_dir(tmp_path: Path) -> Path:
    """A PATH dir whose ``readlink`` implements GNU ``-m`` on BSD-readlink hosts."""
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    (shim / "readlink").write_text(
        "#!/usr/bin/env bash\n"
        "# GNU readlink -m for BSD readlink hosts: canonicalise without\n"
        "# requiring the full path to exist (longest existing prefix + -P).\n"
        "args=()\n"
        "while [ $# -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    -m|--canonicalize-missing) ;;\n"
        "    --) shift; args+=(\"$@\"); break ;;\n"
        "    -*) echo \"shim readlink: unsupported flag $1\" >&2; exit 1 ;;\n"
        "    *) args+=(\"$1\") ;;\n"
        "  esac\n"
        "  shift\n"
        "done\n"
        "set -- \"${args[@]}\"\n"
        "[ $# -eq 1 ] || { echo \"usage: readlink -m FILE\" >&2; exit 1; }\n"
        "p=\"$1\"\n"
        "case \"$p\" in\n"
        "  /*) ;;\n"
        "  *) p=\"$PWD/$p\" ;;\n"
        "esac\n"
        "# Longest existing ANCESTOR dir, then append the missing tail.\n"
        "prefix=\"$(dirname -- \"$p\")\"\n"
        "base=\"$(basename -- \"$p\")\"\n"
        "tail=\"/$base\"\n"
        "while [ ! -d \"$prefix\" ]; do\n"
        "  tail=\"/$(basename -- \"$prefix\")$tail\"\n"
        "  prefix=\"$(dirname -- \"$prefix\")\"\n"
        "done\n"
        "prefix=\"$(cd \"$prefix\" && pwd -P)\"\n"
        "printf '%s%s\\n' \"$prefix\" \"$tail\"\n",
        encoding="utf-8",
    )
    (shim / "readlink").chmod((shim / "readlink").stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return shim


def _gate(install_root: Path, relaunch_target: Path, tmp_path: Path) -> str:
    env = dict(os.environ)
    if not sys.platform.startswith("linux"):
        # BSD readlink lacks -m; prepend a shim so the gate logic itself runs.
        env["PATH"] = f"{_readlink_shim_dir(tmp_path)}{os.pathsep}{env['PATH']}"
    out = subprocess.run(
        ["bash", str(POSIX_SH), "--self-test-gate", "--install-root", str(install_root),
         "--relaunch-target", str(relaunch_target)],
        capture_output=True, text=True, check=True, env=env,
    )
    return out.stdout.strip().split(":", 1)[0]


def test_arch_specific_unpacked_dir_still_relaunches(tmp_path):
    unpacked = tmp_path / "apps" / "desktop" / "release" / "linux-arm64-unpacked"
    unpacked.mkdir(parents=True)
    (unpacked / "Hermes").touch()
    assert _gate(tmp_path, unpacked / "Hermes", tmp_path) == "relaunch"


def test_arch_specific_dir_is_preferred_over_a_stale_x86_64_one(tmp_path):
    """Both dirs can exist after a cross-arch rebuild; the gate must follow the
    dir the running binary actually lives in, not just the first one it finds."""
    release = tmp_path / "apps" / "desktop" / "release"
    stale = release / "linux-unpacked"
    live = release / "linux-arm64-unpacked"
    stale.mkdir(parents=True)
    live.mkdir(parents=True)
    (stale / "Hermes").touch()
    (live / "Hermes").touch()
    assert _gate(tmp_path, live / "Hermes", tmp_path) == "relaunch"


def test_foreign_target_with_arm_dirs_present_is_still_skew(tmp_path):
    """The glob must not over-broaden: a target outside every unpacked dir
    (an AppImage/deb install) keeps gating as skew (#94703)."""
    release = tmp_path / "apps" / "desktop" / "release"
    arm = release / "linux-arm64-unpacked"
    arm.mkdir(parents=True)
    (arm / "Hermes").touch()
    foreign = tmp_path / "opt" / "Hermes"
    foreign.mkdir(parents=True)
    (foreign / "hermes").touch()
    assert _gate(tmp_path, foreign / "hermes", tmp_path) == "skew"
    # A sibling dir sharing the arch prefix must not be mistaken for the unpacked tree.
    assert _gate(tmp_path, Path(str(arm) + "-evil") / "Hermes", tmp_path) == "skew"
