"""Node.js download picks a .tar.gz when the host has no ``xz`` (#11197).

`tar xf node-*.tar.xz` shells out to the xz binary; minimal Debian/DietPi/WSL images ship tar
without it, so the extract died mid-way and the installer then failed on a missing directory.
Both tarball selectors (installer and runtime bootstrap) are driven for real against a stubbed
index page; only the ``xz`` binary's presence differs between the two arms.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX = "node-v26.7.0-linux-x64.tar.xz\nnode-v26.7.0-linux-x64.tar.gz\n"


def _function(path: Path, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\(\) \{{\n.*?^\}}\n", path.read_text(encoding="utf-8"), re.M | re.S)
    assert match, f"{name}() not found in {path}"
    return match.group(0)


def _selected_tarball(tmp_path: Path, *, with_xz: bool, script: Path, fn: str, call: str) -> str:
    """Run the real tarball-selection function with curl stubbed to serve INDEX; the download step
    records the chosen name and aborts, so nothing is extracted."""
    bin_dir = tmp_path / ("bin-xz" if with_xz else "bin-noxz")
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text("#!/bin/sh\nfor a in \"$@\"; do [ \"$a\" = -o ] && { echo \"$4\" >> \"$PICKED\"; exit 1; }; done\n"
                    f"printf '%s' '{INDEX}'\n", encoding="utf-8")
    curl.chmod(curl.stat().st_mode | stat.S_IXUSR)
    # PATH holds only this dir: the host's real xz must not leak into the "no xz" arm.
    for tool in ("grep", "head", "mktemp", "rm", "uname", "sh", "printf"):
        real = shutil.which(tool)
        if real:
            os.symlink(real, bin_dir / tool)
    if with_xz:
        (bin_dir / "xz").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (bin_dir / "xz").chmod(0o755)
    picked = tmp_path / f"picked-{with_xz}-{fn}"
    harness = ("log_info() { :; }; log_warn() { :; }; _nb_log() { :; }; _nb_warn() { :; }\n"
               "HERMES_NODE_TARGET_MAJOR=26\n" + _function(script, fn) + call)
    env = {"PATH": str(bin_dir), "PICKED": str(picked), "HOME": str(tmp_path)}
    subprocess.run([shutil.which("bash") or "/bin/bash", "-c", harness], env=env, check=False, capture_output=True)
    return picked.read_text(encoding="utf-8").strip() if picked.exists() else ""


@pytest.mark.linux_only
@pytest.mark.parametrize("script, fn, call", [
    (REPO_ROOT / "scripts" / "install.sh", "install_node_line", "\ninstall_node_line 26 linux x64\n"),
    (REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh", "_nb_install_bundled_node", "\n_nb_install_bundled_node\n"),
])
def test_tarball_format_follows_xz_availability(tmp_path: Path, script: Path, fn: str, call: str) -> None:
    assert _selected_tarball(tmp_path, with_xz=True, script=script, fn=fn, call=call).endswith(".tar.xz")
    assert _selected_tarball(tmp_path, with_xz=False, script=script, fn=fn, call=call).endswith(".tar.gz")
