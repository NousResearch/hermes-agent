"""Behavioral coverage for install_node's post-install version probe (#87460).

The probe used to be `installed_ver=$(node --version 2>/dev/null)` under
`set -e`: a Node binary that exists but cannot start (Node 26 linux-x64
links libatomic.so.1, missing on minimal Debian/Ubuntu) aborted the whole
installer at exit 127 with the loader's explanation discarded — installs
died mid-sentence with no output at all. The probe must degrade with a
clear error carrying the loader's message, and Debian/Ubuntu installs
must attempt libatomic1 up front.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_install_node() -> str:
    src = INSTALL_SH.read_text(encoding="utf-8")
    match = re.search(r"^install_node\(\) \{\n(.*?)^\}", src, re.S | re.M)
    assert match is not None, "install_node() not found in scripts/install.sh"
    return match.group(1)


def _make_node_tarball(path: Path, node_body: str) -> None:
    """A .tar.xz-named gzip archive — both bsdtar and modern GNU tar sniff
    compression by content, so the extension mismatch is irrelevant here."""
    root = "node-v26.7.0-linux-x64"
    with tarfile.open(path, "w:gz") as tar:
        for name, body, mode in (
            (f"{root}/bin/node", node_body, 0o755),
            (f"{root}/bin/npm", "#!/bin/sh\nexit 0\n", 0o755),
            (f"{root}/bin/npx", "#!/bin/sh\nexit 0\n", 0o755),
        ):
            data = body.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))


def _run_install_node(tmp_path: Path, node_body: str) -> tuple[int, str, str, list[str]]:
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    link_dir = tmp_path / "links"
    fixture = tmp_path / "fixture.tar.xz"
    apt_log = tmp_path / "apt-calls"

    bin_dir.mkdir()
    home.mkdir()
    link_dir.mkdir()
    _make_node_tarball(fixture, node_body)

    def _stub(name: str, body: str) -> None:
        stub = bin_dir / name
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)

    _stub(
        "uname",
        "#!/bin/sh\n"
        'if [ "${1:-}" = "-m" ]; then echo x86_64; else echo Linux; fi\n',
    )
    _stub(
        "curl",
        "#!/bin/sh\n"
        "# Called as: curl -fsSL <index-url>        (stdout -> tarball name)\n"
        "#           curl -fsSL <download-url> -o <path>\n"
        "if [ \"${3:-}\" = \"-o\" ]; then\n"
        '    cp "$FIXTURE" "$4"\n'
        "else\n"
        "    echo 'node-v26.7.0-linux-x64.tar.xz'\n"
        "fi\n",
    )
    _stub("sudo", "#!/bin/sh\nshift_if_env() { :; }\nexec env \"$@\"\n")
    _stub(
        "apt-get",
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$APT_LOG\"\nexit 0\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "FIXTURE": str(fixture),
            "APT_LOG": str(apt_log),
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )

    driver = tmp_path / "driver.sh"
    driver.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "OS=linux\n"
        "DISTRO=ubuntu\n"
        "NODE_VERSION=26\n"
        f"HERMES_HOME={home}\n"
        "HAS_NODE=maybe\n"
        "log_info()    { printf 'INFO %s\\n' \"$*\"; }\n"
        "log_success() { printf 'OK %s\\n' \"$*\"; }\n"
        "log_warn()    { printf 'WARN %s\\n' \"$*\"; }\n"
        "log_error()   { printf 'ERROR %s\\n' \"$*\" >&2; }\n"
        f"get_command_link_dir() {{ echo {link_dir}; }}\n"
        "configure_managed_node_npm_prefix() { return 0; }\n"
        "install_node() {\n"
        f"{_extract_install_node()}"
        "}\n"
        "install_node\n"
        'printf "HAS_NODE=%s\\n" "$HAS_NODE"\n',
        encoding="utf-8",
    )

    proc = subprocess.run(
        ["bash", str(driver)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    apt_calls = (
        apt_log.read_text(encoding="utf-8").splitlines()
        if apt_log.exists()
        else []
    )
    return proc.returncode, proc.stdout, proc.stderr, apt_calls


BROKEN_NODE = (
    "#!/bin/sh\n"
    "echo 'node: error while loading shared libraries: libatomic.so.1: "
    "cannot open shared object file: No such file or directory' >&2\n"
    "exit 127\n"
)

HEALTHY_NODE = "#!/bin/sh\necho v26.7.0\n"


def test_broken_node_degrades_with_clear_error(tmp_path: Path) -> None:
    code, stdout, stderr, _ = _run_install_node(tmp_path, BROKEN_NODE)

    # Old behavior: silent abort at exit 127 with no output (#87460).
    assert code == 0, stderr
    assert "Downloaded Node.js failed to start:" in stderr
    assert "libatomic.so.1" in stderr
    assert "apt-get install -y libatomic1" in stdout + stderr
    assert "HAS_NODE=false" in stdout
    assert "HAS_NODE=true" not in stdout
    # AI-review follow-up: the broken tree and bin links must not linger
    # — retries and later installer steps resolve `node` cleanly.
    home = tmp_path / "home"
    link_dir = tmp_path / "links"
    assert not (home / "node").exists(), "broken managed Node tree left behind"
    assert not (link_dir / "node").exists(), "broken node symlink left behind"
    assert not (link_dir / "npm").exists(), "broken npm symlink left behind"
    # AI-review follow-up: the broken tree and bin links must not linger
    # — retries and later installer steps resolve `node` cleanly.
    assert not (home / "node").exists(), "broken managed Node tree left behind"
    assert not (link_dir / "node").exists(), "broken node symlink left behind"
    assert not (link_dir / "npm").exists(), "broken npm symlink left behind"


def test_healthy_node_reports_success(tmp_path: Path) -> None:
    code, stdout, stderr, _ = _run_install_node(tmp_path, HEALTHY_NODE)

    assert code == 0, stderr
    assert "Node.js v26.7.0 installed" in stdout
    assert "HAS_NODE=true" in stdout


def test_libatomic1_is_preinstalled_on_ubuntu(tmp_path: Path) -> None:
    _, _, _, apt_calls = _run_install_node(tmp_path, HEALTHY_NODE)

    assert any("libatomic1" in call for call in apt_calls), apt_calls
