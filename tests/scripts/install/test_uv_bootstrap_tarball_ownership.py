"""Root uv bootstrap must not adopt the uid/gid recorded in the pinned uv tarball (#81525).

GNU tar restores archived owners when it runs as root, and astral's release tarballs record
uid 1001: the staged uv that root then executes belonged to whichever account holds 1001.

The tests run the real install.sh ``ensure_uv`` and setup-hermes.sh as root inside an
unprivileged user namespace that maps only uid 0. Restoring an archived owner there fails
(uid 1001 is unmapped), so an extraction that still tries to adopt it fails the stage
instead of silently chowning.
"""
from __future__ import annotations

import hashlib
import json
import io
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("linux")

ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = ROOT / "scripts" / "install.sh"


def _userns_root_available() -> bool:
    unshare = shutil.which("unshare")
    if not unshare:
        return False
    probe = subprocess.run([unshare, "-r", "id", "-u"], capture_output=True, text=True, check=False)
    return probe.returncode == 0 and probe.stdout.strip() == "0"


def _uv_archive(path: Path) -> str:
    """A uv-shaped release tarball whose members are owned by 1001, as astral's are."""
    payload = b"#!/bin/sh\necho 'uv 0.0.0'\n"
    with tarfile.open(path, "w:gz") as tf:
        for name, data in (("uv-fixture/", None), ("uv-fixture/uv", payload), ("uv-fixture/uvx", payload)):
            member = tarfile.TarInfo(name.rstrip("/"))
            member.uid = member.gid = 1001
            member.uname = member.gname = "iojs"
            if data is None:
                member.type, member.mode = tarfile.DIRTYPE, 0o755
                tf.addfile(member)
            else:
                member.size, member.mode = len(data), 0o755
                tf.addfile(member, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not _userns_root_available(), reason="needs an unprivileged user namespace (unshare -r)")
def test_root_uv_bootstrap_does_not_adopt_archive_owner(tmp_path: Path) -> None:
    archive = tmp_path / "uv.tar.gz"
    digest = _uv_archive(archive)
    store = tmp_path / "store"
    # Source the real installer, swap only the network and the pin table for the fixture.
    script = f"""
source {shlex.quote(str(INSTALL_SH))} --manifest >/dev/null
uv_bootstrap_target() {{ echo linux-x64; }}
uv_bootstrap_pin() {{ UV_PIN_URL=fixture; UV_PIN_MIRROR=; UV_PIN_SHA256={digest}; }}
curl() {{ while [ "$#" -gt 0 ]; do [ "$1" = -o ] && {{ cp {shlex.quote(str(archive))} "$2"; return; }}; shift; done; }}
ensure_uv
stat -c '%u:%g' "$UV_CMD"
"""
    env = {**os.environ, "HERMES_RUNTIME_DIR": str(store), "HERMES_HOME": str(tmp_path / "home"),
           "TMPDIR": str(tmp_path)}
    result = subprocess.run(["unshare", "-r", "bash", "-c", script], capture_output=True, text=True,
                            env=env, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "0:0"


@pytest.mark.skipif(not _userns_root_available(), reason="needs an unprivileged user namespace (unshare -r)")
def test_root_setup_hermes_uv_staging_does_not_adopt_archive_owner(tmp_path: Path) -> None:
    """setup-hermes.sh stages the same pinned uv tarball through its own extraction."""
    checkout = tmp_path / "checkout"
    (checkout / "pm").mkdir(parents=True)
    shutil.copy2(ROOT / "setup-hermes.sh", checkout / "setup-hermes.sh")
    shutil.copy2(ROOT / "pm" / "artifact-mirror.json", checkout / "pm" / "artifact-mirror.json")
    archive = tmp_path / "uv.tar.gz"
    digest = _uv_archive(archive)
    artifact = {"sha256": digest, "url": "https://example.invalid/uv.tar.gz"}
    lock = {"packages": {"python": {"version": "3.11.0"},
                         "uv": {"artifacts": {f"linux-{a}": artifact for a in ("arm64", "x64")}, "version": "0.0.0"}}}
    (checkout / "pm" / "lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    shims = tmp_path / "shims"
    shims.mkdir()
    curl = shims / "curl"
    curl.write_text(f'#!/bin/sh\nwhile [ "$#" -gt 0 ]; do [ "$1" = -o ] && exec cp {shlex.quote(str(archive))} "$2"; shift; done\n',
                    encoding="utf-8")
    curl.chmod(0o755)
    store = tmp_path / "store"
    env = {**os.environ, "HERMES_RUNTIME_DIR": str(store), "PATH": f"{shims}:{os.environ['PATH']}"}
    # The fixture uv cannot install Python, so the script stops right after staging uv.
    result = subprocess.run(["unshare", "-r", "bash", str(checkout / "setup-hermes.sh")], capture_output=True,
                            text=True, env=env, timeout=60, check=False)
    assert "uv installed" in result.stdout, result.stdout + result.stderr
    staged = next(store.glob("uv-0.0.0-linux-*/uv"))
    assert staged.stat().st_uid == os.getuid()
