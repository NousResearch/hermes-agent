"""Exercise the build mount's real Linux UID permissions, not Android binaries."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
# Only the stdlib facts publisher runs here; the bionic tool bytes are never
# executed. A native Linux image keeps this kernel-permissions test off QEMU.
IMAGE = "python@sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("build_exit", [0, 23], ids=["success", "failed-build"])
def test_assembly_publishes_facts_as_termux_uid_and_returns_host_owned_output(tmp_path, build_exit):
    if not shutil.which("docker"):
        pytest.skip("Docker is required for the bind-mount ownership regression")
    available = subprocess.run(
        ["docker", "info"], capture_output=True, timeout=15,
    )
    if available.returncode:
        pytest.skip("Docker daemon is unavailable")
    image = subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True, timeout=15,
    )
    if image.returncode:
        pytest.skip(f"Pre-pull the native Linux test image: docker pull {IMAGE}")

    payload = tmp_path / "payload"
    payload.mkdir()
    for name in ("python", "uv"):
        entry = payload / name
        entry.mkdir()
        (entry / "bytes").write_bytes(b"read-only staged tool bytes")
    before = {path: (path.stat().st_uid, path.stat().st_gid, path.stat().st_mode)
              for path in payload.rglob("*")}
    result = subprocess.run(
        ["bash", "-c", r'''
set -euo pipefail
IMAGE="$1"; PAYLOAD_ABS="$2"; REPO="$3"
source "$REPO/scripts/termux/assembly_permissions.sh"
prepare_assembly
trap restore_assembly_owner EXIT
docker run --rm --user 1000:1000 --network none \
    -v "$ASSEMBLY:/assembly" \
    -v "$PAYLOAD_ABS/python:/assembly/tools/python:ro" \
    -v "$PAYLOAD_ABS/uv:/assembly/tools/uv:ro" \
    -v "$REPO:/app:ro" \
    -e HERMES_HOME=/tmp/hermes \
    "$IMAGE" sh -ec '
        python -B /app/scripts/termux/build_environment.py prepare-tools \
            --root /assembly --source-tools /assembly/tools
        mkdir /assembly/venv /assembly/pm-runtime
        umask 077
        printf "private build output" > /assembly/pm-runtime/receipt
        stat -c "%u:%g" /assembly/tools/facts.json > /assembly/writer
        exit "$1"
    ' sh "$4"
''', "assembly-permissions-test", IMAGE, str(payload), str(ROOT), str(build_exit)],
        cwd=ROOT, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == build_exit, result.stdout + result.stderr
    assembly, = payload.glob(".environments-*")
    facts = json.loads((assembly / "tools/facts.json").read_text())
    assert set(facts["packages"]) == {"python", "uv"}
    assert (assembly / "writer").read_text().strip() == "1000:1000"
    assert (assembly / "pm-runtime/receipt").read_text() == "private build output"
    for path in [assembly, *assembly.rglob("*")]:
        info = path.stat()
        assert (info.st_uid, info.st_gid) == (os.getuid(), os.getgid()), path
        assert not info.st_mode & 0o022, path
    assert {path: (path.stat().st_uid, path.stat().st_gid, path.stat().st_mode)
            for path in before} == before
    shutil.rmtree(assembly)
