"""Install on a musl libc host (Alpine) through the real ``scripts/install.sh``.

Failure class: libc. Hermes's PM store pins glibc (``-linux-gnu``) builds of uv, CPython and
Node. On a musl host those binaries either fail to exec or, with the ``gcompat`` shim, load and
segfault, so an install or update that publishes them leaves every later ``hermes`` command dead
(#123682). The acceptable outcomes are: musl-compatible tools that run, or a refusal that tells
the user why (names musl) and publishes nothing unrunnable.

The host shape is a real Alpine userland in a container (``docker run alpine``) with only what
the documented one-liner needs (bash, git, curl). The installer's clone goes to a read-only
bind-mounted bare origin at the commit under test; everything else (uv, Python, Node artifacts)
comes from the network exactly as a user's install does. Skips with a reason when docker is not
usable on this host.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.hosts import _hosts as X

IMAGE = "alpine:3.22"


def _docker_reason() -> str | None:
    exe = shutil.which("docker")
    if exe is None:
        return "docker CLI not installed; the musl host shape runs in an Alpine container"
    try:
        cp = subprocess.run([exe, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"docker daemon unreachable ({exc})"
    if cp.returncode != 0:
        return f"docker daemon unusable: {cp.stderr.strip()[:200]}"
    return None


DOCKER_REASON = _docker_reason()

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.skipif(DOCKER_REASON is not None, reason=str(DOCKER_REASON)),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
]

# Runs as root inside the container. Everything the user would see goes to stdout; the probe
# section after the install is fenced so the assertions can split it off.
SCRIPT = r"""
set -u
apk add --no-cache -q bash git curl >/dev/null 2>&1 || { echo "HARNESS: apk add failed"; exit 90; }
git config --global safe.directory '*'
git config --global url."file://$ORIGIN".insteadOf https://github.com/NousResearch/hermes-agent.git
echo "=== libc: $(ldd --version 2>&1 | head -n1)"
bash /work/install.sh --non-interactive </dev/null
echo "=== install rc=$?"
echo "=== probes"
for exe in "$HOME"/.hermes/tools/*/uv "$HOME"/.hermes/tools/*/bin/python3 "$HOME"/.hermes/tools/*/bin/node \
           "$HOME"/.local/bin/hermes; do
    [ -e "$exe" ] || continue
    timeout 60 "$exe" --version >/dev/null 2>&1
    echo "probe rc=$? $exe"
done
"""


@pytest.fixture(scope="module")
def alpine_install(tmp_path_factory):
    root = tmp_path_factory.mktemp("musl")
    origin = X.make_origin(root)
    work = root / "work"
    work.mkdir()
    shutil.copy(H.WORKTREE / "scripts" / "install.sh", work / "install.sh")
    # The --shared origin borrows objects from this checkout's object store: mount it too.
    objects = Path(I.git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=H.WORKTREE)) / "objects"
    pull = subprocess.run(["docker", "pull", "-q", IMAGE], capture_output=True, text=True, timeout=600, check=False)
    assert pull.returncode == 0, f"docker pull {IMAGE} failed (network?):\n{pull.stderr[-2000:]}"
    argv = ["docker", "run", "--rm", "-e", f"ORIGIN={origin}",
            "-v", f"{origin}:{origin}:ro", "-v", f"{objects}:{objects}:ro", "-v", f"{work}:/work:ro",
            IMAGE, "sh", "-c", SCRIPT]
    cp = subprocess.run(argv, capture_output=True, text=True, timeout=1800, check=False)
    assert cp.returncode != 90, "harness: apk could not install bash/git/curl in the container:\n" + I.describe(cp)
    return cp


def _section(out: str) -> tuple[int | None, list[tuple[int, str]]]:
    rc = None
    probes: list[tuple[int, str]] = []
    for line in out.splitlines():
        if line.startswith("=== install rc="):
            rc = int(line.rsplit("=", 1)[1])
        elif line.startswith("probe rc="):
            code, path = line[len("probe rc="):].split(" ", 1)
            probes.append((int(code), path))
    return rc, probes


def test_musl_host_gets_runnable_tools_or_a_refusal_naming_musl(alpine_install):
    cp = alpine_install
    out = cp.stdout + cp.stderr
    assert "=== libc:" in out and "musl" in out.split("=== libc:", 1)[1].splitlines()[0].lower(), (
        "harness: the container is not a musl userland:\n" + I.describe(cp))
    rc, probes = _section(cp.stdout)
    assert rc is not None, "harness: the installer never returned inside the container:\n" + I.describe(cp)
    # What install.sh itself printed: after the harness's libc banner, before its rc line.
    head = cp.stdout.partition("=== install rc=")[0]
    after_banner = head.partition("=== libc:")[2]
    install_log = after_banner.partition("\n")[2] + cp.stderr
    assert I.TRACEBACK not in install_log, "install.sh crashed with a traceback on musl:\n" + I.describe(cp)
    dead = [f"{path} (rc={code})" for code, path in probes if code != 0]
    if rc != 0:
        assert "musl" in install_log.lower(), (
            f"musl host: installer failed without naming musl as the reason (rc={rc}):\n" + I.describe(cp))
    assert not dead, ("musl host: published tool binaries that cannot execute: " + ", ".join(dead)
                      + "\n" + I.describe(cp))
    if rc == 0:
        hermes = [p for code, p in probes if p.endswith("/.local/bin/hermes")]
        assert hermes, "musl host: install exited 0 but published no hermes command:\n" + I.describe(cp)

