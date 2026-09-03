"""Harness: in-container integration tests for S6ServiceManager.

The unit tests in tests/hermes_cli/test_service_manager.py exercise the
class against a tmp-path scandir with a stubbed ``subprocess.run``.
These tests run the real class inside a real container against the
real s6-svc / s6-svscanctl binaries, validating end-to-end.

Phase 3 only registers the service slot — it doesn't depend on the
gateway actually starting (the binary will refuse to start without a
valid profile config). The full register → start → supervised-restart
→ unregister cycle is covered by Phase 4 once profile create/delete
hooks land.

Every ``docker exec`` here runs as the unprivileged ``hermes`` user
(via :func:`docker_exec` in conftest); see the conftest module
docstring. ``/run/service`` is chowned hermes-writable by the
``02-reconcile-profiles`` cont-init.d script, so register/unregister
operations work correctly under UID 10000.
"""
from __future__ import annotations

from tests.docker.conftest import docker_exec, start_container


_REGISTER_SCRIPT = """
import sys
sys.path.insert(0, "/opt/hermes")
from hermes_cli.service_manager import S6ServiceManager
S6ServiceManager().register_profile_gateway("phase3test")
# Don't worry about whether the gateway actually starts — we only care
# that the supervision slot was created. The gateway run script will
# likely error out (no profile config exists) but that's expected.
print("REGISTERED")
"""

_UNREGISTER_SCRIPT = """
import sys
sys.path.insert(0, "/opt/hermes")
from hermes_cli.service_manager import S6ServiceManager
S6ServiceManager().unregister_profile_gateway("phase3test")
print("UNREGISTERED")
"""


def test_s6_register_creates_service_dir_in_live_container(
    built_image: str, container_name: str,
) -> None:
    """S6ServiceManager.register_profile_gateway must create
    ``/run/service/gateway-<profile>/`` and trigger s6-svscan rescan
    against the real s6 supervision tree."""
    start_container(built_image, container_name, cmd="sleep 120")

    r = docker_exec(container_name, "python3", "-c", _REGISTER_SCRIPT, timeout=30)
    assert "REGISTERED" in r.stdout, (
        f"register failed: stderr={r.stderr!r} stdout={r.stdout!r}"
    )

    # Service directory exists with the expected structure.
    r = docker_exec(container_name, "test", "-d", "/run/service/gateway-phase3test")
    assert r.returncode == 0, "service directory not created"

    r = docker_exec(container_name, "test", "-f", "/run/service/gateway-phase3test/run")
    assert r.returncode == 0, "run script not created"

    r = docker_exec(container_name, "test", "-f",
              "/run/service/gateway-phase3test/log/run")
    assert r.returncode == 0, "log/run script not created"

    # s6-svscan picked it up — s6-svstat works against the dir.
    # `docker exec` doesn't put /command/ on PATH (only the supervision
    # tree does), so call s6-svstat by absolute path.
    r = docker_exec(container_name, "/command/s6-svstat",
              "/run/service/gateway-phase3test")
    assert r.returncode == 0, f"s6-svstat failed: {r.stderr or r.stdout}"

    # list_profile_gateways picks it up.
    r = docker_exec(container_name, "python3", "-c", (
        "from hermes_cli.service_manager import S6ServiceManager;"
        "print(S6ServiceManager().list_profile_gateways())"
    ))
    assert "phase3test" in r.stdout, f"list output: {r.stdout!r}"




# Shell probe: build a service-shaped staging dir under the live scandir
# with a given NAME, fire a real `s6-svscanctl -a` rescan, wait, and
# report whether s6-svscan supervised it (which would create a root-owned
# supervise/ dir). Used to prove the dot-prefixed staging name is INVISIBLE
# to a concurrent rescan while a non-dotted one is not.
#
# Echoes one of: SUPERVISED / NOT-SUPERVISED, plus the supervise/ owner.
_SVSCAN_PICKUP_PROBE = r"""
set -eu
NAME="$1"
SCANDIR=/run/service
DIR="$SCANDIR/$NAME"
rm -rf "$DIR"
mkdir -p "$DIR"
printf 'longrun\n' > "$DIR/type"
printf '#!/command/execlineb -P\n/command/s6-sleep 600\n' > "$DIR/run"
chmod 755 "$DIR/run"
# Trigger a full rescan, exactly as register/reconcile do.
/command/s6-svscanctl -a "$SCANDIR"
# Give s6-svscan time to act (its scan is async; 200ms is the manager's
# own settle delay, use 2s here to be comfortably past it on any arch).
/command/s6-sleep 2
if [ -d "$DIR/supervise" ]; then
    owner=$(stat -c '%U' "$DIR/supervise" 2>/dev/null || echo '?')
    echo "SUPERVISED owner=$owner"
else
    echo "NOT-SUPERVISED"
fi
# Best-effort teardown so the probe leaves no live supervisor behind.
/command/s6-svc -d "$DIR" 2>/dev/null || true
/command/s6-svscanctl -an "$SCANDIR" 2>/dev/null || true
/command/s6-sleep 1
rm -rf "$DIR" 2>/dev/null || true
"""


