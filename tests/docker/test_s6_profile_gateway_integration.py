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


def test_s6_unregister_removes_service_dir_in_live_container(
    built_image: str, container_name: str,
) -> None:
    """unregister_profile_gateway must stop the service, remove the
    directory, and trigger s6-svscan rescan so the supervise process
    is dropped."""
    start_container(built_image, container_name, cmd="sleep 120")

    # First register so we have something to unregister.
    r = docker_exec(container_name, "python3", "-c", _REGISTER_SCRIPT, timeout=30)
    assert "REGISTERED" in r.stdout

    # Then unregister.
    r = docker_exec(container_name, "python3", "-c", _UNREGISTER_SCRIPT, timeout=30)
    assert "UNREGISTERED" in r.stdout, (
        f"unregister failed: stderr={r.stderr!r} stdout={r.stdout!r}"
    )

    # Directory is gone.
    r = docker_exec(container_name, "test", "-d", "/run/service/gateway-phase3test")
    assert r.returncode != 0, "service directory still exists after unregister"

    # list_profile_gateways no longer includes it.
    r = docker_exec(container_name, "python3", "-c", (
        "from hermes_cli.service_manager import S6ServiceManager;"
        "print(S6ServiceManager().list_profile_gateways())"
    ))
    assert "phase3test" not in r.stdout


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


def test_s6_dotfile_staging_dir_is_ignored_by_svscan_rescan(
    built_image: str, container_name: str,
) -> None:
    """Regression for the arm64 register-seed race.

    The register path builds the slot in a sibling staging dir and then
    atomically renames it to the live ``gateway-<profile>`` name. That
    staging dir lives INSIDE the scandir s6-svscan watches, so its NAME
    decides whether a concurrent ``s6-svscanctl -a`` rescan (fired by the
    cont-init reconciler registering ``gateway-default``, or by another
    register) supervises the half-built slot.

    - A NON-dotted name (the old ``gateway-<p>.tmp``) IS picked up: once it
      has a valid ``type``/``run``, s6-svscan spawns ``s6-supervise`` AS
      ROOT, creating a root-owned ``supervise/`` — which makes the in-flight
      ``_seed_supervise_skeleton`` EACCES on ``mkdir supervise/event``. That
      is the arm64-only flake (the native-arm runner's wider scheduling
      jitter lets the rescan land inside the seed window).
    - A DOT-prefixed name (the fix, ``.gateway-<p>.tmp``) is SKIPPED by
      s6-svscan and never supervised, so no root-owned ``supervise/`` can
      appear under the staging dir.

    This proves the mechanism directly and is arch-independent (it does not
    rely on hitting the narrow timing window — it forces the rescan and
    checks pickup), so it guards the fix on the amd64 job too.
    """
    start_container(built_image, container_name, cmd="sleep 120")

    # Control: a NON-dotted service-shaped dir IS supervised by the rescan
    # (root-owned supervise/). This is the pre-fix staging-name behaviour and
    # confirms the probe actually exercises s6-svscan pickup.
    r = docker_exec(
        container_name, "sh", "-c", _SVSCAN_PICKUP_PROBE, "probe",
        "gateway-raceprobe.tmp", user="root", timeout=30,
    )
    assert "SUPERVISED" in r.stdout and "NOT-SUPERVISED" not in r.stdout, (
        "control failed: a non-dotted staging dir should be picked up by "
        f"s6-svscan. stdout={r.stdout!r} stderr={r.stderr!r}"
    )

    # The fix: a DOT-prefixed staging dir (the name register/reconcile now
    # use) must be IGNORED by the same rescan — no supervisor, no root-owned
    # supervise/, so the in-flight seed can never EACCES.
    r = docker_exec(
        container_name, "sh", "-c", _SVSCAN_PICKUP_PROBE, "probe",
        ".gateway-raceprobe.tmp", user="root", timeout=30,
    )
    assert "NOT-SUPERVISED" in r.stdout, (
        "dot-prefixed staging dir was supervised by s6-svscan — the race "
        f"that EACCESes the seed is still reachable. stdout={r.stdout!r} "
        f"stderr={r.stderr!r}"
    )
