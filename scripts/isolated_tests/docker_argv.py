"""Builds the only docker argv lists the launcher ever executes.

Nothing from the caller is appended verbatim: the image is a validated local
image ID, the socket a validated Unix path, the command a fixed script. The
test shell starts under `env -i`, so baked image ENV is not inherited by the
test processes; `/usr/bin/env` itself still starts with the image ENV, which is
why image.py refuses images with loader-risk ENV keys before any payload. The
entrypoint is overridden so a baked ENTRYPOINT never runs.
"""

from __future__ import annotations

from pathlib import Path

from isolated_tests.inputs import Refused
from isolated_tests.policy import (
    CONTAINER_GID, CONTAINER_PREREQ_EXIT, CONTAINER_PYTHON_RE, CONTAINER_UID,
    IMAGE_ID_RE, LABEL_KEY, LIMITS, NAME_PREFIX, RUN_ID_RE, SCRATCH, SRC_MOUNT, TMPFS,
)


def check_image(image: str) -> str:
    if not IMAGE_ID_RE.match(image or ""):
        raise Refused(f"image: must be a local image ID 'sha256:<64 hex>', got {image!r}")
    return image


def check_container_python(python: str) -> str:
    if not CONTAINER_PYTHON_RE.match(python or "") or "/../" in python or "/./" in python:
        raise Refused(f"container python: must be a plain absolute path, got {python!r}")
    if python.startswith(tuple(m + "/" for m in (SRC_MOUNT, *TMPFS))):
        raise Refused("container python: must live in the read-only image, not /src or a tmpfs")
    return python


def container_name(run_id: str) -> str:
    if not RUN_ID_RE.match(run_id):
        raise Refused(f"run id: malformed {run_id!r}")
    return NAME_PREFIX + run_id


def host_env(cli_home: Path) -> dict[str, str]:
    """Allowlisted, otherwise empty environment for the docker CLI process."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(cli_home),
        "DOCKER_CONFIG": str(cli_home / "docker-config"),
        "LANG": "C",
    }


def container_env(python: str) -> list[str]:
    bindir = python.rsplit("/", 1)[0]
    return [
        f"PATH={bindir}:/usr/local/bin:/usr/bin:/bin",
        f"HOME={SCRATCH}/home",
        f"TMPDIR={SCRATCH}/tmp",
        f"XDG_CACHE_HOME={SCRATCH}/cache",
        f"HERMES_HOME={SCRATCH}/hermes_home",
        f"HERMES_RUNTIME_DIR={SCRATCH}/runtime",
        f"HERMES_PYTHON={python}",
        "PYTHONDONTWRITEBYTECODE=1",
        "PIP_NO_INDEX=1",
        "UV_OFFLINE=1",
        "LANG=C.UTF-8",
        "TZ=UTC",
        f"HERMES_TEST_WORKERS={LIMITS['test_workers']}",
    ]


def tmpfs_flags() -> list[str]:
    return [f"--tmpfs={path}:rw,nosuid,nodev,size={LIMITS[key]},mode=1777"
            for path, key in TMPFS.items()]


# Fails with a distinct code instead of letting run_tests.sh try to
# provision (download) a test environment on a network-less container.
PREFLIGHT = (
    'set -eu; mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$HERMES_HOME" "$HERMES_RUNTIME_DIR"; '
    'command -v bash >/dev/null 2>&1 || { echo "prerequisite missing: bash" >&2; exit %(code)d; }; '
    '[ -x "$HERMES_PYTHON" ] || { echo "prerequisite missing: HERMES_PYTHON $HERMES_PYTHON" >&2; exit %(code)d; }; '
    '"$HERMES_PYTHON" -I -c "import pytest" >/dev/null 2>&1 || '
    '{ echo "prerequisite missing: pytest in $HERMES_PYTHON (image must ship an offline test env)" >&2; exit %(code)d; }; '
) % {"code": CONTAINER_PREREQ_EXIT}


def _common(docker: Path, socket: Path, cli_home: Path) -> list[str]:
    return [str(docker), "--config", str(cli_home / "docker-config"), "--host", f"unix://{socket}"]


def run_argv(*, docker: Path, socket: Path, cli_home: Path, image: str, python: str,
             snapshot: Path, run_id: str, script: str,
             probe_env: tuple[str, ...] = ()) -> list[str]:
    lim = LIMITS
    if "," in str(snapshot) or "=" in str(snapshot):
        raise Refused(f"mount source contains ',' or '=': {snapshot}")
    if any(not e.startswith("ISOTEST_") or "=" not in e for e in probe_env):
        raise Refused("probe env: only ISOTEST_* assignments are allowed")
    return _common(docker, socket, cli_home) + [
        "run", "--rm", "--pull=never",
        "--name", container_name(run_id), "--label", f"{LABEL_KEY}={run_id}",
        "--network=none", "--ipc=private", f"--shm-size={lim['shm_bytes']}", "--init",
        "--log-driver=none",
        "--read-only", f"--user={CONTAINER_UID}:{CONTAINER_GID}",
        "--cap-drop=ALL", "--security-opt=no-new-privileges",
        f"--cpus={lim['cpus']}", f"--memory={lim['memory_bytes']}",
        f"--memory-swap={lim['memory_swap_bytes']}", f"--pids-limit={lim['pids']}",
        f"--ulimit=nofile={lim['nofile']}:{lim['nofile']}",
        f"--ulimit=fsize={lim['fsize_bytes']}:{lim['fsize_bytes']}",
        "--ulimit=core=0:0",
        *tmpfs_flags(),
        f"--mount=type=bind,source={snapshot},target={SRC_MOUNT},readonly",
        "--workdir", SRC_MOUNT, "--hostname", "isotest", "--no-healthcheck",
        "--entrypoint", "/usr/bin/env",
        check_image(image),
        "-i", *container_env(check_container_python(python)), *probe_env,
        "/bin/sh", "-c", script,
    ]


def target_script(suite_paths: tuple[str, ...]) -> str:
    for p in suite_paths:
        if not all(c.isalnum() or c in "/._-" for c in p):
            raise Refused(f"suite path has unsafe characters: {p!r}")
    return PREFLIGHT + "exec bash scripts/run_tests.sh " + " ".join(suite_paths)


def probe_script(probe_source: str) -> str:
    return PREFLIGHT + 'exec "$HERMES_PYTHON" -I -S -c ' + _sh_quote(probe_source)


def _sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def rm_argv(*, docker: Path, socket: Path, cli_home: Path, run_id: str) -> list[str]:
    """Removes exactly this run's container by its exact name; never prune.

    Only executed after label_argv confirmed the container carries this run's
    label (executor.cleanup_owned)."""
    return _common(docker, socket, cli_home) + ["rm", "--force", container_name(run_id)]


def label_argv(*, docker: Path, socket: Path, cli_home: Path, run_id: str) -> list[str]:
    return _common(docker, socket, cli_home) + [
        "container", "inspect", "--format", "{{index .Config.Labels \"%s\"}}" % LABEL_KEY,
        container_name(run_id)]


def image_inspect_argv(*, docker: Path, socket: Path, cli_home: Path, image: str) -> list[str]:
    """Offline (local store only) metadata read; runs before any payload."""
    return _common(docker, socket, cli_home) + [
        "image", "inspect", "--format", "{{json .}}", check_image(image)]


def server_version_argv(*, docker: Path, socket: Path, cli_home: Path) -> list[str]:
    return _common(docker, socket, cli_home) + ["version", "--format", "{{json .Server}}"]
