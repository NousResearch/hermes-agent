"""Fixed containment policy for the isolated test runner.

Every limit here is a constant on purpose: no CLI flag, env var or config key
can loosen it. Changing a value changes the policy hash, which invalidates
every existing human attestation (see attestation.py).
"""

from __future__ import annotations

import hashlib
import json
import re
from types import MappingProxyType

EXIT_OK = 0
EXIT_TARGET_FAILED = 1
EXIT_REFUSED = 2
EXIT_PREREQUISITE_MISSING = 3
EXIT_UNSUPPORTED_RUNTIME = 4
EXIT_ATTESTATION = 5
EXIT_PROOF_FAILED = 6
EXIT_INTERNAL = 7  # unexpected launcher error; never reported as a target result
# Run happened but its result cannot be trusted (cleanup unverified, output cap
# exceeded, snapshot changed during the run, unknown early CLI exit): valid=false.
EXIT_INVALID = 8
EXIT_TIMEOUT = 124

# In-container marker exit code for "offline toolchain missing" (mapped to 3).
CONTAINER_PREREQ_EXIT = 87

SUPPORTED_RUNTIMES = ("docker",)
SUPPORTED_HOST_OS = ("darwin", "linux")

LIMITS = MappingProxyType(
    {
        "cpus": "1",
        "memory_bytes": 2 * 1024**3,
        "memory_swap_bytes": 2 * 1024**3,  # == memory: no extra swap
        "pids": 128,
        "nofile": 256,
        "fsize_bytes": 64 * 1024**2,
        # Three fixed tmpfs mounts; their sum (512 MiB) is the whole writable
        # disk budget and is charged to the memory cgroup.
        "scratch_tmpfs_bytes": 256 * 1024**2,
        "var_tmp_tmpfs_bytes": 192 * 1024**2,
        "tmp_tmpfs_bytes": 64 * 1024**2,
        "shm_bytes": 64 * 1024**2,
        "test_workers": 1,
        "wall_seconds": 600,
        "stdout_cap_bytes": 4 * 1024**2,
        "stderr_cap_bytes": 1 * 1024**2,
        "cleanup_timeout_seconds": 30,
    }
)

# Snapshot preparation bounds (prepare.py). Finite on purpose; exceeding any
# one refuses the preparation. Not part of policy_document(): the receipt
# carries their digest instead.
PREPARE_LIMITS = MappingProxyType(
    {
        "max_files": 40_000,
        "max_file_bytes": 64 * 1024**2,
        "max_total_bytes": 1 * 1024**3,
        "max_archive_bytes": 1536 * 1024**2,
        "max_patch_bytes": 16 * 1024**2,
        "max_json_bytes": 8 * 1024**2,
        "max_exceptions": 32,
        "max_path_bytes": 1024,
        "git_timeout_seconds": 300,
        "prepare_wall_seconds": 1200,
    }
)

CONTAINER_UID = 65534
CONTAINER_GID = 65534
SRC_MOUNT = "/src"
SCRATCH = "/scratch"
# Fixed tmpfs mounts (path -> LIMITS key). Nothing else may be writable.
TMPFS = MappingProxyType({
    SCRATCH: "scratch_tmpfs_bytes",
    "/var/tmp": "var_tmp_tmpfs_bytes",
    "/tmp": "tmp_tmpfs_bytes",
})
# Mounts Docker itself adds that MAY be writable: an upper bound, not a
# requirement. The probe records whether each one is rw; it passes when one is
# ro or absent (stricter: Docker Desktop mounts /etc/hosts, /etc/hostname and
# /etc/resolv.conf ro under --read-only), and does not claim the rw ones safe.
# Every TMPFS mount must be present and rw; any other rw mount, or a missing
# TMPFS mount, refuses the proof.
DOCKER_RESIDUAL_RW = ("/dev/shm", "/etc/hosts", "/etc/hostname", "/etc/resolv.conf",
                      "/dev/mqueue")
# Mount points the probe's rw inventory leaves out; the probe records each one it
# skipped. TREES drop the point and everything below it (kernel pseudo
# filesystems); POINTS drop that exact point only, so /dev/shm and /dev/mqueue
# stay in the inventory. /dev is not assumed safe: the dev_write_denied check
# must show uid CONTAINER_UID cannot create a file there.
MOUNT_INVENTORY_EXCLUDED_TREES = ("/proc", "/sys", "/dev/pts")
MOUNT_INVENTORY_EXCLUDED_POINTS = ("/dev",)
TMPFS_TOTAL_MAX = 512 * 1024**2

# The only suites `run` may execute; values are repo-relative paths handed to
# scripts/run_tests.sh. Adding a suite is a reviewed source change.
SUITES = MappingProxyType(
    {
        "isolated-selftests": ("tests/scripts/isolated_tests/",),
        # The media-delivery candidate's targeted tests (exact files): its four new
        # tests, then the existing tests that drive the functions it changes.
        "media-delivery": (
            "tests/cron/test_cron_standalone_delivery_unconfirmed.py",
            "tests/gateway/outbound/test_send_partial_delivery.py",
            "tests/hermes_cli/test_send_cmd_partial_delivery.py",
            "tests/plugins/platforms/telegram/test_standalone_delivery_unconfirmed.py",
            "tests/hermes_cli/test_send_cmd_warnings.py",
            "tests/cron/test_cron_reconnect_only_rejection.py",
            "tests/cron/test_cron_live_delivery_confirmation.py",
            "tests/cron/test_cron_standalone_no_duplicate_replay.py",
            "tests/cron/test_cron_standalone_queue_gates.py",
            "tests/gateway/outbound/test_send_message_tool.py",
            "tests/gateway/relay/test_relay_media_ack_lost_consumers.py",
        ),
    }
)

IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CONTAINER_PYTHON_RE = re.compile(r"^/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+$")
RUN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
NAME_PREFIX = "hermes-isotest-"
LABEL_KEY = "org.hermes.isotest.run"


def policy_document() -> dict:
    return {
        "limits": dict(LIMITS),
        "uid": CONTAINER_UID,
        "gid": CONTAINER_GID,
        "suites": {k: list(v) for k, v in SUITES.items()},
        "runtimes": list(SUPPORTED_RUNTIMES),
        "tmpfs": dict(TMPFS),
        "log_driver": "none",
    }


def tmpfs_total() -> int:
    return sum(LIMITS[key] for key in TMPFS.values())


def sha256_json(doc: object) -> str:
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def policy_hash() -> str:
    return sha256_json(policy_document())
