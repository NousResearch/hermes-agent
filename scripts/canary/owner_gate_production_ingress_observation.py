#!/usr/bin/env python3
"""Trusted, read-only production ingress observation for the owner gate.

The same source file has two deliberately separate roles:

* when streamed over the pinned production IAP transport, it runs as root on
  ``ai-platform-runtime-01`` and emits one fixed, secret-free observation of
  the exact active legacy v1 unit/process and effective legacy Caddy route;
  and
* on the owner Mac, it validates that observation, binds the exact transport
  authority, and signs the envelope with the fork-pinned owner-gate release
  key under a dedicated signature domain.

The remote collector accepts no paths, units, hosts, commands, or upstreams.
It never mutates systemd or Caddy and never emits or digests raw Caddy
configuration.  The Caddyfile, adapted documents, and two reads of the live
admin config are compared only in memory.  Only the secret-free live route
projection is digested into the signed report, so credentials cannot create a
persisted secret digest.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import http.client
import ipaddress
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from scripts.canary import owner_gate_production_ingress_contract as contract


OBSERVATION_SCHEMA = contract.OBSERVATION_SCHEMA
ENVELOPE_SCHEMA = contract.ENVELOPE_SCHEMA
FAILURE_SCHEMA = "muncho-owner-gate-production-ingress-observation-failure.v1"
SIGNATURE_DOMAIN = contract.SIGNATURE_DOMAIN

PROJECT = contract.PROJECT
ZONE = contract.ZONE
VM_NAME = contract.VM_NAME
INSTANCE_ID = contract.INSTANCE_ID
REMOTE_PYTHON = "/opt/adventico-ai-platform/hermes-agent/.venv/bin/python"

OLD_V1_UNIT = contract.OLD_V1_UNIT
OLD_V1_FRAGMENT_PATH = contract.OLD_V1_FRAGMENT_PATH
OLD_V1_FRAGMENT_SHA256 = contract.OLD_V1_FRAGMENT_SHA256
OLD_V1_FRAGMENT_MODE = contract.OLD_V1_FRAGMENT_MODE
OLD_V1_USER = contract.OLD_V1_USER
OLD_V1_GROUP = contract.OLD_V1_GROUP
OLD_V1_UID = contract.OLD_V1_UID
OLD_V1_GID = contract.OLD_V1_GID
OLD_V1_EXEC_START_ARGV = contract.OLD_V1_EXEC_START_ARGV
OLD_V1_PROCESS_CMDLINE = contract.OLD_V1_PROCESS_CMDLINE
CADDY_UNIT = contract.CADDY_UNIT
CADDY_UNIT_FRAGMENT = contract.CADDY_UNIT_FRAGMENT
CADDY_EXECUTABLE = "/usr/bin/caddy"
CADDYFILE_PATH = contract.CADDYFILE_PATH
PUBLIC_ORIGIN = contract.PUBLIC_ORIGIN
PUBLIC_HOST = contract.PUBLIC_HOST
PRIVATE_V2_UPSTREAM = contract.PRIVATE_V2_UPSTREAM
LEGACY_V1_UPSTREAM = contract.LEGACY_V1_UPSTREAM

SYSTEMCTL = "/usr/bin/systemctl"
OLD_V1_SYSTEMD_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "FragmentPath",
    "DropInPaths",
    "MainPID",
    "ExecMainPID",
    "NeedDaemonReload",
    "User",
    "Group",
    "ExecStart",
)
CADDY_SYSTEMD_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "FragmentPath",
    "DropInPaths",
    "MainPID",
    "ExecStart",
)
# Compatibility alias used by the fixed test fixture and remote source tests.
SYSTEMD_PROPERTIES = OLD_V1_SYSTEMD_PROPERTIES
CADDY_EXPECTED_ARGV = (
    CADDY_EXECUTABLE,
    "run",
    "--environ",
    "--config",
    str(CADDYFILE_PATH),
)
CADDY_ADMIN_PORT = 2019
CADDY_ADMIN_PATH = "/config/"
CADDY_ADMIN_IPV4_PROC_ADDRESS = "0100007F"
CADDY_ADMIN_IPV6_PROC_ADDRESS = "00000000000000000000000001000000"
EXPECTED_ROOT_UID = contract.EXPECTED_ROOT_UID
EXPECTED_ROOT_GID = contract.EXPECTED_ROOT_GID
CADDYFILE_MODE = contract.CADDYFILE_MODE
FRESHNESS_SECONDS = contract.FRESHNESS_SECONDS
MAX_SIGNING_DELAY_SECONDS = contract.MAX_SIGNING_DELAY_SECONDS
COMMAND_TIMEOUT_SECONDS = 30.0
MAX_SYSTEMCTL_OUTPUT_BYTES = 64 * 1024
MAX_CADDYFILE_BYTES = contract.MAX_CADDYFILE_BYTES
MAX_OLD_V1_FRAGMENT_BYTES = contract.MAX_OLD_V1_FRAGMENT_BYTES
MAX_ADAPTED_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_PROC_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_REMOTE_SOURCE_BYTES = 512 * 1024
MAX_REMOTE_OUTPUT_BYTES = 2 * 1024 * 1024

_REVISION = contract._REVISION
_SHA256 = contract._SHA256
ProductionIngressObservationError = contract.ProductionIngressObservationError
_error = contract._error
_canonical = contract._canonical
_sha256 = contract._sha256
_decode_json = contract._decode_json
_binding = contract._binding
_validate_transport_authority = contract._validate_transport_authority
_require_pinned_release_key = contract._require_pinned_release_key
validate_production_ingress_observation = (
    contract.validate_production_ingress_observation
)
validate_signed_production_ingress_observation = (
    contract.validate_signed_production_ingress_observation
)


@dataclass(frozen=True)
class _StableFile:
    raw: bytes
    identity: tuple[int, ...]
    public_projection: Mapping[str, Any]


def _stable_caddyfile() -> _StableFile:
    path = CADDYFILE_PATH
    if not path.is_absolute() or ".." in path.parts:
        _error("owner_gate_production_ingress_caddyfile_invalid")
    descriptor: int | None = None
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink != 1
            or opened.st_uid != EXPECTED_ROOT_UID
            or opened.st_gid != EXPECTED_ROOT_GID
            or stat.S_IMODE(opened.st_mode) != CADDYFILE_MODE
            or not 0 < opened.st_size <= MAX_CADDYFILE_BYTES
        ):
            _error("owner_gate_production_ingress_caddyfile_invalid")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                _error("owner_gate_production_ingress_caddyfile_changed")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        final = path.lstat()
        identity = (
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if identity != (
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (final.st_dev, final.st_ino) != (opened.st_dev, opened.st_ino):
            _error("owner_gate_production_ingress_caddyfile_changed")
        return _StableFile(
            raw=b"".join(chunks),
            identity=identity,
            public_projection={
                "config_path": str(path),
                "config_uid": opened.st_uid,
                "config_gid": opened.st_gid,
                "config_mode": f"{stat.S_IMODE(opened.st_mode):04o}",
                "config_size": opened.st_size,
            },
        )
    except ProductionIngressObservationError:
        raise
    except OSError as exc:
        _error("owner_gate_production_ingress_caddyfile_invalid", exc)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                _error("owner_gate_production_ingress_caddyfile_invalid", exc)


def _stable_old_v1_fragment() -> Mapping[str, Any]:
    path = OLD_V1_FRAGMENT_PATH
    if not path.is_absolute() or ".." in path.parts:
        _error("owner_gate_production_ingress_v1_fragment_invalid")
    descriptor: int | None = None
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink != 1
            or opened.st_uid != EXPECTED_ROOT_UID
            or opened.st_gid != EXPECTED_ROOT_GID
            or stat.S_IMODE(opened.st_mode) != OLD_V1_FRAGMENT_MODE
            or not 0 < opened.st_size <= MAX_OLD_V1_FRAGMENT_BYTES
        ):
            _error("owner_gate_production_ingress_v1_fragment_invalid")
        raw = bytearray()
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, opened.st_size - len(raw))
            if not chunk:
                _error("owner_gate_production_ingress_v1_fragment_changed")
            raw.extend(chunk)
        after = os.fstat(descriptor)
        final = path.lstat()
        if (
            len(raw) != opened.st_size
            or _sha256(bytes(raw)) != OLD_V1_FRAGMENT_SHA256
            or (
                opened.st_mode,
                opened.st_uid,
                opened.st_gid,
                opened.st_dev,
                opened.st_ino,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ) != (
                after.st_mode,
                after.st_uid,
                after.st_gid,
                after.st_dev,
                after.st_ino,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (final.st_dev, final.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            _error("owner_gate_production_ingress_v1_fragment_changed")
        return {
            "fragment_path": str(path),
            "fragment_uid": opened.st_uid,
            "fragment_gid": opened.st_gid,
            "fragment_mode": f"{stat.S_IMODE(opened.st_mode):04o}",
            "fragment_size": opened.st_size,
            "fragment_sha256": OLD_V1_FRAGMENT_SHA256,
            "stable_nofollow_fragment_verified": True,
        }
    except ProductionIngressObservationError:
        raise
    except OSError as exc:
        _error("owner_gate_production_ingress_v1_fragment_invalid", exc)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                _error(
                    "owner_gate_production_ingress_v1_fragment_invalid",
                    exc,
                )


def _run_command(argv: tuple[str, ...], *, maximum_output_bytes: int) -> bytes:
    if (
        not argv
        or any(not isinstance(item, str) or not item for item in argv)
        or not 0 < maximum_output_bytes <= MAX_ADAPTED_OUTPUT_BYTES
    ):
        _error("owner_gate_production_ingress_command_invalid")
    try:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env={
                "HOME": "/nonexistent",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            },
            shell=False,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _error("owner_gate_production_ingress_command_failed", exc)
    if (
        completed.returncode != 0
        or not isinstance(completed.stdout, bytes)
        or not isinstance(completed.stderr, bytes)
        or len(completed.stdout) > maximum_output_bytes
        or len(completed.stderr) > MAX_SYSTEMCTL_OUTPUT_BYTES
    ):
        _error("owner_gate_production_ingress_command_failed")
    return completed.stdout


def _systemctl_command(unit: str) -> tuple[str, ...]:
    if unit not in {OLD_V1_UNIT, CADDY_UNIT}:
        _error("owner_gate_production_ingress_unit_invalid")
    properties = (
        CADDY_SYSTEMD_PROPERTIES
        if unit == CADDY_UNIT
        else OLD_V1_SYSTEMD_PROPERTIES
    )
    return (
        SYSTEMCTL,
        "show",
        "--no-pager",
        *(f"--property={name}" for name in properties),
        "--",
        unit,
    )


def _systemd_projection(unit: str) -> Mapping[str, Any]:
    raw = _run_command(
        _systemctl_command(unit),
        maximum_output_bytes=MAX_SYSTEMCTL_OUTPUT_BYTES,
    )
    values: dict[str, str] = {}
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as exc:
        _error("owner_gate_production_ingress_unit_invalid", exc)
    for line in lines:
        if "=" not in line:
            _error("owner_gate_production_ingress_unit_invalid")
        key, value = line.split("=", 1)
        if key in values:
            _error("owner_gate_production_ingress_unit_invalid")
        values[key] = value
    properties = (
        CADDY_SYSTEMD_PROPERTIES
        if unit == CADDY_UNIT
        else OLD_V1_SYSTEMD_PROPERTIES
    )
    if set(values) != set(properties):
        _error("owner_gate_production_ingress_unit_invalid")
    projection: dict[str, Any] = {
        "unit": unit,
        "load_state": values["LoadState"],
        "active_state": values["ActiveState"],
        "sub_state": values["SubState"],
        "unit_file_state": values["UnitFileState"],
        "fragment_path": values["FragmentPath"],
        "drop_in_paths": values["DropInPaths"].split()
        if values["DropInPaths"]
        else [],
    }
    if unit == CADDY_UNIT:
        try:
            main_pid = int(values["MainPID"], 10)
        except ValueError as exc:
            _error("owner_gate_production_ingress_caddy_service_unsafe", exc)
        expected_command = " ".join(CADDY_EXPECTED_ARGV)
        exec_start = values["ExecStart"]
        expected_prefix = (
            f"{{ path={CADDY_EXECUTABLE} ; "
            f"argv[]={expected_command} ; "
        )
        if (
            main_pid <= 0
            or len(exec_start) > MAX_SYSTEMCTL_OUTPUT_BYTES
            or "\x00" in exec_start
            or not exec_start.startswith(expected_prefix)
            or not exec_start.endswith(" }")
            or exec_start.count("path=") != 1
            or exec_start.count("argv[]=") != 1
        ):
            _error("owner_gate_production_ingress_caddy_service_unsafe")
        projection["main_pid"] = main_pid
        projection["exec_start_argv"] = list(CADDY_EXPECTED_ARGV)
    else:
        try:
            main_pid = int(values["MainPID"], 10)
            exec_main_pid = int(values["ExecMainPID"], 10)
        except ValueError as exc:
            _error("owner_gate_production_ingress_old_v1_unsafe", exc)
        expected_command = " ".join(OLD_V1_EXEC_START_ARGV)
        exec_start = values["ExecStart"]
        expected_prefix = (
            f"{{ path={OLD_V1_EXEC_START_ARGV[0]} ; "
            f"argv[]={expected_command} ; "
        )
        if (
            values["Id"] != OLD_V1_UNIT
            or values["User"] != OLD_V1_USER
            or values["Group"] != OLD_V1_GROUP
            or values["NeedDaemonReload"] != "no"
            or main_pid <= 1
            or exec_main_pid != main_pid
            or len(exec_start) > MAX_SYSTEMCTL_OUTPUT_BYTES
            or "\x00" in exec_start
            or not exec_start.startswith(expected_prefix)
            or not exec_start.endswith(" }")
            or exec_start.count("path=") != 1
            or exec_start.count("argv[]=") != 1
        ):
            _error("owner_gate_production_ingress_old_v1_unsafe")
        projection.update(
            {
                "main_pid": main_pid,
                "exec_main_pid": exec_main_pid,
                "exec_start_path": OLD_V1_EXEC_START_ARGV[0],
                "exec_start_argv": list(OLD_V1_EXEC_START_ARGV),
                "service_user": values["User"],
                "service_group": values["Group"],
                "need_daemon_reload": values["NeedDaemonReload"] == "yes",
            }
        )
    return projection


def _old_v1_projection() -> Mapping[str, Any]:
    service = _systemd_projection(OLD_V1_UNIT)
    expected = {
        "unit": OLD_V1_UNIT,
        "load_state": "loaded",
        "active_state": "active",
        "sub_state": "running",
        "unit_file_state": "enabled",
        "fragment_path": str(OLD_V1_FRAGMENT_PATH),
        "drop_in_paths": [],
        "exec_start_path": OLD_V1_EXEC_START_ARGV[0],
        "exec_start_argv": list(OLD_V1_EXEC_START_ARGV),
        "service_user": OLD_V1_USER,
        "service_group": OLD_V1_GROUP,
        "need_daemon_reload": False,
    }
    if (
        {name: service.get(name) for name in expected} != expected
        or type(service.get("main_pid")) is not int
        or service["main_pid"] <= 1
        or service.get("exec_main_pid") != service["main_pid"]
    ):
        _error("owner_gate_production_ingress_old_v1_unsafe")
    fragment = _stable_old_v1_fragment()
    process = _old_v1_process_snapshot(service)
    if process.pid != service["main_pid"]:
        _error("owner_gate_production_ingress_old_v1_process_unsafe")
    return {
        **service,
        **fragment,
        "process_cmdline": list(OLD_V1_PROCESS_CMDLINE),
        "process_uid": process.uid,
        "process_gid": process.gid,
        "process_start_time_ticks": process.start_time_ticks,
        "process_cgroup_unit_verified": True,
        "active_process_stable": True,
        "trusted_for_v2": False,
    }


def _caddy_service_projection() -> Mapping[str, Any]:
    service = _systemd_projection(CADDY_UNIT)
    expected = {
        "unit": CADDY_UNIT,
        "load_state": "loaded",
        "active_state": "active",
        "sub_state": "running",
        "unit_file_state": "enabled",
        "fragment_path": CADDY_UNIT_FRAGMENT,
        "drop_in_paths": [],
    }
    if (
        {name: service.get(name) for name in expected} != expected
        or type(service.get("main_pid")) is not int
        or service["main_pid"] <= 0
        or service.get("exec_start_argv") != list(CADDY_EXPECTED_ARGV)
    ):
        _error("owner_gate_production_ingress_caddy_service_unsafe")
    return service


@dataclass(frozen=True)
class _OldV1ProcessSnapshot:
    pid: int
    uid: int
    gid: int
    start_time_ticks: int


@dataclass(frozen=True)
class _CaddyProcessSnapshot:
    pid: int
    identity: tuple[Any, ...]
    admin_host: str
    admin_socket_inode: str


def _bounded_proc_read(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(64 * 1024, MAX_PROC_OUTPUT_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_PROC_OUTPUT_BYTES:
                _error("owner_gate_production_ingress_caddy_process_unsafe")
        raw = b"".join(chunks)
        if not raw:
            _error("owner_gate_production_ingress_caddy_process_unsafe")
        return raw
    except ProductionIngressObservationError:
        raise
    except OSError as exc:
        _error("owner_gate_production_ingress_caddy_process_unsafe", exc)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                _error(
                    "owner_gate_production_ingress_caddy_process_unsafe",
                    exc,
                )


def _bounded_old_v1_proc_read(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(64 * 1024, MAX_PROC_OUTPUT_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_PROC_OUTPUT_BYTES:
                _error("owner_gate_production_ingress_old_v1_process_unsafe")
        raw = b"".join(chunks)
        if not raw:
            _error("owner_gate_production_ingress_old_v1_process_unsafe")
        return raw
    except ProductionIngressObservationError:
        raise
    except OSError as exc:
        _error("owner_gate_production_ingress_old_v1_process_unsafe", exc)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                _error(
                    "owner_gate_production_ingress_old_v1_process_unsafe",
                    exc,
                )


def _old_v1_process_start_time(raw: bytes) -> int:
    try:
        text = raw.decode("ascii", errors="strict")
        close = text.rfind(")")
        fields = text[close + 2 :].split()
        start_time = int(fields[19], 10)
    except (IndexError, UnicodeError, ValueError) as exc:
        _error("owner_gate_production_ingress_old_v1_process_unsafe", exc)
    if close <= 0 or start_time <= 0:
        _error("owner_gate_production_ingress_old_v1_process_unsafe")
    return start_time


def _old_v1_process_snapshot(
    service: Mapping[str, Any],
) -> _OldV1ProcessSnapshot:
    pid = service.get("main_pid")
    if type(pid) is not int or pid <= 1 or service.get("exec_main_pid") != pid:
        _error("owner_gate_production_ingress_old_v1_process_unsafe")
    proc = Path(f"/proc/{pid}")
    status = _bounded_old_v1_proc_read(proc / "status")
    cmdline = _bounded_old_v1_proc_read(proc / "cmdline")
    cgroup = _bounded_old_v1_proc_read(proc / "cgroup")
    first_start = _old_v1_process_start_time(
        _bounded_old_v1_proc_read(proc / "stat")
    )
    final_start = _old_v1_process_start_time(
        _bounded_old_v1_proc_read(proc / "stat")
    )
    uid_match = re.search(
        rb"(?m)^Uid:\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*$",
        status,
    )
    gid_match = re.search(
        rb"(?m)^Gid:\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*$",
        status,
    )
    expected_cmdline = b"\x00".join(
        item.encode("utf-8", errors="strict")
        for item in OLD_V1_PROCESS_CMDLINE
    ) + b"\x00"
    try:
        cgroup_lines = cgroup.decode("ascii", errors="strict").splitlines()
    except UnicodeError as exc:
        _error("owner_gate_production_ingress_old_v1_process_unsafe", exc)
    cgroup_unit_verified = any(
        line.rpartition(":")[2].split("/")[-1] == OLD_V1_UNIT
        for line in cgroup_lines
        if line and line.count(":") >= 2
    )
    if (
        uid_match is None
        or {int(item) for item in uid_match.groups()} != {OLD_V1_UID}
        or gid_match is None
        or {int(item) for item in gid_match.groups()} != {OLD_V1_GID}
        or cmdline != expected_cmdline
        or not cgroup_unit_verified
        or first_start != final_start
        or service.get("exec_start_argv") != list(OLD_V1_EXEC_START_ARGV)
    ):
        _error("owner_gate_production_ingress_old_v1_process_unsafe")
    return _OldV1ProcessSnapshot(
        pid=pid,
        uid=OLD_V1_UID,
        gid=OLD_V1_GID,
        start_time_ticks=first_start,
    )


def _process_start_time(pid: int) -> int:
    raw = _bounded_proc_read(Path(f"/proc/{pid}/stat"))
    try:
        text = raw.decode("ascii", errors="strict")
        close = text.rfind(")")
        fields = text[close + 2 :].split()
        start_time = int(fields[19], 10)
    except (IndexError, UnicodeError, ValueError) as exc:
        _error("owner_gate_production_ingress_caddy_process_unsafe", exc)
    if close <= 0 or start_time <= 0:
        _error("owner_gate_production_ingress_caddy_process_unsafe")
    return start_time


def _pid_socket_inodes(pid: int) -> frozenset[str]:
    root = Path(f"/proc/{pid}/fd")
    try:
        names = os.listdir(root)
    except OSError as exc:
        _error("owner_gate_production_ingress_caddy_process_unsafe", exc)
    if len(names) > 65_536:
        _error("owner_gate_production_ingress_caddy_process_unsafe")
    inodes: set[str] = set()
    for name in names:
        if not name.isdigit():
            _error("owner_gate_production_ingress_caddy_process_unsafe")
        try:
            target = os.readlink(root / name)
        except FileNotFoundError:
            # A non-admin connection can close while the fixed listener stays.
            continue
        except OSError as exc:
            _error("owner_gate_production_ingress_caddy_process_unsafe", exc)
        if target.startswith("socket:[") and target.endswith("]"):
            inode = target[8:-1]
            if not inode.isdigit():
                _error("owner_gate_production_ingress_caddy_process_unsafe")
            inodes.add(inode)
    return frozenset(inodes)


def _admin_listener_for_pid(pid: int) -> tuple[str, str]:
    owned = _pid_socket_inodes(pid)
    listeners: list[tuple[str, str]] = []
    expected_port = f"{CADDY_ADMIN_PORT:04X}"
    for path, allowed in (
        (
            Path(f"/proc/{pid}/net/tcp"),
            {CADDY_ADMIN_IPV4_PROC_ADDRESS: "127.0.0.1"},
        ),
        (
            Path(f"/proc/{pid}/net/tcp6"),
            {CADDY_ADMIN_IPV6_PROC_ADDRESS: "::1"},
        ),
    ):
        raw = _bounded_proc_read(path)
        try:
            lines = raw.decode("ascii", errors="strict").splitlines()
        except UnicodeError as exc:
            _error("owner_gate_production_ingress_caddy_admin_unsafe", exc)
        if not lines or "local_address" not in lines[0]:
            _error("owner_gate_production_ingress_caddy_admin_unsafe")
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10 or ":" not in fields[1]:
                _error("owner_gate_production_ingress_caddy_admin_unsafe")
            address, port = fields[1].rsplit(":", 1)
            inode = fields[9]
            if (
                inode not in owned
                or port.upper() != expected_port
                or fields[3] != "0A"
            ):
                continue
            host = allowed.get(address.upper())
            if host is None:
                _error("owner_gate_production_ingress_caddy_admin_unsafe")
            listeners.append((host, inode))
    if len(listeners) != 1:
        _error("owner_gate_production_ingress_caddy_admin_unsafe")
    return listeners[0]


def _caddy_process_snapshot(
    service: Mapping[str, Any],
) -> _CaddyProcessSnapshot:
    pid = service.get("main_pid")
    if type(pid) is not int or pid <= 0:
        _error("owner_gate_production_ingress_caddy_process_unsafe")
    proc_root = Path(f"/proc/{pid}")
    exe = proc_root / "exe"
    try:
        executable_before = os.readlink(exe)
        process_executable_before = os.stat(exe)
        installed_executable_before = Path(CADDY_EXECUTABLE).lstat()
        cmdline = _bounded_proc_read(proc_root / "cmdline")
        start_time = _process_start_time(pid)
        admin_host, admin_inode = _admin_listener_for_pid(pid)
        executable_after = os.readlink(exe)
        process_executable_after = os.stat(exe)
        installed_executable_after = Path(CADDY_EXECUTABLE).lstat()
    except ProductionIngressObservationError:
        raise
    except OSError as exc:
        _error("owner_gate_production_ingress_caddy_process_unsafe", exc)
    expected_cmdline = b"\x00".join(
        item.encode("utf-8", errors="strict") for item in CADDY_EXPECTED_ARGV
    ) + b"\x00"
    if (
        executable_before != CADDY_EXECUTABLE
        or executable_after != CADDY_EXECUTABLE
        or executable_before != executable_after
        or cmdline != expected_cmdline
        or service.get("exec_start_argv") != list(CADDY_EXPECTED_ARGV)
        or not stat.S_ISREG(installed_executable_before.st_mode)
        or stat.S_ISLNK(installed_executable_before.st_mode)
        or installed_executable_before.st_uid != EXPECTED_ROOT_UID
        or installed_executable_before.st_gid != EXPECTED_ROOT_GID
        or stat.S_IMODE(installed_executable_before.st_mode) & 0o022
        or (process_executable_before.st_dev, process_executable_before.st_ino)
        != (
            installed_executable_before.st_dev,
            installed_executable_before.st_ino,
        )
        or (
            process_executable_before.st_dev,
            process_executable_before.st_ino,
            process_executable_before.st_ctime_ns,
        )
        != (
            process_executable_after.st_dev,
            process_executable_after.st_ino,
            process_executable_after.st_ctime_ns,
        )
        or (
            installed_executable_before.st_dev,
            installed_executable_before.st_ino,
            installed_executable_before.st_mode,
            installed_executable_before.st_uid,
            installed_executable_before.st_gid,
            installed_executable_before.st_size,
            installed_executable_before.st_mtime_ns,
            installed_executable_before.st_ctime_ns,
        )
        != (
            installed_executable_after.st_dev,
            installed_executable_after.st_ino,
            installed_executable_after.st_mode,
            installed_executable_after.st_uid,
            installed_executable_after.st_gid,
            installed_executable_after.st_size,
            installed_executable_after.st_mtime_ns,
            installed_executable_after.st_ctime_ns,
        )
    ):
        _error("owner_gate_production_ingress_caddy_process_unsafe")
    return _CaddyProcessSnapshot(
        pid=pid,
        identity=(
            pid,
            start_time,
            executable_before,
            installed_executable_before.st_dev,
            installed_executable_before.st_ino,
            installed_executable_before.st_ctime_ns,
            tuple(CADDY_EXPECTED_ARGV),
            admin_host,
            admin_inode,
        ),
        admin_host=admin_host,
        admin_socket_inode=admin_inode,
    )


def _read_live_caddy_config(process: _CaddyProcessSnapshot) -> bytes:
    if (
        type(process) is not _CaddyProcessSnapshot
        or process.pid <= 0
        or process.admin_host not in {"127.0.0.1", "::1"}
        or not process.admin_socket_inode.isdigit()
    ):
        _error("owner_gate_production_ingress_caddy_admin_unsafe")
    connection = http.client.HTTPConnection(
        process.admin_host,
        CADDY_ADMIN_PORT,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    try:
        connection.request(
            "GET",
            CADDY_ADMIN_PATH,
            headers={
                "Accept": "application/json",
                "Connection": "close",
                "Host": f"localhost:{CADDY_ADMIN_PORT}",
            },
        )
        response = connection.getresponse()
        content_encoding = response.getheader("Content-Encoding")
        raw = response.read(MAX_ADAPTED_OUTPUT_BYTES + 1)
        if (
            response.status != 200
            or content_encoding not in {None, "", "identity"}
            or not raw
            or len(raw) > MAX_ADAPTED_OUTPUT_BYTES
        ):
            _error("owner_gate_production_ingress_caddy_admin_unsafe")
        return raw
    except ProductionIngressObservationError:
        raise
    except (OSError, http.client.HTTPException) as exc:
        _error("owner_gate_production_ingress_caddy_admin_unsafe", exc)
    finally:
        connection.close()


def _admin_config_matches_process(
    value: Mapping[str, Any],
    process: _CaddyProcessSnapshot,
) -> bool:
    admin = value.get("admin")
    if admin is None:
        # Caddy's documented default is localhost:2019; socket ownership is
        # independently proven from /proc before the request is trusted.
        return True
    if not isinstance(admin, Mapping) or admin.get("disabled") is True:
        return False
    listen = admin.get("listen", "localhost:2019")
    return listen in {
        "localhost:2019",
        f"{process.admin_host}:{CADDY_ADMIN_PORT}",
        f"[{process.admin_host}]:{CADDY_ADMIN_PORT}",
    }


def _caddy_adapt_command() -> tuple[str, ...]:
    return (
        CADDY_EXECUTABLE,
        "adapt",
        "--config",
        str(CADDYFILE_PATH),
        "--adapter",
        "caddyfile",
    )


@dataclass(frozen=True)
class _ServerRouteProjection:
    listeners: tuple[tuple[str, str], ...]
    exact_host_route_count: int
    reverse_proxy_handler_count: int
    dials: tuple[str, ...]


def _normalized_host_pattern(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or value != value.strip()
        or "\x00" in value
    ):
        _error("owner_gate_production_ingress_caddy_config_invalid")
    normalized = value.lower().rstrip(".")
    if not normalized:
        _error("owner_gate_production_ingress_caddy_config_invalid")
    return normalized


def _host_pattern_may_match_public_host(pattern: str) -> bool:
    if "{" in pattern or "}" in pattern:
        # Caddy placeholders are runtime values.  They cannot prove exclusion.
        return True
    return fnmatch.fnmatchcase(PUBLIC_HOST, pattern)


def _route_host_capability(route: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    """Return may-match, exact-host, and covers-all-public-host facts.

    Matcher sets are ORed by Caddy.  Constraints other than ``host`` can match
    some request for the public host and therefore cannot prove exclusion.
    Unknown/negative/expression matchers are intentionally treated as
    potentially matching; only a concrete host matcher whose every pattern
    excludes the fixed public host narrows the graph.
    """

    matchers = route.get("match")
    if matchers is None or matchers == []:
        return True, False, True
    if not isinstance(matchers, list):
        _error("owner_gate_production_ingress_caddy_config_invalid")
    may_match = False
    exact_public_host = False
    covers_all_public_host = False
    for matcher_set in matchers:
        if not isinstance(matcher_set, Mapping):
            _error("owner_gate_production_ingress_caddy_config_invalid")
        if "host" not in matcher_set:
            may_match = True
            if not matcher_set:
                covers_all_public_host = True
            continue
        patterns = matcher_set.get("host")
        if not isinstance(patterns, list) or not patterns:
            _error("owner_gate_production_ingress_caddy_config_invalid")
        normalized = tuple(_normalized_host_pattern(item) for item in patterns)
        if PUBLIC_HOST in normalized:
            exact_public_host = True
        if any(_host_pattern_may_match_public_host(item) for item in normalized):
            may_match = True
            if set(matcher_set) == {"host"}:
                covers_all_public_host = True
    return may_match, exact_public_host, covers_all_public_host


def _reverse_proxy_dials(handler: Mapping[str, Any]) -> tuple[str, ...]:
    dynamic = handler.get("dynamic_upstreams")
    transport = handler.get("transport")
    if dynamic is not None or "handle_response" in handler:
        _error("owner_gate_production_ingress_caddy_route_unsafe")
    if transport is not None and (
        not isinstance(transport, Mapping)
        or transport.get("protocol") != "http"
    ):
        _error("owner_gate_production_ingress_caddy_route_unsafe")
    upstreams = handler.get("upstreams")
    if not isinstance(upstreams, list) or not upstreams:
        _error("owner_gate_production_ingress_caddy_route_unsafe")
    dials: list[str] = []
    for upstream in upstreams:
        dial = upstream.get("dial") if isinstance(upstream, Mapping) else None
        if (
            not isinstance(dial, str)
            or not dial
            or len(dial) > 512
            or dial != dial.strip()
            or "\x00" in dial
        ):
            _error("owner_gate_production_ingress_caddy_route_unsafe")
        dials.append(dial)
    return tuple(dials)


_LOCAL_MIDDLEWARE_HANDLERS = frozenset({
    "encode",
    "headers",
    "log_append",
    "request_body",
    "rewrite",
    "vars",
})
_LOCAL_TERMINAL_HANDLERS = frozenset({"file_server", "static_response"})


def _analyze_routes(
    routes: Any,
    *,
    inherited_public_host_capability: bool = True,
    depth: int = 0,
) -> tuple[int, int, tuple[str, ...], bool]:
    if not isinstance(routes, list) or depth > 64:
        _error("owner_gate_production_ingress_caddy_config_invalid")
    exact_host_route_count = 0
    reverse_proxy_handler_count = 0
    dials: list[str] = []
    public_host_chain_open = inherited_public_host_capability
    blocked_groups: set[str] = set()
    consumes_all_public_host = False
    for route in routes:
        if not isinstance(route, Mapping):
            _error("owner_gate_production_ingress_caddy_config_invalid")
        route_capability, exact_host, covers_all = _route_host_capability(route)
        group = route.get("group")
        terminal = route.get("terminal", False)
        if group is not None and (
            not isinstance(group, str)
            or not group
            or len(group) > 512
            or "\x00" in group
        ):
            _error("owner_gate_production_ingress_caddy_config_invalid")
        if type(terminal) is not bool:
            _error("owner_gate_production_ingress_caddy_config_invalid")
        capable = (
            public_host_chain_open
            and route_capability
            and (group is None or group not in blocked_groups)
        )
        if capable and exact_host:
            exact_host_route_count += 1
        handlers = route.get("handle", [])
        if not isinstance(handlers, list):
            _error("owner_gate_production_ingress_caddy_config_invalid")
        handler_chain_consumes = False
        for index, handler in enumerate(handlers):
            if not isinstance(handler, Mapping):
                _error("owner_gate_production_ingress_caddy_config_invalid")
            handler_name = handler.get("handler")
            if not isinstance(handler_name, str) or not handler_name:
                _error("owner_gate_production_ingress_caddy_config_invalid")
            if not capable:
                continue
            if handler_name == "subroute":
                if index != len(handlers) - 1:
                    _error("owner_gate_production_ingress_caddy_route_unsafe")
                (
                    nested_exact,
                    nested_proxies,
                    nested_dials,
                    nested_consumes,
                ) = _analyze_routes(
                    handler.get("routes"),
                    inherited_public_host_capability=capable,
                    depth=depth + 1,
                )
                exact_host_route_count += nested_exact
                reverse_proxy_handler_count += nested_proxies
                dials.extend(nested_dials)
                handler_chain_consumes = nested_consumes
                continue
            if "routes" in handler:
                # Do not silently miss a route-bearing third-party handler.
                _error("owner_gate_production_ingress_caddy_route_unsafe")
            if handler_name == "reverse_proxy":
                if index != len(handlers) - 1:
                    _error("owner_gate_production_ingress_caddy_route_unsafe")
                reverse_proxy_handler_count += 1
                dials.extend(_reverse_proxy_dials(handler))
                handler_chain_consumes = True
                continue
            if handler_name in _LOCAL_TERMINAL_HANDLERS:
                if index != len(handlers) - 1:
                    _error("owner_gate_production_ingress_caddy_route_unsafe")
                handler_chain_consumes = True
                continue
            if handler_name not in _LOCAL_MIDDLEWARE_HANDLERS:
                # Unknown/custom handlers can initiate network traffic or
                # terminate the chain.  Neither property is safely inferable.
                _error("owner_gate_production_ingress_caddy_route_unsafe")
        if capable and group is not None and covers_all:
            blocked_groups.add(group)
        if capable and covers_all and (terminal or handler_chain_consumes):
            public_host_chain_open = False
            consumes_all_public_host = True
    return (
        exact_host_route_count,
        reverse_proxy_handler_count,
        tuple(dials),
        consumes_all_public_host,
    )


def _listener_projection(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value:
        return (("unknown", ""),)
    listeners: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 512
            or item != item.strip()
            or "\x00" in item
        ):
            _error("owner_gate_production_ingress_caddy_config_invalid")
        lowered = item.lower()
        if lowered.startswith(("unix/", "unixgram/")):
            listeners.append(("unix", lowered.split("/", 1)[1]))
            continue
        address = lowered
        if "/" in address:
            network, address = address.split("/", 1)
            if network not in {"tcp", "tcp4", "tcp6"}:
                listeners.append(("unknown", ""))
                continue
        _host, separator, port = address.rpartition(":")
        if not separator or not port.isdigit() or not 0 < int(port) <= 65535:
            listeners.append(("unknown", ""))
            continue
        listeners.append(("tcp", str(int(port))))
    return tuple(listeners)


def _listeners_may_overlap(
    left: tuple[tuple[str, str], ...],
    right: tuple[tuple[str, str], ...],
) -> bool:
    for left_kind, left_value in left:
        for right_kind, right_value in right:
            if "unknown" in {left_kind, right_kind}:
                return True
            if left_kind == right_kind == "tcp" and left_value == right_value:
                # Same port is conservatively overlapping even when one side
                # names a concrete interface and the other a wildcard.
                return True
            if left_kind == right_kind == "unix" and left_value == right_value:
                return True
    return False


def _dial_ip_and_port(dial: str) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int] | None:
    if dial.startswith("["):
        closing = dial.find("]")
        if closing <= 1 or dial[closing + 1 : closing + 2] != ":":
            return None
        host = dial[1:closing]
        port = dial[closing + 2 :]
    else:
        host, separator, port = dial.rpartition(":")
        if not separator or not host:
            return None
    if not port.isdigit() or not 0 < int(port) <= 65535:
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    return address, int(port)


def _dial_targets_private_v2(dial: str) -> bool:
    endpoint = _dial_ip_and_port(dial)
    expected = _dial_ip_and_port(PRIVATE_V2_UPSTREAM)
    if expected is None:
        _error("owner_gate_production_ingress_caddy_config_invalid")
    if endpoint is None:
        return False
    address, port = endpoint
    expected_address, expected_port = expected
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address == expected_address and port == expected_port


def _dial_is_on_current_host(dial: str) -> bool:
    for prefix in ("unix/", "unix+h2c/"):
        if dial.startswith(prefix):
            socket_path = dial[len(prefix) :]
            parsed = PurePosixPath(socket_path)
            return parsed.is_absolute() and ".." not in parsed.parts
    endpoint = _dial_ip_and_port(dial)
    if endpoint is None:
        return False
    address, _port = endpoint
    return address.is_loopback


def _secret_free_caddy_projection(raw: bytes) -> Mapping[str, Any]:
    value = _decode_json(raw, canonical=False)
    apps = value.get("apps")
    if (
        not isinstance(apps, Mapping)
        or "http" not in apps
        or any(name not in {"http", "pki", "tls"} for name in apps)
    ):
        _error("owner_gate_production_ingress_caddy_config_invalid")
    http = apps.get("http") if isinstance(apps, Mapping) else None
    servers = http.get("servers") if isinstance(http, Mapping) else None
    if not isinstance(servers, Mapping) or not servers:
        _error("owner_gate_production_ingress_caddy_config_invalid")
    projections: list[_ServerRouteProjection] = []
    for server in servers.values():
        if not isinstance(server, Mapping):
            _error("owner_gate_production_ingress_caddy_config_invalid")
        if server.get("listener_wrappers") not in (None, []):
            _error("owner_gate_production_ingress_caddy_route_unsafe")
        routes = server.get("routes", [])
        exact_count, proxy_count, dials, _consumes_all = _analyze_routes(routes)
        errors = server.get("errors")
        if errors is not None:
            if not isinstance(errors, Mapping):
                _error("owner_gate_production_ingress_caddy_config_invalid")
            (
                error_exact,
                error_proxies,
                error_dials,
                _error_consumes_all,
            ) = _analyze_routes(errors.get("routes", []))
            exact_count += error_exact
            proxy_count += error_proxies
            dials += error_dials
        projections.append(
            _ServerRouteProjection(
                listeners=_listener_projection(server.get("listen")),
                exact_host_route_count=exact_count,
                reverse_proxy_handler_count=proxy_count,
                dials=dials,
            )
        )
    exact_total = sum(item.exact_host_route_count for item in projections)
    if exact_total != 1 and any(
        _dial_targets_private_v2(dial)
        for item in projections
        for dial in item.dials
    ):
        _error("owner_gate_production_ingress_private_v2_already_active")
    if exact_total != 1:
        _error("owner_gate_production_ingress_caddy_route_unsafe")
    anchor = next(item for item in projections if item.exact_host_route_count)
    effective = [
        item
        for item in projections
        if _listeners_may_overlap(anchor.listeners, item.listeners)
    ]
    reverse_proxy_handler_count = sum(
        item.reverse_proxy_handler_count for item in effective
    )
    dials = tuple(dial for item in effective for dial in item.dials)
    if any(_dial_targets_private_v2(dial) for dial in dials):
        _error("owner_gate_production_ingress_private_v2_already_active")
    if reverse_proxy_handler_count != 1 or dials != (LEGACY_V1_UPSTREAM,):
        _error("owner_gate_production_ingress_caddy_route_unsafe")
    still_on_current_host = all(_dial_is_on_current_host(dial) for dial in dials)
    if not still_on_current_host:
        _error("owner_gate_production_ingress_caddy_route_unsafe")
    return {
        "auth_host_route_count": 1,
        "reverse_proxy_handler_count": reverse_proxy_handler_count,
        "reverse_proxy_upstream_count": len(dials),
        "reverse_proxy_upstreams": list(dials),
        "legacy_v1_upstream_active": True,
        "still_on_current_host": still_on_current_host,
        "private_v2_upstream_active": False,
    }


def _caddy_projection() -> Mapping[str, Any]:
    first_file = _stable_caddyfile()
    first_service = _caddy_service_projection()
    first_process = _caddy_process_snapshot(first_service)
    first_adapted = _run_command(
        _caddy_adapt_command(),
        maximum_output_bytes=MAX_ADAPTED_OUTPUT_BYTES,
    )
    first_adapted_value = _decode_json(first_adapted, canonical=False)
    first_adapted_projection = _secret_free_caddy_projection(first_adapted)
    first_live = _read_live_caddy_config(first_process)
    first_live_value = _decode_json(first_live, canonical=False)
    first_live_projection = _secret_free_caddy_projection(first_live)
    middle_file = _stable_caddyfile()
    second_adapted = _run_command(
        _caddy_adapt_command(),
        maximum_output_bytes=MAX_ADAPTED_OUTPUT_BYTES,
    )
    second_adapted_value = _decode_json(second_adapted, canonical=False)
    second_adapted_projection = _secret_free_caddy_projection(second_adapted)
    second_live = _read_live_caddy_config(first_process)
    second_live_value = _decode_json(second_live, canonical=False)
    second_live_projection = _secret_free_caddy_projection(second_live)
    final_file = _stable_caddyfile()
    final_service = _caddy_service_projection()
    final_process = _caddy_process_snapshot(final_service)
    if (
        first_file.identity != middle_file.identity
        or first_file.identity != final_file.identity
        or first_file.raw != middle_file.raw
        or first_file.raw != final_file.raw
        or _canonical(first_adapted_value)
        != _canonical(second_adapted_value)
        or _canonical(first_adapted_projection)
        != _canonical(second_adapted_projection)
        or _canonical(first_live_value) != _canonical(second_live_value)
        or _canonical(first_live_projection)
        != _canonical(second_live_projection)
        or _canonical(first_live_value) != _canonical(first_adapted_value)
        or _canonical(second_live_value) != _canonical(second_adapted_value)
        or not _admin_config_matches_process(first_live_value, first_process)
        or not _admin_config_matches_process(second_live_value, first_process)
        or first_file.public_projection != middle_file.public_projection
        or first_file.public_projection != final_file.public_projection
        or first_service != final_service
        or first_process.identity != final_process.identity
    ):
        _error("owner_gate_production_ingress_caddy_changed")
    projection_sha256 = _sha256(_canonical(first_live_projection))
    return {
        **first_service,
        **first_file.public_projection,
        "public_origin": PUBLIC_ORIGIN,
        **first_live_projection,
        "process_executable": CADDY_EXECUTABLE,
        "process_cmdline": list(CADDY_EXPECTED_ARGV),
        "admin_endpoint": (
            f"[{first_process.admin_host}]:{CADDY_ADMIN_PORT}"
            if ":" in first_process.admin_host
            else f"{first_process.admin_host}:{CADDY_ADMIN_PORT}"
        ),
        "live_route_projection_sha256": projection_sha256,
        "effective_unit_inventory_closed": True,
        "active_process_stable": True,
        "admin_listener_owned_by_main_pid": True,
        "live_config_matches_adapted_config": True,
        "double_live_config_projection_identical": True,
        "config_validated": True,
        "stable_nofollow_config_verified": True,
        "double_adapt_projection_identical": True,
        "rollback_mode": "pre_migration_v1_only",
    }


def collect_production_ingress_observation(
    *,
    phase: str,
    release_revision: str,
    plan_sha256: str,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Collect the one fixed safe pre-cutover projection on production."""

    checked_phase, revision, plan_digest = _binding(
        phase=phase,
        release_revision=release_revision,
        plan_sha256=plan_sha256,
    )
    if not sys.platform.startswith("linux") or os.geteuid() != 0:  # windows-footgun: ok — Linux-only production observer
        _error("owner_gate_production_ingress_remote_identity_invalid")
    collected = int(time.time()) if now_unix is None else now_unix
    if type(collected) is not int or collected <= 0:
        _error("owner_gate_production_ingress_time_invalid")
    old_v1 = _old_v1_projection()
    caddy = _caddy_projection()
    final_old_v1 = _old_v1_projection()
    if old_v1 != final_old_v1:
        _error("owner_gate_production_ingress_old_v1_changed")
    completed = int(time.time()) if now_unix is None else now_unix
    fresh_through = collected + FRESHNESS_SECONDS
    if completed < collected or completed > fresh_through:
        _error("owner_gate_production_ingress_time_invalid")
    unsigned = {
        "schema": OBSERVATION_SCHEMA,
        "phase": checked_phase,
        "release_revision": revision,
        "plan_sha256": plan_digest,
        "target": {
            "project": PROJECT,
            "zone": ZONE,
            "vm": VM_NAME,
            "instance_id": INSTANCE_ID,
        },
        "collected_at_unix": collected,
        "completed_at_unix": completed,
        "fresh_through_unix": fresh_through,
        "old_v1": old_v1,
        "caddy": caddy,
        "collector_authority": "production_root_read_only_fixed_projection",
        "caller_selected_input_accepted": False,
        "cloud_mutation_performed": False,
        "service_mutation_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    report = {**unsigned, "report_sha256": _sha256(_canonical(unsigned))}
    return validate_production_ingress_observation(
        report,
        phase=checked_phase,
        release_revision=revision,
        plan_sha256=plan_digest,
        now_unix=completed,
    )


def _observer_source(release_revision: str) -> tuple[bytes, str]:
    _binding(
        phase="inert",
        release_revision=release_revision,
        plan_sha256="0" * 64,
    )
    from scripts.canary import full_canary_owner_launcher as launcher

    class ObserverProvenance(launcher.LocalLauncherProvenance):
        _RELATIVE_MODULE = (
            "scripts/canary/owner_gate_production_ingress_observation.py"
        )

    class ContractProvenance(launcher.LocalLauncherProvenance):
        _RELATIVE_MODULE = (
            "scripts/canary/owner_gate_production_ingress_contract.py"
        )

    # The imported support modules intentionally live below the sealed
    # owner-support root, which is not a Git checkout.  Prove the corresponding
    # tracked blobs through the canonical launcher checkout, then require the
    # sealed bytes actually streamed below to match those Git digests.
    checkout_root = Path(launcher.__file__).absolute().parents[2]
    observer_path = Path(__file__).resolve()
    contract_path = Path(contract.__file__).resolve()
    observer_digest = ObserverProvenance(
        module_path=checkout_root / ObserverProvenance._RELATIVE_MODULE
    )(release_revision)
    contract_digest = ContractProvenance(
        module_path=checkout_root / ContractProvenance._RELATIVE_MODULE
    )(release_revision)

    def stable_component(path: Path, digest: str) -> bytes:
        try:
            before = path.lstat()
            raw = path.read_bytes()
            after = path.lstat()
        except OSError as exc:
            _error("owner_gate_production_ingress_source_invalid", exc)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or not 0 < before.st_size <= MAX_REMOTE_SOURCE_BYTES
            or len(raw) != before.st_size
            or _sha256(raw) != digest
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            _error("owner_gate_production_ingress_source_invalid")
        return raw

    observer_raw = stable_component(observer_path, observer_digest)
    contract_raw = stable_component(contract_path, contract_digest)
    bundle = _self_contained_observer_bundle(
        observer_source=observer_raw,
        contract_source=contract_raw,
    )
    return bundle, _sha256(bundle)


def _self_contained_observer_bundle(
    *,
    observer_source: bytes,
    contract_source: bytes,
) -> bytes:
    if (
        not isinstance(observer_source, bytes)
        or not observer_source
        or len(observer_source) > MAX_REMOTE_SOURCE_BYTES
        or not isinstance(contract_source, bytes)
        or not contract_source
        or len(contract_source) > MAX_REMOTE_SOURCE_BYTES
    ):
        _error("owner_gate_production_ingress_source_invalid")
    bundle = (
        b"import sys as _sys, types as _types\n"
        b"_scripts = _types.ModuleType('scripts')\n"
        b"_scripts.__path__ = ()\n"
        b"_canary = _types.ModuleType('scripts.canary')\n"
        b"_canary.__path__ = ()\n"
        b"_scripts.canary = _canary\n"
        b"_sys.modules['scripts'] = _scripts\n"
        b"_sys.modules['scripts.canary'] = _canary\n"
        b"_contract = _types.ModuleType("
        b"'scripts.canary.owner_gate_production_ingress_contract')\n"
        b"_contract.__file__ = '<owner-gate-production-ingress-contract>'\n"
        b"_contract.__package__ = 'scripts.canary'\n"
        b"_sys.modules["
        b"'scripts.canary.owner_gate_production_ingress_contract'"
        b"] = _contract\n"
        b"_canary.owner_gate_production_ingress_contract = _contract\n"
        b"exec(compile("
        + repr(contract_source).encode("ascii")
        + b", _contract.__file__, 'exec'), _contract.__dict__)\n"
        b"_observer_globals = {"
        b"'__name__': '__main__', "
        b"'__file__': '<owner-gate-production-ingress-observation>', "
        b"'__package__': None}\n"
        b"exec(compile("
        + repr(observer_source).encode("ascii")
        + b", _observer_globals['__file__'], 'exec'), _observer_globals)\n"
    )
    if len(bundle) > MAX_REMOTE_SOURCE_BYTES:
        _error("owner_gate_production_ingress_source_invalid")
    return bundle


def _stable_owner_file(path: Path, *, maximum: int) -> bytes:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or ".." in path.parts
        or type(maximum) is not int
        or maximum <= 0
    ):
        _error("owner_gate_production_ingress_owner_file_invalid")
    descriptor: int | None = None
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink != 1
            or opened.st_uid not in {0, os.geteuid()}  # windows-footgun: ok — Linux-only production observer
            or opened.st_mode & 0o022
            or not 0 < opened.st_size <= maximum
        ):
            _error("owner_gate_production_ingress_owner_file_invalid")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                _error("owner_gate_production_ingress_owner_file_changed")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        final = path.lstat()
        expected = (
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if expected != (
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (final.st_dev, final.st_ino) != (opened.st_dev, opened.st_ino):
            _error("owner_gate_production_ingress_owner_file_changed")
        return b"".join(chunks)
    except ProductionIngressObservationError:
        raise
    except OSError as exc:
        _error("owner_gate_production_ingress_owner_file_invalid", exc)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                _error("owner_gate_production_ingress_owner_file_invalid", exc)


class OwnerGateProductionIngressTransport:
    """Pinned IAP/root transport for the fixed production projection."""

    def __init__(
        self,
        transport: Any | None = None,
        *,
        revision: str | None = None,
    ) -> None:
        if transport is None:
            from scripts.canary import full_canary_owner_launcher as launcher
            from scripts.canary import production_cutover_owner_launcher as cutover

            if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
                _error("owner_gate_production_ingress_transport_invalid")
            trusted = launcher.require_trusted_owner_runtime(revision)
            configuration = launcher.PinnedGcloudConfiguration()
            identity = launcher.GcloudOwnerAccessToken(
                gcloud_executable=trusted,
                gcloud_configuration=configuration,
            )
            identity.account_for_read_only_preflight()
            transport = cutover.ProductionCutoverTransport(
                identity,
                gcloud_executable=trusted,
                gcloud_configuration=configuration,
            )
        required = (
            "_owner_identity",
            "_authorization_snapshot",
            "_run_remote_input",
            "_fixed_remote_environment",
            "_known_hosts",
        )
        if any(not hasattr(transport, name) for name in required):
            _error("owner_gate_production_ingress_transport_invalid")
        self._transport = transport

    def observe(
        self,
        *,
        phase: str,
        release_revision: str,
        plan_sha256: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        checked_phase, revision, plan_digest = _binding(
            phase=phase,
            release_revision=release_revision,
            plan_sha256=plan_sha256,
        )
        source, source_sha256 = _observer_source(revision)
        inner = self._transport
        account = inner._owner_identity.account_for_read_only_preflight()
        inner._owner_identity.require_stable()
        authorization = inner._authorization_snapshot(account)
        if (
            not isinstance(authorization, tuple)
            or len(authorization) != 3
            or any(
                not isinstance(item, str) or _SHA256.fullmatch(item) is None
                for item in authorization
            )
        ):
            _error("owner_gate_production_ingress_transport_invalid")
        command = (
            *inner._fixed_remote_environment(chdir="/"),
            REMOTE_PYTHON,
            "-B",
            "-I",
            "-",
            checked_phase,
            "--release-revision",
            revision,
            "--plan-sha256",
            plan_digest,
        )
        try:
            completed = inner._run_remote_input(
                command,
                account=account,
                input_bytes=source,
                timeout_seconds=120,
                maximum_input_bytes=MAX_REMOTE_SOURCE_BYTES,
                maximum_output_bytes=MAX_REMOTE_OUTPUT_BYTES,
            )
        except Exception as exc:
            _error("owner_gate_production_ingress_transport_failed", exc)
        inner._owner_identity.require_stable()
        if inner._authorization_snapshot(account) != authorization:
            _error("owner_gate_production_ingress_transport_changed")
        raw = completed.stdout
        if (
            not isinstance(raw, bytes)
            or not raw.endswith(b"\n")
            or b"\n" in raw[:-1]
        ):
            _error("owner_gate_production_ingress_transport_output_invalid")
        observation = _decode_json(raw[:-1], canonical=True)
        now_unix = int(time.time())
        checked = validate_production_ingress_observation(
            observation,
            phase=checked_phase,
            release_revision=revision,
            plan_sha256=plan_digest,
            now_unix=now_unix,
        )
        known_hosts_path = Path(inner._known_hosts.absolute_path())
        known_hosts_raw = _stable_owner_file(
            known_hosts_path,
            maximum=256 * 1024,
        )
        authority = {
            "kind": "pinned_owner_gcloud_iap_ssh_read_only",
            "project": PROJECT,
            "zone": ZONE,
            "vm": VM_NAME,
            "instance_id": INSTANCE_ID,
            "known_hosts_file_sha256": _sha256(known_hosts_raw),
            "observer_source_sha256": source_sha256,
            "instance_authorization_sha256": authorization[0],
            "project_authorization_sha256": authorization[1],
            "oslogin_authorization_sha256": authorization[2],
        }
        return checked, authority


def collect_and_sign_production_ingress_observation(
    transport: Any,
    *,
    phase: str,
    release_revision: str,
    plan_sha256: str,
    release_private_key: Any,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Collect over pinned IAP and release-sign the exact canonical envelope."""

    checked_phase, revision, plan_digest = _binding(
        phase=phase,
        release_revision=release_revision,
        plan_sha256=plan_sha256,
    )
    if not isinstance(transport, OwnerGateProductionIngressTransport):
        _error("owner_gate_production_ingress_transport_invalid")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as exc:
        _error("owner_gate_production_ingress_signer_invalid", exc)
    if not isinstance(release_private_key, Ed25519PrivateKey):
        _error("owner_gate_production_ingress_signer_invalid")
    observation, transport_authority = transport.observe(
        phase=checked_phase,
        release_revision=revision,
        plan_sha256=plan_digest,
    )
    signed_at = int(time.time()) if now_unix is None else now_unix
    checked_observation = validate_production_ingress_observation(
        observation,
        phase=checked_phase,
        release_revision=revision,
        plan_sha256=plan_digest,
        now_unix=signed_at,
    )
    checked_transport = _validate_transport_authority(transport_authority)
    completed = checked_observation["completed_at_unix"]
    fresh = checked_observation["fresh_through_unix"]
    if (
        type(signed_at) is not int
        or signed_at < completed
        or signed_at - completed > MAX_SIGNING_DELAY_SECONDS
        or signed_at > fresh
    ):
        _error("owner_gate_production_ingress_signing_time_invalid")
    signer_key_id = _require_pinned_release_key(
        release_private_key.public_key()
    )
    unsigned = {
        "schema": ENVELOPE_SCHEMA,
        "phase": checked_phase,
        "release_revision": revision,
        "plan_sha256": plan_digest,
        "observation": checked_observation,
        "observer_report_sha256": checked_observation["report_sha256"],
        "transport_authority": dict(checked_transport),
        "signed_at_unix": signed_at,
        "fresh_through_unix": fresh,
        "signer_key_id": signer_key_id,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    signature = release_private_key.sign(SIGNATURE_DOMAIN + _canonical(unsigned))
    if len(signature) != 64:
        _error("owner_gate_production_ingress_signature_invalid")
    signed = {
        **unsigned,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }
    envelope = {**signed, "envelope_sha256": _sha256(_canonical(signed))}
    return validate_signed_production_ingress_observation(
        envelope,
        phase=checked_phase,
        release_revision=revision,
        plan_sha256=plan_digest,
        release_public_key=release_private_key.public_key(),
        now_unix=signed_at,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect fixed owner-gate ingress safety facts on production",
    )
    parser.add_argument("phase", choices=("inert", "post_iam"))
    parser.add_argument("--release-revision", required=True)
    parser.add_argument("--plan-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        value = collect_production_ingress_observation(
            phase=arguments.phase,
            release_revision=arguments.release_revision,
            plan_sha256=arguments.plan_sha256,
        )
    except ProductionIngressObservationError as exc:
        failure = {
            "schema": FAILURE_SCHEMA,
            "ok": False,
            "error_code": str(exc),
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        print(_canonical(failure).decode("ascii"))
        return 1
    except BaseException:
        failure = {
            "schema": FAILURE_SCHEMA,
            "ok": False,
            "error_code": "owner_gate_production_ingress_unexpected_failure",
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        print(_canonical(failure).decode("ascii"))
        return 1
    print(_canonical(value).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENVELOPE_SCHEMA",
    "FRESHNESS_SECONDS",
    "OBSERVATION_SCHEMA",
    "OwnerGateProductionIngressTransport",
    "ProductionIngressObservationError",
    "collect_and_sign_production_ingress_observation",
    "collect_production_ingress_observation",
    "validate_production_ingress_observation",
    "validate_signed_production_ingress_observation",
]
