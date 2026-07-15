#!/usr/bin/env python3
"""Digest-bound systemd rail for reviewed Muncho mechanical jobs.

This module is deliberately outside Hermes cron.  It has no model, provider,
Discord, prompt, plugin, or tool dependency.  A job can run only when its
exact identifier, source paths, argument vector, and source digests are baked
into a release-addressed systemd unit.  Adding another job therefore requires
a reviewed code and package change; persisted Hermes scripts are never
discovered or admitted dynamically.

The first and currently only job is the fork-only upstream sync PR routine.
It may create a branch and PR in ``lomliev/hermes-agent``.  The packaged rail
does not set the separate auto-merge/deploy approval, so it cannot merge,
deploy, restart services, or mutate the public upstream repository.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping


RAIL_SCHEMA = "muncho-mechanical-job-rail.v1"
RUN_RECEIPT_SCHEMA = "muncho-mechanical-job-run-receipt.v1"
MANIFEST_SCHEMA = "muncho-mechanical-job-package.v1"
HOST_FACTS_SCHEMA = "muncho-mechanical-job-host-facts.v1"
JOB_ID = "fork_upstream_auto_sync_pr"
SERVICE_UNIT = "muncho-fork-upstream-auto-sync.service"
TIMER_UNIT = "muncho-fork-upstream-auto-sync.timer"
SERVICE_USER = "muncho-fork-auto-sync"
SERVICE_GROUP = "muncho-fork-auto-sync"
STATE_DIRECTORY_NAME = "muncho-fork-auto-sync"
STATE_ROOT = Path("/var/lib") / STATE_DIRECTORY_NAME
RUNTIME_ROOT = Path("/run") / STATE_DIRECTORY_NAME
PACKAGE_ROOT = Path(
    "/var/lib/muncho-production-legacy-cutover/staged/mechanical-job-rail"
)
CREDENTIAL_NAME = "github-token"
CREDENTIAL_SOURCE = Path("/etc/muncho/fork-auto-sync/github-token")
GH_PATH = Path("/usr/bin/gh")
GIT_PATH = Path("/usr/bin/git")
RELEASES_ROOT = Path("/opt/adventico-ai-platform/hermes-agent-releases")
ROUTINE_RELATIVE = Path(
    "ops/muncho/runtime/fork_upstream_auto_sync_pr_routine.py"
)
HARDENING_RELATIVE = Path("ops/muncho/runtime/auto_sync_hardening.py")
RAIL_RELATIVE = Path("ops/muncho/runtime/mechanical_job_rail.py")
SOURCE_MARKER_RELATIVE = Path(".codex-source-commit")
RUN_TIMEOUT_SECONDS = 45 * 60
MAX_CAPTURE_BYTES = 8 * 1024 * 1024

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SYSTEMD_IDENTITY = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_]{20,4096}$")
_INVOCATION = re.compile(r"^[0-9a-f]{32}$")
_UTC = re.compile(r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


class MechanicalJobRailError(RuntimeError):
    """Stable failure for package or execution contract violations."""


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise MechanicalJobRailError("mechanical_job_json_not_canonical") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular_file(
    path: Path,
    *,
    maximum: int,
    expected_mode: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MechanicalJobRailError("mechanical_job_file_unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
            or (
                expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode
            )
        ):
            raise MechanicalJobRailError("mechanical_job_file_metadata_invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    value = b"".join(chunks)
    if len(value) != before.st_size or identity(before) != identity(after):
        raise MechanicalJobRailError("mechanical_job_file_changed_while_reading")
    return value


def _digest_file(path: Path, *, maximum: int = 4 * 1024 * 1024) -> str:
    return _sha256_bytes(_read_regular_file(path, maximum=maximum))


def _validate_digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MechanicalJobRailError(f"{label}_invalid")
    return value


def _release_root(revision: str) -> Path:
    if not isinstance(revision, str) or _SHA40.fullmatch(revision) is None:
        raise MechanicalJobRailError("mechanical_job_revision_invalid")
    return RELEASES_ROOT / f"hermes-agent-{revision[:12]}"


def _identity(value: str, label: str) -> str:
    if not isinstance(value, str) or _SYSTEMD_IDENTITY.fullmatch(value) is None:
        raise MechanicalJobRailError(f"{label}_invalid")
    return value


@dataclass(frozen=True)
class MechanicalJobPackage:
    revision: str
    release_root: Path
    rail_sha256: str
    routine_sha256: str
    hardening_sha256: str
    host_facts_sha256: str
    gh_sha256: str
    git_sha256: str
    service_bytes: bytes
    service_sha256: str
    timer_bytes: bytes
    timer_sha256: str
    manifest_bytes: bytes
    manifest_sha256: str


def _service_unit(
    *,
    revision: str,
    release: Path,
    interpreter: Path,
    rail_sha256: str,
    routine_sha256: str,
    hardening_sha256: str,
    host_facts_sha256: str,
    gh_sha256: str,
    git_sha256: str,
    service_user: str,
    service_group: str,
) -> bytes:
    rail = release / RAIL_RELATIVE
    routine = release / ROUTINE_RELATIVE
    hardening = release / HARDENING_RELATIVE
    lines = [
        "# Exact release-addressed Muncho mechanical-job rail; do not edit.",
        f"# ReleaseRevision={revision}",
        f"# RailSHA256={rail_sha256}",
        f"# RoutineSHA256={routine_sha256}",
        f"# HardeningSHA256={hardening_sha256}",
        f"# HostFactsSHA256={host_facts_sha256}",
        f"# GitHubCLISHA256={gh_sha256}",
        f"# GitSHA256={git_sha256}",
        "[Unit]",
        "Description=Muncho fork-only upstream sync PR mechanical rail",
        "Wants=network-online.target",
        "After=network-online.target",
        "AssertPathExists=/usr/bin/git",
        "AssertPathExists=/usr/bin/gh",
        f"AssertPathExists={interpreter}",
        f"AssertPathExists={release / SOURCE_MARKER_RELATIVE}",
        f"AssertPathExists={rail}",
        f"AssertPathExists={routine}",
        f"AssertPathExists={hardening}",
        f"AssertPathExists={CREDENTIAL_SOURCE}",
        "",
        "[Service]",
        "Type=oneshot",
        "DynamicUser=yes",
        f"User={service_user}",
        f"Group={service_group}",
        f"LoadCredential={CREDENTIAL_NAME}:{CREDENTIAL_SOURCE}",
        f"StateDirectory={STATE_DIRECTORY_NAME}",
        "StateDirectoryMode=0700",
        f"RuntimeDirectory={STATE_DIRECTORY_NAME}",
        "RuntimeDirectoryMode=0700",
        "RuntimeDirectoryPreserve=no",
        "WorkingDirectory=/",
        "Environment=HOME=/var/lib/muncho-fork-auto-sync",
        "Environment=LANG=C.UTF-8",
        "Environment=LC_ALL=C.UTF-8",
        "Environment=PATH=/usr/bin:/bin",
        "Environment=TZ=UTC",
        (
            f"ExecStart={interpreter} -I -S -B {rail} run "
            f"--job-id {JOB_ID} --revision {revision} "
            f"--rail-sha256 {rail_sha256} "
            f"--routine-sha256 {routine_sha256} "
            f"--hardening-sha256 {hardening_sha256} "
            f"--gh-sha256 {gh_sha256} --git-sha256 {git_sha256}"
        ),
        "TimeoutStartSec=3000s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "LimitCORE=0",
        "UMask=0077",
        "NoNewPrivileges=yes",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "LockPersonality=yes",
        "MemoryDenyWriteExecute=yes",
        "PrivateDevices=yes",
        "PrivateTmp=yes",
        "ProtectClock=yes",
        "ProtectControlGroups=yes",
        "ProtectHome=yes",
        "ProtectHostname=yes",
        "ProtectKernelLogs=yes",
        "ProtectKernelModules=yes",
        "ProtectKernelTunables=yes",
        "ProtectProc=invisible",
        "ProtectSystem=strict",
        "ProcSubset=pid",
        "RemoveIPC=yes",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "RestrictNamespaces=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "SystemCallArchitectures=native",
        "IPAddressDeny=169.254.169.254/32",
        f"ReadOnlyPaths={release}",
        "ReadOnlyPaths=/usr/bin/git",
        "ReadOnlyPaths=/usr/bin/gh",
        f"ReadWritePaths={STATE_ROOT}",
        f"ReadWritePaths={RUNTIME_ROOT}",
        "InaccessiblePaths=-/opt/adventico-ai-platform/hermes-home/.env",
        "InaccessiblePaths=-/opt/adventico-ai-platform/hermes-home/auth.json",
        "InaccessiblePaths=-/var/lib/hermes-gateway/.hermes/auth.json",
        "InaccessiblePaths=-/etc/muncho/fork-auto-sync",
        "InaccessiblePaths=-/etc/muncho/discord-connector-credentials",
        "InaccessiblePaths=-/etc/muncho/discord-edge-credentials",
        "InaccessiblePaths=-/etc/muncho/mac-ops-edge-credentials",
        "InaccessiblePaths=-/run/credentials/hermes-cloud-gateway.service",
        "InaccessiblePaths=-/run/credentials/muncho-discord-connector.service",
        "InaccessiblePaths=-/run/credentials/muncho-discord-egress.service",
        "StandardOutput=null",
        "StandardError=journal",
    ]
    result = ("\n".join(lines) + "\n").encode("utf-8", errors="strict")
    _validate_service_unit(
        result,
        revision=revision,
        release=release,
        rail_sha256=rail_sha256,
        routine_sha256=routine_sha256,
        hardening_sha256=hardening_sha256,
        host_facts_sha256=host_facts_sha256,
        gh_sha256=gh_sha256,
        git_sha256=git_sha256,
    )
    return result


def _timer_unit() -> bytes:
    lines = [
        "# Timer activation waits 30 minutes; package/deploy never runs sync inline.",
        "[Unit]",
        "Description=Schedule Muncho fork-only upstream sync PR mechanical rail",
        "",
        "[Timer]",
        f"Unit={SERVICE_UNIT}",
        "OnActiveSec=30m",
        "OnUnitActiveSec=3h",
        "AccuracySec=1m",
        "RandomizedDelaySec=5m",
        "Persistent=false",
        "",
        "[Install]",
        "WantedBy=timers.target",
    ]
    result = ("\n".join(lines) + "\n").encode("utf-8", errors="strict")
    _validate_timer_unit(result)
    return result


def _validate_service_unit(
    value: bytes,
    *,
    revision: str,
    release: Path,
    rail_sha256: str,
    routine_sha256: str,
    hardening_sha256: str,
    host_facts_sha256: str,
    gh_sha256: str,
    git_sha256: str,
) -> None:
    """Validate the rendered unit as an exact security/provenance contract."""

    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise MechanicalJobRailError("mechanical_job_service_unit_invalid") from exc
    required_once = (
        f"# ReleaseRevision={revision}\n",
        f"# RailSHA256={rail_sha256}\n",
        f"# RoutineSHA256={routine_sha256}\n",
        f"# HardeningSHA256={hardening_sha256}\n",
        f"# HostFactsSHA256={host_facts_sha256}\n",
        f"# GitHubCLISHA256={gh_sha256}\n",
        f"# GitSHA256={git_sha256}\n",
        "Type=oneshot\n",
        "DynamicUser=yes\n",
        f"User={SERVICE_USER}\n",
        f"Group={SERVICE_GROUP}\n",
        f"LoadCredential={CREDENTIAL_NAME}:{CREDENTIAL_SOURCE}\n",
        "NoNewPrivileges=yes\n",
        "CapabilityBoundingSet=\n",
        "AmbientCapabilities=\n",
        "ProtectSystem=strict\n",
        "ProtectHome=yes\n",
        "RestrictNamespaces=yes\n",
        "StandardOutput=null\n",
        "StandardError=journal\n",
    )
    exec_prefix = (
        f"ExecStart={release / '.venv/bin/python'} -I -S -B "
        f"{release / RAIL_RELATIVE} run --job-id {JOB_ID} "
        f"--revision {revision} --rail-sha256 {rail_sha256} "
        f"--routine-sha256 {routine_sha256} "
        f"--hardening-sha256 {hardening_sha256} "
        f"--gh-sha256 {gh_sha256} --git-sha256 {git_sha256}\n"
    )
    forbidden = (
        "EnvironmentFile=",
        "PassEnvironment=",
        "OnFailure=",
        "Restart=",
        "GH_TOKEN=",
        "GITHUB_TOKEN=",
        "OPENAI_API_KEY",
        "DISCORD_BOT_TOKEN",
        "FORK_UPSTREAM_AUTO_SYNC_AUTO_MERGE_DEPLOY_APPROVED",
        "muncho-auto-deploy-release",
        "NousResearch/hermes-agent.git HEAD:",
    )
    if (
        not text.endswith("\n")
        or "\x00" in text
        or "\r" in text
        or text.count(exec_prefix) != 1
        or any(text.count(marker) != 1 for marker in required_once)
        or any(marker in text for marker in forbidden)
    ):
        raise MechanicalJobRailError("mechanical_job_service_unit_invalid")


def _validate_timer_unit(value: bytes) -> None:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise MechanicalJobRailError("mechanical_job_timer_unit_invalid") from exc
    required_once = (
        f"Unit={SERVICE_UNIT}\n",
        "OnActiveSec=30m\n",
        "OnUnitActiveSec=3h\n",
        "AccuracySec=1m\n",
        "RandomizedDelaySec=5m\n",
        "Persistent=false\n",
        "WantedBy=timers.target\n",
    )
    if (
        not text.endswith("\n")
        or "\x00" in text
        or "\r" in text
        or any(text.count(marker) != 1 for marker in required_once)
        or any(
            marker in text
            for marker in (
                "OnBootSec=",
                "OnCalendar=",
                "Persistent=true",
                "WakeSystem=true",
            )
        )
    ):
        raise MechanicalJobRailError("mechanical_job_timer_unit_invalid")


def _host_binary_fact(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MechanicalJobRailError("mechanical_job_host_binary_unavailable") from exc
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or mode & 0o022
        or not mode & 0o111
    ):
        raise MechanicalJobRailError("mechanical_job_host_binary_metadata_invalid")
    return {
        "path": str(path),
        "regular": True,
        "nlink": 1,
        "uid": 0,
        "gid": 0,
        "mode": f"{mode:04o}",
        "group_or_other_writable": False,
        "sha256": _digest_file(path, maximum=128 * 1024 * 1024),
    }


def collect_host_facts(
    *,
    gh_path: Path = GH_PATH,
    git_path: Path = GIT_PATH,
    credential_source: Path = CREDENTIAL_SOURCE,
) -> dict[str, Any]:
    """Collect redaction-safe host facts; never open the credential file."""

    if (
        gh_path != GH_PATH
        or git_path != GIT_PATH
        or credential_source != CREDENTIAL_SOURCE
    ):
        raise MechanicalJobRailError("mechanical_job_host_path_not_exact")
    try:
        credential = credential_source.lstat()
    except OSError as exc:
        raise MechanicalJobRailError(
            "mechanical_job_github_credential_unavailable"
        ) from exc
    if (
        not stat.S_ISREG(credential.st_mode)
        or credential.st_nlink != 1
        or credential.st_uid != 0
        or credential.st_gid != 0
        or stat.S_IMODE(credential.st_mode) != 0o400
    ):
        raise MechanicalJobRailError(
            "mechanical_job_github_credential_metadata_invalid"
        )
    unsigned = {
        "schema": HOST_FACTS_SCHEMA,
        "collected_at": _now(),
        "github_cli": _host_binary_fact(gh_path),
        "git": _host_binary_fact(git_path),
        "github_credential": {
            "path": str(credential_source),
            "regular": True,
            "nlink": 1,
            "uid": 0,
            "gid": 0,
            "mode": "0400",
            "content_recorded": False,
            "size_recorded": False,
            "digest_recorded": False,
        },
        "provider_or_model_credential_observed": False,
        "discord_credential_observed": False,
    }
    return {**unsigned, "host_facts_sha256": _sha256_bytes(_canonical(unsigned))}


def validate_host_facts(
    facts: Mapping[str, Any], *, expected_sha256: str | None = None
) -> dict[str, Any]:
    if expected_sha256 is None and isinstance(facts, Mapping):
        candidate = facts.get("host_facts_sha256")
        expected_sha256 = candidate if isinstance(candidate, str) else ""
    expected = _validate_digest(
        expected_sha256, "mechanical_job_host_facts_sha256"
    )
    if (
        not isinstance(facts, Mapping)
        or set(facts)
        != {
            "schema",
            "collected_at",
            "github_cli",
            "git",
            "github_credential",
            "provider_or_model_credential_observed",
            "discord_credential_observed",
            "host_facts_sha256",
        }
        or facts.get("schema") != HOST_FACTS_SCHEMA
    ):
        raise MechanicalJobRailError("mechanical_job_host_facts_invalid")
    unsigned = {key: value for key, value in facts.items() if key != "host_facts_sha256"}
    if facts.get("host_facts_sha256") != expected or _sha256_bytes(
        _canonical(unsigned)
    ) != expected:
        raise MechanicalJobRailError("mechanical_job_host_facts_invalid")
    expected_binary_paths = {
        "github_cli": GH_PATH,
        "git": GIT_PATH,
    }
    for name, path in expected_binary_paths.items():
        value = facts.get(name)
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "path",
                "regular",
                "nlink",
                "uid",
                "gid",
                "mode",
                "group_or_other_writable",
                "sha256",
            }
            or value.get("path") != str(path)
            or value.get("regular") is not True
            or value.get("nlink") != 1
            or value.get("uid") != 0
            or value.get("gid") != 0
            or not isinstance(value.get("mode"), str)
            or re.fullmatch(r"0[0-7]{3}", value["mode"]) is None
            or int(value["mode"], 8) & 0o022
            or int(value["mode"], 8) & 0o111 == 0
            or value.get("group_or_other_writable") is not False
            or not isinstance(value.get("sha256"), str)
            or _SHA256.fullmatch(value["sha256"]) is None
        ):
            raise MechanicalJobRailError("mechanical_job_host_facts_invalid")
    credential = facts.get("github_credential")
    if credential != {
        "path": str(CREDENTIAL_SOURCE),
        "regular": True,
        "nlink": 1,
        "uid": 0,
        "gid": 0,
        "mode": "0400",
        "content_recorded": False,
        "size_recorded": False,
        "digest_recorded": False,
    }:
        raise MechanicalJobRailError("mechanical_job_host_facts_invalid")
    if (
        facts.get("provider_or_model_credential_observed") is not False
        or facts.get("discord_credential_observed") is not False
        or not isinstance(facts.get("collected_at"), str)
        or _UTC.fullmatch(facts["collected_at"]) is None
    ):
        raise MechanicalJobRailError("mechanical_job_host_facts_invalid")
    return dict(facts)


def build_package(
    *,
    revision: str,
    host_facts: Mapping[str, Any],
    expected_host_facts_sha256: str,
    service_user: str = SERVICE_USER,
    service_group: str = SERVICE_GROUP,
) -> MechanicalJobPackage:
    """Render an exact package without installing or starting anything."""

    release = _release_root(revision)
    try:
        resolved_release = release.resolve(strict=True)
    except OSError as exc:
        raise MechanicalJobRailError("mechanical_job_release_unavailable") from exc
    if resolved_release != release:
        raise MechanicalJobRailError("mechanical_job_release_not_final_address")
    marker = _read_regular_file(release / SOURCE_MARKER_RELATIVE, maximum=128)
    try:
        marker_revision = marker.decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise MechanicalJobRailError(
            "mechanical_job_release_marker_mismatch"
        ) from exc
    if marker_revision != revision:
        raise MechanicalJobRailError("mechanical_job_release_marker_mismatch")
    user = _identity(service_user, "mechanical_job_service_user")
    group = _identity(service_group, "mechanical_job_service_group")
    if user != SERVICE_USER or group != SERVICE_GROUP:
        raise MechanicalJobRailError("mechanical_job_service_identity_not_exact")
    trusted_host_facts = validate_host_facts(
        host_facts,
        expected_sha256=expected_host_facts_sha256,
    )
    gh_sha = str(trusted_host_facts["github_cli"]["sha256"])
    git_sha = str(trusted_host_facts["git"]["sha256"])
    interpreter = release / ".venv/bin/python"
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise MechanicalJobRailError("mechanical_job_interpreter_unavailable")
    rail_sha = _digest_file(release / RAIL_RELATIVE)
    routine_sha = _digest_file(release / ROUTINE_RELATIVE)
    hardening_sha = _digest_file(release / HARDENING_RELATIVE)
    service = _service_unit(
        revision=revision,
        release=release,
        interpreter=interpreter,
        rail_sha256=rail_sha,
        routine_sha256=routine_sha,
        hardening_sha256=hardening_sha,
        host_facts_sha256=expected_host_facts_sha256,
        gh_sha256=gh_sha,
        git_sha256=git_sha,
        service_user=user,
        service_group=group,
    )
    timer = _timer_unit()
    service_sha = _sha256_bytes(service)
    timer_sha = _sha256_bytes(timer)
    unsigned = {
        "schema": MANIFEST_SCHEMA,
        "rail_schema": RAIL_SCHEMA,
        "release_revision": revision,
        "release_root": str(release),
        "job_allowlist": [
            {
                "job_id": JOB_ID,
                "argv": ["--execute"],
                "routine": str(release / ROUTINE_RELATIVE),
                "routine_sha256": routine_sha,
                "hardening": str(release / HARDENING_RELATIVE),
                "hardening_sha256": hardening_sha,
                "fork_repository": "lomliev/hermes-agent",
                "upstream_repository_read_only": "NousResearch/hermes-agent",
                "auto_merge_or_deploy_enabled": False,
            }
        ],
        "rail_sha256": rail_sha,
        "host_facts_sha256": expected_host_facts_sha256,
        "host_binaries": {
            str(GH_PATH): gh_sha,
            str(GIT_PATH): git_sha,
        },
        "units": {
            SERVICE_UNIT: service_sha,
            TIMER_UNIT: timer_sha,
        },
        "credential_name": CREDENTIAL_NAME,
        "credential_value_recorded": False,
        "provider_or_model_dependency": False,
        "discord_dependency": False,
        "timer_started_by_package": False,
    }
    manifest_sha = _sha256_bytes(_canonical(unsigned))
    manifest = _canonical({**unsigned, "manifest_sha256": manifest_sha}) + b"\n"
    return MechanicalJobPackage(
        revision=revision,
        release_root=release,
        rail_sha256=rail_sha,
        routine_sha256=routine_sha,
        hardening_sha256=hardening_sha,
        host_facts_sha256=expected_host_facts_sha256,
        gh_sha256=gh_sha,
        git_sha256=git_sha,
        service_bytes=service,
        service_sha256=service_sha,
        timer_bytes=timer,
        timer_sha256=timer_sha,
        manifest_bytes=manifest,
        manifest_sha256=manifest_sha,
    )


def validate_package_manifest(
    manifest: Mapping[str, Any],
    *,
    revision: str,
    host_facts_sha256: str,
) -> dict[str, Any]:
    """Pure exact validator for owner/cutover binding; reads no filesystem."""

    release = _release_root(revision)
    host_digest = _validate_digest(
        host_facts_sha256, "mechanical_job_host_facts_sha256"
    )
    expected_fields = {
        "schema",
        "rail_schema",
        "release_revision",
        "release_root",
        "job_allowlist",
        "rail_sha256",
        "host_facts_sha256",
        "host_binaries",
        "units",
        "credential_name",
        "credential_value_recorded",
        "provider_or_model_dependency",
        "discord_dependency",
        "timer_started_by_package",
        "manifest_sha256",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != expected_fields
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("rail_schema") != RAIL_SCHEMA
        or manifest.get("release_revision") != revision
        or manifest.get("release_root") != str(release)
        or manifest.get("host_facts_sha256") != host_digest
        or manifest.get("credential_name") != CREDENTIAL_NAME
        or manifest.get("credential_value_recorded") is not False
        or manifest.get("provider_or_model_dependency") is not False
        or manifest.get("discord_dependency") is not False
        or manifest.get("timer_started_by_package") is not False
        or not isinstance(manifest.get("manifest_sha256"), str)
        or _SHA256.fullmatch(manifest["manifest_sha256"]) is None
        or _sha256_bytes(
            _canonical(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
            )
        )
        != manifest["manifest_sha256"]
    ):
        raise MechanicalJobRailError("mechanical_job_package_manifest_invalid")
    allowlist = manifest.get("job_allowlist")
    if not isinstance(allowlist, list) or len(allowlist) != 1:
        raise MechanicalJobRailError("mechanical_job_package_manifest_invalid")
    job = allowlist[0]
    if (
        not isinstance(job, Mapping)
        or set(job)
        != {
            "job_id",
            "argv",
            "routine",
            "routine_sha256",
            "hardening",
            "hardening_sha256",
            "fork_repository",
            "upstream_repository_read_only",
            "auto_merge_or_deploy_enabled",
        }
        or job.get("job_id") != JOB_ID
        or job.get("argv") != ["--execute"]
        or job.get("routine") != str(release / ROUTINE_RELATIVE)
        or job.get("hardening") != str(release / HARDENING_RELATIVE)
        or job.get("fork_repository") != "lomliev/hermes-agent"
        or job.get("upstream_repository_read_only")
        != "NousResearch/hermes-agent"
        or job.get("auto_merge_or_deploy_enabled") is not False
        or _SHA256.fullmatch(str(job.get("routine_sha256") or "")) is None
        or _SHA256.fullmatch(str(job.get("hardening_sha256") or "")) is None
        or _SHA256.fullmatch(str(manifest.get("rail_sha256") or "")) is None
    ):
        raise MechanicalJobRailError("mechanical_job_package_manifest_invalid")
    binaries = manifest.get("host_binaries")
    if (
        not isinstance(binaries, Mapping)
        or set(binaries) != {str(GH_PATH), str(GIT_PATH)}
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in binaries.values()
        )
    ):
        raise MechanicalJobRailError("mechanical_job_package_manifest_invalid")
    units = manifest.get("units")
    if (
        not isinstance(units, Mapping)
        or set(units) != {SERVICE_UNIT, TIMER_UNIT}
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in units.values()
        )
    ):
        raise MechanicalJobRailError("mechanical_job_package_manifest_invalid")
    return dict(manifest)


def package_public_manifest(package: MechanicalJobPackage) -> dict[str, Any]:
    if not isinstance(package, MechanicalJobPackage):
        raise MechanicalJobRailError("mechanical_job_package_invalid")
    try:
        manifest = json.loads(package.manifest_bytes.decode("ascii", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MechanicalJobRailError("mechanical_job_package_manifest_invalid") from exc
    trusted = validate_package_manifest(
        manifest,
        revision=package.revision,
        host_facts_sha256=package.host_facts_sha256,
    )
    if trusted["manifest_sha256"] != package.manifest_sha256:
        raise MechanicalJobRailError("mechanical_job_package_manifest_invalid")
    return trusted


def _atomic_private(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_artifact(path: Path, value: bytes, *, mode: int) -> None:
    if mode not in {0o400, 0o440, 0o444, 0o600, 0o640, 0o644}:
        raise MechanicalJobRailError("mechanical_job_artifact_mode_invalid")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise MechanicalJobRailError("mechanical_job_package_root_is_symlink")
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_package(
    package: MechanicalJobPackage,
    *,
    output_root: Path = PACKAGE_ROOT,
) -> dict[str, Any]:
    """Stage exact artifacts; never install, enable, start, or run the job."""

    if not isinstance(package, MechanicalJobPackage):
        raise MechanicalJobRailError("mechanical_job_package_invalid")
    package_public_manifest(package)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise MechanicalJobRailError("mechanical_job_package_root_is_symlink")
    _atomic_artifact(root / SERVICE_UNIT, package.service_bytes, mode=0o444)
    _atomic_artifact(root / TIMER_UNIT, package.timer_bytes, mode=0o444)
    _atomic_artifact(root / "manifest.json", package.manifest_bytes, mode=0o444)
    return verify_package(package, output_root=root)


def verify_package(
    package: MechanicalJobPackage,
    *,
    output_root: Path = PACKAGE_ROOT,
) -> dict[str, Any]:
    """Verify byte identity without installing or activating systemd units."""

    root = Path(output_root)
    package_public_manifest(package)
    expected = {
        SERVICE_UNIT: package.service_bytes,
        TIMER_UNIT: package.timer_bytes,
        "manifest.json": package.manifest_bytes,
    }
    observed: dict[str, str] = {}
    for name, value in expected.items():
        path = root / name
        raw = _read_regular_file(path, maximum=2 * 1024 * 1024)
        if raw != value or stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise MechanicalJobRailError("mechanical_job_package_artifact_drifted")
        observed[name] = _sha256_bytes(raw)
    return {
        "schema": MANIFEST_SCHEMA,
        "release_revision": package.revision,
        "package_root": str(root),
        "artifacts": observed,
        "manifest_sha256": package.manifest_sha256,
        "installed": False,
        "timer_enabled": False,
        "timer_started": False,
        "job_executed": False,
    }


def _write_receipt(state_root: Path, receipt: Mapping[str, Any]) -> None:
    encoded = _canonical(dict(receipt)) + b"\n"
    receipts = state_root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(receipts, 0o700)
    receipt_id = str(receipt["receipt_id"])
    archive = receipts / f"{receipt_id}.json"
    if archive.exists():
        current = _read_regular_file(archive, maximum=128 * 1024)
        if current != encoded:
            raise MechanicalJobRailError("mechanical_job_receipt_replay_mismatch")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(archive, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            archive.unlink(missing_ok=True)
            raise
    _atomic_private(state_root / "latest.json", encoded)


def _capture_digest(stream: BinaryIO) -> tuple[int, str, bool]:
    stream.seek(0)
    digest = hashlib.sha256()
    total = 0
    truncated = False
    while True:
        chunk = stream.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        digest.update(chunk)
        if total > MAX_CAPTURE_BYTES:
            truncated = True
    return total, digest.hexdigest(), truncated


def _load_routine_status(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    try:
        raw = _read_regular_file(path, maximum=2 * 1024 * 1024)
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (MechanicalJobRailError, UnicodeError, json.JSONDecodeError):
        return None, None
    status = payload.get("status") if isinstance(payload, dict) else None
    return (status if isinstance(status, str) else None), _sha256_bytes(raw)


def _credential() -> str:
    directory = os.environ.get("CREDENTIALS_DIRECTORY")
    if not directory or not os.path.isabs(directory):
        raise MechanicalJobRailError("mechanical_job_credentials_directory_invalid")
    path = Path(directory) / CREDENTIAL_NAME
    raw = _read_regular_file(path, maximum=4096)
    try:
        value = raw.decode("ascii", errors="strict").strip()
    except UnicodeError as exc:
        raise MechanicalJobRailError("mechanical_job_github_credential_invalid") from exc
    if _TOKEN.fullmatch(value) is None:
        raise MechanicalJobRailError("mechanical_job_github_credential_invalid")
    return value


def _invocation_id() -> str:
    observed = os.environ.get("INVOCATION_ID", "")
    if _INVOCATION.fullmatch(observed):
        return observed
    return uuid.uuid4().hex


def _attest_release(
    *,
    revision: str,
    rail_sha256: str,
    routine_sha256: str,
    hardening_sha256: str,
) -> tuple[Path, Path, Path]:
    release = _release_root(revision)
    try:
        resolved = release.resolve(strict=True)
    except OSError as exc:
        raise MechanicalJobRailError("mechanical_job_release_unavailable") from exc
    if resolved != release:
        raise MechanicalJobRailError("mechanical_job_release_not_final_address")
    marker = _read_regular_file(
        release / SOURCE_MARKER_RELATIVE,
        maximum=128,
    )
    if marker.decode("ascii", errors="strict").strip() != revision:
        raise MechanicalJobRailError("mechanical_job_release_marker_mismatch")
    rail = release / RAIL_RELATIVE
    routine = release / ROUTINE_RELATIVE
    hardening = release / HARDENING_RELATIVE
    expected = {
        rail: _validate_digest(rail_sha256, "mechanical_job_rail_sha256"),
        routine: _validate_digest(
            routine_sha256, "mechanical_job_routine_sha256"
        ),
        hardening: _validate_digest(
            hardening_sha256, "mechanical_job_hardening_sha256"
        ),
    }
    for path, digest in expected.items():
        if _digest_file(path) != digest:
            raise MechanicalJobRailError("mechanical_job_source_digest_mismatch")
    if Path(__file__).resolve(strict=True) != rail:
        raise MechanicalJobRailError("mechanical_job_launcher_not_release_addressed")
    return release, routine, hardening


def _attest_host_binaries(*, gh_sha256: str, git_sha256: str) -> None:
    expected = {
        GH_PATH: _validate_digest(gh_sha256, "mechanical_job_gh_sha256"),
        GIT_PATH: _validate_digest(git_sha256, "mechanical_job_git_sha256"),
    }
    for path, digest in expected.items():
        fact = _host_binary_fact(path)
        if fact["sha256"] != digest:
            raise MechanicalJobRailError("mechanical_job_host_binary_digest_drifted")


def run_job(args: argparse.Namespace) -> int:
    """Run the one exact allowlisted routine and publish a bounded receipt."""

    if args.job_id != JOB_ID:
        raise MechanicalJobRailError("mechanical_job_not_allowlisted")
    release, routine, _hardening = _attest_release(
        revision=args.revision,
        rail_sha256=args.rail_sha256,
        routine_sha256=args.routine_sha256,
        hardening_sha256=args.hardening_sha256,
    )
    _attest_host_binaries(
        gh_sha256=args.gh_sha256,
        git_sha256=args.git_sha256,
    )
    state_root = STATE_ROOT
    runtime_root = RUNTIME_ROOT
    if os.environ.get("STATE_DIRECTORY") not in (None, "", str(state_root)):
        raise MechanicalJobRailError("mechanical_job_state_directory_drifted")
    if os.environ.get("RUNTIME_DIRECTORY") not in (None, "", str(runtime_root)):
        raise MechanicalJobRailError("mechanical_job_runtime_directory_drifted")
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = _credential()
    invocation = _invocation_id()
    receipt_id = _sha256_bytes(
        _canonical(
            {
                "invocation_id": invocation,
                "job_id": JOB_ID,
                "release_revision": args.revision,
                "routine_sha256": args.routine_sha256,
            }
        )
    )
    lock_path = runtime_root / "rail.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            receipt = {
                "schema": RUN_RECEIPT_SCHEMA,
                "rail_schema": RAIL_SCHEMA,
                "receipt_id": receipt_id,
                "created_at": _now(),
                "job_id": JOB_ID,
                "release_revision": args.revision,
                "outcome": "blocked_already_running",
                "routine_started": False,
                "provider_or_model_invoked": False,
                "discord_delivery_attempted": False,
                "secret_material_recorded": False,
            }
            _write_receipt(state_root, receipt)
            return 0

        started_at = _now()
        report_state = state_root / "routine-state"
        report_public = state_root / "routine-reports"
        worktrees = state_root / "worktrees"
        for path in (report_state, report_public, worktrees):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)
        child_environment = {
            "FORK_UPSTREAM_AUTO_SYNC_EXECUTE_APPROVED": "1",
            "FORK_UPSTREAM_AUTO_SYNC_GH": str(GH_PATH),
            "FORK_UPSTREAM_AUTO_SYNC_REPORT_DIR": str(report_public),
            "FORK_UPSTREAM_AUTO_SYNC_STATE_DIR": str(report_state),
            "FORK_UPSTREAM_AUTO_SYNC_WORKTREE_ROOT": str(worktrees),
            "GH_HOST": "github.com",
            "GH_PROMPT_DISABLED": "1",
            "GH_TOKEN": token,
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": str(state_root),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "TZ": "UTC",
        }
        return_code: int | None = None
        timed_out = False
        with tempfile.TemporaryFile(dir=runtime_root) as stdout, tempfile.TemporaryFile(
            dir=runtime_root
        ) as stderr:
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-B",
                        str(routine),
                        "--execute",
                    ],
                    cwd="/",
                    env=child_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    timeout=RUN_TIMEOUT_SECONDS,
                )
                return_code = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
            stdout_bytes, stdout_sha, stdout_truncated = _capture_digest(stdout)
            stderr_bytes, stderr_sha, stderr_truncated = _capture_digest(stderr)
        routine_status, routine_report_sha = _load_routine_status(
            report_state / "auto-sync-pr-latest.json"
        )
        if timed_out:
            outcome = "failed_timeout"
        elif return_code == 0:
            outcome = "completed"
        elif return_code == 2:
            outcome = "blocked_receipt_recorded"
        else:
            outcome = "failed_process"
        receipt = {
            "schema": RUN_RECEIPT_SCHEMA,
            "rail_schema": RAIL_SCHEMA,
            "receipt_id": receipt_id,
            "created_at": _now(),
            "started_at": started_at,
            "job_id": JOB_ID,
            "release_revision": args.revision,
            "release_root": str(release),
            "rail_sha256": args.rail_sha256,
            "routine_sha256": args.routine_sha256,
            "hardening_sha256": args.hardening_sha256,
            "gh_sha256": args.gh_sha256,
            "git_sha256": args.git_sha256,
            "outcome": outcome,
            "routine_started": True,
            "routine_return_code": return_code,
            "routine_report_status": routine_status,
            "routine_report_sha256": routine_report_sha,
            "stdout": {
                "bytes": stdout_bytes,
                "sha256": stdout_sha,
                "over_capture_bound": stdout_truncated,
                "content_recorded": False,
            },
            "stderr": {
                "bytes": stderr_bytes,
                "sha256": stderr_sha,
                "over_capture_bound": stderr_truncated,
                "content_recorded": False,
            },
            "fork_repository": "lomliev/hermes-agent",
            "upstream_repository_mutation_allowed": False,
            "auto_merge_or_deploy_approved": False,
            "provider_or_model_invoked": False,
            "discord_delivery_attempted": False,
            "secret_material_recorded": False,
        }
        _write_receipt(state_root, receipt)
        return 0 if outcome in {"completed", "blocked_receipt_recorded"} else 1
    finally:
        os.close(lock_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Muncho mechanical-job rail")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--job-id", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--rail-sha256", required=True)
    run.add_argument("--routine-sha256", required=True)
    run.add_argument("--hardening-sha256", required=True)
    run.add_argument("--gh-sha256", required=True)
    run.add_argument("--git-sha256", required=True)
    collect = subparsers.add_parser("collect-host-facts")
    collect.add_argument("--output", type=Path, required=True)
    package = subparsers.add_parser("package")
    package.add_argument("--revision", required=True)
    package.add_argument("--host-facts", type=Path, required=True)
    package.add_argument("--expected-host-facts-sha256", required=True)
    package.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    verify = subparsers.add_parser("verify-package")
    verify.add_argument("--revision", required=True)
    verify.add_argument("--host-facts", type=Path, required=True)
    verify.add_argument("--expected-host-facts-sha256", required=True)
    verify.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    return parser


def _record_launcher_failure(args: argparse.Namespace, code: str) -> None:
    """Best-effort stable receipt for failures before the child can start."""

    revision = getattr(args, "revision", None)
    if not isinstance(revision, str) or _SHA40.fullmatch(revision) is None:
        revision = None
    digests: dict[str, str | None] = {}
    for field in (
        "rail_sha256",
        "routine_sha256",
        "hardening_sha256",
        "gh_sha256",
        "git_sha256",
    ):
        value = getattr(args, field, None)
        digests[field] = (
            value
            if isinstance(value, str) and _SHA256.fullmatch(value) is not None
            else None
        )
    invocation = _invocation_id()
    receipt_id = _sha256_bytes(
        _canonical(
            {
                "invocation_id": invocation,
                "job_id": getattr(args, "job_id", None),
                "release_revision": revision,
                "failure_code": code,
            }
        )
    )
    receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "rail_schema": RAIL_SCHEMA,
        "receipt_id": receipt_id,
        "created_at": _now(),
        "job_id": (
            JOB_ID if getattr(args, "job_id", None) == JOB_ID else None
        ),
        "release_revision": revision,
        **digests,
        "outcome": "failed_launcher_contract",
        "failure_code": code,
        "routine_started": False,
        "provider_or_model_invoked": False,
        "discord_delivery_attempted": False,
        "secret_material_recorded": False,
    }
    try:
        STATE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        _write_receipt(STATE_ROOT, receipt)
    except (OSError, MechanicalJobRailError):
        return


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        try:
            return run_job(args)
        except MechanicalJobRailError as exc:
            _record_launcher_failure(args, str(exc))
            raise
    if args.command == "collect-host-facts":
        _atomic_private(args.output, _canonical(collect_host_facts()) + b"\n")
        return 0
    if args.command == "package":
        facts = json.loads(
            _read_regular_file(args.host_facts, maximum=128 * 1024).decode(
                "ascii", errors="strict"
            )
        )
        package = build_package(
            revision=args.revision,
            host_facts=facts,
            expected_host_facts_sha256=args.expected_host_facts_sha256,
        )
        write_package(package, output_root=args.output_root)
        return 0
    if args.command == "verify-package":
        facts = json.loads(
            _read_regular_file(args.host_facts, maximum=128 * 1024).decode(
                "ascii", errors="strict"
            )
        )
        package = build_package(
            revision=args.revision,
            host_facts=facts,
            expected_host_facts_sha256=args.expected_host_facts_sha256,
        )
        verify_package(package, output_root=args.output_root)
        return 0
    raise MechanicalJobRailError("mechanical_job_command_invalid")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MechanicalJobRailError as exc:
        # Stable codes only; credential values and child output never reach the
        # journal.  Execution-path failures write their detailed bounded
        # receipt before returning whenever the state directory is available.
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None


__all__ = [
    "CREDENTIAL_SOURCE",
    "GH_PATH",
    "GIT_PATH",
    "HOST_FACTS_SCHEMA",
    "JOB_ID",
    "MANIFEST_SCHEMA",
    "MechanicalJobPackage",
    "MechanicalJobRailError",
    "PACKAGE_ROOT",
    "SERVICE_UNIT",
    "TIMER_UNIT",
    "build_package",
    "collect_host_facts",
    "package_public_manifest",
    "validate_host_facts",
    "validate_package_manifest",
    "verify_package",
    "write_package",
]
