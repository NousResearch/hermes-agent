#!/usr/bin/env python3
"""Exact, credential-isolated rail for production cron evidence collection.

The rail is intentionally mechanical.  A code-owned job identifier selects one
reviewed command or one bounded filesystem reader.  The result is written as a
private, self-digesting packet; it is never interpreted, routed, delivered, or
turned into an incident by this process.  Companion Hermes cron records give
the packet to the pinned primary model, which remains the only semantic
authority.

The package builder renders one release-addressed service/timer pair per
reviewed source job.  It does not install or start them.  Every private runtime
dependency is path- and digest-bound in the package manifest and re-attested on
each run.  Provider, gateway, and Discord credentials are deliberately absent.
"""

from __future__ import annotations

import argparse
import grp
import hashlib
import json
import os
import pwd
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


RAIL_SCHEMA = "muncho-trusted-cron-collector-rail.v1"
PACKAGE_SCHEMA = "muncho-trusted-cron-collector-package.v1"
PACKET_SCHEMA = "muncho-trusted-cron-raw-packet.v1"
VOICE_STAGE_SCHEMA = "muncho-voice-context-stage.v1"
VOICE_DELIVERY_PROOF_SCHEMA = "muncho-voice-context-delivery-proof.v1"
VOICE_CURSOR_SCHEMA = "muncho-voice-context-cursor.v1"
EXECUTION_READINESS_SCHEMA = "muncho-trusted-cron-execution-readiness.v1"
EXECUTION_BOUNDARY_ISOLATED = "isolated_read_only"
EXECUTION_BOUNDARY_SCOPED = "credential_scoped_ops_edge"

RELEASES_ROOT = Path("/opt/adventico-ai-platform/hermes-agent-releases")
RAIL_RELATIVE = Path("ops/muncho/runtime/trusted_cron_collector_rail.py")
MANIFEST_PATH = Path("/etc/muncho/cron-collectors/manifest.json")
STATE_ROOT = Path("/var/lib/muncho-cron-collectors")
PACKET_ROOT = STATE_ROOT / "packets"
VOICE_ROOT = STATE_ROOT / "voice-context"
# Reuse the owner-bound, non-login projector identity created by the production
# host foundation.  A ninth ad-hoc Unix principal would otherwise escape the
# signed UID/GID authority.  systemd overrides its process group to the
# existing gateway group solely so 0640 packets are readable by the gateway.
SERVICE_USER = "muncho-projector"
READER_GROUP = "hermes-cloud-gateway"
SERVICE_GROUP = READER_GROUP

CANONICAL_ROOT = Path("/opt/adventico-ai-platform/canonical-brain")
HERMES_HOME = Path("/opt/adventico-ai-platform/hermes-home")
PYTHON = Path("/usr/bin/python3")
GH_WRAPPER = HERMES_HOME / "bin/gh-hermes"

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_PACKET_BYTES = 2 * 1024 * 1024
MAX_COMMAND_CAPTURE = 1024 * 1024
MAX_JSON_TREE_FILES = 1_000
MAX_JSON_TREE_FILE_BYTES = 128 * 1024
MAX_VOICE_EVENTS = 30
MAX_VOICE_EVENT_CHARS = 1_800
MAX_VOICE_TOTAL_CHARS = 18_000

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_RAIL_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SNOWFLAKE = re.compile(r"^[0-9]{17,20}$")
_SECRET_VALUE = re.compile(
    r"(?i)(?:bearer\s+[A-Za-z0-9._~+/=-]{8,}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,}|glpat-[A-Za-z0-9_-]{8,}|"
    r"xox[a-z]-[A-Za-z0-9-]{12,}|"
    r"-----BEGIN\s+(?:RSA|OPENSSH|PRIVATE)\s+KEY-----)"
)
_SECRET_KEY_PARTS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)


class TrustedCronCollectorError(RuntimeError):
    """Stable, non-secret collector/package failure."""


@dataclass(frozen=True)
class CollectorSpec:
    source_job_id: str
    rail_id: str
    schedule: Mapping[str, Any]
    mode: str
    model_review_required: bool
    command: tuple[str, ...] | None = None
    dependency_paths: tuple[str, ...] = ()
    json_roots: tuple[str, ...] = ()
    json_include_content: bool = False
    # Byte-exact legacy provenance only. It is never an authorization or an
    # executable destination for this credential-isolated collector.
    historical_source_delivery: str = "local"

    @property
    def execution_boundary(self) -> str:
        if self.mode in {"mechanical_command", "git_refs"}:
            return EXECUTION_BOUNDARY_SCOPED
        return EXECUTION_BOUNDARY_ISOLATED


def _spec(
    source_job_id: str,
    rail_id: str,
    schedule: Mapping[str, Any],
    mode: str,
    *,
    model_review: bool,
    command: Sequence[str] | None = None,
    dependencies: Sequence[Path] = (),
    json_roots: Sequence[Path] = (),
    json_include_content: bool = False,
    historical_source_delivery: str = "local",
) -> CollectorSpec:
    return CollectorSpec(
        source_job_id=source_job_id,
        rail_id=rail_id,
        schedule=dict(schedule),
        mode=mode,
        model_review_required=model_review,
        command=tuple(command) if command is not None else None,
        dependency_paths=tuple(str(item) for item in dependencies),
        json_roots=tuple(str(item) for item in json_roots),
        json_include_content=json_include_content,
        historical_source_delivery=historical_source_delivery,
    )


_HB = CANONICAL_ROOT / "bin/cloud_heartbeat_writer_no_enforcement_gate.py"
_PROJECTIONS = CANONICAL_ROOT / "bin/canonical_brain_projections_v1_1_refresh.py"
_ROUTEBACK_GATE = CANONICAL_ROOT / "bin/handoff_route_back_violation_alert_gate.py"
_PARITY = CANONICAL_ROOT / "scripts/cloud_local_git_drift_monitor.py"
_KNOWLEDGE_STATUS = CANONICAL_ROOT / "bin/knowledge_artifact_sync_status.py"
_KNOWLEDGE_CONFLICT = CANONICAL_ROOT / "bin/knowledge_artifact_conflict_report.py"
_ROUTEBACK_WATCHDOG = CANONICAL_ROOT / "bin/free_hermes_route_back_state_watchdog.py"
_PERSISTENCE_WATCHDOG = CANONICAL_ROOT / "bin/free_hermes_operational_persistence_watchdog.py"
_WATCHTOWER = HERMES_HOME / "scripts/devops_watchtower_phase1.py"
_CONTABO = HERMES_HOME / "scripts/contabo_observer.py"
_ALWYZON = HERMES_HOME / "scripts/alwyzon_phoenix_observer.py"
_SKYVISION_COUNT = HERMES_HOME / "scripts/skyvision_from_heart_weekly_count.py"
_SKYVISION_DB = HERMES_HOME / "bin/skyvision_db_readonly"
_LEARNING_ROOT = CANONICAL_ROOT / "state/private/learning_loop"
_REPORTS_ROOT = CANONICAL_ROOT / "state/reports"
_DOCS_ROOT = CANONICAL_ROOT / "docs"
_PRIVATE_SKILLS = HERMES_HOME / "skills/devops"
_VOICE_INPUT = HERMES_HOME / "state/voice_context"


# Exact IDs and schedules come from the digest-bound July 2026 production
# review.  This table is execution selection, not semantic routing: an unknown
# ID fails closed and no packet content influences which function runs.
COLLECTOR_SPECS: tuple[CollectorSpec, ...] = (
    _spec("8d09136f7da5", "canonical-brain-heartbeat", {"kind": "interval", "minutes": 2}, "mechanical_command", model_review=False, command=(str(PYTHON), str(_HB)), dependencies=(_HB,)),
    _spec("81ed8a3ea0d9", "canonical-brain-projection-refresh", {"kind": "interval", "minutes": 30}, "mechanical_command", model_review=False, command=(str(PYTHON), str(_PROJECTIONS), "--mode", "incremental", "--active-write"), dependencies=(_PROJECTIONS,)),
    _spec("58344347b373", "routeback-violation-monitor", {"kind": "interval", "minutes": 5}, "mechanical_command", model_review=True, command=(str(PYTHON), str(_ROUTEBACK_GATE)), dependencies=(_ROUTEBACK_GATE,)),
    _spec("a05143c24275", "muncho-parity-drift-monitor", {"kind": "cron", "expr": "0 5 * * *"}, "mechanical_command", model_review=True, command=(str(PYTHON), str(_PARITY), "--summary-only", "--report-always"), dependencies=(_PARITY,)),
    _spec("06ef64d72891", "fork-upstream-sync-drift-monitor", {"kind": "cron", "expr": "30 6 * * *"}, "git_refs", model_review=True, dependencies=(GH_WRAPPER,), historical_source_delivery="discord:1504852355588423801"),
    _spec("a7b15e3dea75", "canonical-brain-daily-ops-brief", {"kind": "cron", "expr": "0 3 * * 1-5"}, "json_tree", model_review=True, json_roots=(_REPORTS_ROOT,), json_include_content=True),
    _spec("a77d64526f9a", "knowledge-artifact-skyvision-voucher-monitor", {"kind": "interval", "minutes": 360}, "mechanical_command", model_review=True, command=(str(PYTHON), str(_KNOWLEDGE_STATUS), "--artifact", "skyvision-voucher-terms-invoicing"), dependencies=(_KNOWLEDGE_STATUS,)),
    _spec("0d446b5df20c", "knowledge-artifact-all-monitor", {"kind": "interval", "minutes": 360}, "mechanical_command", model_review=True, command=(str(PYTHON), str(_KNOWLEDGE_STATUS)), dependencies=(_KNOWLEDGE_STATUS, _KNOWLEDGE_CONFLICT)),
    _spec("54976db7a384", "canonical-routeback-state-watchdog", {"kind": "interval", "minutes": 15}, "mechanical_command", model_review=True, command=(str(PYTHON), str(_ROUTEBACK_WATCHDOG), "--stale-minutes", "30"), dependencies=(_ROUTEBACK_WATCHDOG,)),
    _spec("d84d45a86b80", "canonical-operational-persistence-watchdog", {"kind": "interval", "minutes": 10}, "mechanical_command", model_review=True, command=(str(PYTHON), str(_PERSISTENCE_WATCHDOG)), dependencies=(_PERSISTENCE_WATCHDOG,)),
    _spec("b35767232c37", "private-skill-manifest-monitor", {"kind": "interval", "minutes": 60}, "json_tree", model_review=True, json_roots=(_PRIVATE_SKILLS, _LEARNING_ROOT / "private-skill-expected-manifest.v1.json"), json_include_content=False),
    _spec("27ab4a64f8ad", "learning-loop-model-authored-review", {"kind": "cron", "expr": "15 7 * * *"}, "json_tree", model_review=True, json_roots=(_LEARNING_ROOT,), json_include_content=True),
    _spec("2b2035630202", "learning-loop-model-authored-review", {"kind": "cron", "expr": "25 7 * * *"}, "json_tree", model_review=True, json_roots=(_LEARNING_ROOT,), json_include_content=True),
    _spec("2a9f6be53fec", "knowledge-ingestion-model-authored-review", {"kind": "cron", "expr": "0 7 * * *"}, "filesystem_metadata", model_review=True, json_roots=(_REPORTS_ROOT, _DOCS_ROOT), json_include_content=False),
    _spec("90857403360d", "knowledge-ingestion-model-authored-review", {"kind": "cron", "expr": "0 8 * * 1"}, "filesystem_metadata", model_review=True, json_roots=(_REPORTS_ROOT, _DOCS_ROOT), json_include_content=False),
    _spec("e62f55ca93ca", "voice-context-model-authored-digest", {"kind": "interval", "minutes": 60}, "voice_stage", model_review=True, json_roots=(_VOICE_INPUT,), historical_source_delivery="discord:1504852355588423801:1524321461714681976"),
    _spec("7e4a90bdeff0", "devops-watchtower-public", {"kind": "cron", "expr": "*/5 * * * *"}, "mechanical_command", model_review=True, command=(str(PYTHON), str(_WATCHTOWER), "--mode", "fast", "--no-dispatch"), dependencies=(_WATCHTOWER,)),
    _spec("27f7f59fa0ca", "devops-watchtower-infrastructure", {"kind": "cron", "expr": "*/10 * * * *"}, "mechanical_command", model_review=True, command=(str(PYTHON), str(_WATCHTOWER), "--mode", "infra", "--no-dispatch"), dependencies=(_WATCHTOWER, _CONTABO, _ALWYZON)),
    _spec("6faf380f3512", "devops-watchtower-tls-dns", {"kind": "cron", "expr": "0 * * * *"}, "mechanical_command", model_review=True, command=(str(PYTHON), str(_WATCHTOWER), "--mode", "hourly", "--no-dispatch"), dependencies=(_WATCHTOWER,)),
    _spec("90ac99d45130", "devops-watchtower-digest", {"kind": "cron", "expr": "0 * * * *"}, "mechanical_command", model_review=True, command=(str(PYTHON), str(_WATCHTOWER), "--mode", "digest", "--no-dispatch"), dependencies=(_WATCHTOWER,)),
    _spec("dee523e6f47b", "skyvision-from-heart-weekly-count", {"kind": "cron", "expr": "59 * * * 0"}, "mechanical_command", model_review=True, command=(str(PYTHON), str(_SKYVISION_COUNT)), dependencies=(_SKYVISION_COUNT, _SKYVISION_DB), historical_source_delivery="origin"),
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
        raise TrustedCronCollectorError("trusted_cron_json_invalid") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _catalog() -> dict[str, CollectorSpec]:
    result = {item.source_job_id: item for item in COLLECTOR_SPECS}
    if len(result) != len(COLLECTOR_SPECS) or len(result) != 21:
        raise TrustedCronCollectorError("trusted_cron_catalog_invalid")
    rail_pairs = {(item.source_job_id, item.rail_id) for item in COLLECTOR_SPECS}
    for item in COLLECTOR_SPECS:
        if (
            _JOB_ID.fullmatch(item.source_job_id) is None
            or _RAIL_ID.fullmatch(item.rail_id) is None
            or item.mode
            not in {
                "mechanical_command",
                "git_refs",
                "json_tree",
                "filesystem_metadata",
                "voice_stage",
            }
            or item.mode == "mechanical_command" and not item.command
            or item.mode != "mechanical_command" and item.command is not None
            or item.mode in {"json_tree", "filesystem_metadata", "voice_stage"}
            and not item.json_roots
            or not isinstance(item.schedule, Mapping)
            or item.execution_boundary
            not in {EXECUTION_BOUNDARY_ISOLATED, EXECUTION_BOUNDARY_SCOPED}
        ):
            raise TrustedCronCollectorError("trusted_cron_catalog_invalid")
    if len(rail_pairs) != len(COLLECTOR_SPECS):
        raise TrustedCronCollectorError("trusted_cron_catalog_invalid")
    return result


def catalog_public_contract() -> list[dict[str, Any]]:
    """Return the content-free exact package contract for every collector."""

    _catalog()
    return [
        {
            "source_job_id": item.source_job_id,
            "rail_id": item.rail_id,
            "schedule": dict(item.schedule),
            "mode": item.mode,
            "model_review_required": item.model_review_required,
            "historical_source_delivery": item.historical_source_delivery,
            "historical_source_delivery_eligible": False,
            "execution_boundary": item.execution_boundary,
            "command_argv_sha256": (
                _sha256(_canonical(list(item.command))) if item.command else None
            ),
            "dependency_paths": list(item.dependency_paths),
            "json_roots": list(item.json_roots),
            "json_include_content": item.json_include_content,
            "semantic_judgment_allowed": False,
            "provider_or_model_allowed": False,
            "direct_discord_allowed": False,
        }
        for item in COLLECTOR_SPECS
    ]


def _validate_dependency_facts(value: Any) -> dict[str, str]:
    expected = {
        path for item in COLLECTOR_SPECS for path in item.dependency_paths
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TrustedCronCollectorError("trusted_cron_dependency_facts_invalid")
    result: dict[str, str] = {}
    for path, digest in value.items():
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise TrustedCronCollectorError(
                "trusted_cron_dependency_facts_invalid"
            )
        result[path] = digest
    return result


def _calendar(schedule: Mapping[str, Any]) -> tuple[str, str]:
    if schedule.get("kind") == "interval":
        minutes = schedule.get("minutes")
        if type(minutes) is not int or not 1 <= minutes <= 30 * 24 * 60:
            raise TrustedCronCollectorError("trusted_cron_schedule_invalid")
        return "OnActiveSec=5m", f"OnUnitActiveSec={minutes}m"
    expr = schedule.get("expr")
    exact = {
        "0 5 * * *": "OnCalendar=*-*-* 05:00:00 UTC",
        "30 6 * * *": "OnCalendar=*-*-* 06:30:00 UTC",
        "0 3 * * 1-5": "OnCalendar=Mon..Fri *-*-* 03:00:00 UTC",
        "15 7 * * *": "OnCalendar=*-*-* 07:15:00 UTC",
        "25 7 * * *": "OnCalendar=*-*-* 07:25:00 UTC",
        "0 7 * * *": "OnCalendar=*-*-* 07:00:00 UTC",
        "0 8 * * 1": "OnCalendar=Mon *-*-* 08:00:00 UTC",
        "*/5 * * * *": "OnCalendar=*-*-* *:0/5:00 UTC",
        "*/10 * * * *": "OnCalendar=*-*-* *:0/10:00 UTC",
        "0 * * * *": "OnCalendar=*-*-* *:00:00 UTC",
        "59 * * * 0": "OnCalendar=Sun *-*-* *:59:00 UTC",
    }
    calendar = exact.get(expr)
    if calendar is None:
        raise TrustedCronCollectorError("trusted_cron_schedule_invalid")
    return calendar, "AccuracySec=1m"


def _unit_names(item: CollectorSpec) -> tuple[str, str]:
    stem = f"muncho-cron-{item.source_job_id}"
    return f"{stem}.service", f"{stem}.timer"


def _render_service(
    item: CollectorSpec,
    *,
    revision: str,
    rail_sha256: str,
    manifest_sha256: str,
) -> bytes:
    release = RELEASES_ROOT / f"hermes-agent-{revision[:12]}"
    rail = release / RAIL_RELATIVE
    service, _timer = _unit_names(item)
    lines = [
        "# Release-addressed trusted cron collector; do not edit.",
        f"# ReleaseRevision={revision}",
        f"# RailSHA256={rail_sha256}",
        f"# PackageManifestSHA256={manifest_sha256}",
        "[Unit]",
        f"Description=Muncho raw collector {item.source_job_id}",
        "After=network-online.target",
        "Wants=network-online.target",
        f"AssertPathExists={rail}",
        f"AssertPathExists={MANIFEST_PATH}",
        "",
        "[Service]",
        "Type=oneshot",
        f"User={SERVICE_USER}",
        f"Group={SERVICE_GROUP}",
        "WorkingDirectory=/",
        "Environment=LANG=C.UTF-8",
        "Environment=LC_ALL=C.UTF-8",
        "Environment=PATH=/usr/bin:/bin",
        "Environment=TZ=UTC",
        (
            f"ExecStart={release / '.venv/bin/python'} -I -S -B {rail} run "
            f"--job-id {item.source_job_id} --revision {revision} "
            f"--rail-sha256 {rail_sha256} --manifest {MANIFEST_PATH} "
            f"--manifest-sha256 {manifest_sha256}"
        ),
        "TimeoutStartSec=900s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "UMask=0027",
        "NoNewPrivileges=yes",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
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
        "RestrictNamespaces=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "SupplementaryGroups=",
        "LockPersonality=yes",
        "MemoryDenyWriteExecute=yes",
        "RemoveIPC=yes",
        "SystemCallArchitectures=native",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        f"ReadOnlyPaths={release}",
        f"ReadOnlyPaths={CANONICAL_ROOT}",
        f"ReadOnlyPaths={HERMES_HOME / 'scripts'}",
        f"ReadOnlyPaths={HERMES_HOME / 'bin'}",
        f"ReadOnlyPaths={HERMES_HOME / 'skills'}",
        f"ReadWritePaths={STATE_ROOT}",
        f"ReadWritePaths={CANONICAL_ROOT / 'state'}",
        f"InaccessiblePaths=-{HERMES_HOME / '.env'}",
        f"InaccessiblePaths=-{HERMES_HOME / 'auth.json'}",
        "InaccessiblePaths=-/run/credentials/hermes-cloud-gateway.service",
        "InaccessiblePaths=-/etc/muncho/discord-connector-credentials",
        "InaccessiblePaths=-/etc/muncho/discord-edge-credentials",
        "StandardOutput=null",
        "StandardError=journal",
    ]
    text = "\n".join(lines) + "\n"
    if (
        text.count("ExecStart=") != 1
        or service not in _unit_names(item)
        or any(
            marker in text
            for marker in (
                "EnvironmentFile=",
                "PassEnvironment=",
                "OPENAI_API_KEY",
                "DISCORD_BOT_TOKEN",
                "LoadCredential=",
            )
        )
    ):
        raise TrustedCronCollectorError("trusted_cron_service_unit_invalid")
    return text.encode("utf-8")


def _render_timer(item: CollectorSpec) -> bytes:
    service, _timer = _unit_names(item)
    first, second = _calendar(item.schedule)
    text = "\n".join(
        [
            "# Package generation does not enable or start this timer.",
            "[Unit]",
            f"Description=Schedule Muncho raw collector {item.source_job_id}",
            "",
            "[Timer]",
            f"Unit={service}",
            first,
            second,
            "RandomizedDelaySec=30s",
            "Persistent=false",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ]
    )
    if "Persistent=true" in text or text.count(f"Unit={service}\n") != 1:
        raise TrustedCronCollectorError("trusted_cron_timer_unit_invalid")
    return text.encode("utf-8")


def build_package_manifest(
    *,
    revision: str,
    rail_sha256: str,
    dependency_facts: Mapping[str, str],
) -> dict[str, Any]:
    """Build a complete content-addressed package without installing it."""

    if _SHA40.fullmatch(revision or "") is None or _SHA256.fullmatch(
        rail_sha256 or ""
    ) is None:
        raise TrustedCronCollectorError("trusted_cron_package_identity_invalid")
    dependencies = _validate_dependency_facts(dependency_facts)
    # The unit bytes bind the manifest digest, while the manifest binds their
    # final digests.  Break the cycle with a content digest over the immutable
    # package inputs, then include that digest in every rendered unit.
    package_binding = _sha256(
        _canonical(
            {
                "schema": PACKAGE_SCHEMA,
                "release_revision": revision,
                "rail_sha256": rail_sha256,
                "collector_contract": catalog_public_contract(),
                "dependencies": dependencies,
            }
        )
    )
    units: dict[str, dict[str, str]] = {}
    for item in COLLECTOR_SPECS:
        service_name, timer_name = _unit_names(item)
        service = _render_service(
            item,
            revision=revision,
            rail_sha256=rail_sha256,
            manifest_sha256=package_binding,
        )
        timer = _render_timer(item)
        units[item.source_job_id] = {
            "service": service_name,
            "service_sha256": _sha256(service),
            "timer": timer_name,
            "timer_sha256": _sha256(timer),
        }
    unsigned = {
        "schema": PACKAGE_SCHEMA,
        "rail_schema": RAIL_SCHEMA,
        "release_revision": revision,
        "release_root": str(RELEASES_ROOT / f"hermes-agent-{revision[:12]}"),
        "rail_path": str(
            RELEASES_ROOT / f"hermes-agent-{revision[:12]}" / RAIL_RELATIVE
        ),
        "rail_sha256": rail_sha256,
        "package_binding_sha256": package_binding,
        "collector_contract": catalog_public_contract(),
        "dependency_sha256": dependencies,
        "units": units,
        "service_user": SERVICE_USER,
        "service_group": SERVICE_GROUP,
        "reader_group": READER_GROUP,
        "packet_root": str(PACKET_ROOT),
        "provider_or_model_dependency": False,
        "gateway_credential_dependency": False,
        "discord_credential_dependency": False,
        "semantic_judgment_allowed": False,
        "direct_delivery_allowed": False,
        "timer_enabled_by_package": False,
        "timer_started_by_package": False,
    }
    return {
        **unsigned,
        "manifest_sha256": _sha256(_canonical(unsigned)),
    }


def validate_package_manifest(
    value: Mapping[str, Any],
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "rail_schema",
        "release_revision",
        "release_root",
        "rail_path",
        "rail_sha256",
        "package_binding_sha256",
        "collector_contract",
        "dependency_sha256",
        "units",
        "service_user",
        "service_group",
        "reader_group",
        "packet_root",
        "provider_or_model_dependency",
        "gateway_credential_dependency",
        "discord_credential_dependency",
        "semantic_judgment_allowed",
        "direct_delivery_allowed",
        "timer_enabled_by_package",
        "timer_started_by_package",
        "manifest_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("schema") != PACKAGE_SCHEMA
        or value.get("rail_schema") != RAIL_SCHEMA
        or _SHA40.fullmatch(str(value.get("release_revision") or "")) is None
        or revision is not None and value.get("release_revision") != revision
        or _SHA256.fullmatch(str(value.get("rail_sha256") or "")) is None
        or _SHA256.fullmatch(str(value.get("package_binding_sha256") or ""))
        is None
        or _SHA256.fullmatch(str(value.get("manifest_sha256") or "")) is None
        or _sha256(
            _canonical(
                {key: item for key, item in value.items() if key != "manifest_sha256"}
            )
        )
        != value.get("manifest_sha256")
        or value.get("collector_contract") != catalog_public_contract()
        or value.get("service_user") != SERVICE_USER
        or value.get("service_group") != SERVICE_GROUP
        or value.get("reader_group") != READER_GROUP
        or value.get("packet_root") != str(PACKET_ROOT)
        or any(
            value.get(field) is not False
            for field in (
                "provider_or_model_dependency",
                "gateway_credential_dependency",
                "discord_credential_dependency",
                "semantic_judgment_allowed",
                "direct_delivery_allowed",
                "timer_enabled_by_package",
                "timer_started_by_package",
            )
        )
    ):
        raise TrustedCronCollectorError("trusted_cron_package_manifest_invalid")
    dependencies = _validate_dependency_facts(value.get("dependency_sha256"))
    expected_binding = _sha256(
        _canonical(
            {
                "schema": PACKAGE_SCHEMA,
                "release_revision": value["release_revision"],
                "rail_sha256": value["rail_sha256"],
                "collector_contract": catalog_public_contract(),
                "dependencies": dependencies,
            }
        )
    )
    if value.get("package_binding_sha256") != expected_binding:
        raise TrustedCronCollectorError(
            "trusted_cron_package_manifest_invalid"
        )
    units = value.get("units")
    if not isinstance(units, Mapping) or set(units) != set(_catalog()):
        raise TrustedCronCollectorError("trusted_cron_package_manifest_invalid")
    for job_id, item in units.items():
        spec = _catalog()[job_id]
        service, timer = _unit_names(spec)
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {"service", "service_sha256", "timer", "timer_sha256"}
            or item.get("service") != service
            or item.get("timer") != timer
            or _SHA256.fullmatch(str(item.get("service_sha256") or "")) is None
            or _SHA256.fullmatch(str(item.get("timer_sha256") or "")) is None
            or item.get("service_sha256")
            != _sha256(
                _render_service(
                    spec,
                    revision=value["release_revision"],
                    rail_sha256=value["rail_sha256"],
                    manifest_sha256=expected_binding,
                )
            )
            or item.get("timer_sha256") != _sha256(_render_timer(spec))
        ):
            raise TrustedCronCollectorError(
                "trusted_cron_package_manifest_invalid"
            )
    expected_release = RELEASES_ROOT / f"hermes-agent-{value['release_revision'][:12]}"
    if (
        value.get("release_root") != str(expected_release)
        or value.get("rail_path") != str(expected_release / RAIL_RELATIVE)
        or dependencies != value.get("dependency_sha256")
    ):
        raise TrustedCronCollectorError("trusted_cron_package_manifest_invalid")
    return dict(value)


def _permission_allows(
    metadata: os.stat_result,
    *,
    uid: int,
    gid: int,
    read: bool,
    execute: bool,
) -> bool:
    mode = stat.S_IMODE(metadata.st_mode)
    shift = 6 if metadata.st_uid == uid else 3 if metadata.st_gid == gid else 0
    bits = (mode >> shift) & 0o7
    return (not read or bool(bits & 0o4)) and (
        not execute or bool(bits & 0o1)
    )


def _path_access_issue(
    path: Path,
    *,
    uid: int,
    gid: int,
    read: bool,
    execute: bool,
    stat_reader: Any,
) -> str | None:
    if not path.is_absolute() or ".." in path.parts:
        return "path_invalid"
    current = Path("/")
    try:
        root_metadata = stat_reader(current)
    except OSError:
        return "ancestor_unavailable"
    if not stat.S_ISDIR(root_metadata.st_mode) or not _permission_allows(
        root_metadata,
        uid=uid,
        gid=gid,
        read=False,
        execute=True,
    ):
        return "ancestor_not_traversable"
    for part in path.parts[1:-1]:
        current /= part
        try:
            metadata = stat_reader(current)
        except OSError:
            return "ancestor_unavailable"
        if not stat.S_ISDIR(metadata.st_mode) or not _permission_allows(
            metadata,
            uid=uid,
            gid=gid,
            read=False,
            execute=True,
        ):
            return "ancestor_not_traversable"
    try:
        metadata = stat_reader(path)
    except OSError:
        return "target_unavailable"
    target_execute = execute or stat.S_ISDIR(metadata.st_mode)
    if not (
        stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)
    ):
        return "target_type_invalid"
    if not _permission_allows(
        metadata,
        uid=uid,
        gid=gid,
        read=read,
        execute=target_execute,
    ):
        return "target_permission_denied"
    return None


def collect_execution_readiness(
    manifest: Mapping[str, Any],
    *,
    operational_edge_receipt: Mapping[str, Any] | None = None,
    expected_boot_id_sha256: str | None = None,
    now_unix: int | None = None,
    account_lookup: Any = pwd.getpwnam,
    group_lookup: Any = grp.getgrnam,
    stat_reader: Any = os.stat,
) -> dict[str, Any]:
    """Read-only proof that packaged collectors can run as their exact user.

    Command-backed jobs require a separately collected, portable operational
    edge receipt proving one meaningful real-user round trip per exact job.
    This checker never reads a credential or changes permissions.
    """

    trusted = validate_package_manifest(manifest)
    try:
        account = account_lookup(SERVICE_USER)
        group = group_lookup(SERVICE_GROUP)
        uid = int(account.pw_uid)
        gid = int(group.gr_gid)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise TrustedCronCollectorError(
            "trusted_cron_service_identity_unavailable"
        ) from exc
    release_root = Path(trusted["release_root"])
    requirements: dict[Path, tuple[bool, bool]] = {
        release_root: (True, True),
        Path(trusted["rail_path"]): (True, False),
        release_root / ".venv/bin/python": (True, True),
        PYTHON: (False, True),
    }

    def require(path: Path, *, read: bool, execute: bool) -> None:
        previous = requirements.get(path, (False, False))
        requirements[path] = (
            previous[0] or read,
            previous[1] or execute,
        )

    for path in trusted["dependency_sha256"]:
        dependency = Path(path)
        require(
            dependency,
            read=True,
            execute=dependency in {GH_WRAPPER, _SKYVISION_DB},
        )
    for row in trusted["collector_contract"]:
        for root in row["json_roots"]:
            require(Path(root), read=True, execute=False)
    blocked_paths: list[dict[str, str]] = []
    for path, (read, execute) in sorted(
        requirements.items(), key=lambda item: str(item[0])
    ):
        issue = _path_access_issue(
            path,
            uid=uid,
            gid=gid,
            read=read,
            execute=execute,
            stat_reader=stat_reader,
        )
        if issue is not None:
            blocked_paths.append({
                "path": str(path),
                "requirement": (
                    "read_execute" if read and execute
                    else "execute" if execute
                    else "read"
                ),
                "reason": issue,
            })
    scoped_jobs = sorted(
        row["source_job_id"]
        for row in trusted["collector_contract"]
        if row["execution_boundary"] == EXECUTION_BOUNDARY_SCOPED
    )
    try:
        from gateway.operational_edge_catalog import required_cron_operations
        from gateway.operational_edge_readiness import (
            validate_operational_edge_readiness,
        )

        required_operations = dict(required_cron_operations())
        if scoped_jobs != sorted(required_operations):
            raise ValueError("scoped collector catalog drifted")
        operational = (
            validate_operational_edge_readiness(
                operational_edge_receipt,
                revision=trusted["release_revision"],
                required_jobs=required_operations,
                expected_boot_id_sha256=expected_boot_id_sha256,
                now_unix=now_unix,
            )
            if operational_edge_receipt is not None
            else None
        )
    except (ImportError, TypeError, ValueError) as exc:
        raise TrustedCronCollectorError(
            "trusted_cron_operational_edge_readiness_invalid"
        ) from exc
    edge_packaged = operational is not None
    unsigned = {
        "schema": EXECUTION_READINESS_SCHEMA,
        "release_revision": trusted["release_revision"],
        "collector_manifest_sha256": trusted["manifest_sha256"],
        "service_user": SERVICE_USER,
        "service_uid": uid,
        "service_group": SERVICE_GROUP,
        "service_gid": gid,
        "checked_path_count": len(requirements),
        "blocked_paths": blocked_paths,
        "blocked_path_count": len(blocked_paths),
        "direct_dependencies_ready": not blocked_paths,
        "scoped_execution_edge_required_job_ids": scoped_jobs,
        "scoped_execution_edge_required_count": len(scoped_jobs),
        "scoped_execution_edge_packaged": edge_packaged,
        "scoped_execution_edge_receipt_sha256": (
            operational["receipt_sha256"] if operational is not None else None
        ),
        "scoped_execution_edge_meaningful_packet_count": (
            operational["job_count"] if operational is not None else 0
        ),
        "activation_ready": not blocked_paths and edge_packaged,
        "permissions_widened": False,
        "credential_content_read": False,
        "secret_material_recorded": False,
    }
    return {
        **unsigned,
        "readiness_sha256": _sha256(_canonical(unsigned)),
    }


def validate_execution_readiness(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    operational_edge_receipt: Mapping[str, Any] | None = None,
    expected_boot_id_sha256: str | None = None,
    now_unix: int | None = None,
) -> dict[str, Any]:
    trusted = validate_package_manifest(manifest)
    try:
        from gateway.operational_edge_catalog import required_cron_operations
        from gateway.operational_edge_readiness import (
            validate_operational_edge_readiness,
        )

        required_operations = dict(required_cron_operations())
        operational = (
            validate_operational_edge_readiness(
                operational_edge_receipt,
                revision=trusted["release_revision"],
                required_jobs=required_operations,
                expected_boot_id_sha256=expected_boot_id_sha256,
                now_unix=now_unix,
            )
            if operational_edge_receipt is not None
            else None
        )
    except (ImportError, TypeError, ValueError) as exc:
        raise TrustedCronCollectorError(
            "trusted_cron_execution_readiness_invalid"
        ) from exc
    expected = {
        "schema", "release_revision", "collector_manifest_sha256",
        "service_user", "service_uid", "service_group", "service_gid",
        "checked_path_count", "blocked_paths", "blocked_path_count",
        "direct_dependencies_ready",
        "scoped_execution_edge_required_job_ids",
        "scoped_execution_edge_required_count",
        "scoped_execution_edge_packaged",
        "scoped_execution_edge_receipt_sha256",
        "scoped_execution_edge_meaningful_packet_count", "activation_ready",
        "permissions_widened", "credential_content_read",
        "secret_material_recorded", "readiness_sha256",
    }
    unsigned = {
        name: item for name, item in value.items() if name != "readiness_sha256"
    } if isinstance(value, Mapping) else {}
    scoped = value.get("scoped_execution_edge_required_job_ids") if isinstance(
        value, Mapping
    ) else None
    paths = value.get("blocked_paths") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or value.get("schema") != EXECUTION_READINESS_SCHEMA
        or value.get("release_revision") != trusted["release_revision"]
        or value.get("collector_manifest_sha256") != trusted["manifest_sha256"]
        or value.get("service_user") != SERVICE_USER
        or value.get("service_group") != SERVICE_GROUP
        or type(value.get("service_uid")) is not int
        or type(value.get("service_gid")) is not int
        or type(value.get("checked_path_count")) is not int
        or not isinstance(paths, list)
        or type(value.get("blocked_path_count")) is not int
        or value.get("blocked_path_count") != len(paths)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "requirement", "reason"}
            or not isinstance(row["path"], str)
            or not Path(row["path"]).is_absolute()
            or row["requirement"] not in {"read", "execute", "read_execute"}
            or row["reason"] not in {
                "path_invalid", "ancestor_unavailable",
                "ancestor_not_traversable", "target_unavailable",
                "target_type_invalid", "target_permission_denied",
            }
            for row in paths
        )
        or not isinstance(scoped, list)
        or scoped != sorted(required_operations)
        or type(value.get("scoped_execution_edge_required_count")) is not int
        or value.get("scoped_execution_edge_required_count") != len(scoped)
        or value.get("direct_dependencies_ready") != (not paths)
        or value.get("scoped_execution_edge_packaged") != (
            operational is not None
        )
        or value.get("scoped_execution_edge_receipt_sha256") != (
            operational["receipt_sha256"] if operational is not None else None
        )
        or type(value.get(
            "scoped_execution_edge_meaningful_packet_count"
        )) is not int
        or value.get("scoped_execution_edge_meaningful_packet_count") != (
            operational["job_count"] if operational is not None else 0
        )
        or value.get("activation_ready") != (
            not paths and operational is not None
        )
        or value.get("permissions_widened") is not False
        or value.get("credential_content_read") is not False
        or value.get("secret_material_recorded") is not False
        or _SHA256.fullmatch(str(value.get("readiness_sha256") or "")) is None
        or value.get("readiness_sha256") != _sha256(_canonical(unsigned))
    ):
        raise TrustedCronCollectorError(
            "trusted_cron_execution_readiness_invalid"
        )
    return dict(value)


def render_package_unit_files(
    manifest: Mapping[str, Any],
) -> dict[str, bytes]:
    """Render and re-attest every packaged unit without installing it."""

    trusted = validate_package_manifest(manifest)
    result: dict[str, bytes] = {}
    for item in COLLECTOR_SPECS:
        service_name, timer_name = _unit_names(item)
        service = _render_service(
            item,
            revision=trusted["release_revision"],
            rail_sha256=trusted["rail_sha256"],
            manifest_sha256=trusted["package_binding_sha256"],
        )
        timer = _render_timer(item)
        expected = trusted["units"][item.source_job_id]
        if (
            _sha256(service) != expected["service_sha256"]
            or _sha256(timer) != expected["timer_sha256"]
            or expected["service"] != service_name
            or expected["timer"] != timer_name
        ):
            raise TrustedCronCollectorError(
                "trusted_cron_package_unit_digest_mismatch"
            )
        result[f"systemd/{service_name}"] = service
        result[f"systemd/{timer_name}"] = timer
    if len(result) != len(COLLECTOR_SPECS) * 2:
        raise TrustedCronCollectorError("trusted_cron_package_manifest_invalid")
    return result


def _stable_read(path: Path, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TrustedCronCollectorError("trusted_cron_input_unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 <= before.st_size <= maximum
        ):
            raise TrustedCronCollectorError("trusted_cron_input_metadata_invalid")
        value = b""
        while len(value) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(value)))
            if not chunk:
                break
            value += chunk
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
    if len(value) > maximum or identity(before) != identity(after):
        raise TrustedCronCollectorError("trusted_cron_input_changed")
    return value


def _redact(value: Any) -> Any:
    """Safety-only redaction; never used for routing or task decisions."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            folded = name.casefold().replace("-", "_")
            if any(part in folded for part in _SECRET_KEY_PARTS):
                result[name] = "[REDACTED_SECRET_FIELD]"
            else:
                result[name] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED_SECRET_VALUE]", value)
    return value


def _json_tree(item: CollectorSpec) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for root_text in item.json_roots:
        root = Path(root_text)
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if len(records) >= MAX_JSON_TREE_FILES or not path.is_file():
                continue
            try:
                metadata = path.stat()
                relative = str(path.relative_to(root)) if path != root else path.name
                raw = _stable_read(path, maximum=MAX_JSON_TREE_FILE_BYTES)
            except (OSError, ValueError, TrustedCronCollectorError):
                continue
            record: dict[str, Any] = {
                "root": str(root),
                "relative_path": relative,
                "size": len(raw),
                "mtime_ns": metadata.st_mtime_ns,
                "sha256": _sha256(raw),
            }
            if item.json_include_content and path.suffix.casefold() == ".json":
                try:
                    parsed = json.loads(raw.decode("utf-8", errors="strict"))
                except (UnicodeError, json.JSONDecodeError):
                    pass
                else:
                    record["json"] = _redact(parsed)
            records.append(record)
    return {
        "roots": list(item.json_roots),
        "record_count": len(records),
        "record_limit": MAX_JSON_TREE_FILES,
        "records": records,
    }


def _run_command(item: CollectorSpec) -> dict[str, Any]:
    assert item.command is not None
    try:
        completed = subprocess.run(
            list(item.command),
            cwd=str(CANONICAL_ROOT) if str(CANONICAL_ROOT) in " ".join(item.command) else "/",
            env={
                "HOME": str(STATE_ROOT),
                "HERMES_HOME": str(HERMES_HOME),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "TZ": "UTC",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=840,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = bytes(exc.stdout or b"")[:MAX_COMMAND_CAPTURE]
        stderr = bytes(exc.stderr or b"")[:MAX_COMMAND_CAPTURE]
        return {
            "timed_out": True,
            "return_code": None,
            "stdout_bytes": len(stdout),
            "stdout_sha256": _sha256(stdout),
            "stdout_utf8": _SECRET_VALUE.sub(
                "[REDACTED_SECRET_VALUE]", stdout.decode("utf-8", errors="replace")
            ),
            "stderr_bytes": len(stderr),
            "stderr_sha256": _sha256(stderr),
            "stderr_utf8": _SECRET_VALUE.sub(
                "[REDACTED_SECRET_VALUE]", stderr.decode("utf-8", errors="replace")
            ),
            "capture_truncated": bool(
                len(exc.stdout or b"") > MAX_COMMAND_CAPTURE
                or len(exc.stderr or b"") > MAX_COMMAND_CAPTURE
            ),
        }
    stdout = completed.stdout[:MAX_COMMAND_CAPTURE]
    stderr = completed.stderr[:MAX_COMMAND_CAPTURE]
    return {
        "timed_out": False,
        "return_code": completed.returncode,
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": _sha256(completed.stdout),
        "stdout_utf8": _SECRET_VALUE.sub(
            "[REDACTED_SECRET_VALUE]", stdout.decode("utf-8", errors="replace")
        ),
        "stderr_bytes": len(completed.stderr),
        "stderr_sha256": _sha256(completed.stderr),
        "stderr_utf8": _SECRET_VALUE.sub(
            "[REDACTED_SECRET_VALUE]", stderr.decode("utf-8", errors="replace")
        ),
        "capture_truncated": bool(
            len(completed.stdout) > MAX_COMMAND_CAPTURE
            or len(completed.stderr) > MAX_COMMAND_CAPTURE
        ),
    }


def _git_refs() -> dict[str, Any]:
    endpoints = {
        "fork_ref": "repos/lomliev/hermes-agent/git/ref/heads/main",
        "upstream_ref": "repos/NousResearch/hermes-agent/git/ref/heads/main",
        "compare": "repos/NousResearch/hermes-agent/compare/main...lomliev:main",
    }
    results: dict[str, Any] = {}
    for label, endpoint in endpoints.items():
        completed = subprocess.run(
            [str(GH_WRAPPER), "api", endpoint],
            cwd="/",
            env={
                "HOME": str(STATE_ROOT),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "TZ": "UTC",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
        if len(completed.stdout) > MAX_COMMAND_CAPTURE:
            raise TrustedCronCollectorError("trusted_cron_capture_oversized")
        try:
            payload = json.loads(completed.stdout.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError):
            payload = None
        results[label] = {
            "endpoint": endpoint,
            "return_code": completed.returncode,
            "response": _redact(payload),
            "stdout_sha256": _sha256(completed.stdout),
            "stderr_sha256": _sha256(completed.stderr),
        }
    return results


def _voice_cursor(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": VOICE_CURSOR_SCHEMA, "files": {}}
    try:
        value = json.loads(_stable_read(path, maximum=128 * 1024))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TrustedCronCollectorError("voice_cursor_invalid") from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != VOICE_CURSOR_SCHEMA
        or not isinstance(value.get("files"), Mapping)
        or any(
            not isinstance(name, str)
            or not isinstance(offset, int)
            or offset < 0
            for name, offset in value["files"].items()
        )
    ):
        raise TrustedCronCollectorError("voice_cursor_invalid")
    return dict(value)


def stage_voice_packet(
    *,
    voice_root: Path,
    cursor_path: Path,
    created_at: str | None = None,
) -> dict[str, Any] | None:
    """Stage bounded transcript evidence without advancing any cursor."""

    cursor = _voice_cursor(cursor_path)
    events: list[dict[str, Any]] = []
    proposals = dict(cursor["files"])
    total_chars = 0
    for path in sorted(voice_root.glob("discord_g*_vc*_*.jsonl")):
        if len(events) >= MAX_VOICE_EVENTS:
            break
        start = int(cursor["files"].get(path.name, 0))
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if start < 0 or start > size:
            start = 0
        with path.open("rb") as stream:
            stream.seek(start)
            while len(events) < MAX_VOICE_EVENTS:
                line_start = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                line_end = stream.tell()
                try:
                    record = json.loads(raw.decode("utf-8", errors="strict"))
                except (UnicodeError, json.JSONDecodeError):
                    continue
                if record.get("type") != "discord_voice_context.transcript":
                    continue
                transcript = str(record.get("transcript") or "").replace("\x00", "").strip()
                if not transcript:
                    continue
                transcript = _SECRET_VALUE.sub("[REDACTED_SECRET_VALUE]", transcript)
                transcript = transcript.replace("@everyone", "@\u200beveryone").replace(
                    "@here", "@\u200bhere"
                )
                transcript = transcript[:MAX_VOICE_EVENT_CHARS]
                if events and total_chars + len(transcript) > MAX_VOICE_TOTAL_CHARS:
                    break
                events.append(
                    {
                        "source_file": path.name,
                        "start_offset": line_start,
                        "end_offset": line_end,
                        "timestamp": str(record.get("timestamp") or ""),
                        "guild_id": str(record.get("guild_id") or ""),
                        "voice_channel_id": str(
                            record.get("voice_channel_id") or ""
                        ),
                        "voice_channel_name": str(
                            record.get("voice_channel_name") or ""
                        )[:256],
                        "text_channel_id": str(record.get("text_channel_id") or ""),
                        "speaker_user_id": str(record.get("user_id") or ""),
                        "transcript": transcript,
                        "raw_audio_retained": False,
                    }
                )
                total_chars += len(transcript)
                proposals[path.name] = line_end
            # Never jump over unselected valid events.  The proposal is the end
            # of the last staged event, not EOF.
    if not events:
        return None
    event_digest = _sha256(_canonical(events))
    unsigned = {
        "schema": VOICE_STAGE_SCHEMA,
        "created_at": created_at or _now(),
        "packet_id": f"voice:{event_digest[:32]}",
        "source_cursor_sha256": _sha256(_canonical(cursor)),
        "events": events,
        "event_count": len(events),
        "proposed_offsets": proposals,
        "cursor_committed": False,
        "requires_connector_send_receipt": True,
        "requires_connector_readback_receipt": True,
        "requires_canonical_terminal_event": "route_back.sent",
        "blocked_terminal_event": "route_back.blocked",
        "semantic_judgment_performed": False,
        "secret_material_recorded": False,
    }
    return {**unsigned, "stage_sha256": _sha256(_canonical(unsigned))}


def validate_voice_stage(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema",
        "created_at",
        "packet_id",
        "source_cursor_sha256",
        "events",
        "event_count",
        "proposed_offsets",
        "cursor_committed",
        "requires_connector_send_receipt",
        "requires_connector_readback_receipt",
        "requires_canonical_terminal_event",
        "blocked_terminal_event",
        "semantic_judgment_performed",
        "secret_material_recorded",
        "stage_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or value.get("schema") != VOICE_STAGE_SCHEMA
        or not isinstance(value.get("events"), list)
        or type(value.get("event_count")) is not int
        or value.get("event_count") != len(value["events"])
        or not 1 <= value["event_count"] <= MAX_VOICE_EVENTS
        or not isinstance(value.get("proposed_offsets"), Mapping)
        or any(
            not isinstance(name, str)
            or type(offset) is not int
            or offset < 0
            for name, offset in value["proposed_offsets"].items()
        )
        or value.get("cursor_committed") is not False
        or value.get("requires_connector_send_receipt") is not True
        or value.get("requires_connector_readback_receipt") is not True
        or value.get("requires_canonical_terminal_event") != "route_back.sent"
        or value.get("blocked_terminal_event") != "route_back.blocked"
        or value.get("semantic_judgment_performed") is not False
        or value.get("secret_material_recorded") is not False
        or _SHA256.fullmatch(str(value.get("source_cursor_sha256") or "")) is None
        or _SHA256.fullmatch(str(value.get("stage_sha256") or "")) is None
        or _sha256(
            _canonical({key: item for key, item in value.items() if key != "stage_sha256"})
        )
        != value.get("stage_sha256")
    ):
        raise TrustedCronCollectorError("voice_stage_invalid")
    return dict(value)


def build_voice_delivery_proof(
    *,
    stage: Mapping[str, Any],
    connector_send_receipt_sha256: str | None,
    connector_readback_receipt_sha256: str | None,
    canonical_event_type: str,
    canonical_event_sha256: str,
) -> dict[str, Any]:
    trusted = validate_voice_stage(stage)
    if canonical_event_type not in {"route_back.sent", "route_back.blocked"}:
        raise TrustedCronCollectorError("voice_delivery_proof_invalid")
    sent = canonical_event_type == "route_back.sent"
    if (
        _SHA256.fullmatch(canonical_event_sha256 or "") is None
        or sent
        and (
            _SHA256.fullmatch(connector_send_receipt_sha256 or "") is None
            or _SHA256.fullmatch(connector_readback_receipt_sha256 or "") is None
        )
        or not sent
        and any(
            receipt is not None and _SHA256.fullmatch(receipt) is None
            for receipt in (
                connector_send_receipt_sha256,
                connector_readback_receipt_sha256,
            )
        )
    ):
        raise TrustedCronCollectorError("voice_delivery_proof_invalid")
    unsigned = {
        "schema": VOICE_DELIVERY_PROOF_SCHEMA,
        "packet_id": trusted["packet_id"],
        "stage_sha256": trusted["stage_sha256"],
        "outcome": "sent" if sent else "blocked",
        "connector_send_receipt_sha256": connector_send_receipt_sha256,
        "connector_readback_receipt_sha256": connector_readback_receipt_sha256,
        "canonical_event_type": canonical_event_type,
        "canonical_event_sha256": canonical_event_sha256,
        "cursor_commit_allowed": sent,
    }
    return {**unsigned, "proof_sha256": _sha256(_canonical(unsigned))}


def commit_voice_cursor(
    *,
    stage: Mapping[str, Any],
    proof: Mapping[str, Any],
    cursor_path: Path,
) -> dict[str, Any]:
    """Commit offsets only after sent + readback + Canonical sent proof."""

    trusted_stage = validate_voice_stage(stage)
    expected_fields = {
        "schema",
        "packet_id",
        "stage_sha256",
        "outcome",
        "connector_send_receipt_sha256",
        "connector_readback_receipt_sha256",
        "canonical_event_type",
        "canonical_event_sha256",
        "cursor_commit_allowed",
        "proof_sha256",
    }
    if not isinstance(proof, Mapping):
        raise TrustedCronCollectorError("voice_delivery_proof_invalid")
    sent_outcome = (
        proof.get("outcome") == "sent"
        and proof.get("cursor_commit_allowed") is True
        and proof.get("canonical_event_type") == "route_back.sent"
        and _SHA256.fullmatch(
            str(proof.get("connector_send_receipt_sha256") or "")
        )
        is not None
        and _SHA256.fullmatch(
            str(proof.get("connector_readback_receipt_sha256") or "")
        )
        is not None
    )
    blocked_outcome = (
        proof.get("outcome") == "blocked"
        and proof.get("cursor_commit_allowed") is False
        and proof.get("canonical_event_type") == "route_back.blocked"
        and all(
            receipt is None
            or _SHA256.fullmatch(str(receipt)) is not None
            for receipt in (
                proof.get("connector_send_receipt_sha256"),
                proof.get("connector_readback_receipt_sha256"),
            )
        )
    )
    if (
        set(proof) != expected_fields
        or proof.get("schema") != VOICE_DELIVERY_PROOF_SCHEMA
        or proof.get("packet_id") != trusted_stage["packet_id"]
        or proof.get("stage_sha256") != trusted_stage["stage_sha256"]
        or _SHA256.fullmatch(str(proof.get("canonical_event_sha256") or ""))
        is None
        or not (sent_outcome or blocked_outcome)
        or _SHA256.fullmatch(str(proof.get("proof_sha256") or "")) is None
        or _sha256(
            _canonical({key: item for key, item in proof.items() if key != "proof_sha256"})
        )
        != proof.get("proof_sha256")
    ):
        raise TrustedCronCollectorError("voice_delivery_proof_invalid")
    current = _voice_cursor(cursor_path)
    if _sha256(_canonical(current)) != trusted_stage["source_cursor_sha256"]:
        raise TrustedCronCollectorError("voice_cursor_drifted")
    if blocked_outcome:
        return {
            "packet_id": trusted_stage["packet_id"],
            "outcome": "blocked_cursor_unchanged",
            "cursor_committed": False,
            "canonical_terminal_event": "route_back.blocked",
        }
    target = {
        "schema": VOICE_CURSOR_SCHEMA,
        "files": dict(trusted_stage["proposed_offsets"]),
        "last_packet_id": trusted_stage["packet_id"],
        "delivery_proof_sha256": proof["proof_sha256"],
    }
    _atomic_private(cursor_path, _canonical(target) + b"\n")
    return {
        "packet_id": trusted_stage["packet_id"],
        "outcome": "sent_cursor_committed",
        "cursor_committed": True,
        "canonical_terminal_event": "route_back.sent",
        "cursor_sha256": _sha256(_canonical(target)),
    }


def _atomic_private(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, 0o640)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_manifest(path: Path, expected_binding_sha256: str) -> dict[str, Any]:
    raw = _stable_read(path, maximum=MAX_MANIFEST_BYTES)
    try:
        value = json.loads(raw.decode("ascii", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TrustedCronCollectorError("trusted_cron_manifest_invalid") from exc
    manifest = validate_package_manifest(value)
    if (
        manifest["package_binding_sha256"] != expected_binding_sha256
        or _SHA256.fullmatch(expected_binding_sha256 or "") is None
    ):
        raise TrustedCronCollectorError("trusted_cron_manifest_digest_mismatch")
    return manifest


def _attest_runtime(
    *,
    revision: str,
    rail_sha256: str,
    manifest: Mapping[str, Any],
) -> None:
    if (
        manifest.get("release_revision") != revision
        or manifest.get("rail_sha256") != rail_sha256
        or _SHA40.fullmatch(revision or "") is None
        or _SHA256.fullmatch(rail_sha256 or "") is None
    ):
        raise TrustedCronCollectorError("trusted_cron_runtime_identity_invalid")
    rail = RELEASES_ROOT / f"hermes-agent-{revision[:12]}" / RAIL_RELATIVE
    if Path(__file__).resolve(strict=True) != rail:
        raise TrustedCronCollectorError("trusted_cron_not_release_addressed")
    if _sha256(_stable_read(rail, maximum=4 * 1024 * 1024)) != rail_sha256:
        raise TrustedCronCollectorError("trusted_cron_rail_digest_drifted")
    for path, digest in manifest["dependency_sha256"].items():
        if _sha256(_stable_read(Path(path), maximum=16 * 1024 * 1024)) != digest:
            raise TrustedCronCollectorError("trusted_cron_dependency_drifted")


def collect_packet(item: CollectorSpec) -> dict[str, Any] | None:
    if item.mode == "mechanical_command":
        evidence = _run_command(item)
    elif item.mode == "git_refs":
        evidence = _git_refs()
    elif item.mode in {"json_tree", "filesystem_metadata"}:
        evidence = _json_tree(item)
    elif item.mode == "voice_stage":
        stage = stage_voice_packet(
            voice_root=Path(item.json_roots[0]),
            cursor_path=VOICE_ROOT / "cursor.json",
        )
        if stage is None:
            return None
        evidence = stage
    else:
        raise TrustedCronCollectorError("trusted_cron_mode_invalid")
    unsigned = {
        "schema": PACKET_SCHEMA,
        "created_at": _now(),
        "source_job_id": item.source_job_id,
        "rail_id": item.rail_id,
        "mode": item.mode,
        "evidence": evidence,
        "model_review_required": item.model_review_required,
        "semantic_judgment_performed": False,
        "delivery_attempted": False,
        "provider_or_model_invoked": False,
        "secret_material_recorded": False,
    }
    packet = {**unsigned, "packet_sha256": _sha256(_canonical(unsigned))}
    if len(_canonical(packet)) > MAX_PACKET_BYTES:
        raise TrustedCronCollectorError("trusted_cron_packet_oversized")
    return packet


def run(args: argparse.Namespace) -> int:
    catalog = _catalog()
    item = catalog.get(args.job_id)
    if item is None:
        raise TrustedCronCollectorError("trusted_cron_job_not_allowlisted")
    manifest = _load_manifest(args.manifest, args.manifest_sha256)
    _attest_runtime(
        revision=args.revision,
        rail_sha256=args.rail_sha256,
        manifest=manifest,
    )
    packet = collect_packet(item)
    if packet is None:
        return 0
    root = PACKET_ROOT / item.source_job_id
    _atomic_private(root / f"{packet['packet_sha256']}.json", _canonical(packet) + b"\n")
    _atomic_private(root / "latest.json", _canonical(packet) + b"\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Muncho trusted cron collector rail")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--job-id", required=True)
    run_parser.add_argument("--revision", required=True)
    run_parser.add_argument("--rail-sha256", required=True)
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--manifest-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        return run(args)
    raise TrustedCronCollectorError("trusted_cron_command_invalid")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrustedCronCollectorError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None


__all__ = [
    "COLLECTOR_SPECS",
    "EXECUTION_BOUNDARY_ISOLATED",
    "EXECUTION_BOUNDARY_SCOPED",
    "EXECUTION_READINESS_SCHEMA",
    "PACKAGE_SCHEMA",
    "PACKET_SCHEMA",
    "RAIL_SCHEMA",
    "TrustedCronCollectorError",
    "VOICE_CURSOR_SCHEMA",
    "VOICE_DELIVERY_PROOF_SCHEMA",
    "VOICE_STAGE_SCHEMA",
    "build_package_manifest",
    "build_voice_delivery_proof",
    "catalog_public_contract",
    "collect_packet",
    "collect_execution_readiness",
    "commit_voice_cursor",
    "render_package_unit_files",
    "stage_voice_packet",
    "validate_package_manifest",
    "validate_execution_readiness",
    "validate_voice_stage",
]
