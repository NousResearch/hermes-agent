#!/usr/bin/env python3
"""Fixed, secret-free read-only observation of the real production runtime.

This module runs only as root on ``ai-platform-runtime-01`` through the
owner-Mac pinned gcloud/IAP transport.  It never reads ``.env`` or auth stores
and never records or digests job prompts/bodies/output, logs, PIDs, counters,
or last-run timestamps.  The cron store is loaded only to project fixed static
fields from every enabled and disabled record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import grp
import pwd
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "muncho-production-capability-production-observation.v1"
PROJECT = "adventico-ai-platform"
ZONE = "europe-west3-a"
VM_NAME = "ai-platform-runtime-01"
INSTANCE_ID = "1094477181810932795"
ACTIVE_LINK = Path("/opt/adventico-ai-platform/hermes-agent")
HERMES_HOME = Path("/opt/adventico-ai-platform/hermes-home")
CANONICAL_BRAIN = Path("/opt/adventico-ai-platform/canonical-brain")
CONFIG_PATH = HERMES_HOME / "config.yaml"
GATEWAY_UNIT = "hermes-cloud-gateway.service"
GATEWAY_UNIT_PATH = Path("/etc/systemd/system") / GATEWAY_UNIT
CONNECTOR_CONFIG_PATH = Path("/etc/muncho/discord-public-connector.json")
JOBS_PATH = HERMES_HOME / "cron/jobs.json"
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
MACHINE_ID = Path("/etc/machine-id")
HOSTNAME = Path("/etc/hostname")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SERVICE_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "FragmentPath",
    "DropInPaths",
    "User",
    "Group",
    "WorkingDirectory",
)
_CODE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "run_agent.py",
    "gateway/run.py",
    "hermes_state.py",
    "cron/jobs.py",
)
_OPERATIONAL_EDGE_DOMAINS = (
    "adventico_email",
    "bitrix",
    "canonical",
    "github",
    "infrastructure",
    "skyvision_db",
    "skyvision_email",
    "skyvision_gitlab",
    "skyvision_panel",
)
_FIXED_USERS = (
    "ai-platform-brain",
    "muncho-canonical-writer",
    "muncho-projector",
    "muncho-discord-egress",
    "muncho-discord-connector",
    "muncho-mac-ops-edge",
    "muncho-capability-browser",
    "muncho-worker",
    *(f"muncho-edge-{domain}" for domain in _OPERATIONAL_EDGE_DOMAINS),
)
_FIXED_GROUPS = (
    *_FIXED_USERS,
    "muncho-writer-client",
    "muncho-worker-clients",
    *(f"muncho-edge-{domain}-c" for domain in _OPERATIONAL_EDGE_DOMAINS),
)
_FIXED_PERMISSION_PATHS = (
    HERMES_HOME,
    CANONICAL_BRAIN,
    Path("/etc/adventico-ai-platform"),
    Path("/etc/muncho"),
    Path("/etc/muncho-canonical-writer"),
    Path("/run/muncho-canonical-writer"),
    Path("/run/muncho-discord-connector"),
    Path("/run/muncho-discord-edge"),
    Path("/run/muncho-mac-ops"),
    Path("/run/muncho-capability-browser"),
    Path("/run/muncho-isolated-worker"),
    Path("/run/muncho-operational-edge"),
)
_OWNER_USER_ID = "1279454038731264061"
_OWNER_GUILD_ID = "1282725267068157972"
_CONTROL_TOWER_CHANNEL_ID = "1504852355588423801"
_NASI_CHANNEL_ID = "1505499746939174993"
_UNUSED_PUBLIC_CHANNEL_ID = "1526870121677848636"
_REVIEWED_CRON_HISTORY_TARGETS = {
    "06ef64d72891": [_CONTROL_TOWER_CHANNEL_ID],
    "e62f55ca93ca": ["1524321461714681976"],
}
_APPROVED_OPERATIONAL_CHANNEL_IDS = tuple(
    sorted(
        {
            _CONTROL_TOWER_CHANNEL_ID,
            "1510888721614901358",
            "1504852408227069993",
            "1504852444407140402",
            "1504852485083496561",
            "1504852553031221391",
            "1504852628373373028",
            _NASI_CHANNEL_ID,
            "1507239177350283274",
            "1507239385010016308",
        }
    )
)
_SNOWFLAKE = re.compile(r"^[1-9][0-9]{16,19}$")
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_SAFE_ROUTE_TEXT = re.compile(r"^[A-Za-z0-9_.:/-]{0,512}$")


class ProductionObservationError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_mapping(
    value: Any,
    fields: set[str] | frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ProductionObservationError(
            f"production_observation_{label}_invalid"
        )
    return value


def _string_list(
    value: Any,
    *,
    label: str,
    snowflakes: bool = False,
) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
        or value != sorted(set(value))
        or (
            snowflakes
            and any(_SNOWFLAKE.fullmatch(item) is None for item in value)
        )
    ):
        raise ProductionObservationError(
            f"production_observation_{label}_invalid"
        )
    return value


def _safe_optional_route_text(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or _SAFE_ROUTE_TEXT.fullmatch(value) is None
        or any(character.isspace() for character in value)
    ):
        raise ProductionObservationError(
            f"production_observation_{label}_invalid"
        )
    return value


def _json_clone(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical(value).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionObservationError(
            f"production_observation_{label}_invalid"
        ) from exc


def _duplicates(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProductionObservationError(
                "production_observation_json_duplicate_key"
            )
        result[key] = value
    return result


def _identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _stable_file(path: Path, *, maximum: int) -> tuple[bytes, os.stat_result]:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 <= before.st_size <= maximum
        ):
            raise ProductionObservationError("production_observation_file_invalid")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        reachable = path.lstat()
    except ProductionObservationError:
        raise
    except OSError as exc:
        raise ProductionObservationError(
            "production_observation_file_unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        len(raw) != before.st_size
        or len(raw) > maximum
        or _identity(before) != _identity(opened)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(reachable)
    ):
        raise ProductionObservationError("production_observation_file_changed")
    return raw, before


def _file_projection(path: Path, *, maximum: int) -> Mapping[str, Any]:
    raw, item = _stable_file(path, maximum=maximum)
    return {
        "path": str(path),
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "size": item.st_size,
        "sha256": _sha(raw),
    }


def _directory_projection(path: Path) -> Mapping[str, Any]:
    try:
        item = path.lstat()
    except OSError as exc:
        raise ProductionObservationError(
            "production_observation_directory_unavailable"
        ) from exc
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
        raise ProductionObservationError("production_observation_directory_invalid")
    return {
        "path": str(path),
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
    }


def _permission_path_projection(path: Path) -> Mapping[str, Any]:
    try:
        item = path.lstat()
    except FileNotFoundError:
        return {
            "path": str(path),
            "state": "absent",
            "uid": None,
            "gid": None,
            "mode": None,
        }
    except OSError as exc:
        raise ProductionObservationError(
            "production_observation_permission_path_unavailable"
        ) from exc
    if stat.S_ISDIR(item.st_mode):
        state = "directory"
    elif stat.S_ISREG(item.st_mode):
        state = "regular_file"
    elif stat.S_ISLNK(item.st_mode):
        state = "symlink"
    elif stat.S_ISSOCK(item.st_mode):
        state = "socket"
    else:
        state = "other"
    return {
        "path": str(path),
        "state": state,
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
    }


def _user_projection(name: str) -> Mapping[str, Any]:
    try:
        item = pwd.getpwnam(name)
    except KeyError:
        return {
            "name": name,
            "presence": "absent",
            "uid": None,
            "gid": None,
            "home": None,
            "shell": None,
            "supplementary_group_names": [],
        }
    supplementary = sorted(
        group.gr_name
        for group in grp.getgrall()
        if name in group.gr_mem and group.gr_gid != item.pw_gid
    )
    return {
        "name": name,
        "presence": "present",
        "uid": item.pw_uid,
        "gid": item.pw_gid,
        "home": item.pw_dir,
        "shell": item.pw_shell,
        "supplementary_group_names": supplementary,
    }


def _group_projection(name: str) -> Mapping[str, Any]:
    try:
        item = grp.getgrnam(name)
    except KeyError:
        return {
            "name": name,
            "presence": "absent",
            "gid": None,
            "members": [],
        }
    return {
        "name": name,
        "presence": "present",
        "gid": item.gr_gid,
        "members": sorted(set(item.gr_mem)),
    }


def _identities_permissions_projection() -> Mapping[str, Any]:
    users = [_user_projection(name) for name in sorted(_FIXED_USERS)]
    groups = [_group_projection(name) for name in sorted(_FIXED_GROUPS)]
    paths = [
        _permission_path_projection(path)
        for path in sorted(_FIXED_PERMISSION_PATHS, key=str)
    ]
    unsigned = {"users": users, "groups": groups, "paths": paths}
    return {**unsigned, "projection_sha256": _sha(_canonical(unsigned))}


def _active_release() -> tuple[Path, Mapping[str, Any]]:
    try:
        link = ACTIVE_LINK.lstat()
        target_text = os.readlink(ACTIVE_LINK)
        resolved = ACTIVE_LINK.resolve(strict=True)
    except OSError as exc:
        raise ProductionObservationError(
            "production_observation_active_release_unavailable"
        ) from exc
    release_base = Path("/opt/adventico-ai-platform/hermes-agent-releases")
    try:
        resolved.relative_to(release_base)
    except ValueError as exc:
        raise ProductionObservationError(
            "production_observation_active_release_invalid"
        ) from exc
    if (
        not stat.S_ISLNK(link.st_mode)
        or not target_text.startswith("/opt/adventico-ai-platform/")
        or resolved.is_symlink()
        or not resolved.is_dir()
    ):
        raise ProductionObservationError(
            "production_observation_active_release_invalid"
        )
    marker_raw, _marker = _stable_file(
        resolved / ".codex-source-commit", maximum=128
    )
    revision = marker_raw.decode("ascii", errors="strict").strip()
    if _REVISION.fullmatch(revision) is None:
        raise ProductionObservationError(
            "production_observation_release_revision_invalid"
        )
    return resolved, {
        "link_path": str(ACTIVE_LINK),
        "link_target": target_text,
        "resolved_target": str(resolved),
        "link_uid": link.st_uid,
        "link_gid": link.st_gid,
        "link_mode": f"{stat.S_IMODE(link.st_mode):04o}",
        "release_revision": revision,
    }


def _service_projection() -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            (
                "/usr/bin/systemctl",
                "show",
                *(f"--property={name}" for name in _SERVICE_PROPERTIES),
                "--",
                GATEWAY_UNIT,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            cwd="/",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProductionObservationError(
            "production_observation_service_unavailable"
        ) from exc
    if completed.returncode != 0 or completed.stderr:
        raise ProductionObservationError(
            "production_observation_service_unavailable"
        )
    values: dict[str, str] = {}
    for line in completed.stdout.decode("utf-8", errors="strict").splitlines():
        if "=" not in line:
            raise ProductionObservationError(
                "production_observation_service_invalid"
            )
        key, value = line.split("=", 1)
        if key in values:
            raise ProductionObservationError(
                "production_observation_service_invalid"
            )
        values[key] = value
    if set(values) != set(_SERVICE_PROPERTIES):
        raise ProductionObservationError("production_observation_service_invalid")
    if (
        values["LoadState"] != "loaded"
        or values["ActiveState"] != "active"
        or values["SubState"] != "running"
        or values["FragmentPath"] != str(GATEWAY_UNIT_PATH)
        or values["User"] != "ai-platform-brain"
        or values["Group"] != "ai-platform-brain"
        or values["WorkingDirectory"] != str(ACTIVE_LINK)
    ):
        raise ProductionObservationError("production_observation_service_invalid")
    drop_ins = values["DropInPaths"].split() if values["DropInPaths"] else []
    drop_in_root = Path("/etc/systemd/system/hermes-cloud-gateway.service.d")
    for raw_path in drop_ins:
        candidate = Path(raw_path)
        try:
            candidate.relative_to(drop_in_root)
        except ValueError as exc:
            raise ProductionObservationError(
                "production_observation_service_invalid"
            ) from exc
        if not candidate.is_absolute() or any(character.isspace() for character in raw_path):
            raise ProductionObservationError(
                "production_observation_service_invalid"
            )
    if drop_ins != sorted(set(drop_ins)):
        raise ProductionObservationError("production_observation_service_invalid")
    return {
        "unit": GATEWAY_UNIT,
        "load_state": values["LoadState"],
        "active_state": values["ActiveState"],
        "sub_state": values["SubState"],
        "unit_file_state": values["UnitFileState"],
        "fragment_path": values["FragmentPath"],
        "drop_in_paths": drop_ins,
        "user": values["User"],
        "group": values["Group"],
        "working_directory": values["WorkingDirectory"],
    }


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _config_snowflake_list(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        values = list(value)
    else:
        raise ProductionObservationError(
            f"production_observation_{label}_invalid"
        )
    if any(
        not isinstance(item, str) or _SNOWFLAKE.fullmatch(item) is None
        for item in values
    ):
        raise ProductionObservationError(
            f"production_observation_{label}_invalid"
        )
    return sorted(set(values))


def _authorization_projection(discord: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = ("allowed_users", "allowed_roles", "allowed_channels")
    present = [field for field in fields if field in discord]
    if present and len(present) != len(fields):
        raise ProductionObservationError(
            "production_observation_config_authorization_partial"
        )
    return {
        "source": "config_yaml" if present else "unavailable_not_collected",
        "allowed_user_ids": (
            _config_snowflake_list(
                discord["allowed_users"], label="config_allowed_users"
            )
            if present
            else []
        ),
        "allowed_role_ids": (
            _config_snowflake_list(
                discord["allowed_roles"], label="config_allowed_roles"
            )
            if present
            else []
        ),
        "allowed_channel_ids": (
            _config_snowflake_list(
                discord["allowed_channels"], label="config_allowed_channels"
            )
            if present
            else []
        ),
        "legacy_env_backed_projection_collected": False,
    }


def _connector_policy_projection() -> Mapping[str, Any]:
    raw, _item = _stable_file(CONNECTOR_CONFIG_PATH, maximum=64 * 1024)
    try:
        root = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ProductionObservationError(
                    "production_observation_connector_config_invalid"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProductionObservationError(
            "production_observation_connector_config_invalid"
        ) from exc
    root = _strict_mapping(
        root, {"service", "discord", "journal"}, "connector_config"
    )
    discord = _strict_mapping(
        root["discord"],
        {
            "token_file",
            "credentials_directory",
            "allowed_guild_ids",
            "allowed_channel_ids",
            "allowed_user_ids",
            "allowed_role_ids",
            "free_response_channel_ids",
            "public_only",
            "author_policy",
            "allow_bot_authors",
            "require_mention",
            "auto_thread",
            "thread_require_mention",
            "reviewed_cron_history_targets",
            "ready_timeout_seconds",
            "request_timeout_seconds",
        },
        "connector_discord",
    )
    reviewed_cron_history_targets = discord[
        "reviewed_cron_history_targets"
    ]
    if (
        not isinstance(reviewed_cron_history_targets, Mapping)
        or reviewed_cron_history_targets != _REVIEWED_CRON_HISTORY_TARGETS
        or any(
            not isinstance(job_id, str)
            or _JOB_ID.fullmatch(job_id) is None
            or not isinstance(channel_ids, list)
            or not channel_ids
            or channel_ids != sorted(set(channel_ids))
            or any(
                not isinstance(channel_id, str)
                or _SNOWFLAKE.fullmatch(channel_id) is None
                for channel_id in channel_ids
            )
            for job_id, channel_ids in reviewed_cron_history_targets.items()
        )
    ):
        raise ProductionObservationError(
            "production_observation_connector_policy_invalid"
        )
    projected = {
        "allowed_guild_ids": sorted(set(discord["allowed_guild_ids"])),
        "allowed_channel_ids": sorted(set(discord["allowed_channel_ids"])),
        "allowed_user_ids": sorted(set(discord["allowed_user_ids"])),
        "allowed_role_ids": sorted(set(discord["allowed_role_ids"])),
        "free_response_channel_ids": sorted(
            set(discord["free_response_channel_ids"])
        ),
        "public_only": discord["public_only"],
        "author_policy": discord["author_policy"],
        "allow_bot_authors": discord["allow_bot_authors"],
        "require_mention": discord["require_mention"],
        "auto_thread": discord["auto_thread"],
        "thread_require_mention": discord["thread_require_mention"],
        "reviewed_cron_history_targets_sha256": _sha(
            _canonical(reviewed_cron_history_targets)
        ),
        "dm_messages": False,
        "group_dm_messages": False,
    }
    for field in (
        "allowed_guild_ids",
        "allowed_channel_ids",
        "allowed_user_ids",
        "allowed_role_ids",
        "free_response_channel_ids",
    ):
        original = discord[field]
        if (
            not isinstance(original, list)
            or any(not isinstance(item, str) for item in original)
            or len(original) != len(set(original))
            or any(_SNOWFLAKE.fullmatch(item) is None for item in original)
        ):
            raise ProductionObservationError(
                "production_observation_connector_policy_invalid"
            )
    for field in (
        "allow_bot_authors",
        "require_mention",
        "auto_thread",
        "thread_require_mention",
        "public_only",
    ):
        if type(projected[field]) is not bool:
            raise ProductionObservationError(
                "production_observation_connector_policy_invalid"
            )
    if projected["author_policy"] not in {
        "exact_ids_or_roles",
        "guild_acl",
    }:
        raise ProductionObservationError(
            "production_observation_connector_policy_invalid"
        )
    return projected


def _config_projection() -> Mapping[str, Any]:
    raw, item = _stable_file(CONFIG_PATH, maximum=2 * 1024 * 1024)
    try:
        import yaml

        value = yaml.safe_load(raw)
    except Exception as exc:
        raise ProductionObservationError(
            "production_observation_config_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise ProductionObservationError("production_observation_config_invalid")
    model = _mapping_or_empty(value.get("model"))
    agent = _mapping_or_empty(value.get("agent"))
    goals = _mapping_or_empty(value.get("goals"))
    cron = _mapping_or_empty(value.get("cron"))
    approvals = _mapping_or_empty(value.get("approvals"))
    escalation = _mapping_or_empty(approvals.get("gateway_owner_escalation"))
    platforms = _mapping_or_empty(value.get("platforms"))
    discord = _mapping_or_empty(value.get("discord"))
    voice = _mapping_or_empty(discord.get("voice_context"))
    enabled_platforms = sorted(
        key
        for key, platform in platforms.items()
        if isinstance(key, str)
        and isinstance(platform, Mapping)
        and platform.get("enabled") is True
    )
    selected = {
        "model_route": {
            "default": model.get("default", model.get("model")),
            "provider": model.get("provider"),
        },
        "agent_execution": {
            "reasoning_effort": agent.get("reasoning_effort"),
            "max_turns": agent.get("max_turns"),
            "adaptive_reasoning": _json_clone(
                agent.get("adaptive_reasoning"), label="adaptive_reasoning"
            ),
        },
        "goals": {"max_turns": goals.get("max_turns")},
        "cron": {
            "enabled": cron.get("enabled"),
            "provider": cron.get("provider"),
        },
        "approvals": {
            "mode": approvals.get("mode"),
            "cron_mode": approvals.get("cron_mode"),
            "plan_owner_user_ids": _json_clone(
                approvals.get("plan_owner_user_ids"),
                label="plan_owner_user_ids",
            ),
            "gateway_authorized_user_ids": _json_clone(
                approvals.get("gateway_authorized_user_ids"),
                label="gateway_authorized_user_ids",
            ),
            "gateway_owner_escalation": {
                "enabled": escalation.get("enabled"),
                "owner_user_id": escalation.get("owner_user_id"),
                "owner_guild_id": escalation.get("owner_guild_id"),
                "owner_channel_id": escalation.get("owner_channel_id"),
                "owner_target_type": escalation.get("owner_target_type"),
            },
        },
        "platforms": {
            "enabled_keys": enabled_platforms,
            "discord_relay_only": (
                "relay" in enabled_platforms
                and "discord" not in enabled_platforms
            ),
        },
        "discord_behavior": {
            "require_mention": discord.get("require_mention"),
            "auto_thread": discord.get("auto_thread"),
            "thread_require_mention": discord.get("thread_require_mention"),
            "free_response_channel_ids": _config_snowflake_list(
                discord.get("free_response_channels"),
                label="free_response_channels",
            ),
            "voice_context": {
                "enabled": voice.get("enabled"),
                "text_channel_id": voice.get("text_channel_id"),
                "allowed_channel_ids": _config_snowflake_list(
                    voice.get("allowed_channel_ids")
                    or voice.get("allowed_voice_channel_ids"),
                    label="voice_allowed_channels",
                ),
                "allowed_category_ids": _config_snowflake_list(
                    voice.get("allowed_category_ids")
                    or voice.get("category_id"),
                    label="voice_allowed_categories",
                ),
                "auto_join_channel_ids": _config_snowflake_list(
                    voice.get("auto_join_channel_ids")
                    or voice.get("auto_join_voice_channel_ids"),
                    label="voice_auto_join_channels",
                ),
            },
        },
        "config_authorization": _authorization_projection(discord),
        "connector_policy": _connector_policy_projection(),
    }
    if not isinstance(selected["model_route"]["default"], str):
        raise ProductionObservationError("production_observation_config_invalid")
    return {
        "path": str(CONFIG_PATH),
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "selected_projection": selected,
        "selected_projection_sha256": _sha(_canonical(selected)),
        "full_file_content_or_digest_recorded": False,
    }


def _schedule_projection(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionObservationError("production_observation_schedule_invalid")
    kind = value.get("kind")
    if kind == "interval":
        minutes = value.get("minutes")
        if type(minutes) is not int or minutes < 1:
            raise ProductionObservationError(
                "production_observation_schedule_invalid"
            )
        return {"kind": "interval", "minutes": minutes}
    if kind == "cron":
        expression = value.get("expr")
        if (
            not isinstance(expression, str)
            or len(expression) > 256
            or re.fullmatch(r"[0-9*?,/\- ]+", expression) is None
        ):
            raise ProductionObservationError(
                "production_observation_schedule_invalid"
            )
        return {"kind": "cron", "expr": expression}
    if kind == "once":
        run_at = value.get("run_at")
        if (
            not isinstance(run_at, str)
            or len(run_at) > 64
            or re.fullmatch(r"[0-9T:+.\-Z]+", run_at) is None
        ):
            raise ProductionObservationError(
                "production_observation_schedule_invalid"
            )
        return {"kind": "once", "run_at": run_at}
    raise ProductionObservationError("production_observation_schedule_invalid")


def _origin_public_projection(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ProductionObservationError("production_observation_origin_invalid")
    projected = {
        "platform": value.get("platform"),
        "chat_id": value.get("chat_id"),
        "thread_id": value.get("thread_id"),
        "user_id": value.get("user_id"),
    }
    if (
        projected["platform"] != "discord"
        or _SNOWFLAKE.fullmatch(str(projected["chat_id"] or "")) is None
        or any(
            item is not None and _SNOWFLAKE.fullmatch(str(item)) is None
            for item in (projected["thread_id"], projected["user_id"])
        )
    ):
        raise ProductionObservationError("production_observation_origin_invalid")
    return projected


def _jobs_projection(release: Path) -> Mapping[str, Any]:
    os.environ["HERMES_HOME"] = str(HERMES_HOME)
    sys.path.insert(0, str(release))
    try:
        from cron.jobs import list_jobs

        jobs = list_jobs(include_disabled=True)
    except Exception as exc:
        raise ProductionObservationError("production_observation_jobs_invalid") from exc
    projected: list[Mapping[str, Any]] = []
    for raw in jobs:
        if not isinstance(raw, Mapping):
            raise ProductionObservationError("production_observation_jobs_invalid")
        job_id = raw.get("id")
        if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
            raise ProductionObservationError("production_observation_jobs_invalid")
        toolsets = raw.get("enabled_toolsets")
        skills = raw.get("skills")
        normalized_toolsets = (
            []
            if toolsets is None
            else sorted(set(toolsets))
            if isinstance(toolsets, list)
            and all(isinstance(item, str) and item for item in toolsets)
            else None
        )
        normalized_skills = (
            []
            if skills is None
            else sorted(set(skills))
            if isinstance(skills, list)
            and all(isinstance(item, str) and item for item in skills)
            else None
        )
        if normalized_toolsets is None or normalized_skills is None:
            raise ProductionObservationError("production_observation_jobs_invalid")
        repeat = raw.get("repeat")
        repeat_times = repeat.get("times") if isinstance(repeat, Mapping) else None
        if repeat_times is not None and (
            type(repeat_times) is not int or repeat_times < 1
        ):
            raise ProductionObservationError("production_observation_jobs_invalid")
        script = raw.get("script")
        script_path = _safe_optional_route_text(script, label="script_path")
        workdir = _safe_optional_route_text(raw.get("workdir"), label="workdir")
        if workdir is not None and not Path(workdir).is_absolute():
            raise ProductionObservationError("production_observation_workdir_invalid")
        provider = _safe_optional_route_text(raw.get("provider"), label="provider")
        model = _safe_optional_route_text(raw.get("model"), label="model")
        enabled = raw.get("enabled")
        no_agent = raw.get("no_agent")
        deliver = raw.get("deliver")
        base_url = raw.get("base_url")
        if (
            type(enabled) is not bool
            or type(no_agent) is not bool
            or not isinstance(deliver, str)
            or _SAFE_ROUTE_TEXT.fullmatch(deliver) is None
            or (base_url is not None and not isinstance(base_url, str))
        ):
            raise ProductionObservationError("production_observation_jobs_invalid")
        item = {
            "id": job_id,
            "enabled": enabled,
            "schedule": _schedule_projection(raw.get("schedule")),
            "deliver": deliver,
            "origin_public_projection": _origin_public_projection(
                raw.get("origin")
            ),
            "provider": provider,
            "model": model,
            "base_url_present": bool(base_url),
            "no_agent": no_agent,
            "script_path": script_path,
            "workdir": workdir,
            "enabled_toolsets": normalized_toolsets,
            "skills": normalized_skills,
            "repeat": {"times": repeat_times},
        }
        projected.append(item)
    projected.sort(key=lambda item: str(item["id"]))
    if len({item["id"] for item in projected}) != len(projected):
        raise ProductionObservationError("production_observation_jobs_invalid")
    jobs_file = JOBS_PATH.lstat()
    if (
        stat.S_ISLNK(jobs_file.st_mode)
        or not stat.S_ISREG(jobs_file.st_mode)
        or jobs_file.st_nlink != 1
    ):
        raise ProductionObservationError("production_observation_jobs_invalid")
    return {
        "path": str(JOBS_PATH),
        "uid": jobs_file.st_uid,
        "gid": jobs_file.st_gid,
        "mode": f"{stat.S_IMODE(jobs_file.st_mode):04o}",
        "static_job_count": len(projected),
        "static_jobs": projected,
        "static_projection_sha256": _sha(_canonical(projected)),
        "prompt_body_output_or_dynamic_state_recorded": False,
    }


def _migration_asset_projection(release: Path) -> Mapping[str, Any]:
    del release
    root = CANONICAL_BRAIN / "schemas"
    if not os.path.lexists(root):
        return {
            "root": str(root),
            "state": "absent",
            "file_count": 0,
            "inventory": [],
            "inventory_sha256": _sha(_canonical([])),
            "applied_database_state_recorded": False,
        }
    if root.is_symlink() or not root.is_dir():
        raise ProductionObservationError(
            "production_observation_migrations_invalid"
        )
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if len(files) > 512:
        raise ProductionObservationError("production_observation_migrations_invalid")
    inventory = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        projection = _file_projection(path, maximum=2 * 1024 * 1024)
        inventory.append({"relative_path": relative, **projection})
    return {
        "root": str(root),
        "state": "present",
        "file_count": len(inventory),
        "inventory": inventory,
        "inventory_sha256": _sha(_canonical(inventory)),
        "applied_database_state_recorded": False,
    }


def collect_production_observation(
    *,
    phase: str,
    canary_revision: str,
    capability_plan_sha256: str,
    full_canary_plan_sha256: str,
    fixture_sha256: str,
    run_id: str,
) -> Mapping[str, Any]:
    if (
        phase not in {"before", "after"}
        or _REVISION.fullmatch(canary_revision or "") is None
        or any(
            _DIGEST.fullmatch(value or "") is None
            for value in (
                capability_plan_sha256,
                full_canary_plan_sha256,
                fixture_sha256,
            )
        )
        or _RUN_ID.fullmatch(run_id or "") is None
        or not sys.platform.startswith("linux")
        or os.geteuid() != 0  # windows-footgun: ok — Linux production/canary boundary
    ):
        raise ProductionObservationError("production_observation_input_invalid")
    release, active = _active_release()
    service = _service_projection()
    identities_permissions = _identities_permissions_projection()
    release_inventory = [
        {
            "relative_path": relative,
            **_file_projection(
                release / relative, maximum=64 * 1024 * 1024
            ),
        }
        for relative in _CODE_FILES
    ]
    unit_inventory = [
        _file_projection(Path(path), maximum=512 * 1024)
        for path in [service["fragment_path"], *service["drop_in_paths"]]
    ]
    code_unsigned = {
        "release_revision": active["release_revision"],
        "release_inventory": release_inventory,
        "unit_inventory": unit_inventory,
    }
    host = {
        "boot_id_sha256": _sha(_stable_file(BOOT_ID, maximum=256)[0].strip()),
        "machine_id_sha256": _sha(
            _stable_file(MACHINE_ID, maximum=256)[0].strip()
        ),
        "hostname_sha256": _sha(
            _stable_file(HOSTNAME, maximum=256)[0].strip()
        ),
    }
    surfaces = {
        "code": {
            **code_unsigned,
            "inventory_sha256": _sha(_canonical(code_unsigned)),
        },
        "config": _config_projection(),
        "identities_permissions": identities_permissions,
        "jobs": _jobs_projection(release),
        "migration_assets": _migration_asset_projection(release),
    }
    unsigned = {
        "schema": SCHEMA,
        "phase": phase,
        "canary_revision": canary_revision,
        "capability_plan_sha256": capability_plan_sha256,
        "full_canary_plan_sha256": full_canary_plan_sha256,
        "fixture_sha256": fixture_sha256,
        "run_id": run_id,
        "target": {
            "project": PROJECT,
            "zone": ZONE,
            "vm": VM_NAME,
            "instance_id": INSTANCE_ID,
        },
        "host_identity": host,
        "active_release": active,
        "gateway_service": service,
        "surfaces": surfaces,
        "observed_at_unix_ms": int(time.time() * 1000),
        "collector_authority": "production_root_read_only_fixed_projection",
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_job_content_recorded": False,
    }
    result = {**unsigned, "observation_sha256": _sha(_canonical(unsigned))}
    return validate_production_observation(
        result,
        phase=phase,
        canary_revision=canary_revision,
        capability_plan_sha256=capability_plan_sha256,
        full_canary_plan_sha256=full_canary_plan_sha256,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
        now_unix_ms=int(time.time() * 1000),
    )


def _validate_metadata(
    value: Mapping[str, Any],
    *,
    label: str,
    allow_absent: bool = False,
) -> None:
    if allow_absent and value.get("state") == "absent":
        if any(value.get(field) is not None for field in ("uid", "gid", "mode")):
            raise ProductionObservationError(
                f"production_observation_{label}_invalid"
            )
        return
    if (
        type(value.get("uid")) is not int
        or value["uid"] < 0
        or type(value.get("gid")) is not int
        or value["gid"] < 0
        or re.fullmatch(r"[0-7]{4}", str(value.get("mode") or "")) is None
    ):
        raise ProductionObservationError(
            f"production_observation_{label}_invalid"
        )


def _validate_file_projection(
    value: Any,
    *,
    label: str,
    expected_path: str | None = None,
) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        {"path", "uid", "gid", "mode", "size", "sha256"},
        label,
    )
    path = raw["path"]
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or (expected_path is not None and path != expected_path)
        or type(raw["size"]) is not int
        or raw["size"] < 0
        or _DIGEST.fullmatch(str(raw["sha256"] or "")) is None
    ):
        raise ProductionObservationError(
            f"production_observation_{label}_invalid"
        )
    _validate_metadata(raw, label=label)
    return raw


def _validate_active_release(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        {
            "link_path",
            "link_target",
            "resolved_target",
            "link_uid",
            "link_gid",
            "link_mode",
            "release_revision",
        },
        "active_release",
    )
    release_root = Path("/opt/adventico-ai-platform/hermes-agent-releases")
    try:
        Path(raw["resolved_target"]).relative_to(release_root)
    except (TypeError, ValueError) as exc:
        raise ProductionObservationError(
            "production_observation_active_release_invalid"
        ) from exc
    if (
        raw["link_path"] != str(ACTIVE_LINK)
        or not isinstance(raw["link_target"], str)
        or not Path(raw["link_target"]).is_absolute()
        or _REVISION.fullmatch(str(raw["release_revision"] or "")) is None
        or type(raw["link_uid"]) is not int
        or raw["link_uid"] < 0
        or type(raw["link_gid"]) is not int
        or raw["link_gid"] < 0
        or re.fullmatch(r"[0-7]{4}", str(raw["link_mode"] or "")) is None
    ):
        raise ProductionObservationError(
            "production_observation_active_release_invalid"
        )
    return raw


def _validate_gateway_service(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        {
            "unit",
            "load_state",
            "active_state",
            "sub_state",
            "unit_file_state",
            "fragment_path",
            "drop_in_paths",
            "user",
            "group",
            "working_directory",
        },
        "gateway_service",
    )
    drop_ins = _string_list(
        raw["drop_in_paths"], label="gateway_drop_ins"
    )
    root = Path("/etc/systemd/system/hermes-cloud-gateway.service.d")
    try:
        for item in drop_ins:
            Path(item).relative_to(root)
    except ValueError as exc:
        raise ProductionObservationError(
            "production_observation_gateway_service_invalid"
        ) from exc
    if (
        raw["unit"] != GATEWAY_UNIT
        or raw["load_state"] != "loaded"
        or raw["active_state"] != "active"
        or raw["sub_state"] != "running"
        or raw["fragment_path"] != str(GATEWAY_UNIT_PATH)
        or raw["user"] != "ai-platform-brain"
        or raw["group"] != "ai-platform-brain"
        or raw["working_directory"] != str(ACTIVE_LINK)
        or not isinstance(raw["unit_file_state"], str)
    ):
        raise ProductionObservationError(
            "production_observation_gateway_service_invalid"
        )
    return raw


def _validate_code_surface(
    value: Any,
    *,
    active: Mapping[str, Any],
    service: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        {
            "release_revision",
            "release_inventory",
            "unit_inventory",
            "inventory_sha256",
        },
        "code_surface",
    )
    release_inventory = raw["release_inventory"]
    if not isinstance(release_inventory, list) or len(release_inventory) != len(
        _CODE_FILES
    ):
        raise ProductionObservationError(
            "production_observation_code_surface_invalid"
        )
    release_root = Path(active["resolved_target"])
    for record, relative in zip(release_inventory, _CODE_FILES, strict=True):
        item = _strict_mapping(
            record,
            {
                "relative_path",
                "path",
                "uid",
                "gid",
                "mode",
                "size",
                "sha256",
            },
            "code_file",
        )
        if item["relative_path"] != relative:
            raise ProductionObservationError(
                "production_observation_code_surface_invalid"
            )
        _validate_file_projection(
            {key: item[key] for key in item if key != "relative_path"},
            label="code_file",
            expected_path=str(release_root / relative),
        )
    unit_inventory = raw["unit_inventory"]
    expected_unit_paths = [
        service["fragment_path"], *service["drop_in_paths"]
    ]
    if (
        not isinstance(unit_inventory, list)
        or len(unit_inventory) != len(expected_unit_paths)
    ):
        raise ProductionObservationError(
            "production_observation_code_surface_invalid"
        )
    for item, path in zip(unit_inventory, expected_unit_paths, strict=True):
        _validate_file_projection(
            item, label="gateway_unit_file", expected_path=path
        )
    unsigned = {
        "release_revision": raw["release_revision"],
        "release_inventory": release_inventory,
        "unit_inventory": unit_inventory,
    }
    if (
        raw["release_revision"] != active["release_revision"]
        or raw["inventory_sha256"] != _sha(_canonical(unsigned))
    ):
        raise ProductionObservationError(
            "production_observation_code_surface_invalid"
        )
    return raw


def _validate_config_surface(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        {
            "path",
            "uid",
            "gid",
            "mode",
            "selected_projection",
            "selected_projection_sha256",
            "full_file_content_or_digest_recorded",
        },
        "config_surface",
    )
    _validate_metadata(raw, label="config_surface")
    selected = _strict_mapping(
        raw["selected_projection"],
        {
            "model_route",
            "agent_execution",
            "goals",
            "cron",
            "approvals",
            "platforms",
            "discord_behavior",
            "config_authorization",
            "connector_policy",
        },
        "selected_config",
    )
    model = _strict_mapping(
        selected["model_route"], {"default", "provider"}, "model_route"
    )
    agent = _strict_mapping(
        selected["agent_execution"],
        {"reasoning_effort", "max_turns", "adaptive_reasoning"},
        "agent_execution",
    )
    adaptive = _strict_mapping(
        agent["adaptive_reasoning"],
        {"enabled", "max_effort"},
        "adaptive_reasoning",
    )
    goals = _strict_mapping(selected["goals"], {"max_turns"}, "goals")
    cron = _strict_mapping(selected["cron"], {"enabled", "provider"}, "cron")
    approvals = _strict_mapping(
        selected["approvals"],
        {
            "mode",
            "cron_mode",
            "plan_owner_user_ids",
            "gateway_authorized_user_ids",
            "gateway_owner_escalation",
        },
        "approvals",
    )
    escalation = _strict_mapping(
        approvals["gateway_owner_escalation"],
        {
            "enabled",
            "owner_user_id",
            "owner_guild_id",
            "owner_channel_id",
            "owner_target_type",
        },
        "owner_escalation",
    )
    platforms = _strict_mapping(
        selected["platforms"],
        {"enabled_keys", "discord_relay_only"},
        "platforms",
    )
    discord_behavior = _strict_mapping(
        selected["discord_behavior"],
        {
            "require_mention",
            "auto_thread",
            "thread_require_mention",
            "free_response_channel_ids",
            "voice_context",
        },
        "discord_behavior",
    )
    voice = _strict_mapping(
        discord_behavior["voice_context"],
        {
            "enabled",
            "text_channel_id",
            "allowed_channel_ids",
            "allowed_category_ids",
            "auto_join_channel_ids",
        },
        "voice_context",
    )
    config_authorization = _strict_mapping(
        selected["config_authorization"],
        {
            "source",
            "allowed_user_ids",
            "allowed_role_ids",
            "allowed_channel_ids",
            "legacy_env_backed_projection_collected",
        },
        "config_authorization",
    )
    connector = _strict_mapping(
        selected["connector_policy"],
        {
            "allowed_guild_ids",
            "allowed_channel_ids",
            "allowed_user_ids",
            "allowed_role_ids",
            "free_response_channel_ids",
            "public_only",
            "author_policy",
            "allow_bot_authors",
            "require_mention",
            "auto_thread",
            "thread_require_mention",
            "reviewed_cron_history_targets_sha256",
            "dm_messages",
            "group_dm_messages",
        },
        "connector_policy",
    )
    for field in (
        "allowed_guild_ids",
        "allowed_channel_ids",
        "allowed_user_ids",
        "allowed_role_ids",
        "free_response_channel_ids",
    ):
        _string_list(
            connector[field], label=f"connector_{field}", snowflakes=True
        )
    _string_list(
        platforms["enabled_keys"], label="enabled_platforms"
    )
    for field in (
        "free_response_channel_ids",
    ):
        _string_list(
            discord_behavior[field],
            label=f"discord_{field}",
            snowflakes=True,
        )
    for field in (
        "allowed_channel_ids",
        "allowed_category_ids",
        "auto_join_channel_ids",
    ):
        _string_list(
            voice[field], label=f"voice_{field}", snowflakes=True
        )
    for field in (
        "allowed_user_ids",
        "allowed_role_ids",
        "allowed_channel_ids",
    ):
        _string_list(
            config_authorization[field],
            label=f"config_authorization_{field}",
            snowflakes=True,
        )
    _string_list(
        approvals["plan_owner_user_ids"],
        label="plan_owner_user_ids",
        snowflakes=True,
    )
    _string_list(
        approvals["gateway_authorized_user_ids"],
        label="gateway_authorized_user_ids",
        snowflakes=True,
    )
    if (
        raw["path"] != str(CONFIG_PATH)
        or raw["full_file_content_or_digest_recorded"] is not False
        or raw["selected_projection_sha256"] != _sha(_canonical(selected))
        or model != {"default": "gpt-5.6-sol", "provider": "openai-codex"}
        or agent["reasoning_effort"] != "high"
        or agent["max_turns"] != 90
        or adaptive != {"enabled": True, "max_effort": "max"}
        or goals != {"max_turns": 0}
        or cron != {"enabled": True, "provider": "builtin"}
        or approvals["mode"] != "manual"
        or approvals["cron_mode"] != "deny"
        or approvals["plan_owner_user_ids"] != [_OWNER_USER_ID]
        or approvals["gateway_authorized_user_ids"] != [_OWNER_USER_ID]
        or escalation
        != {
            "enabled": True,
            "owner_user_id": _OWNER_USER_ID,
            "owner_guild_id": _OWNER_GUILD_ID,
            "owner_channel_id": _CONTROL_TOWER_CHANNEL_ID,
            "owner_target_type": "guild_channel",
        }
        or platforms
        != {
            "enabled_keys": ["api_server", "relay"],
            "discord_relay_only": True,
        }
        or discord_behavior["require_mention"] is not True
        or discord_behavior["auto_thread"] is not True
        or discord_behavior["thread_require_mention"] is not False
        or discord_behavior["free_response_channel_ids"]
        != sorted(
            {
                "1504852355588423801",
                "1505499746939174993",
            }
        )
        or voice["enabled"] is not True
        or voice["text_channel_id"] != "1504852355588423801"
        or not voice["allowed_channel_ids"]
        or not voice["auto_join_channel_ids"]
        or not set(voice["auto_join_channel_ids"]).issubset(
            voice["allowed_channel_ids"]
        )
        or config_authorization["source"]
        not in {"config_yaml", "unavailable_not_collected"}
        or config_authorization["legacy_env_backed_projection_collected"]
        is not False
        or (
            config_authorization["source"] == "unavailable_not_collected"
            and any(
                config_authorization[field]
                for field in (
                    "allowed_user_ids",
                    "allowed_role_ids",
                    "allowed_channel_ids",
                )
            )
        )
        or connector["allowed_guild_ids"] != [_OWNER_GUILD_ID]
        or connector["allowed_channel_ids"]
        != list(_APPROVED_OPERATIONAL_CHANNEL_IDS)
        or connector["free_response_channel_ids"]
        != sorted({_CONTROL_TOWER_CHANNEL_ID, _NASI_CHANNEL_ID})
        or _UNUSED_PUBLIC_CHANNEL_ID in connector["allowed_channel_ids"]
        or connector["public_only"] is not False
        or connector["author_policy"] != "guild_acl"
        or any(
            type(connector[field]) is not bool
            for field in (
                "public_only",
                "allow_bot_authors",
                "require_mention",
                "auto_thread",
                "thread_require_mention",
                "dm_messages",
                "group_dm_messages",
            )
        )
        or connector["allow_bot_authors"] is not False
        or connector["require_mention"] is not True
        or connector["auto_thread"] is not True
        or connector["thread_require_mention"] is not False
        or connector["reviewed_cron_history_targets_sha256"]
        != _sha(_canonical(_REVIEWED_CRON_HISTORY_TARGETS))
        or connector["dm_messages"] is not False
        or connector["group_dm_messages"] is not False
    ):
        raise ProductionObservationError(
            "production_observation_config_surface_invalid"
        )
    return raw


def _validate_identities_permissions_surface(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        {"users", "groups", "paths", "projection_sha256"},
        "identities_permissions",
    )
    users = raw["users"]
    groups = raw["groups"]
    paths = raw["paths"]
    if (
        not isinstance(users, list)
        or not isinstance(groups, list)
        or not isinstance(paths, list)
        or [item.get("name") for item in users] != sorted(_FIXED_USERS)
        or [item.get("name") for item in groups] != sorted(_FIXED_GROUPS)
        or [item.get("path") for item in paths]
        != sorted(str(path) for path in _FIXED_PERMISSION_PATHS)
    ):
        raise ProductionObservationError(
            "production_observation_identities_permissions_invalid"
        )
    for item in users:
        user = _strict_mapping(
            item,
            {
                "name",
                "presence",
                "uid",
                "gid",
                "home",
                "shell",
                "supplementary_group_names",
            },
            "user_identity",
        )
        _string_list(
            user["supplementary_group_names"],
            label="supplementary_groups",
        )
        if user["presence"] == "absent":
            if any(
                user[field] is not None
                for field in ("uid", "gid", "home", "shell")
            ):
                raise ProductionObservationError(
                    "production_observation_user_identity_invalid"
                )
        elif (
            user["presence"] != "present"
            or type(user["uid"]) is not int
            or user["uid"] < 0
            or type(user["gid"]) is not int
            or user["gid"] < 0
            or not isinstance(user["home"], str)
            or not Path(user["home"]).is_absolute()
            or not isinstance(user["shell"], str)
            or not Path(user["shell"]).is_absolute()
        ):
            raise ProductionObservationError(
                "production_observation_user_identity_invalid"
            )
    for item in groups:
        group = _strict_mapping(
            item,
            {"name", "presence", "gid", "members"},
            "group_identity",
        )
        _string_list(group["members"], label="group_members")
        if group["presence"] == "absent":
            if group["gid"] is not None or group["members"]:
                raise ProductionObservationError(
                    "production_observation_group_identity_invalid"
                )
        elif (
            group["presence"] != "present"
            or type(group["gid"]) is not int
            or group["gid"] < 0
        ):
            raise ProductionObservationError(
                "production_observation_group_identity_invalid"
            )
    for item in paths:
        path = _strict_mapping(
            item,
            {"path", "state", "uid", "gid", "mode"},
            "permission_path",
        )
        if path["state"] not in {
            "absent",
            "directory",
            "regular_file",
            "symlink",
            "socket",
            "other",
        }:
            raise ProductionObservationError(
                "production_observation_permission_path_invalid"
            )
        _validate_metadata(path, label="permission_path", allow_absent=True)
    unsigned = {"users": users, "groups": groups, "paths": paths}
    if raw["projection_sha256"] != _sha(_canonical(unsigned)):
        raise ProductionObservationError(
            "production_observation_identities_permissions_invalid"
        )
    return raw


def _validate_jobs_surface(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        {
            "path",
            "uid",
            "gid",
            "mode",
            "static_job_count",
            "static_jobs",
            "static_projection_sha256",
            "prompt_body_output_or_dynamic_state_recorded",
        },
        "jobs_surface",
    )
    _validate_metadata(raw, label="jobs_surface")
    jobs = raw["static_jobs"]
    fields = {
        "id",
        "enabled",
        "schedule",
        "deliver",
        "origin_public_projection",
        "provider",
        "model",
        "base_url_present",
        "no_agent",
        "script_path",
        "workdir",
        "enabled_toolsets",
        "skills",
        "repeat",
    }
    if (
        not isinstance(jobs, list)
        or type(raw["static_job_count"]) is not int
        or raw["static_job_count"] != len(jobs)
        or raw["path"] != str(JOBS_PATH)
        or raw["prompt_body_output_or_dynamic_state_recorded"] is not False
        or raw["static_projection_sha256"] != _sha(_canonical(jobs))
    ):
        raise ProductionObservationError(
            "production_observation_jobs_surface_invalid"
        )
    identifiers: list[str] = []
    for item in jobs:
        job = _strict_mapping(item, fields, "job")
        if not isinstance(job["id"], str) or _JOB_ID.fullmatch(job["id"]) is None:
            raise ProductionObservationError("production_observation_job_invalid")
        identifiers.append(job["id"])
        schedule = job["schedule"]
        if not isinstance(schedule, Mapping):
            raise ProductionObservationError("production_observation_job_invalid")
        kind = schedule.get("kind")
        expected_schedule_fields = {
            "interval": {"kind", "minutes"},
            "cron": {"kind", "expr"},
            "once": {"kind", "run_at"},
        }.get(kind)
        if expected_schedule_fields is None or set(schedule) != expected_schedule_fields:
            raise ProductionObservationError("production_observation_job_invalid")
        if _schedule_projection(schedule) != schedule:
            raise ProductionObservationError("production_observation_job_invalid")
        origin = job["origin_public_projection"]
        if origin is not None:
            origin = _strict_mapping(
                origin,
                {"platform", "chat_id", "thread_id", "user_id"},
                "job_origin",
            )
            if _origin_public_projection(origin) != origin:
                raise ProductionObservationError(
                    "production_observation_job_origin_invalid"
                )
        repeat = _strict_mapping(job["repeat"], {"times"}, "job_repeat")
        repeat_times = repeat["times"]
        if repeat_times is not None and (
            type(repeat_times) is not int or repeat_times < 1
        ):
            raise ProductionObservationError("production_observation_job_invalid")
        for field in ("enabled_toolsets", "skills"):
            _string_list(job[field], label=f"job_{field}")
        for field in ("provider", "model", "script_path", "workdir"):
            _safe_optional_route_text(job[field], label=f"job_{field}")
        if job["workdir"] is not None and not Path(job["workdir"]).is_absolute():
            raise ProductionObservationError("production_observation_job_invalid")
        if (
            type(job["enabled"]) is not bool
            or type(job["base_url_present"]) is not bool
            or type(job["no_agent"]) is not bool
            or not isinstance(job["deliver"], str)
            or _SAFE_ROUTE_TEXT.fullmatch(job["deliver"]) is None
        ):
            raise ProductionObservationError("production_observation_job_invalid")
    if identifiers != sorted(set(identifiers)):
        raise ProductionObservationError(
            "production_observation_jobs_surface_invalid"
        )
    return raw


def _validate_migration_assets_surface(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        {
            "root",
            "state",
            "file_count",
            "inventory",
            "inventory_sha256",
            "applied_database_state_recorded",
        },
        "migration_assets",
    )
    inventory = raw["inventory"]
    root = CANONICAL_BRAIN / "schemas"
    if (
        raw["root"] != str(root)
        or raw["state"] not in {"absent", "present"}
        or type(raw["file_count"]) is not int
        or raw["file_count"] < 0
        or not isinstance(inventory, list)
        or raw["file_count"] != len(inventory)
        or raw["inventory_sha256"] != _sha(_canonical(inventory))
        or raw["applied_database_state_recorded"] is not False
        or (raw["state"] == "absent" and inventory)
    ):
        raise ProductionObservationError(
            "production_observation_migration_assets_invalid"
        )
    relative_paths: list[str] = []
    for record in inventory:
        item = _strict_mapping(
            record,
            {
                "relative_path",
                "path",
                "uid",
                "gid",
                "mode",
                "size",
                "sha256",
            },
            "migration_asset",
        )
        relative = item["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ProductionObservationError(
                "production_observation_migration_asset_invalid"
            )
        relative_paths.append(relative)
        _validate_file_projection(
            {key: item[key] for key in item if key != "relative_path"},
            label="migration_asset",
            expected_path=str(root / relative),
        )
    if relative_paths != sorted(set(relative_paths)):
        raise ProductionObservationError(
            "production_observation_migration_assets_invalid"
        )
    return raw


def validate_production_observation(
    value: Any,
    *,
    phase: str,
    canary_revision: str,
    capability_plan_sha256: str,
    full_canary_plan_sha256: str,
    fixture_sha256: str,
    run_id: str,
    now_unix_ms: int,
) -> Mapping[str, Any]:
    """Exact-validate the secret-free projection before owner signing."""

    raw = _strict_mapping(
        value,
        {
            "schema",
            "phase",
            "canary_revision",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "fixture_sha256",
            "run_id",
            "target",
            "host_identity",
            "active_release",
            "gateway_service",
            "surfaces",
            "observed_at_unix_ms",
            "collector_authority",
            "secret_material_recorded",
            "secret_digest_recorded",
            "semantic_job_content_recorded",
            "observation_sha256",
        },
        "root",
    )
    unsigned = {key: item for key, item in raw.items() if key != "observation_sha256"}
    if (
        phase not in {"before", "after"}
        or raw["schema"] != SCHEMA
        or raw["phase"] != phase
        or raw["canary_revision"] != canary_revision
        or raw["capability_plan_sha256"] != capability_plan_sha256
        or raw["full_canary_plan_sha256"] != full_canary_plan_sha256
        or raw["fixture_sha256"] != fixture_sha256
        or raw["run_id"] != run_id
        or raw["target"]
        != {
            "project": PROJECT,
            "zone": ZONE,
            "vm": VM_NAME,
            "instance_id": INSTANCE_ID,
        }
        or raw["collector_authority"]
        != "production_root_read_only_fixed_projection"
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["semantic_job_content_recorded"] is not False
        or type(now_unix_ms) is not int
        or type(raw["observed_at_unix_ms"]) is not int
        or not 0 <= now_unix_ms - raw["observed_at_unix_ms"] <= 300_000
        or raw["observation_sha256"] != _sha(_canonical(unsigned))
    ):
        raise ProductionObservationError("production_observation_root_invalid")
    host = _strict_mapping(
        raw["host_identity"],
        {"boot_id_sha256", "machine_id_sha256", "hostname_sha256"},
        "host_identity",
    )
    if any(_DIGEST.fullmatch(str(item or "")) is None for item in host.values()):
        raise ProductionObservationError(
            "production_observation_host_identity_invalid"
        )
    active = _validate_active_release(raw["active_release"])
    service = _validate_gateway_service(raw["gateway_service"])
    surfaces = _strict_mapping(
        raw["surfaces"],
        {
            "code",
            "config",
            "identities_permissions",
            "jobs",
            "migration_assets",
        },
        "surfaces",
    )
    _validate_code_surface(surfaces["code"], active=active, service=service)
    _validate_config_surface(surfaces["config"])
    _validate_identities_permissions_surface(surfaces["identities_permissions"])
    _validate_jobs_surface(surfaces["jobs"])
    _validate_migration_assets_surface(surfaces["migration_assets"])
    return json.loads(_canonical(raw).decode("utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect fixed public production state for capability canary",
    )
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("--canary-revision", required=True)
    parser.add_argument("--capability-plan-sha256", required=True)
    parser.add_argument("--full-canary-plan-sha256", required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = collect_production_observation(
            phase=args.phase,
            canary_revision=args.canary_revision,
            capability_plan_sha256=args.capability_plan_sha256,
            full_canary_plan_sha256=args.full_canary_plan_sha256,
            fixture_sha256=args.fixture_sha256,
            run_id=args.run_id,
        )
    except Exception as exc:
        failure = {
            "schema": "muncho-production-capability-production-observation-failure.v1",
            "ok": False,
            "error_type": type(exc).__name__,
            "error_sha256": _sha(
                f"{type(exc).__name__}:{exc}".encode(
                    "utf-8", errors="replace"
                )
            ),
        }
        print(_canonical(failure).decode("utf-8"))
        return 1
    print(_canonical(value).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "collect_production_observation"]
