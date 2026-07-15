#!/usr/bin/env python3
"""Apply the exact Cloud Muncho model-sovereignty config remediation.

This fork-only helper performs no task classification or routing. It seals
the approved model-sovereignty configuration values, requires an exact
before hash and plan digest, writes an exact backup, and atomically replaces
the production config without restarting any service.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from gateway.production_access_policy import PRODUCTION_OWNER_DISCORD_USER_ID
from gateway.support_ops_team_registry import (
    SKYVISION_CONTROL_TOWER_CHANNEL_ID,
    SKYVISION_GUILD_ID,
)


CONFIG_PATH = Path("/opt/adventico-ai-platform/hermes-home/config.yaml")
EXPECTED_UID = 999
EXPECTED_GID = 994
EXPECTED_MODE = 0o600
MAX_CONFIG_BYTES = 1024 * 1024
GATEWAY_UNIT = "hermes-cloud-gateway.service"
SYSTEMCTL_PATH = "/usr/bin/systemctl"
_STOPPED_GATEWAY_STATES = frozenset({"failed", "inactive"})

PLAN_SCHEMA = "muncho-production-model-sovereignty-config-plan.v1"
RECEIPT_SCHEMA = "muncho-production-model-sovereignty-config-receipt.v1"
ROLLBACK_RECEIPT_SCHEMA = (
    "muncho-production-model-sovereignty-config-rollback-receipt.v1"
)
MUTATIONS = (
    "agent.adaptive_reasoning={enabled:true,max_effort:max}",
    "agent.background_review_enabled=false",
    "agent.tool_use_enforcement=true",
    "agent.verify_on_stop=false",
    "agent.verification_ledger_enabled=false",
    "agent.environment_hint.remove_stale_gpt_5_5_clause",
    "compression.abort_on_summary_failure=true",
    "auxiliary.compression={provider:openai-codex,model:gpt-5.6-sol}",
    "kanban.auxiliary_planning_enabled=false",
    "kanban.dispatch_in_gateway=false",
    "tools.tool_search={enabled:off}",
    "approvals.owner_authority=exact_discord_id",
    "approvals.gateway_owner_escalation=guild_acl_receipted_control_tower",
    "goals.max_turns=0",
    "command_allowlist=[]",
    "plugins={enabled:[],disabled:[]}",
    "hooks={};hooks_auto_accept=false",
)
_STALE_MODEL_SENTENCE = (
    "gpt-5.6-sol; do not route GPT-5.5 through OPENAI_API_KEY."
)
_CURRENT_MODEL_SENTENCE = "gpt-5.6-sol."
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ConfigGateError(RuntimeError):
    """Stable, non-secret production-config gate failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _StrictLoader(yaml.SafeLoader):
    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise ConfigGateError("config_yaml_alias_forbidden")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
        if not isinstance(node, yaml.nodes.MappingNode):
            raise ConfigGateError("config_yaml_mapping_invalid")
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or not key or key in result:
                raise ConfigGateError("config_yaml_key_invalid")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _StrictLoader.construct_mapping,
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_gateway_stopped() -> Mapping[str, Any]:
    """Prove the production config writer is not running during mutation."""

    if sys.platform != "linux":
        raise ConfigGateError("production_config_gate_requires_linux")
    if os.geteuid() != 0:
        raise ConfigGateError("production_config_gate_requires_root")
    try:
        observed = subprocess.run(
            [
                SYSTEMCTL_PATH,
                "show",
                GATEWAY_UNIT,
                "--property=LoadState",
                "--property=ActiveState",
                "--property=MainPID",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigGateError("gateway_state_unavailable") from exc
    values: dict[str, str] = {}
    for line in observed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise ConfigGateError("gateway_state_invalid")
        values[key] = value
    if (
        observed.returncode != 0
        or values.get("LoadState") != "loaded"
        or values.get("ActiveState") not in _STOPPED_GATEWAY_STATES
        or values.get("MainPID") != "0"
        or set(values) != {"LoadState", "ActiveState", "MainPID"}
    ):
        raise ConfigGateError("gateway_not_stopped")
    return {
        "unit": GATEWAY_UNIT,
        "load_state": "loaded",
        "active_state": values["ActiveState"],
        "main_pid": 0,
    }


def _identity(item: os.stat_result) -> tuple[int, ...]:
    return (
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


def _replacement_identity(item: os.stat_result) -> tuple[int, ...]:
    """Identity fields preserved when an inode is renamed within a directory."""

    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
    )


def _read_exact_config() -> tuple[bytes, os.stat_result]:
    try:
        before = os.lstat(CONFIG_PATH)
    except OSError as exc:
        raise ConfigGateError("config_unavailable") from exc
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ConfigGateError("config_file_type_invalid")
    if before.st_uid != EXPECTED_UID or before.st_gid != EXPECTED_GID:
        raise ConfigGateError("config_owner_invalid")
    if stat.S_IMODE(before.st_mode) != EXPECTED_MODE:
        raise ConfigGateError("config_mode_invalid")
    if before.st_nlink != 1:
        raise ConfigGateError("config_link_count_invalid")
    if not 0 < before.st_size <= MAX_CONFIG_BYTES:
        raise ConfigGateError("config_size_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(CONFIG_PATH, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            remaining = MAX_CONFIG_BYTES + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        reachable = os.lstat(CONFIG_PATH)
    except OSError as exc:
        raise ConfigGateError("config_read_failed") from exc
    if (
        len(raw) != before.st_size
        or len(raw) > MAX_CONFIG_BYTES
        or _identity(before) != _identity(opened)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(reachable)
    ):
        raise ConfigGateError("config_changed_during_read")
    return raw, before


def _load_mapping(raw: bytes) -> dict[str, Any]:
    try:
        value = yaml.load(raw.decode("utf-8", errors="strict"), Loader=_StrictLoader)
    except ConfigGateError:
        raise
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ConfigGateError("config_yaml_invalid") from exc
    if not isinstance(value, dict):
        raise ConfigGateError("config_root_invalid")
    return value


def _required_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    candidate = value.get(name)
    if not isinstance(candidate, dict):
        raise ConfigGateError(f"config_{name}_invalid")
    return candidate


def _target_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    target = copy.deepcopy(dict(value))
    model = _required_mapping(target, "model")
    if model != {
        "default": "gpt-5.6-sol",
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }:
        raise ConfigGateError("config_model_route_drifted")
    agent = _required_mapping(target, "agent")
    if agent.get("reasoning_effort") != "high" or agent.get("max_turns") != 90:
        raise ConfigGateError("config_agent_baseline_drifted")
    if (
        agent.get("task_completion_guidance") is not True
        or agent.get("parallel_tool_call_guidance") is not True
        or agent.get("tool_use_enforcement") not in {"auto", True}
    ):
        raise ConfigGateError("config_agent_execution_policy_drifted")
    hint = agent.get("environment_hint")
    if not isinstance(hint, str) or not hint:
        raise ConfigGateError("config_environment_hint_invalid")
    if _STALE_MODEL_SENTENCE in hint:
        if hint.count(_STALE_MODEL_SENTENCE) != 1:
            raise ConfigGateError("config_environment_hint_ambiguous")
        hint = hint.replace(_STALE_MODEL_SENTENCE, _CURRENT_MODEL_SENTENCE)
    elif "gpt-5.5" in hint.casefold():
        raise ConfigGateError("config_environment_hint_drifted")
    agent["environment_hint"] = hint
    agent["adaptive_reasoning"] = {"enabled": True, "max_effort": "max"}
    if agent.get("background_review_enabled") not in {None, False}:
        raise ConfigGateError("config_background_review_policy_drifted")
    agent["background_review_enabled"] = False
    agent["tool_use_enforcement"] = True
    if agent.get("verify_on_stop") not in {None, False}:
        raise ConfigGateError("config_verify_on_stop_drifted")
    agent["verify_on_stop"] = False
    agent["verification_ledger_enabled"] = False

    compression = _required_mapping(target, "compression")
    if compression.get("abort_on_summary_failure") not in {False, True}:
        raise ConfigGateError("config_compression_policy_drifted")
    compression["abort_on_summary_failure"] = True

    auxiliary = _required_mapping(target, "auxiliary")
    auxiliary_compression = _required_mapping(auxiliary, "compression")
    expected_auxiliary_source = {
        "provider": "auto",
        "model": "",
        "base_url": "",
        "api_key": "",
        "timeout": 120,
    }
    expected_auxiliary_target = {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "base_url": "",
        "api_key": "",
        "timeout": 120,
    }
    if (
        auxiliary_compression != expected_auxiliary_source
        and auxiliary_compression != expected_auxiliary_target
    ):
        raise ConfigGateError("config_auxiliary_compression_drifted")
    auxiliary["compression"] = expected_auxiliary_target
    curator = _required_mapping(target, "curator")
    if (
        curator.get("enabled") is not False
        or curator.get("consolidate") is not False
        or curator.get("prune_builtins") is not False
    ):
        raise ConfigGateError("config_curator_policy_drifted")
    tool_loop = _required_mapping(target, "tool_loop_guardrails")
    if (
        tool_loop.get("warnings_enabled") is not True
        or tool_loop.get("hard_stop_enabled") is not False
    ):
        raise ConfigGateError("config_tool_loop_policy_drifted")
    kanban = _required_mapping(target, "kanban")
    if kanban.get("auto_decompose") is not False:
        raise ConfigGateError("config_kanban_policy_drifted")
    kanban["auxiliary_planning_enabled"] = False
    kanban["dispatch_in_gateway"] = False
    tools = target.setdefault("tools", {})
    if not isinstance(tools, dict):
        raise ConfigGateError("config_tools_surface_drifted")
    tools["tool_search"] = {"enabled": "off"}
    approvals = target.setdefault("approvals", {})
    if not isinstance(approvals, dict):
        raise ConfigGateError("config_approvals_surface_drifted")
    approvals["plan_owner_user_ids"] = [PRODUCTION_OWNER_DISCORD_USER_ID]
    approvals["gateway_authorized_user_ids"] = [
        PRODUCTION_OWNER_DISCORD_USER_ID
    ]
    approvals["gateway_authorized_user_names"] = []
    approvals["gateway_authorized_labels"] = ["Емо"]
    approvals["gateway_owner_escalation"] = {
        "enabled": True,
        "owner_user_id": PRODUCTION_OWNER_DISCORD_USER_ID,
        "owner_guild_id": SKYVISION_GUILD_ID,
        "owner_channel_id": SKYVISION_CONTROL_TOWER_CHANNEL_ID,
        "owner_target_type": "guild_channel",
    }
    goals = target.setdefault("goals", {})
    if not isinstance(goals, dict):
        raise ConfigGateError("config_goals_surface_drifted")
    # Zero removes only the arbitrary cross-turn pause. Per-turn model/tool
    # budgets, exact permission boundaries, and explicit pause/clear remain.
    goals["max_turns"] = 0
    target["command_allowlist"] = []
    plugins = target.get("plugins")
    if plugins is not None and plugins != {} and plugins != {
        "enabled": [],
        "disabled": [],
    }:
        raise ConfigGateError("config_plugin_surface_drifted")
    target["plugins"] = {"enabled": [], "disabled": []}
    hooks = target.get("hooks")
    if hooks is not None and hooks != {}:
        raise ConfigGateError("config_hook_surface_drifted")
    target["hooks"] = {}
    if target.get("hooks_auto_accept") is not False:
        raise ConfigGateError("config_hook_acceptance_drifted")
    return target


def _replace_once(text: str, old: str, new: str, code: str) -> str:
    if text.count(old) != 1:
        raise ConfigGateError(code)
    return text.replace(old, new, 1)


def _transform(raw: bytes) -> bytes:
    source = _load_mapping(raw)
    expected = _target_mapping(source)
    agent = _required_mapping(source, "agent")
    if agent.get("adaptive_reasoning") is not None:
        raise ConfigGateError("config_adaptive_source_drifted")
    if agent.get("background_review_enabled") is not None:
        raise ConfigGateError("config_background_review_source_drifted")
    if agent.get("tool_use_enforcement") != "auto":
        raise ConfigGateError("config_tool_use_source_drifted")
    if agent.get("verify_on_stop") is not None:
        raise ConfigGateError("config_verify_on_stop_source_drifted")
    if _STALE_MODEL_SENTENCE not in str(agent.get("environment_hint") or ""):
        raise ConfigGateError("config_environment_hint_source_drifted")
    kanban = _required_mapping(source, "kanban")
    auxiliary = kanban.get("auxiliary_planning_enabled")
    dispatch = kanban.get("dispatch_in_gateway")
    if dispatch is not True or auxiliary is not None:
        raise ConfigGateError("config_kanban_source_drifted")
    compression = _required_mapping(source, "compression")
    if compression.get("abort_on_summary_failure") is not False:
        raise ConfigGateError("config_compression_source_drifted")
    auxiliary_root = _required_mapping(source, "auxiliary")
    if _required_mapping(auxiliary_root, "compression") != {
        "provider": "auto",
        "model": "",
        "base_url": "",
        "api_key": "",
        "timeout": 120,
    }:
        raise ConfigGateError("config_auxiliary_compression_source_drifted")
    if source.get("plugins") is not None and source.get("plugins") != {}:
        raise ConfigGateError("config_plugin_surface_source_drifted")
    if source.get("hooks") is not None and source.get("hooks") != {}:
        raise ConfigGateError("config_hook_surface_source_drifted")
    if source.get("hooks_auto_accept") is not False:
        raise ConfigGateError("config_hook_acceptance_source_drifted")
    try:
        target_raw = yaml.safe_dump(
            expected,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=4096,
        ).encode("utf-8")
    except (TypeError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigGateError("config_target_serialization_failed") from exc
    if len(target_raw) > MAX_CONFIG_BYTES:
        raise ConfigGateError("config_target_oversized")
    observed = _load_mapping(target_raw)
    if observed != expected:
        raise ConfigGateError("config_target_semantics_drifted")
    if "gpt-5.5" in _required_mapping(observed, "agent")["environment_hint"].casefold():
        raise ConfigGateError("config_environment_hint_not_remediated")
    return target_raw


def _plan_from_source(
    before: bytes,
    metadata: os.stat_result,
) -> tuple[dict[str, Any], bytes]:
    before_sha256 = _sha256(before)
    after = _transform(before)
    backup = CONFIG_PATH.with_name(
        f"{CONFIG_PATH.name}.pre-model-sovereignty-{before_sha256}.bak"
    )
    unsigned = {
        "schema": PLAN_SCHEMA,
        "config_path": str(CONFIG_PATH),
        "config_uid": metadata.st_uid,
        "config_gid": metadata.st_gid,
        "config_mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "before_sha256": before_sha256,
        "after_sha256": _sha256(after),
        "backup_path": str(backup),
        "mutations": list(MUTATIONS),
        "gateway_unit": GATEWAY_UNIT,
        "mutation_requires_gateway_active_state": sorted(
            _STOPPED_GATEWAY_STATES
        ),
        "mutation_requires_gateway_main_pid": 0,
        "service_restart_performed": False,
    }
    plan = {**unsigned, "plan_sha256": _sha256(_canonical_bytes(unsigned))}
    return plan, after


def build_plan(*, expected_before_sha256: str) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_before_sha256) is None:
        raise ConfigGateError("expected_before_sha256_invalid")
    before, metadata = _read_exact_config()
    if _sha256(before) != expected_before_sha256:
        raise ConfigGateError("config_before_sha256_mismatch")
    plan, _ = _plan_from_source(before, metadata)
    return plan


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            raise ConfigGateError("config_write_made_no_progress")
        offset += written


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_exact_backup(path: Path) -> tuple[bytes, os.stat_result]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ConfigGateError("config_backup_unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != EXPECTED_UID
        or before.st_gid != EXPECTED_GID
        or stat.S_IMODE(before.st_mode) != EXPECTED_MODE
        or before.st_nlink != 1
        or not 0 < before.st_size <= MAX_CONFIG_BYTES
    ):
        raise ConfigGateError("config_backup_identity_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            remaining = MAX_CONFIG_BYTES + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        reachable = os.lstat(path)
    except OSError as exc:
        raise ConfigGateError("config_backup_read_failed") from exc
    if (
        len(raw) != before.st_size
        or len(raw) > MAX_CONFIG_BYTES
        or _identity(before) != _identity(opened)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(reachable)
    ):
        raise ConfigGateError("config_backup_changed_during_read")
    return raw, before


def _rename_noreplace(source: Path, target: Path) -> None:
    """Publish a complete same-directory file without replacing a prior one."""

    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise ConfigGateError("atomic_noreplace_unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), target)
        raise OSError(error_number, os.strerror(error_number), target)
    try:
        os.link(source, target, follow_symlinks=False)
    except FileExistsError:
        raise
    os.unlink(source)


def _rename_exchange(source: Path, target: Path) -> None:
    """Atomically exchange two same-filesystem paths on production Linux."""

    if sys.platform != "linux":
        raise ConfigGateError("atomic_exchange_unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ConfigGateError("atomic_exchange_unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        2,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), target)


def _publish_backup(path: Path, raw: bytes) -> None:
    try:
        existing, _ = _read_exact_backup(path)
    except ConfigGateError as exc:
        if exc.code != "config_backup_unavailable":
            raise
    else:
        if existing != raw:
            raise ConfigGateError("config_backup_conflict")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.publish.",
    )
    temporary = Path(temporary_name)
    try:
        os.fchown(descriptor, EXPECTED_UID, EXPECTED_GID)
        os.fchmod(descriptor, EXPECTED_MODE)
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            _rename_noreplace(temporary, path)
        except FileExistsError:
            existing, _ = _read_exact_backup(path)
            if existing != raw:
                raise ConfigGateError("config_backup_conflict")
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _fsync_directory(path.parent)
    observed, _ = _read_exact_backup(path)
    if observed != raw:
        raise ConfigGateError("config_backup_readback_failed")


def _atomic_replace_config(
    *,
    replacement: bytes,
    expected_current: bytes,
    expected_metadata: os.stat_result,
    temporary_prefix: str,
    changed_code: str,
    readback_code: str,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=CONFIG_PATH.parent,
        prefix=temporary_prefix,
    )
    temporary = Path(temporary_name)
    preserve_temporary = False
    try:
        os.fchown(descriptor, expected_metadata.st_uid, expected_metadata.st_gid)
        os.fchmod(descriptor, stat.S_IMODE(expected_metadata.st_mode))
        _write_all(descriptor, replacement)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _require_gateway_stopped()
        current, current_metadata = _read_exact_config()
        if (
            current != expected_current
            or _identity(current_metadata) != _identity(expected_metadata)
        ):
            raise ConfigGateError(changed_code)
        if sys.platform == "linux":
            _rename_exchange(temporary, CONFIG_PATH)
            swapped, swapped_metadata = _read_exact_backup(temporary)
            if (
                swapped != expected_current
                or _replacement_identity(swapped_metadata)
                != _replacement_identity(expected_metadata)
            ):
                _rename_exchange(temporary, CONFIG_PATH)
                restored, _ = _read_exact_config()
                displaced, _ = _read_exact_backup(temporary)
                if restored != swapped or displaced != replacement:
                    preserve_temporary = True
                    raise ConfigGateError("config_exchange_recovery_diverged")
                raise ConfigGateError(changed_code)
            temporary.unlink()
        else:
            # Unit tests run on macOS; production refuses this fallback because
            # its mutation gate is Linux-only and uses RENAME_EXCHANGE above.
            os.replace(temporary, CONFIG_PATH)
        _fsync_directory(CONFIG_PATH.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if not preserve_temporary:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise
    observed, _ = _read_exact_config()
    if observed != replacement:
        raise ConfigGateError(readback_code)


def _apply_receipt(plan: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "ok": True,
        "state": "model_sovereignty_config_applied_no_restart",
        "plan_sha256": plan["plan_sha256"],
        "config_path": str(CONFIG_PATH),
        "before_sha256": plan["before_sha256"],
        "after_sha256": plan["after_sha256"],
        "backup_path": plan["backup_path"],
        "backup_sha256": plan["before_sha256"],
        "mutations": list(MUTATIONS),
        "gateway_unit": GATEWAY_UNIT,
        "gateway_was_stopped": True,
        "service_restart_performed": False,
    }
    return {**unsigned, "receipt_sha256": _sha256(_canonical_bytes(unsigned))}


def _rollback_receipt(
    *,
    plan: Mapping[str, Any],
    recovery: Path,
) -> dict[str, Any]:
    unsigned = {
        "schema": ROLLBACK_RECEIPT_SCHEMA,
        "ok": True,
        "state": "model_sovereignty_config_rolled_back_no_restart",
        "plan_sha256": plan["plan_sha256"],
        "config_path": str(CONFIG_PATH),
        "from_sha256": plan["after_sha256"],
        "to_sha256": plan["before_sha256"],
        "source_backup_path": plan["backup_path"],
        "recovery_backup_path": str(recovery),
        "gateway_unit": GATEWAY_UNIT,
        "gateway_was_stopped": True,
        "service_restart_performed": False,
    }
    return {**unsigned, "receipt_sha256": _sha256(_canonical_bytes(unsigned))}


def apply_plan(
    *, expected_before_sha256: str, approved_plan_sha256: str
) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_before_sha256) is None:
        raise ConfigGateError("expected_before_sha256_invalid")
    if _SHA256.fullmatch(approved_plan_sha256) is None:
        raise ConfigGateError("approved_plan_sha256_invalid")
    current, metadata = _read_exact_config()
    current_sha256 = _sha256(current)
    if current_sha256 == expected_before_sha256:
        before = current
        before_metadata = metadata
    else:
        backup_path = CONFIG_PATH.with_name(
            f"{CONFIG_PATH.name}.pre-model-sovereignty-"
            f"{expected_before_sha256}.bak"
        )
        before, before_metadata = _read_exact_backup(backup_path)
        if _sha256(before) != expected_before_sha256:
            raise ConfigGateError("config_backup_digest_mismatch")
    plan, after = _plan_from_source(before, before_metadata)
    if plan["plan_sha256"] != approved_plan_sha256:
        raise ConfigGateError("approved_plan_sha256_mismatch")
    if _sha256(after) != plan["after_sha256"]:
        raise ConfigGateError("config_target_changed_after_plan")
    _require_gateway_stopped()
    if current == after:
        backup_observed, _ = _read_exact_backup(Path(plan["backup_path"]))
        if backup_observed != before:
            raise ConfigGateError("config_backup_conflict")
        return _apply_receipt(plan)
    if current != before:
        raise ConfigGateError("config_apply_source_mismatch")
    backup = Path(plan["backup_path"])
    _publish_backup(backup, before)
    _atomic_replace_config(
        replacement=after,
        expected_current=before,
        expected_metadata=metadata,
        temporary_prefix=f".{CONFIG_PATH.name}.model-sovereignty.",
        changed_code="config_changed_after_plan",
        readback_code="config_post_write_readback_failed",
    )
    return _apply_receipt(plan)


def rollback_plan(
    *, expected_before_sha256: str, approved_plan_sha256: str
) -> dict[str, Any]:
    """Restore the exact pre-change config bound by an approved apply plan."""

    if _SHA256.fullmatch(expected_before_sha256) is None:
        raise ConfigGateError("expected_before_sha256_invalid")
    if _SHA256.fullmatch(approved_plan_sha256) is None:
        raise ConfigGateError("approved_plan_sha256_invalid")
    backup = CONFIG_PATH.with_name(
        f"{CONFIG_PATH.name}.pre-model-sovereignty-{expected_before_sha256}.bak"
    )
    before, backup_metadata = _read_exact_backup(backup)
    if _sha256(before) != expected_before_sha256:
        raise ConfigGateError("config_backup_digest_mismatch")
    plan, target = _plan_from_source(before, backup_metadata)
    if plan["plan_sha256"] != approved_plan_sha256:
        raise ConfigGateError("approved_plan_sha256_mismatch")
    current, metadata = _read_exact_config()
    _require_gateway_stopped()
    recovery = CONFIG_PATH.with_name(
        f"{CONFIG_PATH.name}.pre-rollback-{plan['after_sha256']}.bak"
    )
    if current == before:
        recovered_target, _ = _read_exact_backup(recovery)
        if recovered_target != target:
            raise ConfigGateError("config_recovery_backup_conflict")
        return _rollback_receipt(plan=plan, recovery=recovery)
    if current != target:
        raise ConfigGateError("config_rollback_source_mismatch")
    _publish_backup(recovery, current)
    _atomic_replace_config(
        replacement=before,
        expected_current=target,
        expected_metadata=metadata,
        temporary_prefix=(
            f".{CONFIG_PATH.name}.model-sovereignty-rollback."
        ),
        changed_code="config_changed_before_rollback",
        readback_code="config_rollback_readback_failed",
    )
    return _rollback_receipt(plan=plan, recovery=recovery)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("command", choices=("plan", "apply", "rollback"))
    parser.add_argument("--expected-before-sha256", required=True)
    parser.add_argument("--approved-plan-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "plan":
            if arguments.approved_plan_sha256 is not None:
                raise ConfigGateError("approved_plan_unexpected")
            result = build_plan(
                expected_before_sha256=arguments.expected_before_sha256
            )
        elif arguments.command == "apply":
            result = apply_plan(
                expected_before_sha256=arguments.expected_before_sha256,
                approved_plan_sha256=arguments.approved_plan_sha256 or "",
            )
        else:
            result = rollback_plan(
                expected_before_sha256=arguments.expected_before_sha256,
                approved_plan_sha256=arguments.approved_plan_sha256 or "",
            )
    except ConfigGateError as exc:
        result = {"ok": False, "error_code": exc.code}
        print(_canonical_bytes(result).decode("ascii"))
        return 2
    print(_canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
