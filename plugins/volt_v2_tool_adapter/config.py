"""Configuration helpers for the Volt V2 tool adapter proof.

The adapter is intentionally opt-in.  Missing config means disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_AUDIT_RELATIVE_PATH = "logs/volt-v2-tool-adapter.jsonl"
DEFAULT_MODE = "observe"
DEFAULT_FAIL_POLICY = "open"
VALID_MODES = {"observe", "transform", "route", "override"}
VALID_FAIL_POLICIES = {"open", "closed_for_allowlisted"}


@dataclass(frozen=True)
class VerificationConfig:
    emit_events: bool = True
    write_audit_jsonl: bool = True
    require_result_marker: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "VerificationConfig":
        if not isinstance(data, Mapping):
            data = {}
        return cls(
            emit_events=_as_bool(data.get("emit_events"), True),
            write_audit_jsonl=_as_bool(data.get("write_audit_jsonl"), True),
            require_result_marker=_as_bool(data.get("require_result_marker"), True),
        )


@dataclass(frozen=True)
class VoltV2ToolAdapterConfig:
    enabled: bool = False
    mode: str = DEFAULT_MODE
    fail_policy: str = DEFAULT_FAIL_POLICY
    artifact_root: str = ""
    audit_path: str = ""
    allowlist_tools: tuple[str, ...] = field(default_factory=tuple)
    allowlist_toolsets: tuple[str, ...] = field(default_factory=tuple)
    allowlist_paths: tuple[str, ...] = field(default_factory=tuple)
    denylist_tools: tuple[str, ...] = field(default_factory=tuple)
    denylist_path_prefixes: tuple[str, ...] = field(default_factory=tuple)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "VoltV2ToolAdapterConfig":
        if not isinstance(data, Mapping):
            data = {}
        allowlist = data.get("allowlist") or {}
        denylist = data.get("denylist") or {}
        if not isinstance(allowlist, Mapping):
            allowlist = {}
        if not isinstance(denylist, Mapping):
            denylist = {}

        mode = str(data.get("mode", DEFAULT_MODE) or DEFAULT_MODE).strip().lower()
        if mode not in VALID_MODES:
            mode = DEFAULT_MODE
        fail_policy = str(data.get("fail_policy", DEFAULT_FAIL_POLICY) or DEFAULT_FAIL_POLICY).strip().lower()
        if fail_policy not in VALID_FAIL_POLICIES:
            fail_policy = DEFAULT_FAIL_POLICY

        artifact_root = _as_path_string(data.get("artifact_root", ""))
        audit_path = _as_path_string(data.get("audit_path", ""))
        if not audit_path:
            audit_path = _default_audit_path()

        return cls(
            enabled=_as_bool(data.get("enabled"), False),
            mode=mode,
            fail_policy=fail_policy,
            artifact_root=artifact_root,
            audit_path=audit_path,
            allowlist_tools=_as_tuple(allowlist.get("tools")),
            allowlist_toolsets=_as_tuple(allowlist.get("toolsets")),
            allowlist_paths=_as_tuple(allowlist.get("paths")),
            denylist_tools=_as_tuple(denylist.get("tools")),
            denylist_path_prefixes=_as_tuple(denylist.get("path_prefixes")),
            verification=VerificationConfig.from_mapping(data.get("verification")),
        )


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if item is not None and str(item).strip())
    return ()


def _as_path_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, Path)):
        return str(value).strip()
    return ""


def _default_audit_path() -> str:
    try:
        from hermes_cli.config import get_hermes_home

        return str(Path(get_hermes_home()) / DEFAULT_AUDIT_RELATIVE_PATH)
    except Exception:
        return str(Path.home() / ".hermes" / DEFAULT_AUDIT_RELATIVE_PATH)


def load_adapter_config() -> VoltV2ToolAdapterConfig:
    """Load adapter config from ``volt_v2.tool_adapter``.

    Missing config or config loading failures are fail-closed for adapter
    activation: return the disabled default so Hermes' normal tool pipeline is
    untouched.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        adapter_cfg = ((cfg.get("volt_v2") or {}).get("tool_adapter") or {})
        if not isinstance(adapter_cfg, Mapping):
            adapter_cfg = {}
        return VoltV2ToolAdapterConfig.from_mapping(adapter_cfg)
    except Exception:
        return VoltV2ToolAdapterConfig()
