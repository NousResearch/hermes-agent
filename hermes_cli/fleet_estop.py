"""Controlled fleet ESTOP for Hermes profile gateway systemd units.

This extends the existing sentinel-based ``hermes pause`` with an operator
control plane for profile gateway units. The default/Pacey gateway is protected
unless the caller uses an explicit override.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

DEFAULT_GATEWAY_UNIT = "hermes-gateway.service"
UNIT_PREFIX = "hermes-gateway"
# Deployment-specific profile service names belong in config, not in the
# upstream code. Configure either:
#   fleet_estop.profile_unit_overrides:
#     tony: hermes-tony-gateway.service
# or env HERMES_FLEET_ESTOP_UNIT_OVERRIDES='{"tony":"hermes-tony-gateway.service"}'.
DEFAULT_PROFILE_UNIT_OVERRIDES: dict[str, str] = {}


def _normalise_profile_name(profile: str) -> str:
    return str(profile or "").strip().lower().replace(" ", "-")


def _valid_unit_name(value: object) -> str | None:
    unit = str(value or "").strip()
    if not unit:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+\.service", unit):
        return None
    if "gateway" not in unit or not unit.startswith("hermes-"):
        return None
    return unit


def _coerce_unit_overrides(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    overrides: dict[str, str] = {}
    for profile, unit in raw.items():
        key = _normalise_profile_name(str(profile))
        value = _valid_unit_name(unit)
        if key and value:
            overrides[key] = value
    return overrides


def profile_unit_overrides() -> dict[str, str]:
    """Return deployment-specific profile→systemd-unit overrides.

    Upstream defaults use the generic ``hermes-gateway-<profile>.service``
    naming scheme. Operators with existing differently named units can set
    ``fleet_estop.profile_unit_overrides`` in config.yaml or the JSON env var
    ``HERMES_FLEET_ESTOP_UNIT_OVERRIDES``. Invalid entries are ignored instead
    of being executed, because unit names are later passed to systemctl.
    """
    overrides = dict(DEFAULT_PROFILE_UNIT_OVERRIDES)
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        fleet_cfg = cfg.get("fleet_estop") if isinstance(cfg, dict) else None
        if isinstance(fleet_cfg, dict):
            overrides.update(_coerce_unit_overrides(fleet_cfg.get("profile_unit_overrides") or fleet_cfg.get("unit_overrides")))
    except Exception:
        pass
    raw_env = os.environ.get("HERMES_FLEET_ESTOP_UNIT_OVERRIDES")
    if raw_env:
        try:
            overrides.update(_coerce_unit_overrides(json.loads(raw_env)))
        except Exception:
            pass
    return overrides


@dataclass(frozen=True)
class FleetGatewayUnit:
    unit: str
    profile: str
    active_state: str = "unknown"
    sub_state: str = "unknown"
    main_pid: int = 0

    @property
    def protected(self) -> bool:
        return self.unit == DEFAULT_GATEWAY_UNIT or self.profile in {"default", "pacey"}


def audit_dir() -> Path:
    path = get_hermes_home() / "workspace" / "fleet-estop"
    path.mkdir(parents=True, exist_ok=True)
    return path


def audit_log_path() -> Path:
    return audit_dir() / "audit.jsonl"


def _systemctl_env() -> dict[str, str]:
    env = dict(os.environ)
    if not env.get("XDG_RUNTIME_DIR"):
        try:
            uid = os.getuid()
        except Exception:
            uid = 1000
        candidate = f"/run/user/{uid}"
        if Path(candidate, "bus").exists():
            env["XDG_RUNTIME_DIR"] = candidate
            env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={candidate}/bus")
    elif not env.get("DBUS_SESSION_BUS_ADDRESS"):
        bus = Path(env["XDG_RUNTIME_DIR"]) / "bus"
        if bus.exists():
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus}"
    return env


def _run_systemctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
        env=_systemctl_env(),
    )


def _run_systemctl_system(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )


def profile_from_unit(unit: str) -> str:
    if unit == DEFAULT_GATEWAY_UNIT:
        return "default"
    for profile, mapped in profile_unit_overrides().items():
        if unit == mapped:
            return profile
    name = unit.removesuffix(".service")
    if name.startswith(f"{UNIT_PREFIX}-"):
        return name[len(UNIT_PREFIX) + 1 :]
    if name.startswith("hermes-") and name.endswith("-gateway"):
        return name[len("hermes-") : -len("-gateway")]
    return name


def unit_from_profile(profile: str) -> str:
    value = str(profile or "").strip()
    if not value or value in {"default", "pacey"}:
        return DEFAULT_GATEWAY_UNIT
    normalised = _normalise_profile_name(value)
    overrides = profile_unit_overrides()
    if normalised in overrides:
        return overrides[normalised]
    compact = re.sub(r"[^A-Za-z0-9_-]+", "", value.replace(" ", ""))
    return f"{UNIT_PREFIX}-{compact}.service"


def parse_systemctl_list(text: str) -> list[str]:
    units: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("UNIT "):
            continue
        first = line.split()[0]
        if first.startswith("hermes-") and first.endswith(".service") and "gateway" in first:
            units.append(first)
    return sorted(set(units), key=lambda u: (u != DEFAULT_GATEWAY_UNIT, u))


def parse_show(unit: str, text: str) -> FleetGatewayUnit:
    data: dict[str, str] = {}
    for raw in (text or "").splitlines():
        if "=" in raw:
            k, v = raw.split("=", 1)
            data[k] = v
    try:
        pid = int(data.get("MainPID") or 0)
    except ValueError:
        pid = 0
    return FleetGatewayUnit(
        unit=unit,
        profile=profile_from_unit(unit),
        active_state=data.get("ActiveState") or "unknown",
        sub_state=data.get("SubState") or "unknown",
        main_pid=pid,
    )


def list_gateway_units() -> list[FleetGatewayUnit]:
    listed = _run_systemctl(["list-units", "--type=service", "--all", "*gateway*", "--no-legend", "--no-pager"])
    units = parse_systemctl_list(listed.stdout)
    result: list[FleetGatewayUnit] = []
    # Ensure the controlling default unit is considered even if list-units did
    # not print unloaded units under a constrained user manager.
    if DEFAULT_GATEWAY_UNIT not in units:
        show = _run_systemctl_system(["show", DEFAULT_GATEWAY_UNIT, "--property=ActiveState,SubState,MainPID", "--no-pager"])
        default_unit = parse_show(DEFAULT_GATEWAY_UNIT, show.stdout)
        result.append(default_unit)
    for unit in units:
        show = _run_systemctl(["show", unit, "--property=ActiveState,SubState,MainPID", "--no-pager"])
        result.append(parse_show(unit, show.stdout))
    return result


def select_units(units: Iterable[FleetGatewayUnit], profiles: list[str] | None) -> list[FleetGatewayUnit]:
    all_units = list(units)
    if not profiles:
        return [u for u in all_units if not u.protected]
    wanted_units = {unit_from_profile(p) for p in profiles}
    wanted_profiles = {profile_from_unit(unit_from_profile(p)) for p in profiles}
    return [u for u in all_units if u.unit in wanted_units or u.profile in wanted_profiles]


def _with_requested_units(inventory: list[FleetGatewayUnit], profiles: list[str] | None) -> list[FleetGatewayUnit]:
    """Ensure explicit targets are present even when list-units omits failed units."""
    if not profiles:
        return inventory
    by_unit = {u.unit: u for u in inventory}
    for profile in profiles:
        unit = unit_from_profile(profile)
        if unit not in by_unit:
            shown = _run_systemctl(["show", unit, "--property=ActiveState,SubState,MainPID", "--no-pager"])
            by_unit[unit] = parse_show(unit, shown.stdout)
    return list(by_unit.values())


def build_plan(
    *,
    action: str,
    profiles: list[str] | None = None,
    include_default: bool = False,
    units: list[FleetGatewayUnit] | None = None,
) -> list[dict[str, object]]:
    inventory = units if units is not None else _with_requested_units(list_gateway_units(), profiles)
    selected = select_units(inventory, profiles)
    plan: list[dict[str, object]] = []
    for unit in selected:
        blocked = unit.protected and not include_default
        plan.append(
            {
                "action": action,
                "unit": unit.unit,
                "profile": unit.profile,
                "protected": unit.protected,
                "blocked": blocked,
                "reason": "default/Pacey gateway protected" if blocked else "selected",
                "active_state": unit.active_state,
                "sub_state": unit.sub_state,
                "main_pid": unit.main_pid,
            }
        )
    return plan


def write_audit(record: dict[str, object]) -> Path:
    path = audit_log_path()
    safe = dict(record)
    safe["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(safe, sort_keys=True) + "\n")
    return path


def execute_plan(plan: list[dict[str, object]], *, dry_run: bool, reason: str | None = None) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for item in plan:
        result = dict(item)
        unit = str(item.get("unit") or "")
        action = str(item.get("action") or "")
        if item.get("blocked"):
            result["status"] = "blocked"
        elif dry_run:
            result["status"] = "dry-run"
        elif action in {"stop", "resume"}:
            verb = "start" if action == "resume" else "stop"
            before = parse_show(unit, _run_systemctl(["show", unit, "--property=ActiveState,SubState,MainPID", "--no-pager"]).stdout)
            proc = _run_systemctl([verb, unit])
            after = parse_show(unit, _run_systemctl(["show", unit, "--property=ActiveState,SubState,MainPID", "--no-pager"]).stdout)
            result.update(
                {
                    "status": "ok" if proc.returncode == 0 else "failed",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr.strip()[-500:],
                    "before_pid": before.main_pid,
                    "after_pid": after.main_pid,
                    "after_active_state": after.active_state,
                    "after_sub_state": after.sub_state,
                }
            )
        else:
            result["status"] = "unsupported-action"
        results.append(result)
    audit = {
        "kind": "fleet-estop",
        "dry_run": dry_run,
        "reason": reason or "",
        "results": results,
    }
    path = write_audit(audit)
    return {"audit_log": str(path), "results": results}


def format_units(units: list[FleetGatewayUnit]) -> str:
    lines = ["Hermes fleet gateway units:"]
    for u in units:
        flag = " protected" if u.protected else ""
        lines.append(f"- {u.profile}: {u.unit} {u.active_state}/{u.sub_state} pid={u.main_pid}{flag}")
    return "\n".join(lines)


def format_plan(plan: list[dict[str, object]]) -> str:
    lines = ["Fleet ESTOP plan:"]
    if not plan:
        lines.append("- no matching gateway units")
        return "\n".join(lines)
    for item in plan:
        blocked = " BLOCKED" if item.get("blocked") else ""
        lines.append(
            f"- {item.get('action')} {item.get('profile')} ({item.get('unit')}) pid={item.get('main_pid')} {item.get('active_state')}/{item.get('sub_state')}{blocked}"
        )
    return "\n".join(lines)


def cmd_fleet_estop(args) -> int:
    action = getattr(args, "fleet_action", None) or "list"
    profiles = list(getattr(args, "target_profile", None) or [])
    include_default = bool(getattr(args, "include_default", False))
    dry_run = bool(getattr(args, "dry_run", False))
    reason = getattr(args, "reason", None)

    if action == "list":
        print(format_units(list_gateway_units()))
        return 0
    if action in {"plan-stop", "plan-resume"}:
        planned_action = "stop" if action == "plan-stop" else "resume"
        print(format_plan(build_plan(action=planned_action, profiles=profiles, include_default=include_default)))
        return 0
    if action in {"stop", "resume"}:
        plan = build_plan(action=action, profiles=profiles, include_default=include_default)
        print(format_plan(plan))
        result = execute_plan(plan, dry_run=dry_run, reason=reason)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1 if any(r.get("status") in {"failed", "blocked"} for r in result["results"]) else 0
    print("Usage: hermes fleet-estop {list|plan-stop|plan-resume|stop|resume}")
    return 2


def build_fleet_estop_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "fleet-estop",
        help="Controlled stop/resume for Hermes profile gateway units",
        description="List, dry-run, stop, or resume Hermes profile gateway systemd user units. The default/Pacey gateway is protected unless --include-default is used.",
    )
    subs = parser.add_subparsers(dest="fleet_action")
    subs.add_parser("list", help="List Hermes gateway units")
    for name in ("plan-stop", "plan-resume", "stop", "resume"):
        p = subs.add_parser(name, help=f"{name} selected non-default profile gateways")
        p.add_argument("--target-profile", dest="target_profile", action="append", help="Profile id to target. Repeat for multiple. Omit to select all non-default profile gateways.")
        p.add_argument("--include-default", action="store_true", help="Allow targeting the protected default/Pacey gateway")
        p.add_argument("--reason", default=None, help="Reason recorded in the audit log")
        p.add_argument("--dry-run", action="store_true", help="Write audit only; do not stop/start units")
    parser.set_defaults(func=cmd_fleet_estop)
