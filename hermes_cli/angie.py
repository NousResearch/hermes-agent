"""Angie production Hermes operational helpers.

This module intentionally stays read-only by default.  The first supported
surface is the production Hermes migration doctor used by the Angie runbook:

    angie doctor hermes --hermes-home /home/angie/.hermes --json

The checks are conservative: they inventory paths, modes, key names and local
service state without printing secret values or mutating runtime state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"xox[baprs]-[A-Za-z0-9-]+"),
    re.compile(r"xapp-[A-Za-z0-9-]+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password|cookie)=([^\s]+)"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"(?i)(postgres(?:ql)?|mysql|redis)://[^\s]+"),
)

_SECRET_ENV_NAME_RE = re.compile(r"(?i)(token|secret|password|api[_-]?key|cookie|credential|auth)")
_AWS_TBOT_ENV_RE = re.compile(r"^(AWS_|TELEPORT|TBOT|TBot|tbot)")
_EXPECTED_PORTS = (3105, 18000)


@dataclass
class DoctorCheck:
    id: str
    severity: str
    status: str
    evidence: str
    owner: str | None = None
    rollback: str | None = None


@dataclass
class DoctorReport:
    status: str
    mode_readiness: dict[str, str]
    checks: list[dict[str, Any]]
    redactions_applied: bool = True


class AngieUsageError(Exception):
    """Invalid operator input for Angie helpers."""


def redact(value: object) -> str:
    """Return a string with secret-looking values removed.

    The doctor should report key names, paths, modes and presence only.  This
    helper is applied to every evidence string before it reaches human or JSON
    output.
    """
    text = str(value)
    text = text.replace("/home/joesu", "<NON_PROD_HOME>")
    for pattern in _SECRET_PATTERNS:
        def repl(match: re.Match[str]) -> str:
            if match.lastindex and match.lastindex >= 2:
                return f"{match.group(1)}=<REDACTED>"
            prefix = match.group(0).split()[0] if match.group(0).lower().startswith("bearer") else "<REDACTED>"
            return prefix if prefix == "<REDACTED>" else f"{prefix} <REDACTED>"
        text = pattern.sub(repl, text)
    return text


def _check(checks: list[DoctorCheck], check_id: str, severity: str, status: str, evidence: object,
           *, owner: str | None = None, rollback: str | None = None) -> None:
    checks.append(
        DoctorCheck(
            id=check_id,
            severity=severity,
            status=status,
            evidence=redact(evidence),
            owner=owner,
            rollback=rollback,
        )
    )


def _is_group_or_world_writable(path: Path) -> bool:
    mode = stat.S_IMODE(path.stat().st_mode)
    return bool(mode & (stat.S_IWGRP | stat.S_IWOTH))


def _mode(path: Path) -> str:
    return oct(stat.S_IMODE(path.stat().st_mode))


def _read_text(path: Path, *, limit: int = 1_000_000) -> str:
    data = path.read_bytes()
    if len(data) > limit:
        data = data[:limit]
    return data.decode("utf-8", errors="replace")


def _load_config(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - exact parser error varies
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return None, "config root is not a mapping"
    return data, None


def _env_key_names(env_path: Path) -> list[str]:
    keys: list[str] = []
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip().removeprefix("export ").strip()
        if key:
            keys.append(key)
    return sorted(set(keys))


def _config_key_paths(data: Any, prefix: str = "") -> list[str]:
    if not isinstance(data, dict):
        return []
    paths: list[str] = []
    for key, value in data.items():
        key_s = str(key)
        path = f"{prefix}.{key_s}" if prefix else key_s
        paths.append(path)
        if isinstance(value, dict):
            paths.extend(_config_key_paths(value, path))
    return paths


def _slack_config_summary(config: dict[str, Any], env_keys: list[str]) -> str:
    gateway_obj = config.get("gateway")
    gateway = gateway_obj if isinstance(gateway_obj, dict) else {}
    slack_obj = gateway.get("slack")
    slack = slack_obj if isinstance(slack_obj, dict) else {}
    slack_paths = [p for p in _config_key_paths(slack, "gateway.slack")]
    slack_env = [k for k in env_keys if "SLACK" in k.upper()]
    secret_like = sorted({k for k in slack_env if _SECRET_ENV_NAME_RE.search(k)})
    non_secret = sorted(set(slack_env) - set(secret_like))
    return (
        f"config_keys={slack_paths or []}; "
        f"env_secret_key_names={secret_like or []}; env_non_secret_key_names={non_secret or []}"
    )


def _run_readonly(command: list[str], *, timeout: float = 5.0) -> tuple[int | None, str]:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except FileNotFoundError:
        return None, f"{command[0]} not found"
    except subprocess.TimeoutExpired:
        return None, f"{' '.join(command)} timed out after {timeout}s"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{type(exc).__name__}: {exc}"


def _check_service(checks: list[DoctorCheck], service_name: str = "hermes-gateway.service") -> None:
    rc, out = _run_readonly(["systemctl", "--user", "is-enabled", service_name])
    if rc is None:
        _check(checks, "gateway.systemd.is_enabled", "warning", "skipped", out)
    else:
        status = "pass" if rc == 0 else "fail"
        sev = "info" if rc == 0 else "warning"
        _check(checks, "gateway.systemd.is_enabled", sev, status, f"rc={rc}; output={out}")

    rc, out = _run_readonly(["systemctl", "--user", "is-active", service_name])
    if rc is None:
        _check(checks, "gateway.systemd.is_active", "warning", "skipped", out)
    else:
        status = "pass" if rc == 0 else "fail"
        sev = "info" if rc == 0 else "warning"
        _check(checks, "gateway.systemd.is_active", sev, status, f"rc={rc}; output={out}")

    rc, out = _run_readonly(["systemctl", "--user", "show", service_name, "-p", "Restart", "-p", "Environment", "-p", "EnvironmentFiles", "--no-pager"])
    if rc is None:
        _check(checks, "gateway.systemd.environment", "warning", "skipped", out)
    else:
        keys: list[str] = []
        env_files: list[str] = []
        for token in re.split(r"\s+", out):
            if token.startswith("EnvironmentFiles="):
                env_files.append(token.split("=", 1)[0])
            elif "=" in token:
                key = token.split("=", 1)[0]
                if _AWS_TBOT_ENV_RE.search(key) or _SECRET_ENV_NAME_RE.search(key):
                    keys.append(key)
        _check(checks, "gateway.systemd.environment", "info", "pass", f"rc={rc}; sensitive_or_aws_tbot_key_names={sorted(set(keys))}; env_files={env_files or []}")

    rc, out = _run_readonly(["loginctl", "show-user", os.environ.get("USER", "angie"), "-p", "Linger"])
    if rc is None:
        _check(checks, "gateway.systemd.linger", "warning", "skipped", out)
    else:
        _check(checks, "gateway.systemd.linger", "info", "pass", f"rc={rc}; {out}")


def _check_ports(checks: list[DoctorCheck]) -> None:
    occupied: list[str] = []
    for port in _EXPECTED_PORTS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        try:
            result = sock.connect_ex(("127.0.0.1", port))
            if result == 0:
                occupied.append(str(port))
        finally:
            sock.close()
    if occupied:
        _check(checks, "network.expected_ports", "warning", "fail", f"localhost ports already listening: {', '.join(occupied)}")
    else:
        _check(checks, "network.expected_ports", "info", "pass", f"localhost ports free: {_EXPECTED_PORTS}")


def _scan_plugin_tree(checks: list[DoctorCheck], hermes_home: Path) -> None:
    plugins_dir = hermes_home / "plugins"
    if not plugins_dir.exists():
        _check(checks, "plugins.inventory", "warning", "skipped", f"{plugins_dir} missing")
        return
    plugin_names = sorted(p.name for p in plugins_dir.iterdir() if p.is_dir())
    _check(checks, "plugins.inventory", "info", "pass", f"plugins={plugin_names}")

    bad_paths: list[str] = []
    secret_hits: list[str] = []
    for path in plugins_dir.rglob("*"):
        if not path.is_file() or path.stat().st_size > 512_000:
            continue
        text = _read_text(path, limit=512_000)
        if "/home/joesu" in text:
            bad_paths.append(str(path))
        if redact(text) != text:
            secret_hits.append(str(path))
    if bad_paths:
        _check(checks, "plugins.non_prod_home_scan", "blocker", "fail", f"files containing non-production home literal: {bad_paths[:20]}")
    else:
        _check(checks, "plugins.non_prod_home_scan", "info", "pass", "no /home/joesu literals found under plugins")
    if secret_hits:
        _check(checks, "plugins.secret_literal_scan", "blocker", "fail", f"secret-looking literals found in plugin files: {secret_hits[:20]}")
    else:
        _check(checks, "plugins.secret_literal_scan", "info", "pass", "no secret-looking literals found under plugins")


def _check_hindsight(checks: list[DoctorCheck], hermes_home: Path, config: dict[str, Any] | None) -> None:
    config_map = config or {}
    memory_obj = config_map.get("memory")
    memory = memory_obj if isinstance(memory_obj, dict) else {}
    provider = memory.get("provider")
    configured = provider == "hindsight"
    compose_candidates = [
        hermes_home / "hindsight-compose" / "compose.yaml",
        hermes_home / "hindsight" / "compose.yaml",
        hermes_home / "deployments" / "hindsight" / "compose.yaml",
    ]
    existing = [p for p in compose_candidates if p.exists()]
    if configured and existing:
        _check(checks, "hindsight.runtime", "info", "pass", f"memory.provider=hindsight; compose={existing[0]}")
    elif configured:
        _check(checks, "hindsight.runtime", "warning", "fail", "memory.provider=hindsight but no local compose template found")
    else:
        _check(checks, "hindsight.runtime", "warning", "skipped", f"memory.provider={provider!r}; Mode B/C blocked unless Hindsight is deployed or Mode A disables memory")


def _check_dashboard(checks: list[DoctorCheck], config: dict[str, Any] | None) -> None:
    config_map = config or {}
    dashboard_obj = config_map.get("dashboard")
    dashboard = dashboard_obj if isinstance(dashboard_obj, dict) else {}
    enabled = bool(dashboard.get("enabled")) if dashboard else False
    bind = str(dashboard.get("host") or dashboard.get("bind") or "") if dashboard else ""
    if not enabled:
        _check(checks, "dashboard.localhost_only", "warning", "skipped", "dashboard not enabled; Mode C deferred")
    elif bind in {"127.0.0.1", "localhost", ""}:
        _check(checks, "dashboard.localhost_only", "info", "pass", f"dashboard enabled with localhost bind={bind or 'default'}")
    else:
        _check(checks, "dashboard.localhost_only", "blocker", "fail", f"dashboard bind is not localhost-only: {bind}")


def _check_mcp(checks: list[DoctorCheck], config: dict[str, Any] | None) -> None:
    config_map = config or {}
    mcp = config_map.get("mcp")
    if not isinstance(mcp, dict) or not mcp:
        mcp = config_map.get("mcp_servers")
    if not isinstance(mcp, dict) or not mcp:
        _check(checks, "mcp.matrix", "warning", "skipped", "no MCP config found")
        return
    servers_obj = mcp.get("servers") or mcp
    server_keys = sorted(str(k) for k in servers_obj.keys()) if isinstance(servers_obj, dict) else []
    _check(checks, "mcp.matrix", "warning", "pass", f"configured MCP entries={server_keys}; smoke tests still required")


def run_hermes_doctor(args: argparse.Namespace) -> int:
    hermes_home = Path(args.hermes_home).expanduser()
    checks: list[DoctorCheck] = []
    config: dict[str, Any] | None = None
    env_keys: list[str] = []

    if not hermes_home.is_absolute():
        _check(checks, "input.hermes_home", "blocker", "fail", f"--hermes-home must be absolute: {hermes_home}")
        return _emit_report(checks, json_output=args.json, invalid_input=True)
    if not hermes_home.exists() or not hermes_home.is_dir():
        _check(checks, "hermes_home.exists", "blocker", "fail", f"missing or not a directory: {hermes_home}")
        return _emit_report(checks, json_output=args.json, invalid_input=True)
    _check(checks, "hermes_home.exists", "info", "pass", f"directory exists: {hermes_home}")

    for parent in [hermes_home, hermes_home.parent]:
        try:
            writable = _is_group_or_world_writable(parent)
            _check(
                checks,
                f"permissions.parent.{parent.name or 'root'}",
                "blocker" if writable else "info",
                "fail" if writable else "pass",
                f"{parent} mode={_mode(parent)} group_or_world_writable={writable}",
            )
        except OSError as exc:
            _check(checks, f"permissions.parent.{parent.name or 'root'}", "blocker", "fail", f"cannot stat {parent}: {exc}")

    config_path = hermes_home / "config.yaml"
    if config_path.exists():
        config, err = _load_config(config_path)
        if err:
            _check(checks, "config.parse", "blocker", "fail", f"{config_path}: {err}")
        else:
            assert config is not None
            _check(checks, "config.parse", "info", "pass", f"{config_path} parses; top_level_keys={sorted(config.keys())}")
            _check(checks, "config.sanitized_diff_ready", "warning", "pass", "config can be read for sanitized diff generation; diff artifact not written by doctor")
    else:
        _check(checks, "config.parse", "blocker", "fail", f"missing {config_path}")
        _check(checks, "config.sanitized_diff_ready", "blocker", "fail", "cannot produce sanitized config diff input without config.yaml")

    env_path = hermes_home / ".env"
    if env_path.exists():
        env_keys = _env_key_names(env_path)
        mode = _mode(env_path)
        worldish = stat.S_IMODE(env_path.stat().st_mode) & (stat.S_IRWXG | stat.S_IRWXO)
        _check(checks, "env.key_inventory", "info", "pass", f"key_names={env_keys}")
        _check(checks, "env.permissions", "blocker" if worldish else "info", "fail" if worldish else "pass", f"{env_path} mode={mode}; values not read into output")
    else:
        _check(checks, "env.key_inventory", "warning", "fail", f"missing {env_path}; missing keys must be reported by key name only")

    for rel in ("auth.json", ".codex", ".config", "credentials.json"):
        path = hermes_home / rel
        if not path.exists():
            continue
        try:
            _check(checks, f"secret_file.{rel}", "info", "pass", f"exists={path.exists()} owner_uid={path.stat().st_uid} mode={_mode(path)}")
        except OSError as exc:
            _check(checks, f"secret_file.{rel}", "blocker", "fail", f"cannot stat {path}: {exc}")

    user_unit = hermes_home.parent / ".config" / "systemd" / "user" / "hermes-gateway.service"
    if user_unit.exists():
        try:
            unit_mode = stat.S_IMODE(user_unit.stat().st_mode)
            writable = bool(unit_mode & (stat.S_IWGRP | stat.S_IWOTH))
            _check(
                checks,
                "gateway.systemd.unit_permissions",
                "blocker" if writable else "info",
                "fail" if writable else "pass",
                f"{user_unit} mode={oct(unit_mode)} group_or_world_writable={writable}",
            )
        except OSError as exc:
            _check(checks, "gateway.systemd.unit_permissions", "warning", "skipped", f"cannot stat {user_unit}: {exc}")
    else:
        _check(checks, "gateway.systemd.unit_permissions", "warning", "skipped", f"missing {user_unit}")

    hermes_bin = shutil.which("hermes")
    if not hermes_bin:
        user_bin = hermes_home.parent / ".local" / "bin" / "hermes"
        if user_bin.exists():
            hermes_bin = str(user_bin)
    _check(checks, "runtime.hermes_command", "info" if hermes_bin else "blocker", "pass" if hermes_bin else "fail", f"hermes={hermes_bin or 'not found'}")
    _check(checks, "runtime.python", "info", "pass", f"python={sys.executable}; version={sys.version.split()[0]}")
    venv_candidates = [hermes_home / ".venv", hermes_home / "venv", Path(sys.prefix)]
    existing_venv = [str(p) for p in venv_candidates if p.exists()]
    _check(checks, "runtime.venv", "info" if existing_venv else "warning", "pass" if existing_venv else "fail", f"venv_candidates_present={existing_venv}")

    if config is not None:
        _check(checks, "slack.config_classification", "warning", "pass", _slack_config_summary(config, env_keys))
    else:
        _check(checks, "slack.config_classification", "warning", "skipped", "config unavailable")

    _check_service(checks)
    _check_ports(checks)

    try:
        usage = shutil.disk_usage(hermes_home)
        free_pct = usage.free / usage.total * 100 if usage.total else 0
        sev = "blocker" if free_pct < 15 else "info"
        _check(checks, "disk.free_space", sev, "fail" if free_pct < 15 else "pass", f"free_pct={free_pct:.1f}; free_bytes={usage.free}")
    except OSError as exc:
        _check(checks, "disk.free_space", "warning", "skipped", f"cannot inspect disk: {exc}")

    rc, out = _run_readonly(["docker", "system", "df"], timeout=10)
    if rc is None:
        _check(checks, "docker.disk_usage", "warning", "skipped", out)
    else:
        _check(checks, "docker.disk_usage", "warning", "pass" if rc == 0 else "skipped", f"rc={rc}; {out[:500]}")

    _scan_plugin_tree(checks, hermes_home)
    _check_hindsight(checks, hermes_home, config)
    _check_dashboard(checks, config)
    _check_mcp(checks, config)
    _check_aws_tbot_negative(checks, hermes_home)

    return _emit_report(checks, json_output=args.json)


def _check_aws_tbot_negative(checks: list[DoctorCheck], hermes_home: Path) -> None:
    present_paths = [str(p) for p in (hermes_home.parent / ".aws", hermes_home.parent / ".tbot", hermes_home.parent / ".tsh") if p.exists()]
    env_keys = sorted(k for k in os.environ if _AWS_TBOT_ENV_RE.search(k))
    if present_paths or env_keys:
        _check(checks, "aws_tbot.negative_verification", "warning", "fail", f"unexpected AWS/Teleport/tbot presence: paths={present_paths}; env_key_names={env_keys}")
    else:
        _check(checks, "aws_tbot.negative_verification", "info", "pass", "no AWS/Teleport/tbot env key names or sibling credential dirs detected")


def _build_report(checks: list[DoctorCheck]) -> DoctorReport:
    serialized = [asdict(c) for c in checks]
    has_blocker = any(c.severity == "blocker" and c.status != "pass" for c in checks)
    has_warning = any(c.severity == "warning" and c.status != "pass" for c in checks)
    status = "blocker" if has_blocker else "warning" if has_warning else "pass"
    by_id = {c.id: c for c in checks}
    mode_a = "blocker" if has_blocker else "warning" if has_warning else "pass"
    hindsight = by_id.get("hindsight.runtime")
    dashboard = by_id.get("dashboard.localhost_only")
    mode_b = mode_a
    if hindsight and hindsight.status != "pass":
        mode_b = "blocker" if mode_a == "pass" else mode_a
    mode_c = mode_b
    if dashboard and dashboard.status != "pass":
        mode_c = "blocker" if mode_b == "pass" else mode_b
    return DoctorReport(
        status=status,
        mode_readiness={"mode_a": mode_a, "mode_b": mode_b, "mode_c": mode_c},
        checks=serialized,
        redactions_applied=True,
    )


def _emit_report(checks: list[DoctorCheck], *, json_output: bool, invalid_input: bool = False) -> int:
    report = _build_report(checks)
    if json_output:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(f"Angie Hermes doctor: {report.status.upper()}")
        print(f"Mode readiness: {report.mode_readiness}")
        for c in checks:
            marker = "✓" if c.status == "pass" else "⚠" if c.severity == "warning" else "✗"
            print(f"  {marker} {c.id} [{c.severity}/{c.status}] {c.evidence}")
    if invalid_input:
        return 2
    return 1 if report.status == "blocker" else 0


def run_plugins_sync(args: argparse.Namespace) -> int:
    hermes_home = Path(args.hermes_home).expanduser()
    source = Path(args.source).expanduser()
    if not hermes_home.is_absolute():
        raise AngieUsageError("--hermes-home must be an absolute path")
    if not source.exists():
        raise AngieUsageError(f"plugin source does not exist: {source}")
    target_root = hermes_home / "plugins"
    inventory = []
    for plugin in sorted(p for p in source.iterdir() if p.is_dir()):
        if plugin.name == "interactive-cli":
            decision = "exclude"
            reason = "interactive-cli is explicitly excluded from production migration"
        else:
            decision = "sync_candidate"
            reason = "dry-run inventory only; production apply requires explicit approval"
        inventory.append({
            "plugin": plugin.name,
            "source": str(plugin),
            "target": str(target_root / plugin.name),
            "decision": decision,
            "reason": reason,
            "env_key_names": _discover_env_key_names(plugin),
            "enabled_state": "unknown_until_production_config_audit",
        })
    payload = {"dry_run": True, "inventory": inventory, "redactions_applied": True}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Plugin sync dry-run: {len(inventory)} plugin directories inspected")
        for item in inventory:
            print(f"  - {item['plugin']}: {item['decision']} -> {item['target']} ({item['reason']})")
    if not args.dry_run:
        print("Refusing to mutate plugins: Phase 1 only supports --dry-run.", file=sys.stderr)
        return 2
    return 0


def _discover_env_key_names(root: Path) -> list[str]:
    names: set[str] = set()
    env_re = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 256_000:
            continue
        text = _read_text(path, limit=256_000)
        for match in env_re.findall(text):
            if _SECRET_ENV_NAME_RE.search(match) or match.startswith(("HERMES_", "SLACK_", "OPENAI_", "ANTHROPIC_", "AWS_", "MCP_")):
                names.add(match)
    return sorted(names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="angie", description="Angie production Hermes operational helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Read-only production readiness diagnostics")
    doctor_sub = doctor.add_subparsers(dest="doctor_target", required=True)
    hermes_doctor = doctor_sub.add_parser("hermes", help="Diagnose a production Hermes home")
    hermes_doctor.add_argument("--hermes-home", required=True, help="Absolute production Hermes home path, e.g. /home/angie/.hermes")
    hermes_doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    hermes_doctor.set_defaults(func=run_hermes_doctor)

    hermes = sub.add_parser("hermes", help="Hermes production migration helpers")
    hermes_sub = hermes.add_subparsers(dest="hermes_command", required=True)
    plugins = hermes_sub.add_parser("plugins", help="Plugin migration helpers")
    plugins_sub = plugins.add_subparsers(dest="plugins_command", required=True)
    sync = plugins_sub.add_parser("sync", help="Inventory plugin sync candidates (dry-run only in Phase 1)")
    sync.add_argument("--hermes-home", required=True, help="Absolute production Hermes home path")
    sync.add_argument("--source", default="plugins/hermes", help="Plugin source directory in the Angie repo")
    sync.add_argument("--dry-run", action="store_true", help="Required; do not copy anything")
    sync.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    sync.set_defaults(func=run_plugins_sync)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except AngieUsageError as exc:
        print(f"angie: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
