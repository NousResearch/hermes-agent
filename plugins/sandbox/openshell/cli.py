"""CLI for the OpenShell sandbox provider."""

from __future__ import annotations

import json
import os
import argparse
import shutil
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from plugins.sandbox.openshell.environment import (
    DEFAULT_MIRROR_EXCLUDES,
    DEFAULT_GATEWAY_NAME,
    DEFAULT_GATEWAY_PORT,
    build_sandbox_name,
    bundled_source_dir,
    ensure_openshell_gateway,
    normalize_openshell_config,
    prepare_openshell_source,
    run_openshell_cli,
)


def register_cli(parser):
    sub = parser.add_subparsers(dest="openshell_command")

    enable = sub.add_parser("enable", help="Configure Hermes OpenShell sandboxing")
    enable.add_argument("--default", action="store_true", help="Make OpenShell the default Hermes sandbox backend")
    enable.add_argument("--install", action="store_true", help="Install or update OpenShell with uv tool install -U openshell")
    enable.add_argument("--mode", choices=["mirror", "remote"], default="mirror")
    enable.add_argument("--from", dest="source", default="", help="OpenShell source name, image, or local source directory")
    enable.add_argument("--policy", default="", help="Optional OpenShell policy YAML path")
    enable.add_argument("--provider", action="append", default=[], help="OpenShell provider to attach at sandbox creation")
    enable.add_argument("--gpu", action="store_true", help="Request OpenShell GPU passthrough")
    enable.add_argument("--no-auto-providers", action="store_true", help="Disable OpenShell auto provider creation")
    enable.add_argument("--gateway-name", default=DEFAULT_GATEWAY_NAME, help="OpenShell gateway name for Hermes sandboxes")
    enable.add_argument("--gateway-port", type=int, default=DEFAULT_GATEWAY_PORT, help="Local OpenShell gateway port")
    enable.add_argument("--gateway-host", default="", help="Gateway host advertised in OpenShell metadata")
    enable.add_argument("--gateway-endpoint", default="", help="Use an existing gateway endpoint instead of starting a local gateway")
    enable.add_argument("--extra-sync", action="append", default=[], help="Additional host path sync as LOCAL[:REMOTE]")
    enable.add_argument("--scope", default="smoke", help="Smoke-test scope name")
    enable.add_argument("--smoke", action="store_true", help="Run smoke validation even if OpenShell is already configured")
    enable.add_argument("--no-smoke", action="store_true", help="Skip create/validate smoke sandbox")

    sub.add_parser("disable", help="Disable OpenShell as the default sandbox backend")
    status = sub.add_parser("status", help="Show OpenShell/Hermes sandbox status")
    _add_config_override_args(status)
    list_p = sub.add_parser("list", help="List OpenShell sandboxes")
    _add_config_override_args(list_p)

    create = sub.add_parser("create", help="Create or validate a Hermes OpenShell sandbox")
    _add_scope_name_args(create)
    _add_config_override_args(create)

    delete = sub.add_parser("delete", help="Delete a Hermes OpenShell sandbox")
    _add_scope_name_args(delete)
    _add_config_override_args(delete)

    shell = sub.add_parser("shell", help="Open an interactive shell in a sandbox")
    _add_scope_name_args(shell)
    _add_config_override_args(shell)

    exec_p = sub.add_parser("exec", help="Run a command in a sandbox")
    _add_scope_name_args(exec_p)
    _add_config_override_args(exec_p)
    exec_p.add_argument("exec_command", nargs=argparse.REMAINDER)

    push = sub.add_parser("push", help="Upload a host path into a sandbox")
    _add_scope_name_args(push)
    _add_config_override_args(push)
    push.add_argument("local_path")
    push.add_argument("remote_path")

    pull = sub.add_parser("pull", help="Download a sandbox path to the host")
    _add_scope_name_args(pull)
    _add_config_override_args(pull)
    pull.add_argument("remote_path")
    pull.add_argument("local_path")

    parser.set_defaults(func=openshell_command)

def _add_scope_name_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scope", default="", help="Hermes scope used to derive the sandbox name")
    group.add_argument("--name", default="", help="Raw OpenShell sandbox name")


def _add_config_override_args(parser):
    parser.add_argument("--gateway-name", default="", help="OpenShell gateway name")
    parser.add_argument("--gateway-port", type=int, default=None, help="Local OpenShell gateway port")
    parser.add_argument("--gateway-host", default="", help="Gateway host advertised in OpenShell metadata")
    parser.add_argument("--gateway-endpoint", default="", help="Existing gateway endpoint")
    parser.add_argument("--source", default="", help="OpenShell source name, image, or local source directory")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="OpenShell operation timeout")


def openshell_command(args):
    cmd = getattr(args, "openshell_command", None)
    if cmd == "enable":
        return _cmd_enable(args)
    if cmd == "disable":
        return _cmd_disable(args)
    if cmd == "status":
        return _cmd_status(args)
    if cmd == "list":
        return _run_passthrough(["sandbox", "list"], args)
    if cmd == "create":
        return _cmd_create(args)
    if cmd == "delete":
        return _run_for_sandbox(args, lambda cfg, name: run_openshell_cli(cfg, ["sandbox", "delete", name], timeout=120))
    if cmd == "shell":
        return _cmd_shell(args)
    if cmd == "exec":
        return _cmd_exec(args)
    if cmd == "push":
        return _run_for_sandbox(
            args,
            lambda cfg, name: run_openshell_cli(
                cfg,
                ["sandbox", "upload", name, args.local_path, args.remote_path],
                timeout=max(cfg["timeout_seconds"], 300),
            ),
        )
    if cmd == "pull":
        return _run_for_sandbox(
            args,
            lambda cfg, name: run_openshell_cli(
                cfg,
                ["sandbox", "download", name, args.remote_path, args.local_path],
                timeout=max(cfg["timeout_seconds"], 300),
            ),
        )

    print("Usage: hermes openshell {enable|disable|status|list|create|delete|shell|exec|push|pull}")
    return None


def _get_dotted(config: dict[str, Any], dotted: str) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _values_equal(current: Any, desired: Any) -> bool:
    if isinstance(desired, bool):
        return bool(current) is desired
    if isinstance(desired, int):
        try:
            return int(current) == desired
        except (TypeError, ValueError):
            return False
    if isinstance(desired, list):
        return list(current or []) == desired
    return str(current or "") == str(desired or "")


def _is_already_configured(config: dict[str, Any], updates: dict[str, Any]) -> bool:
    """Return whether OpenShell provider settings already match.

    Default-on/off state is intentionally ignored: `hermes openshell enable
    --default` may flip the default while reusing an already validated provider
    setup.
    """

    keys = [
        key
        for key in updates
        if key.startswith("sandbox.openshell.") or key in {"sandbox.provider", "sandbox.mode"}
    ]
    if not keys:
        return False
    return all(_values_equal(_get_dotted(config, key), updates[key]) for key in keys)


def _cmd_enable(args):
    previous_config = _load_raw_config()
    if args.install:
        uv = shutil.which("uv")
        if not uv:
            print("uv is required for --install; install uv or install OpenShell manually.", file=sys.stderr)
            sys.exit(1)
        result = subprocess.run([uv, "tool", "install", "-U", "openshell"], text=True)
        if result.returncode != 0:
            sys.exit(result.returncode)

    if not shutil.which("openshell"):
        print("OpenShell CLI not found. Re-run with --install or install it from NVIDIA OpenShell docs.", file=sys.stderr)
        sys.exit(1)

    source = args.source or str(bundled_source_dir())
    updates = {
        "sandbox.enabled": bool(args.default),
        "sandbox.provider": "openshell",
        "sandbox.mode": args.mode,
        "sandbox.openshell.command": "openshell",
        "sandbox.openshell.from": source,
        "sandbox.openshell.policy": args.policy or "",
        "sandbox.openshell.providers": args.provider or [],
        "sandbox.openshell.auto_providers": not args.no_auto_providers,
        "sandbox.openshell.gpu": bool(args.gpu),
        "sandbox.openshell.gateway": args.gateway_name or DEFAULT_GATEWAY_NAME,
        "sandbox.openshell.gateway_endpoint": args.gateway_endpoint or "",
        "sandbox.openshell.gateway_port": int(args.gateway_port or 0),
        "sandbox.openshell.gateway_host": args.gateway_host or "",
        "sandbox.openshell.remote_workspace_dir": "/sandbox",
        "sandbox.openshell.remote_agent_workspace_dir": "/agent",
        "sandbox.openshell.sync_hermes_files": False,
        "sandbox.openshell.extra_syncs": args.extra_sync or [],
    }
    if args.default:
        updates["terminal.backend"] = "openshell"
    already_configured = _is_already_configured(previous_config, updates)
    _update_config(updates)

    if already_configured:
        print("OpenShell sandbox provider was already configured with these settings.")
    else:
        print("OpenShell sandbox provider configured.")
    if args.default:
        print("OpenShell is now the default Hermes tool execution backend.")
    else:
        print("Host execution remains the default. Use `hermes chat --sandbox=openshell` to opt in.")
    print("Disable it with `hermes openshell disable`.")

    should_smoke = not args.no_smoke and (args.smoke or args.install or not already_configured)
    if not should_smoke:
        print("Skipping smoke validation because this OpenShell setup is already configured.")
        print("Use `hermes openshell enable --smoke` to validate the smoke sandbox now.")
        return

    if not args.no_smoke:
        cfg = _load_openshell_config()
        cfg["source"] = source
        cfg["mode"] = args.mode
        cfg["providers"] = args.provider or []
        cfg["policy"] = args.policy or ""
        cfg["auto_providers"] = not args.no_auto_providers
        cfg["gpu"] = bool(args.gpu)
        cfg["gateway"] = args.gateway_name or DEFAULT_GATEWAY_NAME
        cfg["gateway_endpoint"] = args.gateway_endpoint or ""
        cfg["gateway_port"] = int(args.gateway_port or 0)
        cfg["gateway_host"] = args.gateway_host or ""
        cfg["extra_syncs"] = args.extra_sync or []
        cfg = normalize_openshell_config(cfg)
        name = build_sandbox_name(args.scope)
        print(
            f"Validating smoke sandbox {name!r} on gateway {cfg['gateway']!r} "
            f"(port {cfg['gateway_port'] or 'external'})."
        )
        print("First run can take several minutes while OpenShell starts Docker/K3s and builds the image.")
        result = _ensure_sandbox(cfg, name, stream=True)
        if result.returncode != 0:
            print(result.stderr or result.stdout, file=sys.stderr)
            sys.exit(result.returncode)
        print(f"Smoke sandbox ready: {name}")


def _cmd_disable(args):
    _update_config(
        {
            "sandbox.enabled": False,
            "sandbox.provider": "host",
            "terminal.backend": "local",
        }
    )
    print("OpenShell sandboxing disabled; Hermes tool execution defaults to host/local.")
    print("Existing OpenShell sandboxes were not deleted. Use `hermes openshell list` and `hermes openshell delete` to manage them.")


def _cmd_status(args):
    cfg = _load_openshell_config(args)
    print("Hermes OpenShell sandbox provider")
    print(f"  openshell: {shutil.which(cfg['command']) or 'not found'}")
    print(f"  source:    {cfg['source']}")
    print(f"  mode:      {cfg['mode']}")
    print(f"  gateway:   {cfg['gateway']}")
    print(f"  port:      {cfg['gateway_port'] or 'external'}")
    print(f"  workspace: {cfg['remote_workspace_dir']}")
    result = run_openshell_cli(cfg, ["status"], timeout=30)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)


def _cmd_create(args):
    return _run_for_sandbox(args, lambda cfg, name: _ensure_sandbox(cfg, name))


def _cmd_shell(args):
    cfg = _load_openshell_config(args)
    name = _resolve_sandbox_name(args)
    argv = [*_base_cli(cfg), "sandbox", "connect", name]
    os.execvp(argv[0], argv)


def _cmd_exec(args):
    command = list(getattr(args, "exec_command", []) or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("Missing command for `hermes openshell exec`.", file=sys.stderr)
        sys.exit(2)
    return _run_for_sandbox(
        args,
        lambda cfg, name: _exec_via_ssh_config(cfg, name, command),
    )


def _run_passthrough(openshell_args: list[str], args=None):
    cfg = _load_openshell_config(args)
    result = run_openshell_cli(cfg, openshell_args, timeout=120)
    _print_result(result)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _run_for_sandbox(args, fn):
    cfg = _load_openshell_config(args)
    name = _resolve_sandbox_name(args)
    result = fn(cfg, name)
    _print_result(result)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _ensure_sandbox(cfg: dict[str, Any], name: str, *, stream: bool = False):
    if stream and cfg.get("gateway_port") and not cfg.get("gateway_endpoint"):
        print(f"Starting or validating OpenShell gateway {cfg['gateway']!r}...")
    gateway = ensure_openshell_gateway(cfg, capture_output=not stream)
    if gateway is not None and gateway.returncode != 0:
        return gateway

    get_result = run_openshell_cli(cfg, ["sandbox", "get", name], timeout=30)
    if get_result.returncode == 0:
        if stream:
            print(f"Smoke sandbox already exists: {name}")
        return get_result
    source = prepare_openshell_source(cfg, cwd=os.getcwd(), stream=stream)
    create_args = ["sandbox", "create", "--name", name, "--from", source]
    if cfg.get("policy"):
        create_args.extend(["--policy", cfg["policy"]])
    if cfg.get("gpu"):
        create_args.append("--gpu")
    create_args.append("--auto-providers" if cfg.get("auto_providers", True) else "--no-auto-providers")
    for provider in cfg.get("providers", []):
        create_args.extend(["--provider", provider])
    create_args.extend(["--", "true"])
    if stream:
        print(f"Creating OpenShell sandbox {name!r}...")
        return _run_openshell_cli_stream(cfg, create_args, timeout=max(cfg["timeout_seconds"], 300))
    return run_openshell_cli(cfg, create_args, timeout=max(cfg["timeout_seconds"], 300))


def _run_openshell_cli_stream(cfg: dict[str, Any], args: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*_base_cli(cfg), *args],
        cwd=os.getcwd(),
        text=True,
        timeout=timeout,
    )


def _exec_via_ssh_config(cfg: dict[str, Any], name: str, command: list[str]) -> subprocess.CompletedProcess:
    ssh_config = run_openshell_cli(cfg, ["sandbox", "ssh-config", name], timeout=60)
    if ssh_config.returncode != 0:
        return ssh_config

    with tempfile.TemporaryDirectory(prefix="hermes-openshell-cli-") as tmp:
        config_path = Path(tmp) / "ssh_config"
        config_path.write_text(ssh_config.stdout, encoding="utf-8")
        host = name
        for line in ssh_config.stdout.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("host "):
                parts = stripped.split()
                if len(parts) >= 2 and "*" not in parts[1]:
                    host = parts[1]
                    break
        remote_command = " ".join(shlex.quote(part) for part in command)
        return subprocess.run(
            ["ssh", "-F", str(config_path), "-T", host, remote_command],
            capture_output=True,
            text=True,
            timeout=cfg["timeout_seconds"],
        )


def _resolve_sandbox_name(args) -> str:
    if getattr(args, "name", ""):
        return args.name
    return build_sandbox_name(getattr(args, "scope", "") or "default")


def _base_cli(cfg: dict[str, Any]) -> list[str]:
    argv = [cfg["command"]]
    if cfg.get("gateway_endpoint"):
        argv.extend(["--gateway-endpoint", cfg["gateway_endpoint"]])
    elif cfg.get("gateway"):
        argv.extend(["--gateway", cfg["gateway"]])
    return argv


def _print_result(result: subprocess.CompletedProcess):
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)


def _load_raw_config() -> dict:
    from hermes_cli.config import get_config_path

    path = get_config_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _update_config(updates: dict[str, Any]) -> None:
    from hermes_cli.config import ensure_hermes_home, get_config_path, save_env_value
    from utils import atomic_yaml_write

    config = _load_raw_config()
    for key, value in updates.items():
        current = config
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(current.get(part), dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    ensure_hermes_home()
    atomic_yaml_write(get_config_path(), config, sort_keys=False)

    env_sync = {
        "terminal.backend": "TERMINAL_ENV",
        "sandbox.enabled": "HERMES_SANDBOX_ENABLED",
        "sandbox.provider": "HERMES_SANDBOX_PROVIDER",
        "sandbox.mode": "HERMES_SANDBOX_MODE",
        "sandbox.openshell.command": "HERMES_OPENSHELL_COMMAND",
        "sandbox.openshell.from": "HERMES_SANDBOX_SOURCE",
        "sandbox.openshell.policy": "HERMES_OPENSHELL_POLICY",
        "sandbox.openshell.providers": "HERMES_OPENSHELL_PROVIDERS",
        "sandbox.openshell.auto_providers": "HERMES_OPENSHELL_AUTO_PROVIDERS",
        "sandbox.openshell.gpu": "HERMES_OPENSHELL_GPU",
        "sandbox.openshell.gateway": "HERMES_OPENSHELL_GATEWAY",
        "sandbox.openshell.gateway_endpoint": "HERMES_OPENSHELL_GATEWAY_ENDPOINT",
        "sandbox.openshell.gateway_port": "HERMES_OPENSHELL_GATEWAY_PORT",
        "sandbox.openshell.gateway_host": "HERMES_OPENSHELL_GATEWAY_HOST",
        "sandbox.openshell.sync_hermes_files": "HERMES_OPENSHELL_SYNC_HERMES_FILES",
        "sandbox.openshell.extra_syncs": "HERMES_SANDBOX_EXTRA_SYNCS",
    }
    for key, env_var in env_sync.items():
        if key in updates:
            value = updates[key]
            save_env_value(env_var, json.dumps(value) if isinstance(value, list) else str(value))


def _env_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, list):
                return [str(item) for item in loaded]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in stripped.split(",") if part.strip()]


def _load_openshell_config(args=None) -> dict[str, Any]:
    raw = _load_raw_config()
    sandbox = raw.get("sandbox") if isinstance(raw.get("sandbox"), dict) else {}
    openshell = sandbox.get("openshell") if isinstance(sandbox.get("openshell"), dict) else {}
    cfg = {
        "mode": sandbox.get("mode", "mirror"),
        "source": openshell.get("from") or openshell.get("source") or str(bundled_source_dir()),
        "command": openshell.get("command", "openshell"),
        "policy": openshell.get("policy", ""),
        "providers": openshell.get("providers", []),
        "auto_providers": openshell.get("auto_providers", True),
        "gpu": openshell.get("gpu", False),
        "gateway": openshell.get("gateway", ""),
        "gateway_endpoint": openshell.get("gateway_endpoint", ""),
        "gateway_port": openshell.get("gateway_port", DEFAULT_GATEWAY_PORT),
        "gateway_host": openshell.get("gateway_host", ""),
        "remote_workspace_dir": openshell.get("remote_workspace_dir", "/sandbox"),
        "remote_agent_workspace_dir": openshell.get("remote_agent_workspace_dir", "/agent"),
        "timeout_seconds": openshell.get("timeout_seconds", 120),
        "mirror_excludes": openshell.get("mirror_excludes", DEFAULT_MIRROR_EXCLUDES),
        "extra_syncs": openshell.get("extra_syncs", []),
        "sync_hermes_files": openshell.get("sync_hermes_files", False),
    }
    env_map = {
        "HERMES_OPENSHELL_COMMAND": "command",
        "HERMES_SANDBOX_SOURCE": "source",
        "HERMES_OPENSHELL_SOURCE": "source",
        "HERMES_OPENSHELL_POLICY": "policy",
        "HERMES_OPENSHELL_GATEWAY": "gateway",
        "HERMES_OPENSHELL_GATEWAY_ENDPOINT": "gateway_endpoint",
        "HERMES_OPENSHELL_GATEWAY_PORT": "gateway_port",
        "HERMES_OPENSHELL_GATEWAY_HOST": "gateway_host",
        "HERMES_SANDBOX_MODE": "mode",
        "HERMES_SANDBOX_TIMEOUT_SECONDS": "timeout_seconds",
        "HERMES_SANDBOX_EXTRA_SYNCS": "extra_syncs",
    }
    for env_var, key in env_map.items():
        value = os.getenv(env_var)
        if value not in (None, ""):
            cfg[key] = value
    if os.getenv("HERMES_OPENSHELL_PROVIDERS"):
        cfg["providers"] = _env_list(os.environ["HERMES_OPENSHELL_PROVIDERS"])
    if os.getenv("HERMES_OPENSHELL_AUTO_PROVIDERS"):
        cfg["auto_providers"] = _env_truthy(os.environ["HERMES_OPENSHELL_AUTO_PROVIDERS"])
    if os.getenv("HERMES_OPENSHELL_GPU"):
        cfg["gpu"] = _env_truthy(os.environ["HERMES_OPENSHELL_GPU"])
    if os.getenv("HERMES_OPENSHELL_SYNC_HERMES_FILES"):
        cfg["sync_hermes_files"] = _env_truthy(os.environ["HERMES_OPENSHELL_SYNC_HERMES_FILES"])
    if args is not None:
        cli_overrides = {
            "source": getattr(args, "source", ""),
            "gateway": getattr(args, "gateway_name", ""),
            "gateway_endpoint": getattr(args, "gateway_endpoint", ""),
            "gateway_port": getattr(args, "gateway_port", None),
            "gateway_host": getattr(args, "gateway_host", ""),
            "timeout_seconds": getattr(args, "timeout_seconds", None),
        }
        for key, value in cli_overrides.items():
            if value not in (None, ""):
                cfg[key] = value
    return normalize_openshell_config(cfg)
