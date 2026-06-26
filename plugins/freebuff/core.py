"""Hermes bridge for Codebuff Freebuff (https://github.com/CodebuffAI/codebuff)."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hermes_constants import display_hermes_home, get_hermes_home

from . import proxy as proxy_mgr
from . import token as token_bridge

PLUGIN_ID = "freebuff"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
NPM_PACKAGE = "freebuff"
MANICODE_CONFIG_DIR = Path.home() / ".config" / "manicode"
METADATA_PATH = MANICODE_CONFIG_DIR / "freebuff-metadata.json"
GITHUB_URL = "https://github.com/CodebuffAI/codebuff"
NPM_URL = "https://www.npmjs.com/package/freebuff"
DOCS_URL = "https://freebuff.com/cli"


def _json_ok(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def plugin_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        cfg = {}
    plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
    section = plugins.get(PLUGIN_ID) if isinstance(plugins.get(PLUGIN_ID), dict) else {}
    return dict(section)


def binary_name() -> str:
    return "freebuff.exe" if os.name == "nt" else "freebuff"


def default_binary_path() -> Path:
    override = str(plugin_config().get("binary_path") or "").strip()
    if override:
        return Path(override).expanduser()
    return MANICODE_CONFIG_DIR / binary_name()


def resolve_workdir(explicit: str | None = None) -> Path:
    if explicit and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    cfg = plugin_config()
    raw = str(cfg.get("workdir") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
        terminal = loaded.get("terminal") if isinstance(loaded.get("terminal"), dict) else {}
        cwd = str(terminal.get("cwd") or "").strip()
        if cwd:
            return Path(cwd).expanduser().resolve()
    except Exception:
        pass
    return Path.cwd().resolve()


def _which(name: str) -> str | None:
    found = shutil.which(name)
    return found or None


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: float = 120.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s", "cmd": cmd}
    except FileNotFoundError:
        return {"ok": False, "error": f"command not found: {cmd[0]}", "cmd": cmd}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}


def _read_metadata() -> dict[str, Any]:
    if not METADATA_PATH.is_file():
        return {}
    try:
        payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _binary_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "size_bytes": 0}
    try:
        stat = path.stat()
        return {"exists": True, "size_bytes": stat.st_size}
    except OSError as exc:
        return {"exists": False, "size_bytes": 0, "error": str(exc)}


def _node_info() -> dict[str, Any]:
    node = _which("node")
    npm = _which("npm")
    npx = _which("npx")
    node_version = ""
    npm_version = ""
    if node:
        result = _run([node, "--version"], timeout=10.0)
        if result.get("ok"):
            node_version = result.get("stdout") or ""
    if npm:
        result = _run([npm, "--version"], timeout=10.0)
        if result.get("ok"):
            npm_version = result.get("stdout") or ""
    return {
        "node": node,
        "node_version": node_version,
        "npm": npm,
        "npm_version": npm_version,
        "npx": npx,
    }


def _plugin_enabled(config: dict[str, Any] | None = None) -> bool:
    cfg = config if config is not None else _load_yaml(get_hermes_home() / "config.yaml")
    plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
    enabled = plugins.get("enabled") if isinstance(plugins.get("enabled"), list) else []
    return PLUGIN_ID in enabled


def _ensure_plugin_enabled(config: dict[str, Any]) -> bool:
    plugins = config.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        plugins = {}
        config["plugins"] = plugins
    enabled = plugins.setdefault("enabled", [])
    if not isinstance(enabled, list):
        enabled = []
        plugins["enabled"] = enabled
    if PLUGIN_ID in enabled:
        return False
    enabled.append(PLUGIN_ID)
    return True


def _provider_registered() -> tuple[bool, str]:
    try:
        from providers import get_provider_profile

        profile = get_provider_profile("freebuff")
    except Exception as exc:
        return False, f"provider import failed: {exc}"
    if profile is None:
        return False, "freebuff model-provider profile not registered"
    return True, profile.display_name or "Freebuff"


def _ensure_fallback_entry(config: dict[str, Any]) -> bool:
    entries = config.setdefault("fallback_providers", [])
    if not isinstance(entries, list):
        entries = []
        config["fallback_providers"] = entries
    for entry in entries:
        if isinstance(entry, dict) and entry.get("provider") == PLUGIN_ID:
            return False
    entries.insert(0, {"provider": PLUGIN_ID, "model": DEFAULT_MODEL})
    return True


def _ensure_model_defaults(config: dict[str, Any], *, force_default: bool = False) -> tuple[bool, list[str]]:
    model = config.setdefault("model", {})
    if not isinstance(model, dict):
        model = {}
        config["model"] = model
    changed = False
    changes: list[str] = []
    base_url = proxy_mgr.proxy_base_url()
    if model.get("provider") != PLUGIN_ID:
        model["provider"] = PLUGIN_ID
        changed = True
        changes.append(f"model.provider={PLUGIN_ID}")
    current_default = str(model.get("default") or "").strip()
    if force_default or not current_default:
        if model.get("default") != DEFAULT_MODEL:
            model["default"] = DEFAULT_MODEL
            changed = True
            changes.append(f"model.default={DEFAULT_MODEL}")
    if str(model.get("base_url") or "").strip() != base_url:
        model["base_url"] = base_url
        changed = True
        changes.append(f"model.base_url={base_url}")
    return changed, changes


def _ensure_toolset(config: dict[str, Any]) -> bool:
    tools = config.setdefault("tools", {})
    if not isinstance(tools, dict):
        tools = {}
        config["tools"] = tools
    cli_tools = tools.setdefault("cli", {})
    if not isinstance(cli_tools, dict):
        cli_tools = {}
        tools["cli"] = cli_tools
    enabled = cli_tools.setdefault("enabled", [])
    if not isinstance(enabled, list):
        enabled = []
        cli_tools["enabled"] = enabled
    if "freebuff" in enabled:
        return False
    enabled.append("freebuff")
    return True


def launch_command(*, workdir: Path | None = None) -> list[str]:
    """Resolve the best command vector to start Freebuff interactively."""
    cfg = plugin_config()
    package = str(cfg.get("npm_package") or NPM_PACKAGE).strip() or NPM_PACKAGE
    wd = workdir or resolve_workdir()
    binary = default_binary_path()
    if binary.is_file():
        return [str(binary)]
    cli = _which("freebuff")
    if cli:
        return [cli]
    npx = _which("npx")
    if npx:
        return [npx, "--yes", package]
    npm = _which("npm")
    if npm:
        return [npm, "exec", "--yes", package]
    return [package]


def status() -> dict[str, Any]:
    binary = default_binary_path()
    metadata = _read_metadata()
    node_info = _node_info()
    cli_path = _which("freebuff")
    config_path = get_hermes_home() / "config.yaml"
    config = _load_yaml(config_path)
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    registered, provider_label = _provider_registered()
    proxy_state = proxy_mgr.proxy_status()
    upstream_token = token_bridge.resolve_upstream_token()
    proxy_key = token_bridge._read_env_key(token_bridge.ENV_PROXY_KEY)
    return {
        "ok": True,
        "plugin_id": PLUGIN_ID,
        "platform": platform.platform(),
        "arch": platform.machine(),
        "github": GITHUB_URL,
        "npm": NPM_URL,
        "docs": DOCS_URL,
        "config_dir": str(MANICODE_CONFIG_DIR),
        "binary_path": str(binary),
        "binary": _binary_stats(binary),
        "metadata": metadata,
        "cli_on_path": cli_path,
        "launch_command": launch_command(),
        "workdir": str(resolve_workdir()),
        "node": node_info,
        "plugin_enabled": _plugin_enabled(),
        "hermes_home": display_hermes_home(),
        "provider_registered": registered,
        "provider_label": provider_label,
        "model_provider": model.get("provider"),
        "model_default": model.get("default"),
        "model_base_url": model.get("base_url") or proxy_mgr.proxy_base_url(),
        "upstream_token_set": bool(upstream_token),
        "proxy_api_key_set": bool(proxy_key),
        "proxy": proxy_state,
        "interactive_only": True,
        "note": (
            "Freebuff TUI: `hermes freebuff run`. Hermes AI inference: "
            "`hermes freebuff connect --apply-model` (local OpenAI-compatible proxy)."
        ),
    }


def doctor() -> dict[str, Any]:
    st = status()
    issues: list[str] = []
    warnings: list[str] = []

    node_info = st.get("node") if isinstance(st.get("node"), dict) else {}
    if not node_info.get("node"):
        issues.append("Node.js not found on PATH (required for npm install -g freebuff)")
    if not node_info.get("npm") and not node_info.get("npx"):
        warnings.append("npm/npx not found; global install and npx fallback unavailable")

    binary = st.get("binary") if isinstance(st.get("binary"), dict) else {}
    if not binary.get("exists") and not st.get("cli_on_path"):
        issues.append(
            "Freebuff binary/CLI missing. Run `hermes freebuff install` or "
            "`npm install -g freebuff` once, then `freebuff` to download the native binary."
        )
    elif binary.get("exists") and int(binary.get("size_bytes") or 0) < 1024:
        warnings.append("Downloaded binary looks unusually small; consider reinstalling")

    if not _plugin_enabled():
        warnings.append(f"Plugin not in plugins.enabled — run `hermes freebuff setup`")

    wd = resolve_workdir()
    if not wd.is_dir():
        warnings.append(f"Configured workdir does not exist: {wd}")

    if not st.get("upstream_token_set"):
        issues.append(
            "Freebuff auth token missing. Run `hermes freebuff run`, complete GitHub login, "
            "then `hermes freebuff connect` (or set FREEBUFF_TOKEN in ~/.hermes/.env)."
        )
    if not st.get("provider_registered"):
        warnings.append("freebuff model-provider not registered — restart Hermes after plugin enable")

    proxy_state = st.get("proxy") if isinstance(st.get("proxy"), dict) else {}
    if not proxy_state.get("installed"):
        warnings.append(
            "freebuff2api proxy not installed — `hermes freebuff proxy install` "
            "(community OpenAI bridge; not official Codebuff)"
        )
    elif not proxy_state.get("running"):
        warnings.append("Local Freebuff proxy not running — `hermes freebuff connect` or `proxy start`")
    elif not (proxy_state.get("probe") or {}).get("ok"):
        warnings.append(
            f"Proxy health check failed: {(proxy_state.get('probe') or {}).get('error', 'unknown')}"
        )

    if st.get("model_provider") != PLUGIN_ID:
        warnings.append(
            "model.provider is not freebuff — run `hermes freebuff connect --apply-model` "
            "to route Hermes chat through the Freebuff free tier"
        )

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "status": st,
    }


def install(*, global_install: bool = True) -> dict[str, Any]:
    npm = _which("npm")
    if not npm:
        return {
            "ok": False,
            "error": "npm not found on PATH. Install Node.js 18+ first.",
            "docs": DOCS_URL,
        }
    package = str(plugin_config().get("npm_package") or NPM_PACKAGE).strip() or NPM_PACKAGE
    cmd = [npm, "install", "-g", package] if global_install else [npm, "install", package]
    result = _run(cmd, timeout=300.0)
    post = status()
    return {
        "ok": bool(result.get("ok")),
        "install": result,
        "status": post,
        "next_steps": [
            f"cd {resolve_workdir()}",
            "freebuff",
        ],
    }


def connect(
    *,
    apply_model: bool = True,
    start_proxy: bool = True,
    install_proxy: bool = True,
    sync_token: bool = True,
) -> dict[str, Any]:
    """Wire Freebuff free tier into Hermes as model.provider=freebuff."""
    steps: dict[str, Any] = {}
    setup_result = setup()
    steps["setup"] = setup_result

    if sync_token:
        steps["token"] = token_bridge.sync_upstream_token_to_env()
        if not steps["token"].get("ok"):
            return {"ok": False, "steps": steps, "doctor": doctor()}

    steps["proxy_key"] = token_bridge.ensure_proxy_api_key()

    if install_proxy and not proxy_mgr.proxy_status().get("installed"):
        steps["proxy_install"] = proxy_mgr.install_proxy()
        if not steps["proxy_install"].get("ok"):
            return {"ok": False, "steps": steps, "doctor": doctor()}

    if start_proxy:
        steps["proxy_start"] = proxy_mgr.start_proxy()
        if not steps["proxy_start"].get("ok"):
            return {"ok": False, "steps": steps, "doctor": doctor()}

    config_path = get_hermes_home() / "config.yaml"
    config = _load_yaml(config_path)
    model_changes: list[str] = []
    if _ensure_fallback_entry(config):
        model_changes.append("fallback_providers[0]=freebuff")
    if apply_model:
        _, defaults = _ensure_model_defaults(config, force_default=True)
        model_changes.extend(defaults)
    if model_changes:
        _save_yaml(config_path, config)
    steps["model_changes"] = model_changes

    diag = doctor()
    return {
        "ok": diag.get("ok", False),
        "steps": steps,
        "doctor": diag,
        "next_steps": [
            "Restart Hermes CLI/gateway if it was already running",
            f"Chat with model.provider=freebuff and model.default={DEFAULT_MODEL}",
            "Monitor ~/.hermes/logs/freebuff-proxy.log if requests fail",
        ],
    }


def setup(*, enable_plugin: bool = True, enable_toolset: bool = True) -> dict[str, Any]:
    config_path = get_hermes_home() / "config.yaml"
    config = _load_yaml(config_path)
    changed: list[str] = []
    if enable_plugin and _ensure_plugin_enabled(config):
        changed.append("plugins.enabled+=freebuff")
    if enable_toolset and _ensure_toolset(config):
        changed.append("tools.cli.enabled+=freebuff")
    if changed:
        _save_yaml(config_path, config)
    return {
        "ok": True,
        "changed": changed,
        "config_path": str(config_path),
        "doctor": doctor(),
    }


def run(*, workdir: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    wd = resolve_workdir(workdir)
    if not wd.is_dir():
        return {"ok": False, "error": f"workdir not found: {wd}"}

    cmd = launch_command(workdir=wd)
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "workdir": str(wd),
            "command": cmd,
            "shell": subprocess.list2cmdline(cmd) if os.name == "nt" else " ".join(cmd),
        }

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(wd),
            env=os.environ.copy(),
        )
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc), "command": cmd, "workdir": str(wd)}
    except OSError as exc:
        return {"ok": False, "error": str(exc), "command": cmd, "workdir": str(wd)}

    return {
        "ok": True,
        "pid": proc.pid,
        "workdir": str(wd),
        "command": cmd,
        "interactive": True,
        "note": "Freebuff TUI attached to this terminal session.",
    }


def sync_skill(*, force: bool = False) -> dict[str, Any]:
    src = Path(__file__).resolve().parent / "skills" / "freebuff"
    dst = get_hermes_home() / "skills" / "freebuff"
    if not (src / "SKILL.md").is_file():
        return {"ok": False, "error": f"missing bundled skill: {src / 'SKILL.md'}"}
    if dst.exists() or dst.is_symlink():
        if not force:
            return {"ok": True, "skipped": True, "path": str(dst)}
        if dst.is_symlink():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return {"ok": True, "action": "copied", "path": str(dst)}


def handle_status(_args: dict[str, Any], **_: Any) -> str:
    return _json_ok(status())


def handle_doctor(_args: dict[str, Any], **_: Any) -> str:
    return _json_ok(doctor())


def handle_launch(args: dict[str, Any], **_: Any) -> str:
    workdir = str(args.get("workdir") or "").strip() or None
    dry_run = bool(args.get("dry_run", True))
    payload = run(workdir=workdir, dry_run=dry_run)
    if payload.get("ok") and dry_run:
        payload["hint"] = (
            "Freebuff requires an interactive TTY. Run the returned command in "
            "`terminal`, or use `hermes freebuff run` from a real console."
        )
    return _json_ok(payload)


def handle_slash(args: str) -> str:
    text = (args or "").strip()
    if not text or text.lower() in {"help", "?"}:
        return (
            "Freebuff plugin — install/run Codebuff's free coding agent.\n"
            "Usage: /freebuff status | doctor | setup | install | run [path] | connect | proxy {status|install|start|stop}\n"
            "Docs: https://freebuff.com/cli"
        )
    parts = text.split(maxsplit=1)
    verb = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if verb == "status":
        return _json_ok(status())
    if verb == "doctor":
        payload = doctor()
        return _json_ok(payload)
    if verb == "setup":
        return _json_ok(setup())
    if verb == "install":
        return _json_ok(install())
    if verb == "run":
        return _json_ok(run(workdir=rest or None, dry_run=False))
    if verb == "skill":
        force = rest.lower() in {"force", "--force"}
        return _json_ok(sync_skill(force=force))
    if verb == "connect":
        apply_model = rest.lower() not in {"no-model", "--no-model"}
        return _json_ok(connect(apply_model=apply_model))
    if verb == "proxy":
        sub = (rest.split(maxsplit=1)[0].lower() if rest else "status")
        if sub in {"status", ""}:
            return _json_ok(proxy_mgr.proxy_status())
        if sub == "install":
            return _json_ok(proxy_mgr.install_proxy())
        if sub == "start":
            return _json_ok(proxy_mgr.start_proxy())
        if sub == "stop":
            return _json_ok(proxy_mgr.stop_proxy())
        return f"Unknown /freebuff proxy subcommand: {sub}"
    return f"Unknown /freebuff subcommand: {verb}. Try /freebuff help."


STATUS_SCHEMA = {
    "name": "freebuff_status",
    "description": (
        "Report Freebuff CLI/binary install state, metadata version, and launch command. "
        "Freebuff is Codebuff's free ad-supported terminal coding agent."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

DOCTOR_SCHEMA = {
    "name": "freebuff_doctor",
    "description": "Run Freebuff preflight checks (Node/npm, downloaded binary, plugin config).",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

LAUNCH_SCHEMA = {
    "name": "freebuff_launch",
    "description": (
        "Resolve or start Freebuff in a project directory. Defaults to dry_run=true "
        "returning the shell command; set dry_run=false only from an interactive CLI session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workdir": {
                "type": "string",
                "description": "Project directory for Freebuff (default: terminal.cwd or cwd).",
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true (default), return launch command without spawning TUI.",
            },
        },
        "additionalProperties": False,
    },
}
