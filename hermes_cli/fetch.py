"""Fastfetch-style runtime overview for Hermes Agent."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli import __release_date__, __version__
from hermes_cli.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: float = 2.0) -> str:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _compact_path(path: Path) -> str:
    home = Path.home()
    try:
        resolved = str(path)
        home_s = str(home)
        if resolved == home_s:
            return "~"
        if resolved.startswith(home_s + os.sep):
            return "~" + resolved[len(home_s):]
    except Exception:
        pass
    return str(path)


def _profile_name(hermes_home: Path) -> str:
    env_profile = os.getenv("HERMES_PROFILE", "").strip()
    if env_profile:
        return env_profile
    parts = hermes_home.parts
    if "profiles" in parts:
        try:
            return parts[parts.index("profiles") + 1]
        except Exception:
            pass
    return "default"


def _persona_name(config: dict[str, Any], hermes_home: Path) -> str:
    candidates = [
        config.get("personality"),
        config.get("persona"),
        config.get("soul"),
    ]
    for section in ("profile", "agent"):
        value = config.get(section)
        if isinstance(value, dict):
            candidates.extend([value.get("personality"), value.get("persona")])

    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
        if isinstance(item, dict):
            for key in ("display_name", "name", "id"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    for filename in ("SOUL.md", "soul.md", "PERSONA.md", "persona.md"):
        if (hermes_home / filename).exists():
            return Path(filename).stem
    return "default"


def _model_info(config: dict[str, Any]) -> dict[str, str]:
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        return {
            "provider": str(model_cfg.get("provider") or "auto"),
            "name": str(
                model_cfg.get("default")
                or model_cfg.get("model")
                or model_cfg.get("name")
                or "not set"
            ),
        }
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"provider": "auto", "name": model_cfg.strip()}
    return {"provider": "auto", "name": "not set"}


def _git_info() -> dict[str, Any]:
    branch = _run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT) or "unknown"
    commit = _run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT) or "unknown"
    dirty = bool(_run(["git", "status", "--short"], cwd=PROJECT_ROOT))
    return {"branch": branch, "commit": commit, "dirty": dirty, "path": _compact_path(PROJECT_ROOT)}


def _update_summary() -> str:
    upstream = _run(["git", "rev-parse", "--short", "origin/main"], cwd=PROJECT_ROOT)
    local = _run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT)
    if not upstream or not local:
        return "unknown"
    if upstream == local:
        return "up to date"
    behind = _run(["git", "rev-list", "--count", "HEAD..origin/main"], cwd=PROJECT_ROOT)
    ahead = _run(["git", "rev-list", "--count", "origin/main..HEAD"], cwd=PROJECT_ROOT)
    bits: list[str] = []
    if behind and behind != "0":
        bits.append(f"behind {behind}")
    if ahead and ahead != "0":
        bits.append(f"ahead {ahead}")
    return ", ".join(bits) if bits else f"local {local} / origin {upstream}"


def _gateway_summary() -> str:
    try:
        from hermes_cli.gateway import get_gateway_runtime_snapshot

        snapshot = get_gateway_runtime_snapshot()
        if snapshot.running:
            pids = ",".join(str(pid) for pid in snapshot.gateway_pids[:2])
            manager = snapshot.manager
            if snapshot.has_process_service_mismatch:
                manager = "manual"
            return f"running ({manager}, pid {pids})"
        return f"stopped ({snapshot.manager})"
    except Exception:
        return "unknown"


def _platforms() -> list[str]:
    try:
        from gateway.status import read_runtime_status

        runtime = read_runtime_status() or {}
        platforms = runtime.get("platforms") or {}
        if isinstance(platforms, dict) and platforms:
            names: list[str] = []
            for name, state in sorted(platforms.items()):
                if not isinstance(state, dict):
                    names.append(str(name))
                    continue
                status = state.get("state") or state.get("status") or "configured"
                suffix = "✓" if status == "connected" else str(status)
                names.append(f"{name} {suffix}")
            return names
    except Exception:
        pass

    checks = {
        "telegram": "TELEGRAM_BOT_TOKEN",
        "discord": "DISCORD_BOT_TOKEN",
        "slack": "SLACK_BOT_TOKEN",
        "whatsapp": "WHATSAPP_ENABLED",
        "signal": "SIGNAL_HTTP_URL",
        "email": "EMAIL_ADDRESS",
        "matrix": "MATRIX_HOMESERVER_URL",
        "homeassistant": "HASS_TOKEN",
    }
    return [name for name, env in checks.items() if os.getenv(env)]


def _cron_summary(hermes_home: Path) -> dict[str, int | str]:
    jobs_path = hermes_home / "cron" / "jobs.json"
    if not jobs_path.exists():
        return {"active": 0, "paused": 0, "total": 0}
    try:
        data = json.loads(jobs_path.read_text(encoding="utf-8"))
        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        active = sum(1 for job in jobs if job.get("enabled", True))
        return {"active": active, "paused": len(jobs) - active, "total": len(jobs)}
    except Exception as exc:
        return {"error": str(exc)}


def _count_skills(hermes_home: Path) -> int:
    skills_dir = hermes_home / "skills"
    if not skills_dir.is_dir():
        return 0
    return sum(1 for path in skills_dir.rglob("SKILL.md") if "/.archive/" not in str(path))


def _tools_summary(config: dict[str, Any]) -> str:
    tools_cfg = config.get("tools")
    if isinstance(tools_cfg, dict):
        enabled = tools_cfg.get("enabled") or tools_cfg.get("toolsets") or tools_cfg.get("default")
        if isinstance(enabled, list):
            return f"{len(enabled)} configured"
    try:
        import toolsets

        return f"{len(getattr(toolsets, 'TOOLSETS', {}))} available"
    except Exception:
        return "unknown"


def _memory_summary(config: dict[str, Any]) -> str:
    mem = config.get("memory")
    if not isinstance(mem, dict):
        return "unknown"
    provider = mem.get("provider") or "built-in"
    return f"{'enabled' if mem.get('memory_enabled', True) else 'disabled'} ({provider})"


def collect_fetch_info() -> dict[str, Any]:
    """Collect a secret-safe Hermes runtime overview."""
    hermes_home = get_hermes_home()
    config = load_config()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "release_date": __release_date__,
        "repo": _git_info(),
        "profile": _profile_name(hermes_home),
        "persona": _persona_name(config, hermes_home),
        "hermes_home": _compact_path(hermes_home),
        "model": _model_info(config),
        "gateway": _gateway_summary(),
        "platforms": _platforms(),
        "tools": _tools_summary(config),
        "skills": _count_skills(hermes_home),
        "cron": _cron_summary(hermes_home),
        "memory": _memory_summary(config),
        "mcp_servers": (
            len(config.get("mcp", {}).get("servers", {}))
            if isinstance(config.get("mcp"), dict)
            else 0
        ),
        "host": {
            "system": platform.system() or "unknown",
            "release": platform.release() or "unknown",
            "machine": platform.machine() or "unknown",
            "termux": bool(
                os.getenv("PREFIX", "").endswith("com.termux/files/usr")
                or os.getenv("TERMUX_VERSION")
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "node": _run(["node", "--version"]) or "not found",
            "npm": _run(["npm", "--version"]) or "not found",
        },
        "update": _update_summary(),
    }


_PERSONA_AVATAR = ["   /\\_/\\   ", "  ( o.o )  ", "   > ^ <   ", " persona   "]
_HERMES_AVATAR = ["  __/\\__  ", " /  Hermes\\", " \\__  __/", "    \\/    "]


def _cron_text(cron: dict[str, Any]) -> str:
    if "error" in cron:
        return f"error: {cron['error']}"
    return f"{cron.get('active', 0)} active / {cron.get('total', 0)} total"


def render_fetch_info(
    info: dict[str, Any], *, plain: bool = False, no_persona: bool = False, compact: bool = False
) -> str:
    """Render collected fetch info as text."""
    if plain:
        rows = [
            ("Hermes", f"v{info['version']} ({info['release_date']})"),
            ("Repo", f"{info['repo']['branch']}@{info['repo']['commit']}{' dirty' if info['repo']['dirty'] else ''}"),
            ("Profile", info["profile"]),
            ("Persona", "disabled" if no_persona else info["persona"]),
            ("Model", f"{info['model']['provider']} / {info['model']['name']}"),
            ("Gateway", info["gateway"]),
            ("Platforms", ", ".join(info["platforms"]) or "none configured"),
            ("Tools", info["tools"]),
            ("Skills", str(info["skills"])),
            ("Cron", _cron_text(info["cron"])),
            ("Memory", info["memory"]),
            ("MCP", f"{info['mcp_servers']} configured"),
            (
                "Host",
                f"{info['host']['system']} {info['host']['release']} {info['host']['machine']}"
                f"{' · Termux' if info['host']['termux'] else ''}",
            ),
            (
                "Runtime",
                f"Python {info['runtime']['python']} · Node {info['runtime']['node']} · npm {info['runtime']['npm']}",
            ),
            ("Update", info["update"]),
        ]
        return "\n".join(f"{key + ':':<11} {value}" for key, value in rows)

    avatar = _HERMES_AVATAR if no_persona else _PERSONA_AVATAR
    title = "Hermes Agent"
    subtitle = "neutral overview" if no_persona else f"persona-aware · {info['persona']}"
    rows = [
        ("Version", f"v{info['version']} · {info['repo']['branch']}@{info['repo']['commit']}{' *' if info['repo']['dirty'] else ''}"),
        ("Profile", f"{info['profile']} · {info['hermes_home']}"),
        ("Model", f"{info['model']['provider']} / {info['model']['name']}"),
        ("Gateway", info["gateway"]),
        ("Platforms", ", ".join(info["platforms"]) or "none configured"),
        ("Tools", info["tools"]),
        ("Skills", f"{info['skills']} installed"),
        ("Cron", _cron_text(info["cron"])),
        ("Memory", info["memory"]),
        ("MCP", f"{info['mcp_servers']} configured"),
        ("Host", f"{info['host']['system']} {info['host']['release']}{' · Termux' if info['host']['termux'] else ''}"),
        ("Update", info["update"]),
    ]
    if compact:
        rows = rows[:5] + [rows[7], rows[-1]]

    width = max(len(line) for line in avatar)
    lines: list[str] = []
    for idx, (key, value) in enumerate([(title, subtitle), *rows]):
        left = avatar[idx] if idx < len(avatar) else ""
        lines.append(f"{left:<{width}}  {key:<9} {value}")
    return "\n".join(lines)


def run_fetch(args: argparse.Namespace) -> int:
    info = collect_fetch_info()
    if getattr(args, "json", False):
        print(json.dumps(info, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(
            render_fetch_info(
                info,
                plain=getattr(args, "plain", False),
                no_persona=getattr(args, "no_persona", False),
                compact=getattr(args, "compact", False),
            )
        )
    return 0


def render_fetch_slash_args(raw_args: str = "") -> str:
    """Render /fetch slash-command output from a small shell-like arg string.

    Supported forms intentionally mirror the CLI flags while allowing concise
    slash usage:

    - /fetch
    - /fetch compact
    - /fetch --plain
    - /fetch json --no-persona
    """
    try:
        tokens = shlex.split(raw_args or "")
    except ValueError as exc:
        return f"Usage: /fetch [text|plain|compact|json] [--no-persona]\nParse error: {exc}"

    fmt = "text"
    no_persona = False
    for token in tokens:
        value = token.strip().lower()
        if value in {"--no-persona", "--neutral"}:
            no_persona = True
        elif value in {"--text", "text"}:
            fmt = "text"
        elif value in {"--plain", "plain"}:
            fmt = "plain"
        elif value in {"--compact", "compact", "short"}:
            fmt = "compact"
        elif value in {"--json", "json"}:
            fmt = "json"
        elif value in {"--help", "-h", "help"}:
            return "Usage: /fetch [text|plain|compact|json] [--no-persona]"
        else:
            return (
                f"Unknown /fetch argument: {token}\n"
                "Usage: /fetch [text|plain|compact|json] [--no-persona]"
            )

    info = collect_fetch_info()
    if fmt == "json":
        return json.dumps(info, indent=2, ensure_ascii=False, sort_keys=True)
    return render_fetch_info(
        info,
        plain=(fmt == "plain"),
        compact=(fmt == "compact"),
        no_persona=no_persona,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show a fastfetch-style Hermes runtime overview")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--plain", action="store_true", help="Use neutral plain-text output")
    parser.add_argument("--no-persona", action="store_true", help="Disable persona styling")
    parser.add_argument("--compact", action="store_true", help="Show a shorter overview")
    return run_fetch(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
