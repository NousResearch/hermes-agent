"""Discover the local Hermes deployment and install an operational HEGI config."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from utils import fast_safe_load


@dataclass(slots=True, frozen=True)
class DiscoveredAgent:
    name: str
    db_path: Path


@dataclass(slots=True, frozen=True)
class DiscoveredEnvironment:
    hermes_root: Path
    runtime_home: Path
    curator_env: Path
    chat_id: str
    professor_user_ids: tuple[str, ...]
    agents: tuple[DiscoveredAgent, ...]
    default_project: str


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _config_has_memory_curator(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        raw = fast_safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    servers = raw.get("mcp_servers", {}) if isinstance(raw, dict) else {}
    return isinstance(servers, dict) and any(
        "memory-forest-curator" in str(name) for name in servers
    )


def hermes_root_for(runtime_home: Path | None = None) -> Path:
    if runtime_home is not None:
        resolved = runtime_home.expanduser().resolve()
        if resolved.parent.name == "profiles":
            return resolved.parent.parent
        if resolved.name == ".hermes":
            return resolved
    return (Path.home() / ".hermes").resolve()


def resolve_runtime_home(
    *, hermes_root: Path | None = None, honor_environment: bool = True
) -> Path:
    """Locate the profile that owns Memory Forest and the HEGI Telegram bot."""
    if honor_environment:
        explicit = os.environ.get("HERMES_HOME", "").strip()
        if explicit:
            return Path(explicit).expanduser().resolve()
    root = (hermes_root or hermes_root_for()).expanduser().resolve()
    exact = root / "profiles" / "memory-curator"
    candidates = [exact]
    profiles = root / "profiles"
    if profiles.is_dir():
        candidates.extend(
            path
            for path in sorted(profiles.iterdir())
            if path.is_dir() and path != exact
        )
    candidates.append(root)
    for candidate in candidates:
        if _config_has_memory_curator(candidate / "config.yaml"):
            return candidate
    for candidate in candidates:
        env = _read_env(candidate / ".env")
        if env.get("TELEGRAM_BOT_TOKEN") and "curator" in candidate.name.lower():
            return candidate
    return root


def _candidate_databases(root: Path, runtime_home: Path) -> list[Path]:
    candidates = [root / "state.db"]
    profiles = root / "profiles"
    if profiles.is_dir():
        candidates.extend(sorted(profiles.glob("*/state.db")))
    runtime_db = (runtime_home / "state.db").resolve()
    return [
        path.resolve()
        for path in candidates
        if path.is_file()
        and (
            runtime_home.resolve() == root.resolve()
            or path.resolve() != runtime_db
        )
    ]


def _database_chats(path: Path) -> list[tuple[str, str, int, float]]:
    try:
        uri = f"file:{path}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=3)
        rows = connection.execute(
            """
            SELECT s.chat_id, COALESCE(s.chat_type, ''), COUNT(m.id),
                   COALESCE(MAX(m.timestamp), 0)
            FROM sessions AS s
            JOIN messages AS m ON m.session_id=s.id
            WHERE s.chat_id IS NOT NULL AND s.chat_id != ''
              AND m.role IN ('user', 'assistant', 'agent')
              AND COALESCE(m.active, 1)=1
            GROUP BY s.chat_id, s.chat_type
            """
        ).fetchall()
        connection.close()
    except (OSError, sqlite3.Error):
        return []
    return [
        (str(chat_id), str(chat_type), int(count), float(latest))
        for chat_id, chat_type, count, latest in rows
    ]


def _configured_chat_candidates(root: Path, runtime_home: Path) -> list[str]:
    candidates: list[str] = []
    for env_path in (runtime_home / ".env", root / ".env"):
        value = _read_env(env_path).get("HEGI_GROUP_CHAT_ID", "").strip()
        if value:
            candidates.append(value)
    old_state = root / "hegi-watch" / "state.json"
    if old_state.is_file():
        try:
            value = str(json.loads(old_state.read_text()).get("last_chat_id", "")).strip()
            if value:
                candidates.append(value)
        except (OSError, ValueError):
            pass
    for config_path in (runtime_home / "config.yaml", root / "config.yaml"):
        if not config_path.is_file():
            continue
        try:
            raw = fast_safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        telegram = raw.get("telegram", {}) if isinstance(raw, dict) else {}
        allowed = telegram.get("allowed_chats", "") if isinstance(telegram, dict) else ""
        values = allowed if isinstance(allowed, list) else re.split(r"[\s,]+", str(allowed))
        candidates.extend(str(item).strip() for item in values if str(item).strip().startswith("-"))
    return candidates


def _discover_chat_id(root: Path, runtime_home: Path, databases: list[Path]) -> str:
    configured = _configured_chat_candidates(root, runtime_home)
    if configured:
        return configured[0]
    scores: dict[str, tuple[int, int, float]] = {}
    for path in databases:
        for chat_id, chat_type, count, latest in _database_chats(path):
            if chat_type not in {"group", "forum", "supergroup"} and not chat_id.startswith("-"):
                continue
            database_count, message_count, newest = scores.get(chat_id, (0, 0, 0.0))
            scores[chat_id] = (
                database_count + 1,
                message_count + count,
                max(newest, latest),
            )
    if not scores:
        return ""
    return max(scores, key=lambda key: scores[key])


def _agent_name(path: Path, root: Path) -> str:
    if path == (root / "state.db").resolve():
        return "HeHe"
    slug = path.parent.name.lower()
    if "codex" in slug or slug in {"heco", "co"}:
        return "HeCo"
    if "claude" in slug or slug in {"hecl", "heclaude"}:
        return "HeClaude"
    return path.parent.name


def _db_has_chat(path: Path, chat_id: str) -> bool:
    return any(item[0] == chat_id and item[2] > 0 for item in _database_chats(path))


def _discover_professor_ids(root: Path, runtime_home: Path) -> tuple[str, ...]:
    env_paths = [runtime_home / ".env", root / ".env"]
    profiles = root / "profiles"
    if profiles.is_dir():
        env_paths.extend(sorted(profiles.glob("*/.env")))
    result: list[str] = []
    for path in env_paths:
        env = _read_env(path)
        values = [
            env.get("TELEGRAM_ALLOWED_USERS", ""),
            env.get("TELEGRAM_HOME_CHANNEL", ""),
        ]
        for value in values:
            for candidate in re.split(r"[\s,]+", value):
                candidate = candidate.strip()
                if candidate.isdigit() and candidate not in result:
                    result.append(candidate)
    return tuple(result)


def _discover_project(runtime_home: Path) -> str:
    config_path = runtime_home / "config.yaml"
    roots: list[Path] = []
    if config_path.is_file():
        text = config_path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"MEMORY_FOREST_ROOT:\s*['\"]?([^'\"\n]+)", text):
            roots.append(Path(match.group(1).strip()).expanduser())
        for match in re.finditer(r"command:\s*['\"]?([^'\"\n]+curator[^'\"\n]*)", text):
            command = Path(match.group(1).strip()).expanduser()
            if command.is_file():
                command_text = command.read_text(encoding="utf-8", errors="ignore")
                root_match = re.search(
                    r"MEMORY_FOREST_ROOT[^}]*:-([^}\"']+)", command_text
                )
                if root_match:
                    roots.append(Path(root_match.group(1).strip()).expanduser())
    roots.append(Path.home() / "memory-forests" / "yw-research")
    for root in roots:
        stm = root / "04 stm"
        if not stm.is_dir():
            continue
        projects = sorted(path.name for path in stm.iterdir() if path.is_dir())
        if "media_aesthetics" in projects:
            return "media_aesthetics"
        if projects:
            return projects[0]
    return "research"


def discover_environment(
    *, hermes_root: Path | None = None, runtime_home: Path | None = None
) -> DiscoveredEnvironment:
    root = (hermes_root or hermes_root_for(runtime_home)).expanduser().resolve()
    runtime = (
        runtime_home.expanduser().resolve()
        if runtime_home is not None
        else resolve_runtime_home(hermes_root=root)
    )
    databases = _candidate_databases(root, runtime)
    chat_id = _discover_chat_id(root, runtime, databases)
    selected = [path for path in databases if chat_id and _db_has_chat(path, chat_id)]
    preferred = [
        path
        for path in selected
        if _agent_name(path, root) in {"HeHe", "HeCo", "HeClaude"}
    ]
    if len(preferred) >= 2:
        selected = preferred
    agents = tuple(
        DiscoveredAgent(_agent_name(path, root), path)
        for path in selected
    )
    return DiscoveredEnvironment(
        hermes_root=root,
        runtime_home=runtime,
        curator_env=runtime / ".env",
        chat_id=chat_id,
        professor_user_ids=_discover_professor_ids(root, runtime),
        agents=agents,
        default_project=_discover_project(runtime),
    )


def discovery_errors(discovery: DiscoveredEnvironment) -> list[str]:
    errors: list[str] = []
    if not discovery.curator_env.is_file():
        errors.append(f"Memory Curator env가 없습니다: {discovery.curator_env}")
    elif not _read_env(discovery.curator_env).get("TELEGRAM_BOT_TOKEN"):
        errors.append(f"TELEGRAM_BOT_TOKEN이 없습니다: {discovery.curator_env}")
    if not discovery.chat_id:
        errors.append("Telegram group chat ID를 탐지하지 못했습니다.")
    if not discovery.professor_user_ids:
        errors.append("Professor Telegram user ID를 탐지하지 못했습니다.")
    if len(discovery.agents) < 2:
        errors.append("대상 chat을 포함한 agent DB를 2개 이상 탐지하지 못했습니다.")
    return errors


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = path.with_name(f"{path.name}.backup-{stamp}")
    counter = 1
    while destination.exists():
        destination = path.with_name(f"{path.name}.backup-{stamp}-{counter}")
        counter += 1
    shutil.copy2(path, destination)
    return destination


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _enable_plugin(runtime_home: Path) -> None:
    config_path = runtime_home / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Hermes profile config가 없습니다: {config_path}")
    raw = fast_safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Hermes profile config가 mapping이 아닙니다: {config_path}")
    plugins = raw.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        plugins = {}
        raw["plugins"] = plugins
    enabled = plugins.get("enabled")
    enabled_list = list(enabled) if isinstance(enabled, list) else []
    if "hegi-telegram" not in enabled_list:
        enabled_list.append("hegi-telegram")
        plugins["enabled"] = enabled_list
        _backup(config_path)
        _atomic_yaml(config_path, raw)


def install(
    *, repo_root: Path, hermes_root: Path | None = None, runtime_home: Path | None = None
) -> dict[str, Any]:
    discovery = discover_environment(
        hermes_root=hermes_root, runtime_home=runtime_home
    )
    errors = discovery_errors(discovery)
    if errors:
        raise RuntimeError("; ".join(errors))
    default_path = repo_root / "hegi" / "config" / "default.yaml"
    defaults = fast_safe_load(default_path.read_text(encoding="utf-8")) or {}
    target = discovery.runtime_home / "hegi" / "config.yaml"
    existing: dict[str, Any] = {}
    if target.is_file():
        loaded = fast_safe_load(target.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            existing = loaded
    config = _deep_merge(defaults, existing)
    config["enabled"] = True
    analysis = config.setdefault("analysis", {})
    default_analysis = defaults.get("analysis", {})
    analysis["prompt_version"] = default_analysis.get("prompt_version", "v2.0.2")
    config["telegram"] = {
        **config.get("telegram", {}),
        "chat_id": discovery.chat_id,
        "curator_env": str(discovery.curator_env),
        "enabled": True,
    }
    config["agents"] = [
        {"name": agent.name, "db_path": str(agent.db_path)}
        for agent in discovery.agents
    ]
    memory = config.setdefault("memory", {})
    memory["professor_user_ids"] = list(discovery.professor_user_ids)
    memory["default_project"] = discovery.default_project
    memory["auto_commit"] = False
    memory["auto_draft"] = False
    memory["require_professor_approval"] = True
    archive = config.setdefault("archive", {})
    archive["local_spool"] = str(discovery.runtime_home / "hegi" / "archive")
    backup = _backup(target)
    _atomic_yaml(target, config)

    plugin_source = repo_root / "hegi" / "plugin"
    plugin_target = discovery.runtime_home / "plugins" / "hegi-telegram"
    plugin_target.mkdir(parents=True, exist_ok=True)
    for name in ("plugin.yaml", "__init__.py"):
        shutil.copy2(plugin_source / name, plugin_target / name)
    _enable_plugin(discovery.runtime_home)
    return {
        "config": str(target),
        "backup": str(backup) if backup else None,
        "runtime_home": str(discovery.runtime_home),
        "chat_id": discovery.chat_id,
        "professor_user_ids": list(discovery.professor_user_ids),
        "agents": [
            {"name": agent.name, "db_path": str(agent.db_path)}
            for agent in discovery.agents
        ],
        "plugin": str(plugin_target),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HEGI environment bootstrap")
    subparsers = parser.add_subparsers(dest="command", required=True)
    locate = subparsers.add_parser("locate-home")
    locate.add_argument("--hermes-root", type=Path)
    discover = subparsers.add_parser("discover")
    discover.add_argument("--hermes-root", type=Path)
    discover.add_argument("--runtime-home", type=Path)
    installer = subparsers.add_parser("install")
    installer.add_argument("--repo-root", required=True, type=Path)
    installer.add_argument("--hermes-root", type=Path)
    installer.add_argument("--runtime-home", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "locate-home":
            print(resolve_runtime_home(hermes_root=args.hermes_root))
            return 0
        if args.command == "discover":
            discovery = discover_environment(
                hermes_root=args.hermes_root, runtime_home=args.runtime_home
            )
            print(
                json.dumps(
                    {
                        "runtime_home": str(discovery.runtime_home),
                        "curator_env": str(discovery.curator_env),
                        "chat_id": discovery.chat_id,
                        "professor_user_ids": list(discovery.professor_user_ids),
                        "agents": [
                            {"name": agent.name, "db_path": str(agent.db_path)}
                            for agent in discovery.agents
                        ],
                        "default_project": discovery.default_project,
                        "errors": discovery_errors(discovery),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if not discovery_errors(discovery) else 2
        result = install(
            repo_root=args.repo_root,
            hermes_root=args.hermes_root,
            runtime_home=args.runtime_home,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"HEGI install failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
