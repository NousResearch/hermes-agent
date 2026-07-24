"""HEGI configuration loading and validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import fast_safe_load


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "enabled": False,
    "telegram": {
        "chat_id": "",
        "curator_env": "",
        "enabled": False,
    },
    "agents": [],
    "episode": {
        "quiet_minutes": 10,
        "max_gap_minutes": 30,
        "minimum_agents": 2,
        "minimum_messages": 4,
        "maximum_messages": 10000,
        "timestamp_bucket_seconds": 3,
        "initial_lookback_minutes": 240,
    },
    "analysis": {
        "model": "",
        "provider": "",
        "max_input_chars": 100000,
        "chunk_chars": 30000,
        "max_output_tokens": 10000,
        "prompt_version": "v2.0.0",
    },
    "archive": {
        "local_spool": "",
        "nas_root": "",
        "sync_when_available": True,
        "markdown": True,
        "json": True,
    },
    "memory": {
        "enabled": True,
        "read_server": "memory-forest-read",
        "search_tool": "",
        "draft_server": "memory-forest-curator-draft",
        "draft_tool": "",
        "auto_commit": False,
        "auto_draft": False,
        "require_professor_approval": True,
        "professor_user_ids": [],
    },
    "daemon": {"poll_seconds": 60},
    "reports": {"telegram": True},
    "v3": {"enabled": False},
    "v4": {"enabled": False},
    "v5": {"enabled": False},
}


@dataclass(slots=True, frozen=True)
class AgentSourceConfig:
    name: str
    db_path: Path


@dataclass(slots=True)
class HegiConfig:
    path: Path
    raw: dict[str, Any]
    enabled: bool
    chat_id: str
    agents: list[AgentSourceConfig]
    state_db: Path
    local_spool: Path
    curator_env: Path
    nas_root: Path | None

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        return value if isinstance(value, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def default_config_path() -> Path:
    return get_hermes_home() / "hegi" / "config.yaml"


def load_config(path: str | Path | None = None) -> HegiConfig:
    config_path = Path(path).expanduser() if path else default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"HEGI 설정 파일이 없습니다: {config_path}. "
            "hegi/config/default.yaml을 복사하여 값을 설정하세요."
        )
    loaded = fast_safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("HEGI 설정 루트는 YAML mapping이어야 합니다.")
    raw = _deep_merge(DEFAULT_CONFIG, loaded)
    home = get_hermes_home()
    chat_id = str(raw.get("telegram", {}).get("chat_id", "")).strip()
    agents: list[AgentSourceConfig] = []
    for entry in raw.get("agents", []):
        if not isinstance(entry, dict):
            raise ValueError("agents 항목은 name/db_path mapping이어야 합니다.")
        name = str(entry.get("name", "")).strip()
        db_path = str(entry.get("db_path", "")).strip()
        if not name or not db_path:
            raise ValueError("각 agent에는 name과 db_path가 필요합니다.")
        agents.append(AgentSourceConfig(name, _expand_path(db_path, base=home)))
    if len({agent.name for agent in agents}) != len(agents):
        raise ValueError("agent name은 중복될 수 없습니다.")

    archive = raw["archive"]
    telegram = raw["telegram"]
    state_db_value = str(raw.get("state_db", "")).strip()
    state_db = (
        _expand_path(state_db_value, base=home)
        if state_db_value
        else home / "hegi" / "state.db"
    )
    spool_value = str(archive.get("local_spool", "")).strip()
    local_spool = (
        _expand_path(spool_value, base=home)
        if spool_value
        else home / "hegi" / "archive"
    )
    curator_value = str(telegram.get("curator_env", "")).strip()
    curator_env = (
        _expand_path(curator_value, base=home)
        if curator_value
        else home / "profiles" / "memory-curator" / ".env"
    )
    nas_value = str(archive.get("nas_root", "")).strip()
    nas_root = _expand_path(nas_value, base=home) if nas_value else None
    return HegiConfig(
        path=config_path,
        raw=raw,
        enabled=bool(raw.get("enabled", False)),
        chat_id=chat_id,
        agents=agents,
        state_db=state_db,
        local_spool=local_spool,
        curator_env=curator_env,
        nas_root=nas_root,
    )


def validate_config(config: HegiConfig, *, require_runtime: bool = False) -> list[str]:
    errors: list[str] = []
    if not config.chat_id:
        errors.append("telegram.chat_id가 비어 있습니다.")
    if not config.agents:
        errors.append("agents가 비어 있습니다.")
    episode = config.section("episode")
    for key in (
        "quiet_minutes",
        "max_gap_minutes",
        "minimum_agents",
        "minimum_messages",
        "initial_lookback_minutes",
    ):
        try:
            if int(episode.get(key, 0)) <= 0:
                errors.append(f"episode.{key}는 양수여야 합니다.")
        except (TypeError, ValueError):
            errors.append(f"episode.{key}는 정수여야 합니다.")
    memory = config.section("memory")
    if memory.get("auto_commit"):
        errors.append("memory.auto_commit은 안전 경계상 true일 수 없습니다.")
    if memory.get("auto_draft"):
        errors.append("memory.auto_draft는 안전 경계상 true일 수 없습니다.")
    if not memory.get("require_professor_approval", True):
        errors.append("memory.require_professor_approval은 true여야 합니다.")
    if require_runtime:
        for agent in config.agents:
            if not agent.db_path.is_file():
                errors.append(f"{agent.name} DB가 없습니다: {agent.db_path}")
        if config.section("telegram").get("enabled") and not config.curator_env.is_file():
            errors.append(f"Telegram env 파일이 없습니다: {config.curator_env}")
    return errors
