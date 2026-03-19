from __future__ import annotations

import copy
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional


_thread_local = threading.local()
_runtime_var: ContextVar["HermesRuntimeContext | None"] = ContextVar(
    "hermes_runtime_context",
    default=None,
)


def get_global_hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


@dataclass(frozen=True)
class HermesRuntimeContext:
    profile_name: str | None
    global_home: Path
    effective_home: Path
    effective_config: dict[str, Any]
    workspace: str | None
    shared_skills_dir: Path
    private_skills_dir: Path | None

    @property
    def state_db_path(self) -> Path:
        return self.effective_home / "state.db"

    @property
    def sessions_dir(self) -> Path:
        return self.effective_home / "sessions"

    @property
    def memories_dir(self) -> Path:
        return self.effective_home / "memories"

    @property
    def cron_dir(self) -> Path:
        return self.effective_home / "cron"

    @property
    def soul_path(self) -> Path:
        return self.effective_home / "SOUL.md"

    def get_skill_roots(self) -> list[Path]:
        roots: list[Path] = []
        if self.private_skills_dir:
            roots.append(self.private_skills_dir)
        roots.append(self.shared_skills_dir)
        return roots


def get_current_runtime() -> HermesRuntimeContext | None:
    runtime = _runtime_var.get()
    if runtime is not None:
        return runtime
    return getattr(_thread_local, "runtime_context", None)


@contextmanager
def use_runtime(runtime: HermesRuntimeContext | None) -> Iterator[HermesRuntimeContext | None]:
    token = _runtime_var.set(runtime)
    previous = getattr(_thread_local, "runtime_context", None)
    _thread_local.runtime_context = runtime
    try:
        yield runtime
    finally:
        if previous is None:
            try:
                delattr(_thread_local, "runtime_context")
            except AttributeError:
                pass
        else:
            _thread_local.runtime_context = previous
        _runtime_var.reset(token)


def get_effective_home() -> Path:
    runtime = get_current_runtime()
    if runtime is not None:
        return runtime.effective_home
    return get_global_hermes_home()


def get_effective_memories_dir() -> Path:
    runtime = get_current_runtime()
    if runtime is not None:
        return runtime.memories_dir
    return get_global_hermes_home() / "memories"


def get_effective_soul_path() -> Path:
    runtime = get_current_runtime()
    if runtime is not None:
        if runtime.soul_path.exists():
            return runtime.soul_path
        return runtime.global_home / "SOUL.md"
    return get_global_hermes_home() / "SOUL.md"


def get_shared_skills_dir() -> Path:
    runtime = get_current_runtime()
    if runtime is not None:
        return runtime.shared_skills_dir
    return get_global_hermes_home() / "skills"


def get_private_skills_dir() -> Path | None:
    runtime = get_current_runtime()
    if runtime is not None:
        return runtime.private_skills_dir
    return None


def get_effective_skill_roots() -> list[Path]:
    runtime = get_current_runtime()
    if runtime is not None:
        return runtime.get_skill_roots()
    return [get_global_hermes_home() / "skills"]


def get_effective_workspace() -> str | None:
    runtime = get_current_runtime()
    if runtime is not None and runtime.workspace:
        return runtime.workspace
    return None


def get_workspace_isolation_root() -> Path | None:
    runtime = get_current_runtime()
    if runtime is None:
        return None

    isolation_cfg = runtime.effective_config.get("isolation", {})
    if not isinstance(isolation_cfg, dict):
        isolation_cfg = {}

    enabled = isolation_cfg.get("enabled")
    if enabled is None:
        enabled = isolation_cfg.get("isolated")
    if enabled is None:
        enabled = isolation_cfg.get("workspace_only")
    if not enabled:
        return None

    root_value = (
        isolation_cfg.get("root")
        or isolation_cfg.get("workspace_root")
        or runtime.workspace
    )
    if not isinstance(root_value, str) or not root_value.strip():
        return None
    return Path(root_value).expanduser()


def _normalize_slash_command_name(name: Any) -> str:
    return str(name or "").strip().lower().lstrip("/")


def get_disabled_slash_commands() -> set[str]:
    runtime = get_current_runtime()
    if runtime is None:
        return set()

    config = runtime.effective_config or {}
    disabled_values: list[Any] = []
    for candidate in (
        config.get("disabled_slash_commands"),
        (config.get("messaging", {}) or {}).get("disabled_slash_commands"),
        (config.get("gateway", {}) or {}).get("disabled_slash_commands"),
    ):
        if isinstance(candidate, (list, tuple, set)):
            disabled_values.extend(candidate)

    normalized = {
        _normalize_slash_command_name(value)
        for value in disabled_values
        if _normalize_slash_command_name(value)
    }
    if "all" in normalized:
        normalized.add("*")
    return normalized


def find_skill_root_for_path(path: Path) -> Path | None:
    resolved = path.resolve()
    for root in get_effective_skill_roots():
        try:
            resolved.relative_to(root.resolve())
            return root
        except Exception:
            continue
    return None


def profile_home(global_home: Path, profile_name: str) -> Path:
    return global_home / "profiles" / profile_name


def _resolve_shared_skills_dir(global_home: Path) -> Path:
    default_dir = global_home / "skills"
    try:
        from tools.skills_tool import (
            SKILLS_DIR as configured_skills_dir,
            _DEFAULT_SHARED_SKILLS_DIR,
        )

        configured = Path(configured_skills_dir)
        if configured != Path(_DEFAULT_SHARED_SKILLS_DIR):
            return configured
    except Exception:
        pass
    return default_dir


def ensure_runtime_home(runtime: HermesRuntimeContext) -> None:
    runtime.effective_home.mkdir(parents=True, exist_ok=True)
    for directory in (
        runtime.sessions_dir,
        runtime.memories_dir,
        runtime.cron_dir,
        runtime.effective_home / "skills",
        runtime.effective_home / "workspace",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def build_runtime_context(
    *,
    global_config: dict[str, Any],
    profile_name: str | None = None,
    global_home: Path | None = None,
) -> HermesRuntimeContext:
    from hermes_cli.config import _deep_merge

    global_home = Path(global_home or get_global_hermes_home())
    effective_home = global_home
    effective_config = copy.deepcopy(global_config)
    workspace: str | None = None
    private_skills_dir: Path | None = None

    if profile_name:
        profile_cfg = (
            global_config.get("profiles", {}).get(profile_name, {}).get("config", {})
        )
        if isinstance(profile_cfg, dict):
            effective_config = _deep_merge(effective_config, profile_cfg)
        effective_home = profile_home(global_home, profile_name)
        private_skills_dir = effective_home / "skills"
        default_workspace = effective_home / "workspace"
        workspace_value = None
        if isinstance(profile_cfg, dict):
            workspace_value = profile_cfg.get("workspace")
            if not workspace_value:
                workspace_value = (
                    profile_cfg.get("terminal", {}) or {}
                ).get("cwd")
        if isinstance(workspace_value, str) and workspace_value.strip():
            workspace = workspace_value.strip()
        else:
            workspace = str(default_workspace)

    runtime = HermesRuntimeContext(
        profile_name=profile_name,
        global_home=global_home,
        effective_home=effective_home,
        effective_config=effective_config,
        workspace=workspace,
        shared_skills_dir=_resolve_shared_skills_dir(global_home),
        private_skills_dir=private_skills_dir,
    )
    ensure_runtime_home(runtime)
    return runtime
