from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from gateway.session import SessionStore, SessionSource
from runtime_context import HermesRuntimeContext, build_runtime_context, use_runtime


_gateway_runtime_var: ContextVar["GatewayRuntimeState | None"] = ContextVar(
    "gateway_runtime_state",
    default=None,
)


@dataclass
class GatewayRuntimeState:
    key: str
    profile_name: str | None
    context: HermesRuntimeContext
    session_store: SessionStore
    session_db: Any | None
    running_agents: dict[str, Any] = field(default_factory=dict)
    pending_approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    honcho_managers: dict[str, Any] = field(default_factory=dict)
    honcho_configs: dict[str, Any] = field(default_factory=dict)


class ProfileResolver:
    def __init__(self, global_config: dict[str, Any]):
        self._profiles = global_config.get("profiles", {}) or {}

    @staticmethod
    def _normalize_user_id(user_id: Any) -> str:
        text = str(user_id or "").strip()
        if not text:
            return ""
        normalized = text.lower()
        if "@" in normalized:
            normalized = normalized.split("@", 1)[0]
        return normalized

    def resolve(self, source: SessionSource) -> str | None:
        platform = source.platform.value if source.platform else ""
        if not platform:
            return None

        candidates = {
            self._normalize_user_id(source.user_id),
            self._normalize_user_id(source.user_id_alt),
        }
        candidates.discard("")
        if not candidates:
            return None

        for profile_name, profile_data in self._profiles.items():
            users = profile_data.get("users", {}) if isinstance(profile_data, dict) else {}
            configured = users.get(platform, []) if isinstance(users, dict) else []
            if not isinstance(configured, list):
                continue
            normalized = {self._normalize_user_id(item) for item in configured}
            normalized.discard("")
            if normalized & candidates:
                return profile_name
        return None


class GatewayRuntimeRegistry:
    def __init__(
        self,
        *,
        gateway_config: Any,
        global_config: dict[str, Any],
        global_home: Path,
        has_active_processes_fn,
    ):
        self.gateway_config = gateway_config
        self.global_config = global_config
        self.global_home = Path(global_home)
        self._has_active_processes_fn = has_active_processes_fn
        self._resolver = ProfileResolver(global_config)
        self._runtimes: dict[str, GatewayRuntimeState] = {}
        self._global_runtime = self._create_runtime(None)

    def _create_runtime(self, profile_name: str | None) -> GatewayRuntimeState:
        context = build_runtime_context(
            global_config=self.global_config,
            profile_name=profile_name,
            global_home=self.global_home,
        )
        session_db = None
        try:
            from hermes_state import SessionDB

            session_db = SessionDB(context.state_db_path)
        except Exception:
            session_db = None

        key = profile_name or "global"
        self._runtimes[key] = GatewayRuntimeState(
            key=key,
            profile_name=profile_name,
            context=context,
            session_store=SessionStore(
                context.sessions_dir,
                self.gateway_config,
                has_active_processes_fn=self._has_active_processes_fn,
                db_path=context.state_db_path,
            ),
            session_db=session_db,
        )
        return self._runtimes[key]

    def get_runtime(self, profile_name: str | None) -> GatewayRuntimeState:
        key = profile_name or "global"
        runtime = self._runtimes.get(key)
        if runtime is None:
            runtime = self._create_runtime(profile_name)
        return runtime

    def resolve_runtime(self, source: SessionSource) -> GatewayRuntimeState:
        return self.get_runtime(self._resolver.resolve(source))

    def current_runtime(self) -> GatewayRuntimeState:
        return _gateway_runtime_var.get() or self._global_runtime

    def iter_runtimes(self) -> list[GatewayRuntimeState]:
        runtimes = [self._global_runtime]
        for key, runtime in self._runtimes.items():
            if key == "global":
                continue
            runtimes.append(runtime)
        return runtimes

    @contextmanager
    def use_runtime(
        self,
        *,
        source: SessionSource | None = None,
        runtime: GatewayRuntimeState | None = None,
    ) -> Iterator[GatewayRuntimeState]:
        if runtime is not None:
            selected = runtime
        elif source is not None:
            selected = self.resolve_runtime(source)
        else:
            selected = self.current_runtime()
        token = _gateway_runtime_var.set(selected)
        with use_runtime(selected.context):
            try:
                yield selected
            finally:
                _gateway_runtime_var.reset(token)


class RuntimeObjectProxy:
    def __init__(self, registry: GatewayRuntimeRegistry, attr_name: str):
        self._registry = registry
        self._attr_name = attr_name

    def _target(self):
        return getattr(self._registry.current_runtime(), self._attr_name)

    def __bool__(self) -> bool:
        return bool(self._target())

    def __getattr__(self, name: str):
        target = self._target()
        if target is None:
            raise AttributeError(name)
        return getattr(target, name)


class RuntimeMappingProxy(MutableMapping[str, Any]):
    def __init__(self, registry: GatewayRuntimeRegistry, attr_name: str):
        self._registry = registry
        self._attr_name = attr_name

    def _mapping(self) -> dict[str, Any]:
        return getattr(self._registry.current_runtime(), self._attr_name)

    def __getitem__(self, key: str) -> Any:
        return self._mapping()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._mapping()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._mapping()[key]

    def __iter__(self):
        return iter(self._mapping())

    def __len__(self) -> int:
        return len(self._mapping())

    def clear(self) -> None:
        self._mapping().clear()
