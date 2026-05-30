"""Worker runtime registry for Dev execution plans.

The registry keeps Dev orchestration independent from any single worker
backend. AO is the only production launcher today; fixture exists for
deterministic supervisor tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from gateway.dev_control.runtime_capabilities import attach_capabilities, runtime_capabilities


DEFAULT_RUNTIME = "ao"


@dataclass(frozen=True)
class WorkerRuntimeInfo:
    id: str
    label: str
    available: bool
    launch_supported: bool
    test_only: bool
    supported_actions: tuple[str, ...]
    default_agent: Optional[str] = None
    default_model: Optional[str] = None
    default_reasoning_effort: Optional[str] = None
    configured_mode: Optional[str] = None
    setup_warning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    capabilities: Optional[Dict[str, bool]] = None

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "label": self.label,
            "available": self.available,
            "launch_supported": self.launch_supported,
            "test_only": self.test_only,
            "supported_actions": list(self.supported_actions),
            "default_agent": self.default_agent,
            "default_model": self.default_model,
            "default_reasoning_effort": self.default_reasoning_effort,
            "configured_mode": self.configured_mode,
            "setup_warning": self.setup_warning,
            "metadata": dict(self.metadata or {}),
        }
        capabilities = self.capabilities or runtime_capabilities(
            runtime_id=self.id,
            supported_actions=self.supported_actions,
            test_only=self.test_only,
            launch_supported=self.launch_supported,
        )
        return attach_capabilities(payload, capabilities)


class WorkerRuntimeError(RuntimeError):
    pass


class AOWorkerRuntime:
    id = "ao"

    def __init__(self, bridge: Any = None):
        if bridge is None:
            from tools.ao_bridge import AOBridge

            bridge = AOBridge()
        self.bridge = bridge

    @property
    def info(self) -> WorkerRuntimeInfo:
        return WorkerRuntimeInfo(
            id=self.id,
            label="Agent Orchestrator",
            available=True,
            launch_supported=True,
            test_only=False,
            supported_actions=("spawn", "status", "list", "send", "kill", "capture_output", "runtime_health"),
        )

    def spawn(self, **kwargs: Any) -> Any:
        return self.bridge.spawn(**kwargs)

    def status(self, session_id: str) -> Any:
        return self.bridge.status(session_id)

    def list(self, project_id: Optional[str] = None) -> list[Any]:
        return self.bridge.list(project_id=project_id)

    def send(self, session_id: str, message: str) -> Any:
        return self.bridge.send(session_id, message)

    def kill(self, session_id: str, **kwargs: Any) -> Any:
        return self.bridge.kill(session_id, **kwargs)

    def capture_output(self, session: Any, lines: int = 40) -> str:
        return self.bridge.capture_output(session, lines=lines)

    def runtime_health(self, session: Any) -> Dict[str, Any]:
        return self.bridge.runtime_health(session)


class FixtureWorkerRuntime:
    id = "fixture"

    @property
    def info(self) -> WorkerRuntimeInfo:
        return WorkerRuntimeInfo(
            id=self.id,
            label="Fixture",
            available=True,
            launch_supported=False,
            test_only=True,
            supported_actions=("status",),
            default_agent="fixture",
            default_model="fixture",
            default_reasoning_effort="test",
        )

    def spawn(self, **kwargs: Any) -> Any:
        raise WorkerRuntimeError("Fixture runtime does not launch real workers.")

    def status(self, session_id: str) -> None:
        return None

    def list(self, project_id: Optional[str] = None) -> list[Any]:
        return []

    def send(self, session_id: str, message: str) -> None:
        return None

    def kill(self, session_id: str, **kwargs: Any) -> None:
        return None

    def capture_output(self, session: Any, lines: int = 40) -> str:
        return ""

    def runtime_health(self, session: Any) -> Dict[str, Any]:
        return {"runtime_health": "ok", "runtime_warning": None}


class OpenHandsWorkerRuntime:
    id = "openhands"

    def __init__(self, bridge: Any = None):
        if bridge is None:
            from tools.openhands_bridge import OpenHandsBridge

            bridge = OpenHandsBridge()
        self.bridge = bridge

    @property
    def info(self) -> WorkerRuntimeInfo:
        discovery = self.bridge.discovery()
        supported_actions = ("status", "list", "runtime_health")
        if discovery.get("launch_supported"):
            supported_actions = ("spawn", "status", "list", "send", "kill", "capture_output", "runtime_health")
        return WorkerRuntimeInfo(
            id=self.id,
            label="OpenHands",
            available=bool(discovery.get("available")),
            launch_supported=bool(discovery.get("launch_supported")),
            test_only=False,
            supported_actions=supported_actions,
            default_agent="openhands",
            configured_mode=discovery.get("configured_mode"),
            setup_warning=discovery.get("setup_warning"),
            metadata={
                "server_url": discovery.get("server_url"),
                "sdk_available": discovery.get("sdk_available"),
                "command": discovery.get("command"),
            },
        )

    def spawn(self, **kwargs: Any) -> Any:
        kwargs.pop("minimal_worker_prompt", None)
        return self.bridge.spawn(**kwargs)

    def status(self, session_id: str) -> Any:
        return self.bridge.status(session_id)

    def list(self, project_id: Optional[str] = None) -> list[Any]:
        return self.bridge.list(project_id=project_id)

    def send(self, session_id: str, message: str) -> Any:
        return self.bridge.send(session_id, message)

    def kill(self, session_id: str, **kwargs: Any) -> Any:
        return self.bridge.kill(session_id, **kwargs)

    def capture_output(self, session: Any, lines: int = 40) -> str:
        return self.bridge.capture_output(session, lines=lines)

    def runtime_health(self, session: Any) -> Dict[str, Any]:
        return self.bridge.runtime_health(session)


class WorkerRuntimeRouter:
    def __init__(self, *, ao_bridge: Any = None, openhands_bridge: Any = None):
        self._runtimes: Dict[str, Any] = {
            "ao": AOWorkerRuntime(ao_bridge),
            "fixture": FixtureWorkerRuntime(),
            "openhands": OpenHandsWorkerRuntime(openhands_bridge),
        }

    def get(self, runtime: Optional[str]) -> Any:
        runtime_id = normalize_runtime(runtime)
        adapter = self._runtimes.get(runtime_id)
        if adapter is None:
            raise WorkerRuntimeError(f"Unsupported worker runtime: {runtime_id}")
        return adapter

    def info(self, runtime: Optional[str]) -> Dict[str, Any]:
        return self.get(runtime).info.as_dict()

    def list_runtimes(self) -> list[Dict[str, Any]]:
        return [adapter.info.as_dict() for adapter in self._runtimes.values()]

    def spawn(self, runtime: Optional[str], **kwargs: Any) -> Any:
        adapter = self.get(runtime)
        info = adapter.info
        if not info.launch_supported:
            raise WorkerRuntimeError(info.setup_warning or f"Worker runtime {adapter.id} does not support launch.")
        return adapter.spawn(**kwargs)

    def status(self, runtime: Optional[str], session_id: str) -> Any:
        return self.get(runtime).status(session_id)

    def list(self, runtime: Optional[str], project_id: Optional[str] = None) -> list[Any]:
        return self.get(runtime).list(project_id=project_id)

    def send(self, runtime: Optional[str], session_id: str, message: str) -> Any:
        return self.get(runtime).send(session_id, message)

    def kill(self, runtime: Optional[str], session_id: str, **kwargs: Any) -> Any:
        return self.get(runtime).kill(session_id, **kwargs)

    def capture_output(self, runtime: Optional[str], session: Any, lines: int = 40) -> str:
        return self.get(runtime).capture_output(session, lines=lines)

    def runtime_health(self, runtime: Optional[str], session: Any) -> Dict[str, Any]:
        return self.get(runtime).runtime_health(session)


def normalize_runtime(runtime: Optional[str]) -> str:
    value = str(runtime or DEFAULT_RUNTIME).strip().lower().replace("-", "_")
    return value or DEFAULT_RUNTIME


def list_worker_runtimes() -> list[Dict[str, Any]]:
    return WorkerRuntimeRouter().list_runtimes()
