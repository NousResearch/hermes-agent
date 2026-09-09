"""
Runtime Manager for registered execution runtimes.
"""

from typing import Any, Dict, List, Optional
import logging
from runtime.base import Runtime, RuntimeType
from runtime.local.runtime import LocalRuntime
from runtime.ssh.runtime import SSHRuntime
from runtime.container.runtime import ContainerRuntime
from runtime.sandbox.runtime import SandboxRuntime
from runtime.remote.runtime import RemoteRuntime

logger = logging.getLogger(__name__)


class RuntimeManager:
    """Manages active, available, and remote runtimes."""

    def __init__(self) -> None:
        self._runtimes: Dict[str, Runtime] = {}
        default_local = LocalRuntime()
        self._runtimes[default_local.runtime_id] = default_local
        self.active_runtime_id = default_local.runtime_id

    def register_runtime(self, runtime: Runtime) -> None:
        self._runtimes[runtime.runtime_id] = runtime

    def unregister_runtime(self, runtime_id: str) -> bool:
        if runtime_id in self._runtimes and runtime_id != "local_default":
            del self._runtimes[runtime_id]
            if self.active_runtime_id == runtime_id:
                self.active_runtime_id = "local_default"
            return True
        return False

    def get_runtime(self, runtime_id: Optional[str] = None) -> Optional[Runtime]:
        r_id = runtime_id or self.active_runtime_id
        return self._runtimes.get(r_id)

    def set_active_runtime(self, runtime_id: str) -> bool:
        if runtime_id in self._runtimes:
            self.active_runtime_id = runtime_id
            return True
        return False

    def list_runtimes(self) -> List[Dict[str, Any]]:
        return [r.get_status() for r in self._runtimes.values()]
