"""
Container Runtime Implementation (Docker/Singularity/Podman).
"""

from typing import Any, Dict, Optional
from runtime.base import Runtime, RuntimeType


class ContainerRuntime(Runtime):
    def __init__(self, runtime_id: str, name: str, container_image: str = "ubuntu:22.04") -> None:
        super().__init__(runtime_id, name, RuntimeType.CONTAINER)
        self.container_image = container_image

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def is_connected(self) -> bool:
        return self.connected

    def execute_command(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        return {"exit_code": 0, "stdout": f"Container [{self.container_image}]: {command}", "stderr": ""}

    def get_capabilities(self) -> Dict[str, Any]:
        return {"container": True, "image": self.container_image}
