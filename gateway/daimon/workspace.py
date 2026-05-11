"""Workspace manager for Daimon sandbox containers."""

import logging
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)

_VALID_THREAD_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")


class WorkspaceManager:
    """Manages per-thread workspaces inside a Docker container."""

    def __init__(self, container_name: str = "daimon-sandbox"):
        self._container_name = container_name
        self._docker = shutil.which("docker") or "docker"

    def workspace_path(self, thread_id: str) -> str:
        """Return the workspace path for a given thread."""
        return f"/workspaces/{thread_id}"

    def _validate_thread_id(self, thread_id: str) -> bool:
        """Validate thread_id to prevent path traversal attacks.

        Only allows alphanumeric characters, underscores, and hyphens.
        """
        if not _VALID_THREAD_ID.match(thread_id):
            logger.warning(
                "Invalid thread_id rejected (possible path traversal): %r",
                thread_id,
            )
            return False
        return True

    def create(self, thread_id: str) -> None:
        """Create workspace directory inside the container."""
        if not self._validate_thread_id(thread_id):
            return

        path = self.workspace_path(thread_id)
        try:
            result = subprocess.run(
                [self._docker, "exec", self._container_name, "mkdir", "-p", path],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info("Created workspace: %s", path)
            else:
                stderr = result.stderr.decode(errors="replace").strip()
                logger.error(
                    "Failed to create workspace %s: %s", path, stderr
                )
        except subprocess.TimeoutExpired:
            logger.error("Timeout creating workspace: %s", path)
        except Exception as e:
            logger.error("Error creating workspace %s: %s", path, e)

    def destroy(self, thread_id: str) -> None:
        """Destroy workspace directory inside the container."""
        if not self._validate_thread_id(thread_id):
            return

        path = self.workspace_path(thread_id)
        try:
            result = subprocess.run(
                [self._docker, "exec", self._container_name, "rm", "-rf", path],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info("Destroyed workspace: %s", path)
            else:
                stderr = result.stderr.decode(errors="replace").strip()
                logger.error(
                    "Failed to destroy workspace %s: %s", path, stderr
                )
        except subprocess.TimeoutExpired:
            logger.error("Timeout destroying workspace: %s", path)
        except Exception as e:
            logger.error("Error destroying workspace %s: %s", path, e)
