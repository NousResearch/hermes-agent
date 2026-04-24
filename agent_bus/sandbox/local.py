"""LocalSandbox — simplest provider backing sandbox with host filesystem.

Per-thread directory structure (auto-created on acquire):
    ~/.hermes/threads/{thread_id}/user-data/
        workspace/
        uploads/
        outputs/
"""

from __future__ import annotations

import threading
from pathlib import Path

from agent_bus.sandbox.sandbox import Sandbox, SandboxError, SandboxProvider
from agent_bus.sandbox.virtual_path import translate_virtual_path


class LocalSandbox:
    """Per-thread sandbox backed by host filesystem."""

    id: str
    thread_id: str

    def __init__(self, thread_id: str):
        from agent_bus.sandbox.virtual_path import _thread_base

        self.thread_id = thread_id
        self.id = f"local:{thread_id}"
        self._base = _thread_base(thread_id)
        self._user_data = self._base / "user-data"
        self._workspace = self._user_data / "workspace"
        self._uploads = self._user_data / "uploads"
        self._outputs = self._user_data / "outputs"
        # Create directories lazily on first access
        for d in (self._workspace, self._uploads, self._outputs):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def workspace_dir(self) -> Path:
        return self._workspace

    @property
    def uploads_dir(self) -> Path:
        return self._uploads

    @property
    def outputs_dir(self) -> Path:
        return self._outputs

    def translate(self, virtual_path: str) -> Path:
        return translate_virtual_path(virtual_path, self.thread_id)

    def read_file(self, virtual_path: str) -> str:
        real = self.translate(virtual_path)
        if not real.exists():
            raise SandboxError(f"file not found: {virtual_path}")
        try:
            return real.read_text(encoding="utf-8")
        except Exception as exc:
            raise SandboxError(f"read failed {virtual_path}: {exc}") from exc

    def write_file(self, virtual_path: str, content: str) -> None:
        real = self.translate(virtual_path)
        real.parent.mkdir(parents=True, exist_ok=True)
        try:
            real.write_text(content, encoding="utf-8")
        except Exception as exc:
            raise SandboxError(f"write failed {virtual_path}: {exc}") from exc

    def list_dir(self, virtual_path: str) -> list[str]:
        real = self.translate(virtual_path)
        if not real.exists():
            return []
        if not real.is_dir():
            raise SandboxError(f"not a directory: {virtual_path}")
        return sorted(p.name for p in real.iterdir())


class LocalSandboxProvider:
    """Factory that caches LocalSandbox instances per thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, LocalSandbox] = {}

    def acquire(self, thread_id: str) -> LocalSandbox:
        with self._lock:
            sb = self._cache.get(thread_id)
            if sb is None:
                sb = LocalSandbox(thread_id)
                self._cache[thread_id] = sb
            return sb

    def get(self, thread_id: str) -> LocalSandbox | None:
        with self._lock:
            return self._cache.get(thread_id)

    def release(self, thread_id: str) -> None:
        """Drop cached reference. Does NOT delete directories — that's explicit."""
        with self._lock:
            self._cache.pop(thread_id, None)

    def wipe_thread_dir(self, thread_id: str) -> bool:
        """Actually delete the thread's entire user-data dir. Destructive — use sparingly."""
        sb = self.get(thread_id)
        if sb is None:
            sb = LocalSandbox(thread_id)
        import shutil as _shutil

        base = sb._base  # noqa: SLF001
        if not base.exists():
            return False
        _shutil.rmtree(base)
        self.release(thread_id)
        return True


# Module-level default singleton
_default_provider: LocalSandboxProvider | None = None


def get_default_provider() -> LocalSandboxProvider:
    global _default_provider
    if _default_provider is None:
        _default_provider = LocalSandboxProvider()
    return _default_provider
