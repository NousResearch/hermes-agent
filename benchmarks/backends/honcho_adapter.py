"""Benchmark adapter for the Honcho memory plugin.

This adapter targets the real Honcho-backed memory flow, but it requires an
available Honcho client configuration (API key and/or base URL). Unlike purely
local adapters, live benchmark runs for Honcho are environment-dependent.

Tests can inject a fake session manager to validate adapter behavior without
network access.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import BenchmarkableStore

BACKEND_NAME = "honcho"
BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,
)

_MODULE_CACHE: dict[str, Any] = {}


def _git_show(pathspec: str) -> str:
    result = subprocess.run(
        ["git", "show", pathspec],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _load_honcho_runtime_modules() -> tuple[Any, Any]:
    cached = _MODULE_CACHE.get("modules")
    if cached:
        return cached

    temp_root = Path(tempfile.mkdtemp(prefix="benchmark-honcho-"))
    package_name = "benchmark_honcho_runtime"
    package_dir = temp_root / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("")
    (package_dir / "client.py").write_text(_git_show("main:plugins/memory/honcho/client.py"))
    session_src = _git_show("main:plugins/memory/honcho/session.py")
    session_src = session_src.replace(
        "from plugins.memory.honcho.client import get_honcho_client",
        f"from {package_name}.client import get_honcho_client",
    )
    (package_dir / "session.py").write_text(session_src)

    importlib.invalidate_caches()
    sys.path.insert(0, str(temp_root))
    client_mod = importlib.import_module(f"{package_name}.client")
    session_mod = importlib.import_module(f"{package_name}.session")
    _MODULE_CACHE["modules"] = (client_mod, session_mod)
    _MODULE_CACHE["temp_root"] = temp_root
    return client_mod, session_mod


@dataclass
class _MinimalConfig:
    workspace_id: str = "hermes-benchmark"
    ai_peer: str = "hermes"
    peer_name: Optional[str] = None
    write_frequency: str = "turn"
    memory_mode: str = "hybrid"
    peer_memory_modes: dict[str, str] = None
    context_tokens: int = 800
    dialectic_reasoning_level: str = "low"
    dialectic_max_chars: int = 600
    recall_mode: str = "hybrid"
    observation_mode: str = "unified"

    def __post_init__(self):
        if self.peer_memory_modes is None:
            self.peer_memory_modes = {}


class HonchoBenchmarkAdapter(BenchmarkableStore):
    """Benchmark adapter for a configured Honcho deployment."""

    def __init__(self, **kwargs):
        self._workspace_id = kwargs.get("workspace_id") or os.environ.get("HONCHO_BENCHMARK_WORKSPACE") or "hermes-benchmark"
        self._session_key = kwargs.get("session_key", "benchmark-session")
        self._manager = kwargs.get("manager")
        self._owns_manager = False
        self._available = True
        self._error = ""
        if self._manager is None:
            self._init_real_manager(kwargs)

    def _init_real_manager(self, kwargs: dict[str, Any]) -> None:
        api_key = kwargs.get("api_key") or os.environ.get("HONCHO_API_KEY")
        base_url = kwargs.get("base_url") or os.environ.get("HONCHO_BASE_URL")
        if not api_key and not base_url:
            self._available = False
            self._error = "Honcho benchmark adapter requires HONCHO_API_KEY or HONCHO_BASE_URL"
            return

        client_mod, session_mod = _load_honcho_runtime_modules()
        cfg = _MinimalConfig(workspace_id=self._workspace_id)
        try:
            client = client_mod.get_honcho_client(
                cfg if hasattr(cfg, "base_url") else None
            )
        except Exception:
            # Build client directly if config helper cannot be used with minimal config
            from honcho import Honcho
            effective_api_key = api_key or ("local" if base_url and "localhost" in base_url else None)
            kwargs_client = {
                "workspace_id": self._workspace_id,
                "api_key": effective_api_key,
                "environment": "production",
            }
            if base_url:
                kwargs_client["base_url"] = base_url
            client = Honcho(**kwargs_client)

        self._manager = session_mod.HonchoSessionManager(
            honcho=client,
            context_tokens=800,
            config=cfg,
        )
        self._owns_manager = True
        self._manager.get_or_create(self._session_key)

    def _ensure_available(self) -> None:
        if not self._available or self._manager is None:
            raise RuntimeError(self._error or "Honcho benchmark adapter is not configured")

    def store(self, content: str, category: str = "factual",
              scope: str = "global", importance: float = 0.5) -> None:
        del category, scope, importance
        self._ensure_available()
        session = self._manager.get_or_create(self._session_key)
        session.add_message("user", content)
        self._manager.save(session)
        self._manager.flush_all()

    def recall(self, query: str, top_k: int = 10,
               scope: Optional[str] = None) -> list[str]:
        del top_k, scope
        self._ensure_available()
        result = self._manager.search_context(self._session_key, query, max_tokens=800)
        if not result:
            result = self._manager.dialectic_query(self._session_key, query, peer="user")
        if not result:
            return []
        if isinstance(result, str):
            return [line.strip() for line in result.splitlines() if line.strip()][:10] or [result]
        return [str(result)]

    def simulate_time(self, days: float) -> None:
        del days
        return None

    def simulate_access(self, content_substring: str) -> None:
        del content_substring
        return None

    def consolidate(self) -> None:
        return None

    def get_stats(self) -> dict[str, Any]:
        return {
            "session_key": self._session_key,
            "workspace_id": self._workspace_id,
            "configured": self._available,
        }

    def reset(self) -> None:
        if self._manager is None:
            return
        try:
            self._manager.delete(self._session_key)
        except Exception:
            pass
        if self._available:
            self._manager.get_or_create(self._session_key)


BACKEND_CLASS = HonchoBenchmarkAdapter
