"""RLM memory provider — adapter to the existing local RLM/GBrain/Librarian memory plane."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from agent.memory_provider import MemoryProvider


def _provider_source() -> Path:
    configured = os.getenv("RLM_HERMES_MEMORY_PROVIDER")
    if configured:
        return Path(configured).expanduser()
    local_adapter = Path(__file__).with_name("hermes_memory_provider.py")
    if local_adapter.is_file():
        return local_adapter
    release = os.getenv("RLM_RUNTIME_RELEASE", "").strip()
    root = Path.home() / "Library/Application Support/Eve/RLM/runtime"
    if release:
        return root / release / "integrations/hermes_memory_provider.py"
    current = Path.home() / "Library/Application Support/Eve/RLM/state/current-runtime.json"
    if current.exists():
        import json

        data = json.loads(current.read_text(encoding="utf-8"))
        release = str(data.get("release") or data.get("release_id") or "")
        if release:
            return root / release / "integrations/hermes_memory_provider.py"
    return root / "current/integrations/hermes_memory_provider.py"


def _load_provider_class():
    source = _provider_source()
    if not source.is_file():
        raise RuntimeError(f"RLM Hermes memory adapter not found: {source}")
    spec = importlib.util.spec_from_file_location("_hermes_rlm_memory_provider", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load RLM Hermes memory adapter: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider_class = getattr(module, "HermesRLMMemoryProvider", None)
    if not isinstance(provider_class, type) or not issubclass(provider_class, MemoryProvider):
        raise RuntimeError("RLM Hermes adapter does not implement MemoryProvider")
    return provider_class


def register(ctx) -> None:
    ctx.register_memory_provider(_load_provider_class()())
