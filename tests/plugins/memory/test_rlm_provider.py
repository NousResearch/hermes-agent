from __future__ import annotations

import importlib
import sys
from pathlib import Path

from agent.memory_provider import MemoryProvider


def test_rlm_provider_loads_runtime_adapter(monkeypatch, tmp_path):
    adapter = tmp_path / "hermes_memory_provider.py"
    adapter.write_text(
        "from agent.memory_provider import MemoryProvider\n"
        "class HermesRLMMemoryProvider(MemoryProvider):\n"
        "    name = 'rlm'\n"
        "    def is_available(self): return True\n"
        "    def initialize(self, session_id, **kwargs): pass\n"
        "    def get_tool_schemas(self): return []\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RLM_HERMES_MEMORY_PROVIDER", str(adapter))
    sys.modules.pop("plugins.memory.rlm", None)
    module = importlib.import_module("plugins.memory.rlm")

    provider = module._load_provider_class()()

    assert isinstance(provider, MemoryProvider)
    assert provider.name == "rlm"
