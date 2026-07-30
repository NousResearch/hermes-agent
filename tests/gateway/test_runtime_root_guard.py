"""Gateway lazy imports stay within the active source tree."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

from gateway import run as gateway_run


def test_runtime_agent_resolves_from_gateway_source_root():
    agent_class = gateway_run._load_runtime_ai_agent_class()
    runtime_root = Path(gateway_run.__file__).resolve().parent.parent

    assert agent_class.__name__ == "AIAgent"
    assert Path(sys.modules["run_agent"].__file__).resolve() == (
        runtime_root / "run_agent.py"
    )


def test_foreign_sys_path_entry_cannot_win(tmp_path, monkeypatch):
    (tmp_path / "run_agent.py").write_text(
        "class AIAgent: pass\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "run_agent", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    gateway_run._load_runtime_ai_agent_class()

    runtime_root = Path(gateway_run.__file__).resolve().parent.parent
    assert Path(sys.modules["run_agent"].__file__).resolve() == (
        runtime_root / "run_agent.py"
    )


def test_preloaded_foreign_run_agent_fails_closed(tmp_path, monkeypatch):
    foreign = ModuleType("run_agent")
    foreign.__file__ = str(tmp_path / "run_agent.py")
    foreign.AIAgent = type("AIAgent", (), {})
    monkeypatch.setitem(sys.modules, "run_agent", foreign)

    with pytest.raises(RuntimeError, match="already loaded"):
        gateway_run._load_runtime_ai_agent_class()


def test_preloaded_foreign_agent_init_fails_closed(tmp_path, monkeypatch):
    gateway_run._load_runtime_ai_agent_class()
    foreign = ModuleType("agent.agent_init")
    foreign.__file__ = str(tmp_path / "agent" / "agent_init.py")
    monkeypatch.setitem(sys.modules, "agent.agent_init", foreign)

    with pytest.raises(RuntimeError, match="agent.agent_init resolved"):
        gateway_run._load_runtime_ai_agent_class()


def test_api_server_create_agent_uses_runtime_root_loader():
    """API server must not bypass the runtime-root pin with a bare import."""
    import inspect
    from gateway.platforms import api_server as api_mod

    src = inspect.getsource(api_mod.APIServerAdapter._create_agent)
    assert "_load_runtime_ai_agent_class" in src
    assert "from run_agent import AIAgent" not in src
