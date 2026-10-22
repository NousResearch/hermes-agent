"""The gateway ``/skills`` handler must resolve the session cache key it peeks.

Approving a staged skill write is where the write actually commits, so the handler threads
the session agent's memory manager into the shared handler to mirror it to external memory
providers. That cache peek read a bare ``session_key`` that was never bound, so a populated
agent cache raised ``NameError`` out of the handler — an absent cache (the shape the
routed-profile tests build) skips the peek and hid it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools import write_approval as wa

_SKILL = "---\nname: gateway-skill\ndescription: d\n---\n\nbody\n"


class _RecordingProvider(MemoryProvider):
    """Minimal external provider that records on_skill_write calls."""

    def __init__(self) -> None:
        self.calls = []

    @property
    def name(self) -> str:
        return "recording"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def shutdown(self) -> None:
        pass

    def on_skill_write(self, action, name, content, metadata=None):
        self.calls.append({
            "action": action,
            "name": name,
            "content": content,
            "metadata": dict(metadata or {}),
        })


class _Event:
    """MessageEvent stand-in: the handler reads ``source`` and the command args only."""

    def __init__(self, args: str):
        self._args = args
        self.source = SessionSource(
            platform=Platform.TELEGRAM, user_id="u1", chat_id="c1",
            user_name="tester", chat_type="dm")

    def get_command_args(self) -> str:
        return self._args


def _runner(agent_cache):
    class _Runner(GatewayRunner):
        """Bare runner (no __init__): only the /skills handler + the cache helpers are exercised."""

        def __init__(self):
            self.config = GatewayConfig()
            self._agent_cache = agent_cache
            self._agent_cache_lock = threading.Lock()

        def _session_key_for_source(self, _source):
            return "session-key"

        def _write_approval_setter(self, _section, _event):
            return lambda _enabled: None

    return _Runner()


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_skills_cmd_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _stage_skill_write():
    return wa.stage_write(
        wa.SKILLS, {"action": "create", "name": "gateway-skill", "content": _SKILL},
        summary="create gateway-skill", origin="foreground")


@pytest.mark.asyncio
async def test_skills_command_mirrors_through_cached_agent_memory_manager(hermes_home):
    provider = _RecordingProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    rec = _stage_skill_write()

    # Cache values are the real ``(agent, config_signature)`` tuples.
    runner = _runner({"session-key": (SimpleNamespace(_memory_manager=manager), "sig")})
    out = await runner._handle_skills_command(_Event(f"approve {rec['id']}"))

    assert "Approved 1" in out
    assert [c["name"] for c in provider.calls] == ["gateway-skill"]
    assert provider.calls[0]["metadata"]["execution_context"] == "approval_replay"
    assert os.path.isfile(os.path.join(hermes_home, "skills", "gateway-skill", "SKILL.md"))


@pytest.mark.asyncio
async def test_skills_command_without_cached_agent_still_approves(hermes_home):
    rec = _stage_skill_write()

    out = await _runner({})._handle_skills_command(_Event(f"approve {rec['id']}"))

    assert "Approved 1" in out
    assert os.path.isfile(os.path.join(hermes_home, "skills", "gateway-skill", "SKILL.md"))
