#!/usr/bin/env python3
"""Minimal repro for CI failures (is_wsl mocks + background review thread).

Run from repo root: python scripts/_repro_bug.py
Exit 0 prints PASS; nonzero on failure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stub_wslinterop_exists(orig_exists):
    def _exists(path):
        if str(path).replace("\\", "/").rstrip("/").endswith("WSLInterop"):
            return False
        return orig_exists(path)

    return _exists


def repro_is_wsl() -> None:
    import hermes_constants

    hermes_constants._wsl_detected = None
    fake = (
        "Linux version 5.15.146.1-microsoft-standard-WSL2 "
        "(gcc (GCC) 11.2.0) #1 SMP Thu Jan 11 04:09:03 UTC 2024\n"
    )

    hc = sys.modules["hermes_constants"]
    orig_exists = hc.os.path.exists
    m = mock_open(read_data=fake)
    with patch.object(
        hc.os.path,
        "exists",
        side_effect=_stub_wslinterop_exists(orig_exists),
    ):
        with patch.object(hc, "_builtin_open", m), patch("builtins.open", m):
            assert hc.is_wsl() is True, "is_wsl + mocked /proc/version"


class _SyncThread:
    __slots__ = ("_target", "_daemon", "_name")

    def __init__(self, *, target=None, daemon=None, name=None, args=(), kwargs=None):
        self._target = target
        self._daemon = daemon
        self._name = name

    def start(self):
        if self._target:
            self._target()


def repro_background_review() -> None:
    import run_agent
    from run_agent import AIAgent

    captured: dict = {}

    def _make_agent_stub():
        agent = object.__new__(AIAgent)
        agent.model = "test-model"
        agent.platform = "test"
        agent.provider = "openai"
        agent.session_id = "sess-123"
        agent.quiet_mode = True
        agent._memory_store = None
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._memory_nudge_interval = 5
        agent._skill_nudge_interval = 5
        agent.background_review_callback = None
        agent.status_callback = None
        agent._MEMORY_REVIEW_PROMPT = "review memory"
        agent._SKILL_REVIEW_PROMPT = "review skills"
        agent._COMBINED_REVIEW_PROMPT = "review both"
        return agent

    def _capture_init(self, *args, **kwargs):
        captured["enabled_toolsets"] = kwargs.get("enabled_toolsets")
        raise RuntimeError("stop after capturing init args")

    agent = _make_agent_stub()
    with patch.object(AIAgent, "__init__", _capture_init), patch.object(
        run_agent.threading, "Thread", _SyncThread
    ):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "enabled_toolsets" in captured, "AIAgent.__init__ was not reached"
    assert sorted(captured["enabled_toolsets"]) == ["memory", "skills"]


def main() -> int:
    try:
        repro_is_wsl()
        print("repro_is_wsl: PASS")
        repro_background_review()
        print("repro_background_review: PASS")
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
