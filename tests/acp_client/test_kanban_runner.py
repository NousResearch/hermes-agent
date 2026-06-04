"""Tests for acp_client.kanban_runner — default-off launch plan + lifecycle.

The pure planners (``build_launch_plan``/``decide_writeback``) and the
``ProgressWriter`` need nothing launched.  ``run_acp_lane`` is driven through an
injected fake connection factory, so no real ``claude``/``codex`` process is
ever started.
"""

import contextlib
import json
from types import SimpleNamespace

import pytest

from acp_client import AcpClientUnavailable
from acp_client.kanban_runner import (
    LaunchPlan,
    ProgressWriter,
    build_launch_plan,
    decide_writeback,
    run_acp_lane,
)
from acp_client.transport import (
    LAUNCH_GUARD_ENV_VAR,
    TRANSPORT_ACP,
    TRANSPORT_ENV_VAR,
    TRANSPORT_PTY,
)

_ACP_ENV = {TRANSPORT_ENV_VAR: "acp", LAUNCH_GUARD_ENV_VAR: "1"}


def _force_acp(monkeypatch):
    """Make ``resolve_transport``'s default acp-availability check return True."""
    monkeypatch.setattr("acp_client.acp_available", lambda: True, raising=True)


class TestBuildLaunchPlan:
    def test_default_env_is_pty(self):
        plan = build_launch_plan(workspace="/tmp/ws", env={})
        assert plan.transport == TRANSPORT_PTY
        assert not plan.use_acp
        assert not plan.fell_back

    def test_acp_plan_with_both_gates(self, monkeypatch):
        _force_acp(monkeypatch)
        plan = build_launch_plan(workspace="/tmp/ws", env=_ACP_ENV, backend="claude")
        assert plan.transport == TRANSPORT_ACP
        assert plan.use_acp
        assert plan.command == "claude"
        assert plan.args == ["--acp"]
        # env is allowlisted — no credential keys leak through.
        plan2 = build_launch_plan(
            workspace="/tmp/ws",
            env={**_ACP_ENV, "ANTHROPIC_API_KEY": "sk-x", "PATH": "/usr/bin"},
            backend="claude",
        )
        assert "ANTHROPIC_API_KEY" not in (plan2.env or {})
        assert plan2.env.get("PATH") == "/usr/bin"

    def test_unknown_backend_falls_back(self, monkeypatch):
        _force_acp(monkeypatch)
        plan = build_launch_plan(workspace="/tmp/ws", env=_ACP_ENV, backend="nope")
        assert plan.transport == TRANSPORT_PTY
        assert plan.fell_back
        assert plan.refusal and "nope" in plan.refusal

    def test_unknown_backend_strict_raises(self, monkeypatch):
        _force_acp(monkeypatch)
        with pytest.raises(AcpClientUnavailable):
            build_launch_plan(
                workspace="/tmp/ws", env=_ACP_ENV, backend="nope", strict=True
            )

    def test_strict_raises_when_acp_unavailable(self, monkeypatch):
        monkeypatch.setattr("acp_client.acp_available", lambda: False, raising=True)
        with pytest.raises(AcpClientUnavailable):
            build_launch_plan(workspace="/tmp/ws", env=_ACP_ENV, strict=True)


class TestDecideWriteback:
    @pytest.mark.parametrize(
        "stop_reason,action,lane_status",
        [
            ("end_turn", "complete", "done"),
            ("max_tokens", "block", "blocked"),
            ("max_turns", "block", "blocked"),
            ("cancelled", "block", "blocked"),
            ("refusal", "block", "blocked"),
            (None, "block", "blocked"),
            ("something_unknown", "block", "blocked"),
        ],
    )
    def test_mapping(self, stop_reason, action, lane_status):
        d = decide_writeback(stop_reason, "summary text")
        assert d.action == action
        assert d.lane_status == lane_status

    def test_summary_is_tail_trimmed(self):
        d = decide_writeback("end_turn", "x" * 5000)
        assert len(d.summary) == 3500


class TestProgressWriter:
    def test_appends_jsonl(self, tmp_path):
        w = ProgressWriter(str(tmp_path))
        w.write({"event": "lane_start"})
        w.write({"event": "lane_end", "action": "complete"})
        lines = (tmp_path / "progress.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "lane_start"
        assert json.loads(lines[1])["action"] == "complete"

    def test_coerces_non_dict_record(self, tmp_path):
        w = ProgressWriter(str(tmp_path))
        w.write(SimpleNamespace(outcome="deny", reason="nope"))
        rec = json.loads((tmp_path / "progress.jsonl").read_text().splitlines()[0])
        assert rec["event"] == "permission"
        assert rec["outcome"] == "deny"


# ---- run_acp_lane (fake-connection driven; launches nothing) ----------------


class _FakeConn:
    def __init__(self, *, stop_reason, chunks, on_event):
        self._stop_reason = stop_reason
        self._chunks = chunks
        self._on_event = on_event
        self.prompts: list[str] = []

    async def create_session(self, *, cwd):
        return SimpleNamespace(session_id="sess-fake-1", cwd=cwd)

    async def prompt(self, session_id, text):
        self.prompts.append(text)
        for c in self._chunks:
            if self._on_event is not None:
                self._on_event({"type": "agent_message_chunk", "text": c})
        return SimpleNamespace(stop_reason=self._stop_reason)


def _make_factory(*, stop_reason="end_turn", chunks=("hello ", "world")):
    captured = {}

    @contextlib.asynccontextmanager
    async def factory(
        backend, *, cwd, workspace_path, registry, base_env, sessions, on_event
    ):
        captured["backend"] = backend
        captured["cwd"] = cwd
        conn = _FakeConn(stop_reason=stop_reason, chunks=chunks, on_event=on_event)
        captured["conn"] = conn
        yield conn

    return factory, captured


class TestRunAcpLane:
    @pytest.mark.asyncio
    async def test_refuses_real_launch_without_allow_flag(self):
        plan = LaunchPlan(transport=TRANSPORT_ACP, reason="x", backend="claude")
        with pytest.raises(AcpClientUnavailable):
            await run_acp_lane(
                plan, workspace="/tmp/ws", prompt_text="hi"
            )  # no connection_factory + allow_launch=False

    @pytest.mark.asyncio
    async def test_rejects_non_acp_plan(self):
        plan = LaunchPlan(transport=TRANSPORT_PTY, reason="x")
        with pytest.raises(ValueError):
            await run_acp_lane(
                plan, workspace="/tmp/ws", prompt_text="hi", allow_launch=True
            )

    @pytest.mark.asyncio
    async def test_drives_fake_connection_end_to_end(self, tmp_path):
        plan = LaunchPlan(
            transport=TRANSPORT_ACP, reason="x", backend="claude", command="claude"
        )
        factory, captured = _make_factory(stop_reason="end_turn")
        progress = ProgressWriter(str(tmp_path))
        decision = await run_acp_lane(
            plan,
            workspace=str(tmp_path),
            prompt_text="do the thing",
            progress=progress,
            connection_factory=factory,
        )
        assert decision.action == "complete"
        assert decision.lane_status == "done"
        assert decision.summary == "hello world"  # accumulated from chunks
        assert captured["conn"].prompts == ["do the thing"]
        # progress.jsonl captured lane_start, the message chunks, and lane_end.
        events = [
            json.loads(line)
            for line in (tmp_path / "progress.jsonl").read_text().splitlines()
        ]
        assert events[0]["event"] == "lane_start"
        assert events[-1]["event"] == "lane_end"
        assert events[-1]["action"] == "complete"

    @pytest.mark.asyncio
    async def test_blocked_stop_reason_maps_to_block(self, tmp_path):
        plan = LaunchPlan(transport=TRANSPORT_ACP, reason="x", backend="claude")
        factory, _ = _make_factory(stop_reason="max_tokens", chunks=())
        decision = await run_acp_lane(
            plan,
            workspace=str(tmp_path),
            prompt_text="hi",
            connection_factory=factory,
        )
        assert decision.action == "block"
        assert decision.lane_status == "blocked"
