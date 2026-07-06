"""Gateway integration tests for the live subagent roster bubble.

Drives the real GatewayRunner._run_agent path with a fake AIAgent that emits
subagent.start / subagent.complete lifecycle events, asserting the roster
bubble is seeded, edited, collapses, and survives the fast-child / stale-run /
burst / seed-failure races the dual-review flagged (B1, B2, Concern 1/3).

Harness mirrors tests/gateway/test_run_progress_interrupt.py.
"""

import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class ProgressCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM, seed_fails=False, flood_seed_fails=0):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.typing = []
        self._seed_fails = seed_fails
        # When > 0, the first N seed sends are rejected with a FLOOD result
        # (known-not-delivered) and then sends succeed — models Telegram flood
        # control clearing. Distinct from seed_fails, which is an AMBIGUOUS
        # failure (might have delivered → must latch, never retry).
        self._flood_seed_fails = flood_seed_fails
        self._send_seq = 0

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self._send_seq += 1
        self.sent.append({"chat_id": chat_id, "content": content})
        if self._flood_seed_fails > 0:
            self._flood_seed_fails -= 1
            # Telegram long flood: success=False, error="flood_control:{wait}".
            return SendResult(success=False, message_id=None, error="flood_control:18")
        if self._seed_fails:
            return SendResult(success=False, message_id=None)
        return SendResult(success=True, message_id=f"roster-{self._send_seq}")

    async def edit_message(self, chat_id, message_id, content, **kwargs) -> SendResult:
        self.edits.append({"message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append(chat_id)

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class NoEditAdapter(ProgressCaptureAdapter):
    """Adapter without message editing — roster must no-op (Concern 5)."""

    # Inherit BasePlatformAdapter.edit_message (the sentinel the consumer checks)
    edit_message = BasePlatformAdapter.edit_message


def _emit_start(cb, sid, goal, idx):
    if cb is None:
        return
    cb("subagent.start", None, goal, subagent_id=sid, goal=goal, task_index=idx, task_count=2)


def _emit_complete(cb, sid, status="completed", duration=1.0):
    if cb is None:
        return
    cb("subagent.complete", None, None, subagent_id=sid, status=status, duration_seconds=duration)


class RosterAgent:
    """Emits a 2-child fan-out: both start, both complete."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        _emit_start(cb, "sa-0", "verify php", 0)
        _emit_start(cb, "sa-1", "verify fe", 1)
        time.sleep(0.5)  # let the idle tick render the running roster
        _emit_complete(cb, "sa-0", "completed", 1.2)
        _emit_complete(cb, "sa-1", "failed", 2.4)
        time.sleep(0.4)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FastChildAgent:
    """B1: emits start+complete then returns immediately (before any tick)."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        _emit_start(cb, "sa-0", "fast child", 0)
        _emit_complete(cb, "sa-0", "completed", 0.1)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class BurstAgent:
    """Concern 1: 10 children all start near-simultaneously."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        if cb is None:
            return {"final_response": "done", "messages": [], "api_calls": 1}
        for i in range(10):
            cb("subagent.start", None, f"child {i}", subagent_id=f"sa-{i}",
               goal=f"child {i}", task_index=i, task_count=10)
        time.sleep(0.5)
        for i in range(10):
            cb("subagent.complete", None, None, subagent_id=f"sa-{i}",
               status="completed", duration_seconds=1.0)
        time.sleep(0.4)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class NoRosterEmittingToolAgent:
    """Emits a normal tool.started — with roster on but tool_progress off,
    NO ordinary tool bubble should appear (Concern 6)."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        cb("tool.started", "web_search", "should not render", {})
        _emit_start(cb, "sa-0", "real child", 0)
        time.sleep(0.4)
        _emit_complete(cb, "sa-0", "completed", 1.0)
        time.sleep(0.3)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class SlowRosterAgent:
    """Like RosterAgent but runs long enough (~2.5s) that, with the roster
    interval at its 1.0s floor, a flood-rejected seed has time to RE-SEED on a
    later idle tick. Used by the sync-path flood-reseed test."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_requested

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        _emit_start(cb, "sa-0", "verify php", 0)
        _emit_start(cb, "sa-1", "verify fe", 1)
        time.sleep(2.5)  # > 1.0s floor: a flooded seed gets a paced re-seed
        _emit_complete(cb, "sa-0", "completed", 1.2)
        _emit_complete(cb, "sa-1", "completed", 2.4)
        time.sleep(0.3)
        return {"final_response": "done", "messages": [], "api_calls": 1}


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


async def _run(monkeypatch, tmp_path, agent_cls, session_id, *,
               adapter=None, roster="on", tool_progress="off", roster_interval=None):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    if adapter is None:
        adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    # Inject display config: roster on, tool_progress off (the key combo).
    cfg = {
        "display": {
            "platforms": {
                "telegram": {
                    "subagent_roster": roster,
                    "tool_progress": tool_progress,
                }
            }
        }
    }
    if roster_interval is not None:
        cfg["display"]["platforms"]["telegram"]["subagent_roster_interval"] = roster_interval
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: cfg)
    monkeypatch.delenv("HERMES_TOOL_PROGRESS_MODE", raising=False)

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", thread_id="17585",
    )
    result = await runner._run_agent(
        message="hi", context_prompt="", history=[], source=source,
        session_id=session_id, session_key="agent:main:telegram:group:-1001:17585",
    )
    return adapter, result


def _all_text(adapter):
    return [c["content"] for c in adapter.sent] + [c["content"] for c in adapter.edits]


@pytest.mark.asyncio
async def test_roster_on_renders_bubble(monkeypatch, tmp_path):
    adapter, result = await _run(monkeypatch, tmp_path, RosterAgent, "sess-roster")
    assert result["final_response"] == "done"
    blob = "\n".join(_all_text(adapter))
    # Roster bubble seeded + collapsed. Throttling may coalesce both starts into
    # one frame, so assert the bubble exists + the final collapse counts are
    # right (2 children, 1 done, 1 failed) rather than every intermediate label.
    assert "🤖" in blob
    assert "verify php" in blob  # at least the first child's label rendered
    assert "2 subagent" in blob and "1 ✓" in blob and "1 ✗" in blob
    # exactly one seed send for the roster bubble
    assert len(adapter.sent) == 1, f"expected one seed send, got {len(adapter.sent)}"


@pytest.mark.asyncio
async def test_roster_off_renders_nothing(monkeypatch, tmp_path):
    adapter, result = await _run(monkeypatch, tmp_path, RosterAgent, "sess-off", roster="off")
    assert result["final_response"] == "done"
    blob = "\n".join(_all_text(adapter))
    assert "🤖" not in blob and "verify php" not in blob
    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_no_ordinary_tool_bubble_in_roster_only_mode(monkeypatch, tmp_path):
    # Concern 6: roster on + tool_progress off must NOT leak normal tool rows.
    adapter, result = await _run(
        monkeypatch, tmp_path, NoRosterEmittingToolAgent, "sess-no-leak"
    )
    blob = "\n".join(_all_text(adapter))
    assert "should not render" not in blob
    assert "🤖" in blob and "real child" in blob  # roster still works


@pytest.mark.asyncio
async def test_fast_child_still_collapses(monkeypatch, tmp_path):
    # B1: child finishes before the first tick; must still emit ONE summary.
    adapter, result = await _run(monkeypatch, tmp_path, FastChildAgent, "sess-fast")
    blob = "\n".join(_all_text(adapter))
    # A roster lead marker must be present: 🤖 (live) or, when the child
    # finished before the first tick, the collapsed finished marker ✅ / ⚠️.
    assert any(m in blob for m in ("🤖", "✅", "⚠️")), (
        "fast child produced no roster output at all (B1 regression)"
    )
    # the collapsed one-liner form
    assert "subagent" in blob


@pytest.mark.asyncio
async def test_burst_does_not_send_many_messages(monkeypatch, tmp_path):
    # Concern 1: 10 simultaneous starts -> one seed send, edits throttled.
    adapter, result = await _run(monkeypatch, tmp_path, BurstAgent, "sess-burst")
    assert len(adapter.sent) == 1, f"burst seeded {len(adapter.sent)} messages, expected 1"
    # edits are throttled (3s interval) — a 0.9s run should produce few edits.
    assert len(adapter.edits) <= 4, f"burst produced {len(adapter.edits)} edits, expected throttled"


@pytest.mark.asyncio
async def test_seed_failure_no_retry_spam(monkeypatch, tmp_path):
    # Concern 3: seed send fails -> no repeated seed attempts (no dupe bubbles).
    adapter = ProgressCaptureAdapter(seed_fails=True)
    _, result = await _run(monkeypatch, tmp_path, RosterAgent, "sess-seedfail", adapter=adapter)
    assert result["final_response"] == "done"
    # multiple ticks happened, but seed must be attempted at most a small number
    # of times, never once-per-tick spam.
    assert len(adapter.sent) <= 2, f"seed retried {len(adapter.sent)} times (spam)"


@pytest.mark.asyncio
async def test_no_edit_adapter_noops(monkeypatch, tmp_path):
    # Concern 5: adapter without edit support -> roster silently no-ops.
    adapter = NoEditAdapter()
    _, result = await _run(monkeypatch, tmp_path, RosterAgent, "sess-noedit", adapter=adapter)
    assert result["final_response"] == "done"
    assert len(adapter.sent) == 0 and len(adapter.edits) == 0


@pytest.mark.asyncio
async def test_sync_roster_reseeds_after_flood(monkeypatch, tmp_path):
    # The sync (in-turn) _publish_roster path: a flood-rejected seed must NOT
    # latch — a later paced idle tick re-seeds, so the bubble still appears once
    # flood clears. roster_interval=1.0 (floor) + a ~2.5s agent gives the
    # re-seed window. The first send floods; the re-seed lands the bubble.
    adapter = ProgressCaptureAdapter(flood_seed_fails=1)
    _, result = await _run(
        monkeypatch, tmp_path, SlowRosterAgent, "sess-sync-flood",
        adapter=adapter, roster_interval=1.0,
    )
    assert result["final_response"] == "done"
    # At least two sends: the flooded seed + a successful re-seed. (No latch.)
    assert len(adapter.sent) >= 2, (
        f"flood seed must re-seed on the sync path, got {len(adapter.sent)} sends"
    )
    # The bubble actually landed: a roster lead marker is present in some frame.
    blob = "\n".join(_all_text(adapter))
    assert any(m in blob for m in ("🤖", "✅", "⚠️")), "re-seeded roster never rendered"


@pytest.mark.asyncio
async def test_sync_roster_latches_on_ambiguous_seed_failure(monkeypatch, tmp_path):
    # Mirror guard: an AMBIGUOUS (non-flood) seed failure on the sync path must
    # STILL latch — never re-seed — even with a long agent + 1.0s interval that
    # would otherwise allow a retry. This locks the nonlocal/latch contract that
    # py_compile cannot catch (an accidental local would re-seed and spam).
    adapter = ProgressCaptureAdapter(seed_fails=True)
    _, result = await _run(
        monkeypatch, tmp_path, SlowRosterAgent, "sess-sync-ambig",
        adapter=adapter, roster_interval=1.0,
    )
    assert result["final_response"] == "done"
    # Latched after the first failure: at most one seed attempt despite a ~2.5s
    # run with sub-second idle ticks (a broken latch would send many).
    assert len(adapter.sent) <= 1, (
        f"ambiguous seed failure must latch, got {len(adapter.sent)} sends (spam)"
    )
