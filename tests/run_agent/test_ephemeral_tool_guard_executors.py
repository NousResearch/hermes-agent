"""The ephemeral tool-write guard must hold on EVERY executor path.

Regression tests for the /temp / --no-session leak where memory,
skill_manage and cronjob writes sailed through the sequential executor:
the guard lived only in ``invoke_tool``, which only the concurrent
executor reaches, while the guarded tools essentially never run there —
single tool calls route sequential unconditionally
(``run_agent._execute_tool_calls``), and none of the guarded tools are
parallel-safe, so batched calls land in sequential barrier segments too.
The enforced boundary is now ``_run_agent_tool_execution_middleware``
(agent/tool_executor.py), which both executors funnel through.

These tests drive the REAL production router (``agent._execute_tool_calls``)
with ``agent.ephemeral = True`` and observe the genuine dispatch surfaces:
``run_agent.handle_function_call`` for registry tools and the memory store
for the inline memory branch. A recorder stands in for
``handle_function_call`` so nothing durable can be created even when the
guard regresses — the leak is then visible as a recorded dispatch instead
of a real cron job.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

BLOCK_MARKER = "blocked in this temporary chat"


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent(*, ephemeral: bool):
    from run_agent import AIAgent

    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs(
                "web_search", "memory", "cronjob", "skill_manage"
            ),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            # The REAL constructor wiring: init_agent derives
            # _persist_disabled, _session_json_enabled, and the temporary-
            # registry entry from this flag — stamping agent.ephemeral after
            # construction would leave those untested.
            ephemeral=ephemeral,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._flush_messages_to_session_db = MagicMock()
    # Inline memory branch dependencies: a mock store observes the write
    # attempt; no external providers.
    agent._memory_store = MagicMock()
    agent._memory_store.add.return_value = {"success": True, "message": "added"}
    agent._memory_manager = None
    return agent


def _mock_tool_call(name: str, arguments: dict, call_id: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _run(agent, tool_calls, *, entry: str = "router") -> list:
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)
    with patch(
        "agent.tool_executor.maybe_persist_tool_result",
        side_effect=lambda **kwargs: kwargs["content"],
    ):
        if entry == "router":
            agent._execute_tool_calls(assistant_message, messages, "task-1")
        elif entry == "concurrent":
            agent._execute_tool_calls_concurrent(
                assistant_message, messages, "task-1"
            )
        else:  # pragma: no cover - test misuse
            raise ValueError(entry)
    return messages


def _result_content(messages: list, call_id: str) -> str:
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id:
            content = m.get("content")
            return content if isinstance(content, str) else str(content)
    raise AssertionError(f"no tool result appended for {call_id}")


# ---------------------------------------------------------------------------
# Single-call writes — the shape every lone write takes in production, and
# exactly the shape that leaked: len(tool_calls) <= 1 routes SEQUENTIAL.
# ---------------------------------------------------------------------------
def test_single_memory_write_blocked_on_sequential_router():
    agent = _make_agent(ephemeral=True)
    messages = _run(
        agent,
        [_mock_tool_call("memory", {"action": "add", "content": "PROBE"}, "c1")],
    )
    content = _result_content(messages, "c1")
    assert not agent._memory_store.add.called, (
        "ephemeral chat wrote to the memory store via the sequential "
        "inline memory branch"
    )
    assert BLOCK_MARKER in content


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("cronjob", {"action": "create", "schedule": "30m", "prompt": "probe"}),
        ("cronjob", {"action": "update", "job_id": "j1", "schedule": "1h"}),
        ("cronjob", {"action": "remove", "job_id": "j1"}),
        ("cronjob", {"action": "run", "job_id": "j1", "prompt": "temp-chat text"}),
        ("skill_manage", {"action": "create", "name": "probe-skill"}),
        ("skill_manage", {"action": "delete", "name": "probe-skill"}),
    ],
)
def test_single_registry_write_blocked_on_sequential_router(tool_name, args):
    agent = _make_agent(ephemeral=True)
    dispatched = []

    def recorder(function_name, function_args, effective_task_id, **kwargs):
        dispatched.append(function_name)
        return json.dumps({"success": True})

    with patch("run_agent.handle_function_call", side_effect=recorder):
        messages = _run(agent, [_mock_tool_call(tool_name, args, "c1")])

    content = _result_content(messages, "c1")
    assert not dispatched, (
        f"ephemeral chat dispatched {tool_name} {args['action']} to "
        "handle_function_call on the sequential path"
    )
    assert BLOCK_MARKER in content


# ---------------------------------------------------------------------------
# Read/write split: read-side actions on guarded tools still execute, and a
# non-ephemeral agent dispatches writes normally. Together these pin that the
# middleware check blocks exactly the write actions and nothing else.
# ---------------------------------------------------------------------------
def test_read_side_action_still_dispatches_in_ephemeral_chat():
    agent = _make_agent(ephemeral=True)
    dispatched = []

    def recorder(function_name, function_args, effective_task_id, **kwargs):
        dispatched.append(function_name)
        return json.dumps({"success": True, "jobs": []})

    with patch("run_agent.handle_function_call", side_effect=recorder):
        messages = _run(agent, [_mock_tool_call("cronjob", {"action": "list"}, "c1")])

    content = _result_content(messages, "c1")
    assert dispatched == ["cronjob"]
    assert BLOCK_MARKER not in content


def test_writes_dispatch_normally_when_not_ephemeral():
    agent = _make_agent(ephemeral=False)
    dispatched = []

    def recorder(function_name, function_args, effective_task_id, **kwargs):
        dispatched.append(function_name)
        return json.dumps({"success": True})

    with patch("run_agent.handle_function_call", side_effect=recorder):
        messages = _run(
            agent,
            [
                _mock_tool_call(
                    "cronjob",
                    {"action": "create", "schedule": "30m", "prompt": "probe"},
                    "c1",
                )
            ],
        )
    assert dispatched == ["cronjob"]
    assert BLOCK_MARKER not in _result_content(messages, "c1")

    agent2 = _make_agent(ephemeral=False)
    messages2 = _run(
        agent2,
        [_mock_tool_call("memory", {"action": "add", "content": "PROBE"}, "c1")],
    )
    assert agent2._memory_store.add.called
    assert BLOCK_MARKER not in _result_content(messages2, "c1")


# ---------------------------------------------------------------------------
# Multi-call batch: guarded tools are never parallel-safe, so they land in a
# sequential barrier segment even when batched with parallel-safe calls. The
# barrier segment must enforce the guard while the safe call still runs.
# ---------------------------------------------------------------------------
def test_write_blocked_in_mixed_batch_segmented_path():
    agent = _make_agent(ephemeral=True)
    dispatched = []

    def recorder(function_name, function_args, effective_task_id, **kwargs):
        dispatched.append(function_name)
        return json.dumps({"success": True})

    with patch("run_agent.handle_function_call", side_effect=recorder):
        messages = _run(
            agent,
            [
                _mock_tool_call("web_search", {"query": "weather"}, "c1"),
                _mock_tool_call("memory", {"action": "add", "content": "PROBE"}, "c2"),
            ],
        )

    assert "web_search" in dispatched, "parallel-safe call should still run"
    assert not agent._memory_store.add.called
    assert BLOCK_MARKER in _result_content(messages, "c2")
    assert BLOCK_MARKER not in _result_content(messages, "c1")


# ---------------------------------------------------------------------------
# Concurrent executor: the historically-guarded path must stay guarded now
# that the middleware is the enforced boundary.
# ---------------------------------------------------------------------------
def test_write_blocked_on_concurrent_path():
    agent = _make_agent(ephemeral=True)
    messages = _run(
        agent,
        [_mock_tool_call("memory", {"action": "add", "content": "PROBE"}, "c1")],
        entry="concurrent",
    )
    assert not agent._memory_store.add.called
    assert BLOCK_MARKER in _result_content(messages, "c1")


# ---------------------------------------------------------------------------
# Guard-function precedence: memory_tool executes `operations` whenever it is
# a non-empty list and ignores any bare `action` also present. The guard must
# follow the tool's precedence, not invent its own.
# ---------------------------------------------------------------------------
def test_operations_batch_checked_even_when_bare_action_present():
    from agent.agent_runtime_helpers import check_ephemeral_tool_block

    smuggled = {
        "action": "view",
        "operations": [{"action": "add", "content": "x"}],
    }
    assert check_ephemeral_tool_block("memory", smuggled) is not None, (
        "a write batch must not ride in behind an innocuous bare action"
    )


def test_operations_batch_without_bare_action_still_blocked():
    from agent.agent_runtime_helpers import check_ephemeral_tool_block

    assert (
        check_ephemeral_tool_block(
            "memory", {"operations": [{"action": "replace", "old_text": "a", "content": "b"}]}
        )
        is not None
    )


def test_empty_operations_falls_back_to_bare_action():
    from agent.agent_runtime_helpers import check_ephemeral_tool_block

    # memory_tool treats an empty operations list as falsy and takes the
    # single-op path, so the bare action decides.
    assert (
        check_ephemeral_tool_block(
            "memory", {"action": "add", "content": "x", "operations": []}
        )
        is not None
    )


def test_batch_with_no_blocked_ops_is_allowed_through():
    from agent.agent_runtime_helpers import check_ephemeral_tool_block

    # Unknown op actions carry no durable write; the tool rejects the batch
    # atomically on its own. The guard must not block what cannot write.
    assert (
        check_ephemeral_tool_block(
            "memory", {"operations": [{"action": "bogus"}]}
        )
        is None
    )


# ---------------------------------------------------------------------------
# delegate_task children: the child gets a FRESH session id that is not in
# the hermes_state temporary registry, so it must inherit the parent's
# ephemeral contract at construction or its transcript (whose prompt
# typically distills the temporary conversation) persists as a normal
# subagent session — and its own memory/skill/cron writes dodge the guard.
# ---------------------------------------------------------------------------
def test_delegate_child_inherits_ephemeral_from_temporary_parent():
    from tools.delegate_tool import _build_child_agent

    parent = _make_agent(ephemeral=True)
    with patch("run_agent.AIAgent") as child_cls:
        _build_child_agent(
            task_index=0,
            goal="probe goal",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=3,
            task_count=1,
            parent_agent=parent,
        )
    assert child_cls.call_count == 1
    assert child_cls.call_args.kwargs.get("ephemeral") is True, (
        "delegate child of a temporary parent was built without the "
        "ephemeral flag — its session would persist"
    )


def test_delegate_child_stays_persistent_for_normal_parent():
    from tools.delegate_tool import _build_child_agent

    parent = _make_agent(ephemeral=False)
    with patch("run_agent.AIAgent") as child_cls:
        _build_child_agent(
            task_index=0,
            goal="probe goal",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=3,
            task_count=1,
            parent_agent=parent,
        )
    assert child_cls.call_count == 1
    assert child_cls.call_args.kwargs.get("ephemeral") is False


# ---------------------------------------------------------------------------
# Compression rotation. The observed leak (session 20260808_215424_903ac2):
# a desktop temporary chat carries a live session_db, and auto-compaction's
# durable commit published the full compacted transcript as a compression
# child. Three layers now stop it, tested independently:
#   1. init_agent registers the AGENT's session id in the hermes_state
#      temporary registry (the TUI registered only its RPC handle, so the
#      DB-layer refusals checked an id that never appeared in any write —
#      token accounting then created its FK row and gave the first rotation
#      a parent to chain from);
#   2. the compression durable-commit block is skipped for
#      persistence-isolated agents (memory-only compaction, id unchanged);
#   3. publish_compression_child refuses registered ids outright.
# ---------------------------------------------------------------------------
def test_init_agent_registers_agent_session_id_and_close_releases_it():
    from hermes_state import is_session_ephemeral

    agent = _make_agent(ephemeral=True)
    assert agent.session_id, "agent must have a session id"
    assert is_session_ephemeral(agent.session_id), (
        "ephemeral agent's own session id must be in the temporary registry — "
        "DB rows are keyed by THIS id, not the TUI's RPC handle"
    )
    agent.close()
    assert not is_session_ephemeral(agent.session_id)

    normal = _make_agent(ephemeral=False)
    assert not is_session_ephemeral(normal.session_id)


def test_compression_rotation_publishes_nothing_for_temporary_chat(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        agent = _make_agent(ephemeral=True)
        agent._session_db = db
        agent.compression_in_place = False
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail question"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        agent.context_compressor = compressor
        original_sid = agent.session_id

        with patch.object(
            db, "publish_compression_child", wraps=db.publish_compression_child
        ) as publish:
            agent._compress_context(
                [{"role": "user", "content": f"m{i}"} for i in range(10)],
                "sys",
                approx_tokens=10_000,
            )

        publish.assert_not_called()
        assert agent.session_id == original_sid, (
            "temporary chats must compact in memory without rotating the id"
        )
        with db._read_ctx() as conn:
            rows = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert rows == 0, "compression left durable rows for a temporary chat"
    finally:
        db.close()


def test_publish_compression_child_refuses_registered_ids(tmp_path):
    from hermes_state import (
        SessionDB,
        mark_session_ephemeral,
        unmark_session_ephemeral,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    parent = "publish-refusal-parent"
    try:
        # The leak's exact shape: a parent row EXISTS (created before the id
        # was protected) — the registry must still refuse the publication.
        db.create_session(parent, "desktop")
        mark_session_ephemeral(parent)
        with pytest.raises(RuntimeError, match="temporary"):
            db.publish_compression_child(
                parent_session_id=parent,
                child_session_id="publish-refusal-child",
                source="desktop",
                messages=[{"role": "user", "content": "x"}],
                require_compression_lease=False,
            )
        with db._read_ctx() as conn:
            child_rows = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id = ?",
                ("publish-refusal-child",),
            ).fetchone()[0]
        assert child_rows == 0
    finally:
        unmark_session_ephemeral(parent)
        unmark_session_ephemeral("publish-refusal-child")
        db.close()


# ---------------------------------------------------------------------------
# Background self-improvement review: the fork's entire output is durable
# state (MEMORY.md, the skill library) distilled from the conversation, so it
# must never spawn for a temporary chat. Its own _persist_disabled is the
# weaker contract — it deliberately still allows memory/skill writes. The
# guard lives inside _spawn_background_review so the automatic post-turn
# trigger, CLI /refine, and gateway /refine are all covered at once.
# ---------------------------------------------------------------------------
def test_background_review_never_spawns_for_temporary_chat():
    agent = _make_agent(ephemeral=True)
    with (
        patch(
            "agent.background_review.spawn_background_review_thread"
        ) as spawn,
        patch("run_agent.threading.Thread") as thread_cls,
    ):
        agent._spawn_background_review(
            messages_snapshot=[{"role": "user", "content": "PROBE"}],
            review_memory=True,
            review_skills=True,
        )
    spawn.assert_not_called()
    thread_cls.assert_not_called()


def test_background_review_still_spawns_for_normal_sessions():
    agent = _make_agent(ephemeral=False)
    with (
        patch(
            "agent.background_review.spawn_background_review_thread",
            return_value=(lambda: None, "prompt"),
        ) as spawn,
        patch("run_agent.threading.Thread") as thread_cls,
    ):
        agent._spawn_background_review(
            messages_snapshot=[{"role": "user", "content": "q"}],
            review_memory=True,
            review_skills=False,
        )
    spawn.assert_called_once()
    thread_cls.assert_called_once()
    thread_cls.return_value.start.assert_called_once()


# ---------------------------------------------------------------------------
# Trajectory capture: a trajectory line is the full message history, so
# persistence-isolated agents must skip it even when the operator runs with
# --save_trajectories.
# ---------------------------------------------------------------------------
def test_trajectory_capture_skipped_for_persist_disabled_agents():
    agent = _make_agent(ephemeral=True)
    agent.save_trajectories = True
    with patch.object(agent, "_convert_to_trajectory_format") as convert:
        agent._save_trajectory([{"role": "user", "content": "PROBE"}], "PROBE", True)
    convert.assert_not_called()


def test_trajectory_capture_still_works_for_normal_agents():
    agent = _make_agent(ephemeral=False)
    agent.save_trajectories = True
    with (
        patch.object(
            agent, "_convert_to_trajectory_format", return_value={"messages": []}
        ) as convert,
        patch("run_agent._save_trajectory_to_file") as save,
    ):
        agent._save_trajectory([{"role": "user", "content": "q"}], "q", True)
    convert.assert_called_once()
    save.assert_called_once()


# ---------------------------------------------------------------------------
# Alternate dispatch surfaces that call handle_function_call directly, WITHOUT
# the executor middleware: the code-execution sandbox bridge and the
# hermes-tools MCP transport. Today neither can reach a guarded tool — the
# sandbox allow-list and the transport's EXPOSED_TOOLS exclude them all — and
# that exclusion is what makes middleware-level enforcement complete. Anyone
# adding a guarded tool to either surface must add an ephemeral check at that
# surface first.
# ---------------------------------------------------------------------------
def test_sandbox_allowlist_disjoint_from_ephemeral_guarded_tools():
    from agent.agent_runtime_helpers import _EPHEMERAL_BLOCKED_TOOL_ACTIONS
    from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS

    overlap = SANDBOX_ALLOWED_TOOLS & set(_EPHEMERAL_BLOCKED_TOOL_ACTIONS)
    assert not overlap, (
        f"{sorted(overlap)} exposed to the code-execution sandbox, whose RPC "
        "bridge dispatches via handle_function_call and bypasses the "
        "ephemeral tool-write guard in _run_agent_tool_execution_middleware. "
        "Add an ephemeral check to the sandbox bridge before exposing these."
    )


def test_mcp_transport_exposure_disjoint_from_ephemeral_guarded_tools():
    from agent.agent_runtime_helpers import _EPHEMERAL_BLOCKED_TOOL_ACTIONS
    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

    overlap = set(EXPOSED_TOOLS) & set(_EPHEMERAL_BLOCKED_TOOL_ACTIONS)
    assert not overlap, (
        f"{sorted(overlap)} exposed via the hermes-tools MCP transport, which "
        "dispatches via handle_function_call and bypasses the ephemeral "
        "tool-write guard in _run_agent_tool_execution_middleware. Add an "
        "ephemeral check to the transport before exposing these."
    )


# ---------------------------------------------------------------------------
# External memory-provider tools (hindsight_retain, supermemory_store, ...).
# Their names come from plugins, so blocking is delegated to the owning
# provider's read_only_tool_names() declaration via
# MemoryManager.is_write_tool(), fail-closed for undeclared tools. Provider
# tools dispatch through the memory-manager branch of the sequential
# executor, which also flows through the guarded middleware.
# ---------------------------------------------------------------------------
class _StubMemoryManager:
    """Duck-typed stand-in exposing exactly what the executor and guard use:
    has_tool / is_write_tool / handle_tool_call."""

    def __init__(self):
        self.calls = []

    def has_tool(self, name):
        return name in {"fake_retain", "fake_recall"}

    def is_write_tool(self, name):
        return name == "fake_retain"

    def handle_tool_call(self, name, args, **kwargs):
        self.calls.append((name, args))
        return json.dumps({"result": "ok"})


def test_provider_write_tool_blocked_on_sequential_router():
    agent = _make_agent(ephemeral=True)
    manager = _StubMemoryManager()
    agent._memory_manager = manager

    messages = _run(agent, [_mock_tool_call("fake_retain", {"content": "PROBE"}, "c1")])
    content = _result_content(messages, "c1")
    assert not manager.calls, (
        "ephemeral chat dispatched a provider write tool to the memory manager"
    )
    assert BLOCK_MARKER in content


def test_provider_read_tool_still_dispatches_in_ephemeral_chat():
    agent = _make_agent(ephemeral=True)
    manager = _StubMemoryManager()
    agent._memory_manager = manager

    messages = _run(agent, [_mock_tool_call("fake_recall", {"query": "q"}, "c1")])
    content = _result_content(messages, "c1")
    assert [c[0] for c in manager.calls] == ["fake_recall"]
    assert BLOCK_MARKER not in content


def test_provider_write_tool_dispatches_when_not_ephemeral():
    agent = _make_agent(ephemeral=False)
    manager = _StubMemoryManager()
    agent._memory_manager = manager

    messages = _run(agent, [_mock_tool_call("fake_retain", {"content": "PROBE"}, "c1")])
    assert [c[0] for c in manager.calls] == ["fake_retain"]
    assert BLOCK_MARKER not in _result_content(messages, "c1")


# ---------------------------------------------------------------------------
# MemoryManager.is_write_tool: fail-closed classification contract.
# ---------------------------------------------------------------------------
def _make_provider_class(**overrides):
    from agent.memory_provider import MemoryProvider

    class _Provider(MemoryProvider):
        def __init__(self, name="fakeprov", tools=()):
            self._name = name
            self._tools = [
                {"name": t, "description": t, "parameters": {"type": "object", "properties": {}}}
                for t in tools
            ]

        @property
        def name(self):
            return self._name

        def is_available(self):
            return True

        def initialize(self, session_id, **kwargs):
            pass

        def get_tool_schemas(self):
            return self._tools

        def handle_tool_call(self, tool_name, args, **kwargs):
            return json.dumps({"handled": tool_name})

    for attr, fn in overrides.items():
        setattr(_Provider, attr, fn)
    return _Provider


def test_is_write_tool_fails_closed_without_declaration():
    from agent.memory_manager import MemoryManager

    mgr = MemoryManager()
    provider_cls = _make_provider_class()
    mgr.add_provider(provider_cls(tools=("prov_store", "prov_search")))
    # No read_only_tool_names override: every tool counts as a write.
    assert mgr.is_write_tool("prov_store") is True
    assert mgr.is_write_tool("prov_search") is True
    # Tools no provider owns are not memory-provider tools at all.
    assert mgr.is_write_tool("unrelated_tool") is False


def test_is_write_tool_respects_declared_read_only():
    from agent.memory_manager import MemoryManager

    mgr = MemoryManager()
    provider_cls = _make_provider_class(
        read_only_tool_names=lambda self: frozenset({"prov_search"})
    )
    mgr.add_provider(provider_cls(tools=("prov_store", "prov_search")))
    assert mgr.is_write_tool("prov_store") is True
    assert mgr.is_write_tool("prov_search") is False


def test_is_write_tool_fails_closed_when_declaration_raises():
    from agent.memory_manager import MemoryManager

    def _broken(self):
        raise RuntimeError("boom")

    mgr = MemoryManager()
    provider_cls = _make_provider_class(read_only_tool_names=_broken)
    mgr.add_provider(provider_cls(tools=("prov_search",)))
    assert mgr.is_write_tool("prov_search") is True


# ---------------------------------------------------------------------------
# Classification pins for the in-repo providers. Verified against each
# handler's implementation; a tool moving between sets is a deliberate
# policy change and must update this table. Instances are created with
# object.__new__ because read_only_tool_names() reads no instance state and
# plugin constructors are not needed for it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("module", "cls_name", "expected_read_only"),
    [
        ("plugins.memory.byterover", "ByteRoverMemoryProvider",
         {"brv_query", "brv_status"}),
        ("plugins.memory.hindsight", "HindsightMemoryProvider",
         {"hindsight_recall", "hindsight_reflect"}),
        # Holographic declares nothing read-only on purpose: fact_store is a
        # read/write hybrid (action=add/update/remove) and fact_feedback
        # mutates trust — the fail-closed default is its true classification.
        ("plugins.memory.holographic", "HolographicMemoryProvider", set()),
        ("plugins.memory.honcho", "HonchoMemoryProvider",
         {"honcho_search", "honcho_reasoning", "honcho_context"}),
        ("plugins.memory.mem0", "Mem0MemoryProvider", {"mem0_search"}),
        ("plugins.memory.openviking", "OpenVikingMemoryProvider",
         {"viking_search", "viking_read", "viking_browse"}),
        ("plugins.memory.retaindb", "RetainDBMemoryProvider",
         {"retaindb_profile", "retaindb_search", "retaindb_context",
          "retaindb_list_files", "retaindb_read_file"}),
        ("plugins.memory.supermemory", "SupermemoryMemoryProvider",
         {"supermemory_search", "supermemory_profile",
          "supermemory-search", "supermemory-profile"}),
    ],
)
def test_in_repo_provider_read_only_classification(module, cls_name, expected_read_only):
    import importlib

    cls = getattr(importlib.import_module(module), cls_name)
    instance = object.__new__(cls)
    declared = cls.read_only_tool_names(instance)
    assert declared == frozenset(expected_read_only)


@pytest.mark.parametrize(
    ("module", "cls_name", "write_tools"),
    [
        ("plugins.memory.byterover", "ByteRoverMemoryProvider", {"brv_curate"}),
        ("plugins.memory.hindsight", "HindsightMemoryProvider", {"hindsight_retain"}),
        ("plugins.memory.holographic", "HolographicMemoryProvider",
         {"fact_store", "fact_feedback"}),
        ("plugins.memory.honcho", "HonchoMemoryProvider",
         {"honcho_profile", "honcho_conclude"}),
        ("plugins.memory.mem0", "Mem0MemoryProvider",
         {"mem0_add", "mem0_update", "mem0_delete"}),
        ("plugins.memory.openviking", "OpenVikingMemoryProvider",
         {"viking_remember", "viking_forget", "viking_add_resource"}),
        ("plugins.memory.retaindb", "RetainDBMemoryProvider",
         {"retaindb_remember", "retaindb_forget", "retaindb_upload_file",
          "retaindb_ingest_file", "retaindb_delete_file"}),
        ("plugins.memory.supermemory", "SupermemoryMemoryProvider",
         {"supermemory_store", "supermemory_forget",
          "supermemory-save", "supermemory-forget"}),
    ],
)
def test_in_repo_provider_write_tools_not_declared_read_only(module, cls_name, write_tools):
    import importlib

    cls = getattr(importlib.import_module(module), cls_name)
    instance = object.__new__(cls)
    declared = cls.read_only_tool_names(instance)
    leaked = declared & write_tools
    assert not leaked, (
        f"{sorted(leaked)} are write tools declared read-only — they would "
        "run in temporary chats and durably mutate external memory"
    )
