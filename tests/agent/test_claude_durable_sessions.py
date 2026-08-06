"""Contract for durable Claude Agent SDK sessions: resume, rewind, recovery.

The property under test throughout is the ownership boundary: Hermes remembers
*which* Claude session belongs to a conversation and never rebuilds Claude's
context.  Concretely —

* a restart resumes the same SDK session rather than replaying history, and
  the visible Hermes transcript gains exactly one user row per turn;
* a rewind forks the SDK session at the message UUID that opened the turn
  being replaced, and leaves the parent transcript untouched;
* a branch forks too, so the child starts warm and isolated;
* a stale binding recovers in exactly one bootstrap+retry and cannot loop;
* a provider switch bootstraps once and resumes forever after — asserting
  that history is NOT rebuilt on turn two is the prompt-cache guarantee.

``claude-agent-sdk`` is an optional extra; a stand-in is installed when it is
absent, and every assertion here is about Hermes' own behaviour so the suite
means the same thing either way.
"""

from __future__ import annotations

import sys
import types
import uuid
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from agent import claude_runtime
from agent.claude_runtime import (
    MAX_SESSION_RECOVERIES,
    ClaudeEventProjector,
    build_claude_agent_options,
    claude_bootstrap_prefix,
    prepare_claude_sdk_session,
    run_claude_agent_sdk_turn,
)
from agent.claude_session_store import RUNTIME
from hermes_state import SessionDB
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# SDK stand-in (used only when the optional extra is not installed)
# ---------------------------------------------------------------------------


@dataclass
class _FakeHookMatcher:
    matcher: object = None
    hooks: list = field(default_factory=list)
    timeout: object = None


@dataclass
class _FakeClaudeAgentOptions:
    system_prompt: object = None
    tools: object = None
    allowed_tools: list = field(default_factory=list)
    mcp_servers: dict = field(default_factory=dict)
    strict_mcp_config: bool = False
    setting_sources: object = None
    cwd: object = None
    env: dict = field(default_factory=dict)
    stderr: object = None
    model: object = None
    include_partial_messages: bool = False
    resume: object = None
    session_store: object = None
    session_store_flush: str = "batched"
    enable_file_checkpointing: bool = False
    hooks: object = None


@pytest.fixture(autouse=True)
def sdk_module(monkeypatch):
    """Yield an importable ``claude_agent_sdk``, faking it when absent."""
    try:  # pragma: no cover - exercised only where the extra is installed
        import claude_agent_sdk  # noqa: F401

        yield sys.modules["claude_agent_sdk"]
        return
    except ImportError:
        pass

    module = types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = _FakeClaudeAgentOptions
    module.HookMatcher = _FakeHookMatcher
    module.project_key_for_directory = lambda directory=None: f"key:{directory}"
    module.fold_session_summary = _fake_fold
    module.tool = lambda *a, **k: (lambda fn: fn)
    module.create_sdk_mcp_server = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.ToolAnnotations = type("ToolAnnotations", (), {})
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    yield module


def _fake_fold(previous, key, entries):
    data = dict((previous or {}).get("data") or {})
    data["entries"] = int(data.get("entries", 0)) + len(entries)
    return {"session_id": key["session_id"], "mtime": 0, "data": data}


# ---------------------------------------------------------------------------
# SDK message shapes (the runtime dispatches on class name)
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    text: str


@dataclass
class AssistantMessage:
    content: list
    session_id: str | None = None


@dataclass
class ResultMessage:
    subtype: str = "success"
    session_id: str = "sdk-1"
    result: str | None = None
    usage: dict | None = None
    total_cost_usd: float | None = None
    terminal_reason: str | None = None
    is_error: bool = False
    errors: list | None = None


@dataclass
class SystemMessage:
    subtype: str
    data: dict = field(default_factory=dict)


@dataclass
class MirrorErrorMessage:
    subtype: str = "mirror_error"
    data: dict = field(default_factory=dict)
    key: dict | None = None
    error: str = ""


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


HERMES_SESSION = "hermes-session-1"
CWD = "/tmp/hermes-claude-test"


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session(HERMES_SESSION, "cli")
    yield session_db
    session_db.close()


def _make_agent(db) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.api_mode = "claude_agent_sdk"
    agent.client = MagicMock()
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.session_id = HERMES_SESSION
    agent.session_cwd = CWD
    agent._session_db = db
    agent._session_db_created = True
    # Persist visible rows the way the real flush does, without dragging the
    # whole run_agent persistence stack into a runtime test.
    agent._flush_messages_to_session_db = lambda messages, *a, **k: _flush(db, messages)
    return agent


_FLUSHED = "_test_flushed"


def _flush(db, messages):
    for message in messages:
        if message.get(_FLUSHED):
            continue
        message[_FLUSHED] = True
        content = message.get("content")
        db.append_message(
            HERMES_SESSION,
            message.get("role", "user"),
            content if isinstance(content, str) else "",
            display_kind=message.get("display_kind"),
        )
    return True


class _StubSession:
    """Replays a scripted message list; optionally fails the first attempt."""

    def __init__(self, script, *, fail_once=None):
        self.script = script
        self.fail_once = fail_once
        self.closed = False
        self.prompts: list[str] = []

    def run_turn(self, prompt, *, on_message, timeout=None):
        self.prompts.append(prompt)
        if self.fail_once is not None:
            failure, self.fail_once = self.fail_once, None
            raise failure
        for message in self.script:
            on_message(message)
        return len(self.script)

    def note_session_id(self, session_id):
        pass

    def close(self):
        self.closed = True


def _script(sdk_session_id: str, text: str = "ok"):
    return [
        AssistantMessage(content=[TextBlock(text)], session_id=sdk_session_id),
        ResultMessage(session_id=sdk_session_id, result=text),
    ]


def _turn(agent, session, messages, user_message="hello"):
    """Drive one Claude turn with the preflight and session build stubbed."""
    messages.append({"role": "user", "content": user_message})
    with (
        patch.object(claude_runtime, "claude_runtime_preflight", return_value=None),
        patch.object(
            claude_runtime, "verify_claude_billing_for_agent", return_value=None
        ),
        patch.object(claude_runtime, "_ensure_session", return_value=session),
    ):
        return run_claude_agent_sdk_turn(
            agent,
            user_message=user_message,
            original_user_message=user_message,
            messages=messages,
            effective_task_id="task-1",
        )


def _mirror(db, agent, sdk_session_id, entries):
    """Stand in for the SDK's transcript mirror landing a batch."""
    db.append_provider_transcript_entries(
        RUNTIME,
        agent._claude_project_key,
        sdk_session_id,
        "",
        entries,
    )


def _user_entry(text="hello"):
    return {
        "type": "user",
        "uuid": str(uuid.uuid4()),
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {"role": "user", "content": text},
    }


def _assistant_entry(text="ok"):
    return {
        "type": "assistant",
        "uuid": str(uuid.uuid4()),
        "timestamp": "2026-01-01T00:00:01.000Z",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_options_carry_an_eager_mirror_and_never_the_sdks_own_checkpoints(db):
    agent = _make_agent(db)
    agent._claude_sdk_resume_id = None

    options = build_claude_agent_options(
        agent, system_prompt="sys", effective_task_id=lambda: "t", cwd=CWD
    )

    assert options.session_store is not None
    # Eager, because Hermes' rewind boundary is a message UUID and a batched
    # flush would leave the newest turn unmappable until the next one.
    assert options.session_store_flush == "eager"
    # The SDK rejects the pair, and Hermes owns checkpointing.
    assert not getattr(options, "enable_file_checkpointing", False)
    assert getattr(options, "resume", None) is None


def test_options_resume_the_session_bound_to_this_conversation(db):
    agent = _make_agent(db)
    db.bind_provider_runtime_session(HERMES_SESSION, RUNTIME, "sdk-abc")

    prepare_claude_sdk_session(agent, CWD)
    options = build_claude_agent_options(
        agent, system_prompt="sys", effective_task_id=lambda: "t", cwd=CWD
    )

    assert options.resume == "sdk-abc"


def test_no_session_db_means_no_mirror_and_no_resume(db):
    """A background-review fork or a persistence-disabled agent still runs."""
    agent = _make_agent(db)
    agent._session_db = None

    state = prepare_claude_sdk_session(agent, CWD)
    options = build_claude_agent_options(
        agent, system_prompt="sys", effective_task_id=lambda: "t", cwd=CWD
    )

    assert state == {"resume": None, "bootstrap": False, "recoveries": 0}
    assert options.session_store is None
    assert getattr(options, "resume", None) is None


# ---------------------------------------------------------------------------
# Restart → resume
# ---------------------------------------------------------------------------


def test_a_restart_resumes_the_same_sdk_session_without_duplicating_the_turn(db):
    first = _make_agent(db)
    messages: list[dict] = []
    session = _StubSession(_script("sdk-1"))
    _turn(first, session, messages, "first question")

    assert db.get_provider_runtime_session(HERMES_SESSION, RUNTIME)[
        "provider_session_id"
    ] == "sdk-1"

    # A new process: a new AIAgent over the same session id and the same DB,
    # with the transcript reloaded rather than carried in memory.
    restarted = _make_agent(db)
    reloaded = [{"role": "user", "content": "first question", _FLUSHED: True},
                {"role": "assistant", "content": "ok", _FLUSHED: True}]
    second_session = _StubSession(_script("sdk-1"))
    _turn(restarted, second_session, reloaded, "second question")

    assert restarted._claude_sdk_resume_id == "sdk-1"
    # The whole point: the prompt is the user's message, NOT a replay of
    # history on top of a session that already has it.
    assert second_session.prompts == ["second question"]
    stored = [m for m in db.get_messages(HERMES_SESSION) if m["role"] == "user"]
    assert [m["content"] for m in stored] == ["first question", "second question"]


def test_the_binding_survives_in_the_db_not_just_on_the_agent(db):
    agent = _make_agent(db)
    _turn(agent, _StubSession(_script("sdk-77")), [], "hi")

    binding = db.get_provider_runtime_session(HERMES_SESSION, RUNTIME)

    assert binding["provider_session_id"] == "sdk-77"
    assert binding["bootstrapped"] is True
    assert binding["project_key"] == agent._claude_project_key


# ---------------------------------------------------------------------------
# Provider switch → bootstrap once, then resume
# ---------------------------------------------------------------------------


def test_switching_into_claude_bootstraps_once_then_resumes(db):
    """Rebuilding history every turn would destroy the upstream prompt cache."""
    agent = _make_agent(db)
    history = [
        {"role": "user", "content": "what is a monad", _FLUSHED: True},
        {"role": "assistant", "content": "a monoid in the category of endofunctors",
         _FLUSHED: True},
    ]
    session_one = _StubSession(_script("sdk-boot"))
    _turn(agent, session_one, history, "explain that again")

    assert "prior_conversation" in session_one.prompts[0]
    assert "monoid in the category of endofunctors" in session_one.prompts[0]
    assert session_one.prompts[0].endswith("explain that again")

    session_two = _StubSession(_script("sdk-boot"))
    _turn(agent, session_two, history, "and again")

    # Turn two resumes. History is NOT rebuilt.
    assert session_two.prompts == ["and again"]
    assert agent._claude_sdk_resume_id == "sdk-boot"


def test_a_conversation_with_no_history_replays_no_prior_conversation(db):
    agent = _make_agent(db)
    session = _StubSession(_script("sdk-1"))

    _turn(agent, session, [], "first message ever")

    prompt = session.prompts[0]
    # Hermes' own context always rides the first turn — in subscription mode
    # it cannot go in the system prompt (decision record §11) — but with no
    # prior history there is nothing to replay.
    assert "<prior_conversation>" not in prompt
    assert "<operating_instructions>" in prompt
    assert prompt.endswith("first message ever")


def test_the_bootstrap_prefix_is_deterministic_and_bounded():
    history = [{"role": "user", "content": f"turn {i}"} for i in range(500)]

    first = claude_bootstrap_prefix(history)
    second = claude_bootstrap_prefix(history)

    assert first == second
    assert len(first) <= claude_runtime.BOOTSTRAP_MAX_CHARS + 1000
    # Tool traffic is Claude's to own from here; replaying it would imply
    # Claude issued calls it never made.
    assert claude_bootstrap_prefix(
        [{"role": "tool", "content": "result", "tool_call_id": "t1"}]
    ) == ""


# ---------------------------------------------------------------------------
# Rewind / edit
# ---------------------------------------------------------------------------


def _prime_two_turns(db):
    """Two mirrored turns with a binding, as if two Claude turns had run."""
    agent = _make_agent(db)
    agent._claude_project_key = claude_runtime.claude_project_key(CWD)
    db.bind_provider_runtime_session(
        HERMES_SESSION, RUNTIME, "sdk-parent", project_key=agent._claude_project_key
    )

    turn_one = [_user_entry("one"), _assistant_entry("answer one")]
    _mirror(db, agent, "sdk-parent", turn_one)
    row_one = db.append_message(HERMES_SESSION, "user", "one")
    db.append_message(HERMES_SESSION, "assistant", "answer one")
    db.record_provider_message_binding(
        HERMES_SESSION, 0, RUNTIME, "sdk-parent",
        message_id=row_one, project_key=agent._claude_project_key, after_entry_id=0,
    )
    db.set_provider_message_binding_uuids(
        HERMES_SESSION, 0,
        provider_message_uuid=turn_one[0]["uuid"], fork_boundary_uuid=None,
    )

    watermark = db.provider_transcript_watermark(
        RUNTIME, agent._claude_project_key, "sdk-parent"
    )
    turn_two = [_user_entry("two"), _assistant_entry("answer two")]
    _mirror(db, agent, "sdk-parent", turn_two)
    row_two = db.append_message(HERMES_SESSION, "user", "two")
    db.append_message(HERMES_SESSION, "assistant", "answer two")
    db.record_provider_message_binding(
        HERMES_SESSION, 1, RUNTIME, "sdk-parent",
        message_id=row_two, project_key=agent._claude_project_key,
        after_entry_id=watermark,
    )
    db.set_provider_message_binding_uuids(
        HERMES_SESSION, 1,
        provider_message_uuid=turn_two[0]["uuid"],
        # Everything up to and including the end of turn one survives.
        fork_boundary_uuid=turn_one[-1]["uuid"],
    )
    return agent, row_two, turn_one, turn_two


def test_rewinding_forks_the_sdk_session_at_the_surviving_boundary(db):
    agent, row_two, turn_one, _turn_two = _prime_two_turns(db)
    forks: list[tuple] = []

    def _fake_fork(store, source, cwd, up_to):
        forks.append((source, up_to))
        return "sdk-fork"

    db.rewind_to_message(HERMES_SESSION, row_two)
    with patch.object(claude_runtime, "_fork_stored_session", _fake_fork):
        state = prepare_claude_sdk_session(agent, CWD)

    # Forked from the parent, inclusive of the last entry the user kept.
    assert forks == [("sdk-parent", turn_one[-1]["uuid"])]
    assert state["resume"] == "sdk-fork"
    assert state["bootstrap"] is False
    binding = db.get_provider_runtime_session(HERMES_SESSION, RUNTIME)
    assert binding["provider_session_id"] == "sdk-fork"
    assert binding["pending_rewind_ordinal"] is None


def test_rewinding_leaves_the_parent_transcript_untouched(db):
    agent, row_two, _turn_one, _turn_two = _prime_two_turns(db)
    before = db.load_provider_transcript_entries(
        RUNTIME, agent._claude_project_key, "sdk-parent"
    )

    db.rewind_to_message(HERMES_SESSION, row_two)
    with patch.object(
        claude_runtime, "_fork_stored_session", lambda *a, **k: "sdk-fork"
    ):
        prepare_claude_sdk_session(agent, CWD)

    assert (
        db.load_provider_transcript_entries(
            RUNTIME, agent._claude_project_key, "sdk-parent"
        )
        == before
    )


def test_rewinding_drops_the_bindings_for_the_turns_being_replaced(db):
    agent, row_two, _turn_one, _turn_two = _prime_two_turns(db)

    db.rewind_to_message(HERMES_SESSION, row_two)
    with patch.object(
        claude_runtime, "_fork_stored_session", lambda *a, **k: "sdk-fork"
    ):
        prepare_claude_sdk_session(agent, CWD)

    assert db.get_provider_message_binding(HERMES_SESSION, 0) is not None
    assert db.get_provider_message_binding(HERMES_SESSION, 1) is None


def test_a_truncating_transcript_rewrite_signals_the_same_boundary(db):
    """prompt.submit's truncate (Desktop edit/regenerate) goes through here."""
    agent, _row_two, _turn_one, _turn_two = _prime_two_turns(db)

    db.replace_messages(
        HERMES_SESSION,
        [{"role": "user", "content": "one"}, {"role": "assistant", "content": "answer one"}],
    )

    binding = db.get_provider_runtime_session(HERMES_SESSION, RUNTIME)
    assert binding["pending_rewind_ordinal"] == 1


def test_rewinding_past_every_turn_starts_a_clean_session(db):
    agent, _row_two, _turn_one, _turn_two = _prime_two_turns(db)
    first_row = db.get_provider_message_binding(HERMES_SESSION, 0)["message_id"]
    forks: list = []

    db.rewind_to_message(HERMES_SESSION, first_row)
    with patch.object(
        claude_runtime, "_fork_stored_session", lambda *a, **k: forks.append(a) or "x"
    ):
        state = prepare_claude_sdk_session(agent, CWD)

    # Nothing survives, so there is nothing to fork.
    assert forks == []
    assert state["resume"] is None
    assert db.get_provider_runtime_session(HERMES_SESSION, RUNTIME)[
        "provider_session_id"
    ] == ""


def test_a_failed_fork_degrades_to_a_clean_session_rather_than_a_wrong_resume(db):
    agent, row_two, _turn_one, _turn_two = _prime_two_turns(db)

    db.rewind_to_message(HERMES_SESSION, row_two)
    with patch.object(claude_runtime, "_fork_stored_session", lambda *a, **k: None):
        state = prepare_claude_sdk_session(agent, CWD)

    assert state["resume"] is None
    # Never resume a cursor that points past discarded history.
    assert db.get_provider_runtime_session(HERMES_SESSION, RUNTIME)[
        "provider_session_id"
    ] == ""


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------


def test_branching_a_session_forks_the_sdk_session_instead_of_replaying(db):
    parent_agent, _row, _t1, _t2 = _prime_two_turns(db)
    branch_id = "hermes-branch-1"

    db.create_session(
        branch_id, "cli", model_config={"_branched_from": HERMES_SESSION}
    )

    branch_agent = _make_agent(db)
    branch_agent.session_id = branch_id
    forks: list[tuple] = []
    with patch.object(
        claude_runtime,
        "_fork_stored_session",
        lambda store, source, cwd, up_to: forks.append((source, up_to)) or "sdk-branch",
    ):
        state = prepare_claude_sdk_session(branch_agent, CWD)

    # A full fork (no boundary): the branch keeps the parent's whole context.
    assert forks == [("sdk-parent", None)]
    assert state["resume"] == "sdk-branch"
    assert state["bootstrap"] is False
    # The parent's own binding is untouched — the two sessions are isolated.
    assert db.get_provider_runtime_session(HERMES_SESSION, RUNTIME)[
        "provider_session_id"
    ] == "sdk-parent"


def test_an_ordinary_new_session_is_not_treated_as_a_branch(db):
    _prime_two_turns(db)
    db.create_session("hermes-plain", "cli")

    assert db.get_provider_runtime_session("hermes-plain", RUNTIME) is None


# ---------------------------------------------------------------------------
# Stale / deleted SDK session
# ---------------------------------------------------------------------------


def test_a_stale_session_recovers_in_exactly_one_bootstrap_retry(db):
    agent = _make_agent(db)
    db.bind_provider_runtime_session(HERMES_SESSION, RUNTIME, "sdk-gone")
    history = [{"role": "user", "content": "earlier", _FLUSHED: True},
               {"role": "assistant", "content": "earlier answer", _FLUSHED: True}]
    session = _StubSession(
        _script("sdk-new"),
        fail_once=RuntimeError("No conversation found with session ID sdk-gone"),
    )

    result = _turn(agent, session, history, "next")

    assert result["completed"] is True
    assert len(session.prompts) == 2, "the user's turn is submitted once, then retried"
    assert session.prompts[0] == "next"
    # The retry bootstraps canonical history into the fresh session.
    assert "prior_conversation" in session.prompts[1]
    assert db.get_provider_runtime_session(HERMES_SESSION, RUNTIME)[
        "provider_session_id"
    ] == "sdk-new"


def test_recovery_does_not_loop_when_the_fresh_session_also_fails(db):
    agent = _make_agent(db)
    db.bind_provider_runtime_session(HERMES_SESSION, RUNTIME, "sdk-gone")

    class _AlwaysFails(_StubSession):
        def run_turn(self, prompt, *, on_message, timeout=None):
            self.prompts.append(prompt)
            raise RuntimeError("No conversation found")

    session = _AlwaysFails([])
    result = _turn(agent, session, [], "next")

    # One recovery, then the failure is reported as a failure.
    assert len(session.prompts) == 2
    assert result["completed"] is False
    assert result["error"]


def test_the_recovery_cap_stops_a_permanently_broken_binding(db):
    agent = _make_agent(db)
    db.bind_provider_runtime_session(HERMES_SESSION, RUNTIME, "sdk-gone")
    for _ in range(MAX_SESSION_RECOVERIES):
        db.clear_provider_runtime_session(HERMES_SESSION, RUNTIME)
    db.bind_provider_runtime_session(HERMES_SESSION, RUNTIME, "sdk-gone")

    session = _StubSession(_script("sdk-new"), fail_once=RuntimeError("session not found"))
    result = _turn(agent, session, [], "next")

    assert len(session.prompts) == 1, "the cap must stop the retry"
    assert result["completed"] is False


def test_a_connect_that_fails_while_resuming_is_treated_as_stale(db):
    """The structural signal: the CLI could not open the transcript at all."""
    agent = _make_agent(db)
    db.bind_provider_runtime_session(HERMES_SESSION, RUNTIME, "sdk-gone")
    good = _StubSession(_script("sdk-new"))
    attempts = {"n": 0}

    def _ensure(_agent, _task_id):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("Failed to start Claude Code")
        return good

    messages: list[dict] = [{"role": "user", "content": "hi"}]
    with (
        patch.object(claude_runtime, "claude_runtime_preflight", return_value=None),
        patch.object(
            claude_runtime, "verify_claude_billing_for_agent", return_value=None
        ),
        patch.object(claude_runtime, "_ensure_session", _ensure),
    ):
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages,
            effective_task_id="task-1",
        )

    assert attempts["n"] == 2
    assert result["completed"] is True
    assert db.get_provider_runtime_session(HERMES_SESSION, RUNTIME)[
        "provider_session_id"
    ] == "sdk-new"


def test_a_rewind_retires_the_live_session_so_the_fork_actually_takes_effect(db):
    """A connected client still points at the parent transcript."""
    agent, row_two, _t1, _t2 = _prime_two_turns(db)
    live = _StubSession([])
    agent._claude_session = live

    db.rewind_to_message(HERMES_SESSION, row_two)
    with patch.object(
        claude_runtime, "_fork_stored_session", lambda *a, **k: "sdk-fork"
    ):
        prepare_claude_sdk_session(agent, CWD)

    assert live.closed is True
    assert getattr(agent, "_claude_session", None) is None


def test_an_unrelated_turn_failure_is_not_treated_as_a_stale_session(db):
    agent = _make_agent(db)
    db.bind_provider_runtime_session(HERMES_SESSION, RUNTIME, "sdk-live")

    session = _StubSession(_script("sdk-live"), fail_once=RuntimeError("connection reset"))
    result = _turn(agent, session, [], "next")

    assert len(session.prompts) == 1
    assert result["completed"] is False
    # The binding is not thrown away over a network blip.
    assert db.get_provider_runtime_session(HERMES_SESSION, RUNTIME)[
        "provider_session_id"
    ] == "sdk-live"


def test_a_fresh_session_that_fails_is_a_failure_not_a_recovery(db):
    agent = _make_agent(db)
    session = _StubSession(_script("sdk-1"), fail_once=RuntimeError("session not found"))

    result = _turn(agent, session, [], "next")

    assert len(session.prompts) == 1
    assert result["completed"] is False


# ---------------------------------------------------------------------------
# Mirror errors
# ---------------------------------------------------------------------------


def test_a_mirror_error_warns_and_does_not_kill_the_turn(db, caplog):
    agent = _make_agent(db)
    statuses: list[str] = []
    agent._emit_status = statuses.append
    session = _StubSession(
        [
            AssistantMessage(content=[TextBlock("still here")], session_id="sdk-1"),
            MirrorErrorMessage(
                error="store append failed",
                data={"error": "store append failed"},
                key={"project_key": "p", "session_id": "sdk-1"},
            ),
            ResultMessage(session_id="sdk-1", result="still here"),
        ]
    )

    with caplog.at_level("WARNING", logger="agent.claude_runtime"):
        result = _turn(agent, session, [], "hi")

    assert result["completed"] is True
    assert result["final_response"] == "still here"
    assert any("mirror" in record.message.lower() for record in caplog.records)
    assert statuses, "the user is told durability degraded"


def test_a_mirror_error_delivered_as_a_plain_system_message_still_warns(db, caplog):
    agent = _make_agent(db)
    projector = ClaudeEventProjector(agent)

    with caplog.at_level("WARNING", logger="agent.claude_runtime"):
        projector(SystemMessage("mirror_error", {"error": "boom"}))

    assert projector.mirror_errors == ["boom"]


# ---------------------------------------------------------------------------
# Alternation
# ---------------------------------------------------------------------------


def _roles(messages):
    return [m["role"] for m in messages]


def _assert_alternating(messages):
    """No two same-role messages in a row outside a tool round."""
    previous = None
    for message in messages:
        role = message["role"]
        if role in ("user", "assistant"):
            assert not (
                previous == role and not message.get("tool_calls")
            ), f"role alternation broken: {_roles(messages)}"
            previous = role
        elif role == "tool":
            previous = "tool"


def test_alternation_holds_across_a_restart(db):
    agent = _make_agent(db)
    messages: list[dict] = []
    _turn(agent, _StubSession(_script("sdk-1", "first answer")), messages, "one")

    restarted = _make_agent(db)
    _turn(restarted, _StubSession(_script("sdk-1", "second answer")), messages, "two")

    _assert_alternating(messages)
    assert _roles(messages) == ["user", "assistant", "user", "assistant"]


def test_alternation_holds_across_a_rewind_and_re_ask(db):
    agent, row_two, _t1, _t2 = _prime_two_turns(db)
    db.rewind_to_message(HERMES_SESSION, row_two)
    with patch.object(
        claude_runtime, "_fork_stored_session", lambda *a, **k: "sdk-fork"
    ):
        prepare_claude_sdk_session(agent, CWD)

    messages = [{"role": "user", "content": "one", _FLUSHED: True},
                {"role": "assistant", "content": "answer one", _FLUSHED: True}]
    _turn(agent, _StubSession(_script("sdk-fork", "edited answer")), messages, "two-edited")

    _assert_alternating(messages)
    assert _roles(messages) == ["user", "assistant", "user", "assistant"]


def test_alternation_holds_across_a_native_compaction(db):
    agent = _make_agent(db)
    messages: list[dict] = []

    session = _StubSession(
        [
            SystemMessage("compact_boundary", {"session_id": "sdk-1"}),
            AssistantMessage(content=[TextBlock("after compaction")], session_id="sdk-1"),
            ResultMessage(session_id="sdk-1", result="after compaction"),
        ]
    )
    _turn(agent, session, messages, "long conversation")
    _turn(agent, _StubSession(_script("sdk-1", "next")), messages, "and more")

    _assert_alternating(messages)
    # Compaction is Claude's; Hermes' visible transcript is not rewritten.
    assert _roles(messages) == ["user", "assistant", "user", "assistant"]
