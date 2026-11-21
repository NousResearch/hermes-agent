import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


class FakeAgent:
    def __init__(self):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self._supports_active_turn_redirect = True
        self.steers = []
        self.redirects = []
        self.runs = []

    def steer(self, text):
        self.steers.append(text)
        return True

    def redirect(self, text):
        self.redirects.append(text)
        return True

    def run_conversation(self, *, user_message, conversation_history, task_id, **kwargs):
        self.runs.append(user_message)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        final = f"ran: {user_message}"
        messages.append({"role": "assistant", "content": final})
        return {"final_response": final, "messages": messages}


class CaptureConn:
    def __init__(self):
        self.updates = []

    async def session_update(self, *args, **kwargs):
        if kwargs:
            self.updates.append((kwargs.get("session_id"), kwargs.get("update")))
        else:
            self.updates.append((args[0], args[1]))

    async def request_permission(self, *args, **kwargs):
        return SimpleNamespace(outcome="allow")


class NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None


def make_agent_and_state():
    fake = FakeAgent()
    manager = SessionManager(agent_factory=lambda **kwargs: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    conn = CaptureConn()
    acp_agent.on_connect(conn)
    return acp_agent, state, fake, conn


def test_acp_real_agent_gets_session_db_for_recall(monkeypatch):
    """ACP sessions persist to SessionDB; recall must receive the same DB handle."""
    captured = {}
    sentinel_db = NoopDb()

    class CapturingAgent(FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    def mod(name, **attrs):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=CapturingAgent))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "m", "provider": "p"}})
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "provider": "p",
                "api_mode": "chat_completions",
                "base_url": "u",
                "api_key": "k",
                "command": None,
                "args": [],
            },
        ),
    )

    manager = SessionManager(db=sentinel_db)
    agent = manager._make_agent(session_id="acp-session", cwd=".")

    assert isinstance(agent, CapturingAgent)
    assert captured["session_db"] is sentinel_db
    assert captured["platform"] == "acp"
    assert captured["session_id"] == "acp-session"


@pytest.mark.asyncio
async def test_acp_steer_slash_command_injects_into_running_agent():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/steer prefer the simpler fix")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.steers == ["prefer the simpler fix"]
    assert fake.runs == []


@pytest.mark.asyncio
async def test_acp_reset_rejected_while_turn_running():
    """prompt() dispatches slash commands before the is_running claim; /reset
    must be refused there rather than clearing state.history mid-turn."""
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True
    state.history = [{"role": "user", "content": "earlier"}]

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/reset")],
    )

    assert response.stop_reason == "end_turn"
    assert state.history == [{"role": "user", "content": "earlier"}]
    assert fake.runs == []
    assert state.queued_prompts == []


@pytest.mark.asyncio
async def test_acp_compress_rejected_while_turn_running():
    """Mid-turn /compress would compress a torn history and rebind
    state.history while the live turn still appends to the old list."""
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True
    state.history = [{"role": "user", "content": "earlier"}]
    sentinel_db = object()
    fake._session_db = sentinel_db
    fake._cached_system_prompt = "sys"
    fake._compress_context = lambda *a, **k: ([{"role": "user", "content": "summary"}], "new-sys")

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/compress")],
    )

    assert response.stop_reason == "end_turn"
    assert state.history == [{"role": "user", "content": "earlier"}]
    assert fake._session_db is sentinel_db
    assert fake.runs == []


@pytest.mark.asyncio
async def test_acp_prompt_during_mutating_command_queues_then_runs():
    """The command_op flag closes the check-then-act window: a prompt arriving while
    /reset is mid-flight must queue behind it, then run on the cleared history."""
    acp_agent, state, fake, _conn = make_agent_and_state()
    loop = asyncio.get_running_loop()
    state.history = [{"role": "user", "content": "earlier"}]

    # Inject inside _cmd_reset: at that point _handle_slash_command already holds
    # command_op, so the concurrent prompt must queue rather than claim the turn.
    orig_reset = acp_agent._cmd_reset

    def patched_reset(args, st):
        fut = asyncio.run_coroutine_threadsafe(
            acp_agent.prompt(
                session_id=st.session_id,
                prompt=[TextContentBlock(type="text", text="follow-up")],
            ), loop)
        fut.result(timeout=10)
        return orig_reset(args, st)

    with patch.object(acp_agent, "_cmd_reset", patched_reset):
        response = await acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="/reset")],
        )

    assert response.stop_reason == "end_turn"
    # The follow-up queued behind the op, then ran on the cleared history.
    assert fake.runs == ["follow-up"]
    assert state.history == [
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "ran: follow-up"},
    ]
    assert state.command_op is False








@pytest.mark.asyncio
async def test_acp_cancel_publishes_hard_stop_while_holding_runtime_lock():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True
    state.current_prompt_text = "original request"
    observed = {}

    def interrupt():
        acquired = state.runtime_lock.acquire(blocking=False)
        observed["lock_held"] = not acquired
        if acquired:
            state.runtime_lock.release()

    fake.interrupt = interrupt

    await acp_agent.cancel(state.session_id)

    assert observed["lock_held"] is True
    assert state.cancel_event.is_set()
    assert state.interrupted_prompt_text == "original request"








def _mixin_state(history=None):
    return SimpleNamespace(session_id="s", history=history if history is not None else [], agent=object())


def test_acp_skill_command_expands_into_the_model_prompt(monkeypatch):
    """`/skill <name>` must become the turn's prompt: the skill body has to reach the model,
    not the client, or the agent only ever sees the literal "/skill ..." text."""
    from acp_adapter.commands import SlashCommandsMixin

    monkeypatch.setattr(
        "agent.skill_commands.resolve_skill_command_key", lambda name: "/demo" if name == "demo" else None
    )
    monkeypatch.setattr(
        "agent.skill_commands.build_skill_invocation_message",
        lambda key, user_instruction="", task_id=None: f"SKILL {key} :: {user_instruction}",
    )

    out = SlashCommandsMixin()._expand_skill_command("/skill demo draw a cat", _mixin_state())

    assert out == ("prompt", "SKILL /demo :: draw a cat")


def test_acp_bare_skill_slug_expands_too(monkeypatch):
    from acp_adapter.commands import SlashCommandsMixin

    monkeypatch.setattr("agent.skill_commands.resolve_skill_command_key", lambda name: "/demo" if name == "demo" else None)
    monkeypatch.setattr(
        "agent.skill_commands.build_skill_invocation_message",
        lambda key, user_instruction="", task_id=None: f"SKILL {key}",
    )

    assert SlashCommandsMixin()._expand_skill_command("/demo", _mixin_state()) == ("prompt", "SKILL /demo")


def test_acp_unknown_skill_reports_instead_of_running_a_turn(monkeypatch):
    from acp_adapter.commands import SlashCommandsMixin

    monkeypatch.setattr("agent.skill_commands.resolve_skill_command_key", lambda name: None)

    out = SlashCommandsMixin()._expand_skill_command("/skill nope", _mixin_state())

    assert out[0] == "message"
    assert "No skill named 'nope'" in out[1]


def test_acp_non_skill_slash_command_is_left_to_the_command_table(monkeypatch):
    """A plain command must fall through, not be swallowed by skill expansion."""
    from acp_adapter.commands import SlashCommandsMixin

    monkeypatch.setattr("agent.skill_commands.resolve_skill_command_key", lambda name: None)

    assert SlashCommandsMixin()._expand_skill_command("/queue say hi", _mixin_state()) is None


def test_acp_advertises_skill_and_btw_commands():
    """The composer only offers what the agent advertises; /skill and /btw were missing."""
    from acp_adapter.commands import SlashCommandsMixin

    names = {command.name for command in SlashCommandsMixin._available_commands()}
    assert {"skill", "skills", "btw", "queue"} <= names


def test_acp_btw_answers_without_touching_history(monkeypatch):
    """`/btw` is a side question: answer from a snapshot and leave the live history alone."""
    from acp_adapter.commands import SlashCommandsMixin

    history = [{"role": "user", "content": "earlier"}]
    state = SimpleNamespace(session_id="s", history=history, agent=object())
    seen = {}

    def answer(question, snapshot, *, parent_agent=None, **kwargs):
        seen["question"] = question
        seen["snapshot"] = snapshot
        return "side answer"

    monkeypatch.setattr("agent.side_question.answer_side_question", answer)

    reply = SlashCommandsMixin._cmd_btw(SlashCommandsMixin(), "what changed?", state)

    assert reply == "side answer"
    assert seen["question"] == "what changed?"
    assert seen["snapshot"] == history and seen["snapshot"] is not history
    assert state.history == [{"role": "user", "content": "earlier"}]


def test_acp_queue_on_idle_session_runs_as_the_turn():
    """`/queue` on an idle session has nothing to wait for: its text becomes the turn, so the
    reply is one message instead of a queue note with a drained turn concatenated behind it."""
    from acp_adapter.commands import SlashCommandsMixin

    idle = SimpleNamespace(is_running=False, command_op=False)
    busy = SimpleNamespace(is_running=True, command_op=False)

    assert SlashCommandsMixin()._take_idle_queue_prompt("/queue say apple", idle) == "say apple"
    assert SlashCommandsMixin()._take_idle_queue_prompt("/queue say apple", busy) is None
    assert SlashCommandsMixin()._take_idle_queue_prompt("/queue", idle) is None
    assert SlashCommandsMixin()._take_idle_queue_prompt("/help", idle) is None


def test_acp_advertises_each_skill_as_its_own_command(monkeypatch):
    """The composer lists only what is advertised: `/1password` has to be there, not just
    `/skill 1password`, or a user cannot discover or invoke it from the menu."""
    from acp_adapter.commands import SlashCommandsMixin

    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/1password": {"name": "1password", "description": "Manage 1Password items"}},
    )

    commands = {command.name: command for command in SlashCommandsMixin._available_commands()}

    assert commands["1password"].description == "Manage 1Password items"
    assert {"help", "skill", "skills", "btw"} <= set(commands)


def test_acp_core_command_wins_a_skill_name_collision(monkeypatch):
    """A skill named `help` must not shadow /help; it stays reachable as /skill help."""
    from acp_adapter.commands import SlashCommandsMixin

    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/help": {"name": "help", "description": "A skill called help"}},
    )
    monkeypatch.setattr(
        "agent.skill_commands.resolve_skill_command_key", lambda name: "/help" if name == "help" else None
    )

    advertised = [command.name for command in SlashCommandsMixin._available_commands()]

    assert advertised.count("help") == 1
    assert SlashCommandsMixin()._expand_skill_command("/help", _mixin_state()) is None


class _ResumeDB:
    def __init__(self, rows, history):
        self._rows = rows
        self._history = history
        self.loaded = []

    def list_sessions_rich(self, **_kwargs):
        return self._rows

    def get_messages_as_conversation(self, session_id, **_kwargs):
        self.loaded.append(session_id)
        return list(self._history)


def _resume_state():
    return SimpleNamespace(
        session_id="acp-1", history=[], is_running=False, command_op=False, agent=object(),
    )


def test_acp_resume_lists_sessions_from_every_surface(monkeypatch):
    """/resume has to reach sessions Hermes kept for the CLI/TUI too, not only T3 threads."""
    from acp_adapter.commands import SlashCommandsMixin

    db = _ResumeDB(
        [
            {"id": "20261020_162956_cfc159", "title": "MR Review 82", "message_count": 411, "source": "tui"},
            {"id": "61df244d-0aea-4e50", "title": "1Password skill", "message_count": 4, "source": "acp"},
            {"id": "sub-1", "title": "child", "message_count": 2, "source": "subagent"},
        ],
        [],
    )
    mixin = SlashCommandsMixin()
    monkeypatch.setattr(mixin, "_resume_db", lambda: db)

    reply = mixin._cmd_resume("", _resume_state())

    assert "MR Review 82" in reply and "1Password skill" in reply
    assert "sub-1" not in reply  # subagent transcripts are not resumable conversations
    assert "| 1 |" in reply and "| 2 |" in reply


def test_acp_resume_loads_history_into_the_thread(monkeypatch):
    """Loading a session must replace this thread's history, so the conversation continues
    with its context and not as a fresh thread."""
    from acp_adapter.commands import SlashCommandsMixin

    history = [{"role": "user", "content": "earlier question"}, {"role": "assistant", "content": "earlier answer"}]
    db = _ResumeDB([{"id": "abc12345-rest", "title": "Old", "message_count": 2, "source": "tui"}], history)
    mixin = SlashCommandsMixin()
    monkeypatch.setattr(mixin, "_resume_db", lambda: db)
    state = _resume_state()

    assert "| 1 |" in mixin._cmd_resume("", state)
    reply = mixin._cmd_resume("1", state)

    assert "2 messages loaded" in reply
    assert state.history == history
    # The transcript itself is the reply: a client cannot be handed past turns any other way.
    assert "earlier question" in reply and "earlier answer" in reply


def test_acp_resume_accepts_an_id_prefix(monkeypatch):
    from acp_adapter.commands import SlashCommandsMixin

    db = _ResumeDB([{"id": "abc12345-rest", "title": "Old", "message_count": 1, "source": "tui"}], [{"role": "user", "content": "x"}])
    mixin = SlashCommandsMixin()
    monkeypatch.setattr(mixin, "_resume_db", lambda: db)
    state = _resume_state()
    mixin._cmd_resume("", state)

    assert "Resumed" in mixin._cmd_resume("abc1", state)
    assert "No session matching" in mixin._cmd_resume("zzz", state)


def test_acp_resume_ignores_the_composer_echo(monkeypatch):
    """The T3 composer re-inserts the command name, so a bare submit arrives as
    '/resume /resume'; that must list, not report a missing session."""
    from acp_adapter.commands import SlashCommandsMixin

    db = _ResumeDB([{"id": "abc12345", "title": "Old", "message_count": 1, "source": "tui"}], [])
    mixin = SlashCommandsMixin()
    monkeypatch.setattr(mixin, "_resume_db", lambda: db)

    assert "| 1 |" in mixin._cmd_resume("/resume", _resume_state())


class _SearchDB(_ResumeDB):
    def __init__(self, rows, hits, history=None):
        super().__init__(rows, history if history is not None else [{"role": "user", "content": "earlier"}])
        self._hits = hits
        self.queries = []

    def search_messages(self, query, **_kwargs):
        self.queries.append(query)
        return self._hits.get(query, [])

    def get_session(self, session_id):
        return next((row for row in self._rows if row.get("id") == session_id), None)


def _search_mixin(monkeypatch, rows, hits):
    from acp_adapter.commands import SlashCommandsMixin

    db = _SearchDB(rows, hits)
    mixin = SlashCommandsMixin()
    monkeypatch.setattr(mixin, "_resume_db", lambda: db)
    return mixin, db


def test_acp_sessions_searches_titles_and_transcripts(monkeypatch):
    """/sessions has to find a session by its title as well as by what was said in it."""
    rows = [
        {"id": "aaa11111-x", "title": "Hermes agent setup notes", "source": "cli", "last_active": 1000, "message_count": 9},
        {"id": "bbb22222-y", "title": "Unrelated", "source": "acp", "last_active": 2000, "message_count": 4},
        {"id": "ccc33333-z", "title": "Also unrelated", "source": "tui", "last_active": 500, "message_count": 2},
    ]
    hits = {'"Hermes agent setup"': [{"session_id": "ccc33333-z", "snippet": "we ran the >>>Hermes agent setup<<< wizard"}]}
    mixin, db = _search_mixin(monkeypatch, rows, hits)

    reply = mixin._cmd_sessions("Hermes agent setup", _resume_state())

    assert '"Hermes agent setup"' in db.queries  # phrase first, not any-word
    assert "Hermes agent setup notes" in reply          # title match
    assert "we ran the **Hermes agent setup** wizard" in reply  # transcript match
    assert "Unrelated" not in reply.replace("Also unrelated", "")


def test_acp_sessions_orders_by_newest_activity(monkeypatch):
    rows = [
        {"id": "old", "title": "Hermes old", "source": "cli", "last_active": 100, "message_count": 1},
        {"id": "new", "title": "Hermes new", "source": "cli", "last_active": 900, "message_count": 1},
    ]
    mixin, _db = _search_mixin(monkeypatch, rows, {})

    lines = [line for line in mixin._cmd_sessions("Hermes", _resume_state()).splitlines() if line.startswith("| 1 ")]

    assert "Hermes new" in lines[0]


def test_acp_sessions_falls_back_to_any_word(monkeypatch):
    """When the exact phrase is absent, the loose query still runs and the reply says so."""
    rows = [{"id": "s1", "title": "x", "source": "cli", "last_active": 5, "message_count": 1}]
    hits = {"delegation progress": [{"session_id": "s1", "snippet": "**delegation** and **progress**"}]}
    mixin, _db = _search_mixin(monkeypatch, rows, hits)

    reply = mixin._cmd_sessions("delegation progress", _resume_state())

    assert "mentioning any of" in reply


def test_acp_sessions_reports_no_match(monkeypatch):
    mixin, _db = _search_mixin(monkeypatch, [{"id": "s1", "title": "x", "source": "cli", "last_active": 5, "message_count": 1}], {})

    assert "No session matches" in mixin._cmd_sessions("zzzznope", _resume_state())


def test_acp_sessions_skips_subagent_transcripts(monkeypatch):
    rows = [
        {"id": "child", "title": "Subagent: do a thing", "source": "desktop", "last_active": 900, "message_count": 3},
        {"id": "parent", "title": "Parent", "source": "desktop", "last_active": 100, "message_count": 3},
    ]
    hits = {"child": [
        {"session_id": "child", "snippet": "**child** work"},
        {"session_id": "parent", "snippet": "spawned a **child**"},
    ]}
    mixin, _db = _search_mixin(monkeypatch, rows, hits)

    reply = mixin._cmd_sessions("child", _resume_state())

    assert "Parent" in reply
    assert "Subagent:" not in reply


def test_acp_sessions_prints_ids_that_resume_accepts(monkeypatch):
    """A same-day collision ('20260811…') must widen the printed id, or the id in the table
    cannot be pasted back into /resume."""
    rows = [
        {"id": "20260811_113129_98b2a2", "title": "Hermes one", "source": "discord", "last_active": 900, "message_count": 3},
        {"id": "20260811_110351_c92284", "title": "Hermes two", "source": "discord", "last_active": 800, "message_count": 3},
    ]
    mixin, _db = _search_mixin(monkeypatch, rows, {})
    state = _resume_state()

    reply = mixin._cmd_sessions("Hermes", state)

    printed = [line.rsplit("`", 2)[-2] for line in reply.splitlines() if line.startswith("| 1 ") or line.startswith("| 2 ")]
    assert len(printed) == 2 and printed[0] != printed[1]
    assert all(len(value) > 8 for value in printed)
    assert "Resumed" in mixin._cmd_resume(printed[0], state)


def test_acp_resume_reports_an_ambiguous_prefix(monkeypatch):
    rows = [
        {"id": "20260811_113129_98b2a2", "title": "a", "source": "discord", "last_active": 1, "message_count": 1},
        {"id": "20260811_110351_c92284", "title": "b", "source": "discord", "last_active": 2, "message_count": 1},
    ]
    mixin, _db = _search_mixin(monkeypatch, rows, {})
    state = _resume_state()

    reply = mixin._cmd_resume("20260811", state)

    assert "matches 2 sessions" in reply
    assert state.history == []  # never guesses which transcript to load


def test_acp_sessions_skips_search_echoes_and_title_calls(monkeypatch):
    """Earlier `/sessions` runs and auto-title calls quote the thread's own text; matching them
    would make every past search look like a result."""
    rows = [
        {"id": "echo", "title": "(untitled)", "source": "acp", "last_active": 900, "message_count": 2},
        {"id": "titled", "title": "(untitled)", "source": "acp", "last_active": 800, "message_count": 2},
        {"id": "real", "title": "Real work", "source": "cli", "last_active": 100, "message_count": 5},
    ]
    hits = {'"Hermes agent setup"': [
        {"session_id": "echo", "snippet": "...User message:\n/sessions >>>Hermes agent setup<<<"},
        {"session_id": "titled", "snippet": '{"title": ">>>Hermes agent setup<<<", "needsRefinement": false}'},
        {"session_id": "real", "snippet": "installed the >>>Hermes agent setup<<< wizard"},
    ]}
    mixin, _db = _search_mixin(monkeypatch, rows, hits)

    reply = mixin._cmd_sessions("Hermes agent setup", _resume_state())

    assert "Real work" in reply
    assert "1 session(s)" in reply
