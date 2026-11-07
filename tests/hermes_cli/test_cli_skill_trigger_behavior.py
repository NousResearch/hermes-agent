"""What a classic CLI user gets from ``metadata.hermes.triggers``: which plain turns load a skill,
what the model is sent, and what the transcript keeps.

Assertions are on observable surfaces only: the message ``run_conversation`` receives, the bytes a
real ``AIAgent`` sends to a recording fake provider, the rows written to the session store, and what
an explicit ``/skill`` queues. Skill files and plugins are real files under the test's home.
"""
from __future__ import annotations

import os
import queue
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import skill_commands
from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

REPO = Path(__file__).resolve().parents[2]
# Bundled skills that ship an informal top-level ``triggers:`` list (prose, not an opt-in).
TOP_LEVEL_TRIGGER_SKILLS = {
    "songwriting-and-ai-music": REPO / "skills" / "creative" / "songwriting-and-ai-music",
    "popular-web-designs": REPO / "skills" / "creative" / "popular-web-designs",
    "web-pentest": REPO / "optional-skills" / "security" / "web-pentest",
    "oss-forensics": REPO / "optional-skills" / "security" / "oss-forensics",
}


def _write_skill(skills_dir: Path, name: str, frontmatter: str, body: str) -> None:
    skill = skills_dir / name
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill.\n{frontmatter}---\n{body}\n", encoding="utf-8")


def _home() -> Path:
    return Path(os.environ["HERMES_HOME"])


def _write_config(text: str) -> None:
    home = _home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    from tools import skills_tool

    skills = tmp_path / "skills"
    skills.mkdir()
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills)
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_sig_by_key", {})
    return skills


def _shell(agent=None):
    return SimpleNamespace(
        session_id="trigger-behavior", conversation_history=[],
        agent=agent or SimpleNamespace(run_conversation=MagicMock(return_value={})),
        _sudo_password_callback=None, _approval_callback=None, _secret_capture_callback=None,
        _flush_credit_notices=lambda: None, _vault_unlock_callback=None,
        _vault_save_login_callback=None, _vault_code_callback=None,
    )


def _send(message, shell=None):
    """Run one classic-CLI agent turn for ``message``; return what run_conversation received."""
    shell = shell or _shell()
    shell.conversation_history = [*shell.conversation_history, {"role": "user", "content": message}]
    CLIChatTurnMixin._chat_run_agent(shell, SimpleNamespace(voice_prefix="", stream_callback=None, result=None),
                                     message)
    return shell.agent.run_conversation.call_args.kwargs


def _loaded(kwargs, body):
    return body in kwargs["user_message"] and kwargs["persist_user_message"] is not None


# (a) top-level ``triggers:`` is ignored ---------------------------------------------------------

def test_top_level_triggers_never_auto_load_but_metadata_triggers_do(skills_dir):
    _write_skill(skills_dir, "top-only", "triggers: [weather report]\n", "TOP-ONLY-BODY")
    _write_skill(skills_dir, "both-keys",
                 "triggers: [stock tips]\nmetadata:\n  hermes:\n    triggers: [market outlook]\n", "BOTH-BODY")

    for message in ("Give me the weather report", "Any stock tips today?"):
        kwargs = _send(message)
        assert kwargs["user_message"] == message
        assert kwargs["persist_user_message"] is None

    assert _loaded(_send("What is the market outlook?"), "BOTH-BODY")
    # The skills stay invocable by name either way.
    assert {"/top-only", "/both-keys"} <= set(skill_commands.get_skill_commands())


# (b) plugin skills are matched ------------------------------------------------------------------

@pytest.fixture
def plugin_skill(skills_dir):
    from hermes_cli import plugins

    plugin = _home() / "plugins" / "zztest-probe"
    md = plugin / "skills" / "fx-desk" / "SKILL.md"
    md.parent.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: zztest-probe\nversion: 0.1.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "from pathlib import Path\ndef register(ctx):\n"
        "    ctx.register_skill('fx-desk', Path(__file__).parent / 'skills' / 'fx-desk' / 'SKILL.md')\n",
        encoding="utf-8")
    md.write_text("---\nname: fx-desk\ndescription: FX desk.\n"
                  "metadata:\n  hermes:\n    triggers: [currency desk]\n---\nPLUGIN-FX-BODY\n", encoding="utf-8")
    _write_config("plugins:\n  enabled: [zztest-probe]\n")
    plugins._reset_plugin_managers_for_tests()
    yield
    plugins._reset_plugin_managers_for_tests()


def test_plugin_skill_trigger_loads_the_plugin_skill(plugin_skill):
    kwargs = _send("Open the currency desk for me")

    assert _loaded(kwargs, "PLUGIN-FX-BODY")
    assert kwargs["persist_user_message"] == "Open the currency desk for me"
    # Plugin skills stay out of the filesystem map that native menus and the gateway use.
    assert "/zztest-probe:fx-desk" not in skill_commands.get_skill_commands()


@pytest.mark.parametrize("local_name,winner", [
    ("fx", "PLUGIN-FX-BODY"),                        # shorter than ``zztest-probe:fx-desk``
    ("fx-desk-local-override", "LOCAL-FX-BODY"),     # longer
])
def test_plugin_and_filesystem_skill_on_the_same_trigger_pick_one_deterministically(
        plugin_skill, skills_dir, local_name, winner):
    _write_skill(skills_dir, local_name, "metadata:\n  hermes:\n    triggers: [currency desk]\n", "LOCAL-FX-BODY")

    kwargs = _send("Open the currency desk for me")
    loser = ({"PLUGIN-FX-BODY", "LOCAL-FX-BODY"} - {winner}).pop()
    assert _loaded(kwargs, winner)
    assert loser not in kwargs["user_message"]


# (c) skills.auto_triggers=false ------------------------------------------------------------------

def test_auto_triggers_false_stops_matching_but_explicit_slash_skill_still_loads(skills_dir):
    from cli import HermesCLI

    _write_skill(skills_dir, "market-watch", "metadata:\n  hermes:\n    triggers: [market]\n", "MARKET-BODY")
    _write_config("skills:\n  auto_triggers: false\n")

    kwargs = _send("What is the market doing today?")
    assert kwargs["user_message"] == "What is the market doing today?"
    assert kwargs["persist_user_message"] is None

    cli = HermesCLI()
    cli.agent = None
    cli._pending_input = queue.Queue()
    cli.process_command("/market-watch what is the market doing")
    queued = cli._pending_input.get_nowait()
    assert "MARKET-BODY" in queued
    assert skill_commands.extract_user_instruction_from_skill_message(queued) == "what is the market doing"

    _write_config("skills:\n  auto_triggers: true\n")
    assert _loaded(_send("What is the market doing today?"), "MARKET-BODY")


@pytest.mark.parametrize("skills_config,auto_loads", [
    ('skills:\n  auto_triggers: "false"\n', False),
    ('skills:\n  auto_triggers: "off"\n', False),
    ('skills:\n  auto_triggers: "0"\n', False),
    ('skills:\n  auto_triggers: "true"\n', True),
    ("skills: {}\n", True),  # unset
], ids=["quoted-false", "quoted-off", "quoted-0", "quoted-true", "unset"])
def test_quoted_auto_triggers_values_are_read_as_booleans(skills_dir, skills_config, auto_loads):
    _write_skill(skills_dir, "market-watch", "metadata:\n  hermes:\n    triggers: [market]\n", "MARKET-BODY")
    _write_config(skills_config)

    message = "What is the market doing today?"
    kwargs = _send(message)
    if auto_loads:
        assert _loaded(kwargs, "MARKET-BODY")
    else:
        assert kwargs["user_message"] == message
        assert kwargs["persist_user_message"] is None


# (d) transcript keeps the user's text; the model gets (and is resent) the skill ---------------

def test_transcript_keeps_user_text_while_the_expansion_is_sent_and_replayed(skills_dir):
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home

    _write_skill(skills_dir, "market-watch", "metadata:\n  hermes:\n    triggers: [market]\n",
                 "Use verified market sources.")
    message = "What is the market doing today?"
    with FakeLLMServer([Text("Markets are flat."), Text("You're welcome.")]) as srv:
        write_hermes_home(_home(), srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        db = SessionDB(db_path=_home() / "zztest-state.db")
        sid = "trigger-transcript"

        def make_agent():
            return AIAgent(provider="custom", base_url=srv.base_url, api_key="sk-fake", model="fake-model",
                           session_id=sid, session_db=db, quiet_mode=True, platform="cli", skip_memory=True,
                           skip_context_files=True, enabled_toolsets=["memory"])

        agent = make_agent()
        try:
            shell = _shell(agent)
            shell.session_id = sid
            shell.conversation_history = [{"role": "user", "content": message}]
            turn = SimpleNamespace(voice_prefix="", stream_callback=None, result=None)
            CLIChatTurnMixin._chat_run_agent(shell, turn, message)
            assert turn.result.get("final_response") == "Markets are flat.", turn.result

            sent = [m for m in srv.main_requests()[0]["messages"] if m["role"] == "user"][-1]["content"]
            assert "Use verified market sources." in sent
            assert skill_commands.extract_user_instruction_from_skill_message(sent) == message

            rows = [r for r in db.get_messages(sid) if r["role"] == "user"]
            assert [r["content"] for r in rows] == [message]
            assert rows[0]["api_content"] == sent

            # Next turn (fresh agent, history reloaded from the store): the skill turn is replayed
            # byte-for-byte, so the model still has the skill body.
            history = db.get_messages_as_conversation(sid)
            agent.close()
            agent = make_agent()
            agent.run_conversation("thanks", conversation_history=history, task_id=sid)
            replayed = [m for m in srv.main_requests()[1]["messages"] if m["role"] == "user"]
            assert replayed[0]["content"] == sent
            assert replayed[-1]["content"] == "thanks"
        finally:
            agent.close()
            db.close()


# (e) bundled skills with top-level triggers never auto-load ------------------------------------

def _top_level_phrases(skill_md: Path) -> list[str]:
    from agent.skill_utils import parse_frontmatter

    data, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    assert not (data.get("metadata") or {}).get("hermes", {}).get("triggers"), skill_md
    return [p.replace("[URL]", "https://example.com").replace("[owner/repo]", "octo/repo")
            for p in data["triggers"]]


def test_bundled_skills_with_top_level_triggers_never_auto_load(skills_dir):
    for name, source in TOP_LEVEL_TRIGGER_SKILLS.items():
        shutil.copytree(source, skills_dir / name)

    commands = skill_commands.get_skill_commands()
    assert {f"/{name}" for name in TOP_LEVEL_TRIGGER_SKILLS} <= set(commands)
    for name in TOP_LEVEL_TRIGGER_SKILLS:
        phrases = _top_level_phrases(skills_dir / name / "SKILL.md")
        assert phrases, name
        for phrase in phrases:
            message = f"Please help: {phrase}"
            kwargs = _send(message)
            assert kwargs["user_message"] == message, (name, phrase)
            assert kwargs["persist_user_message"] is None, (name, phrase)


# (f) expansion shape and pass-through (merged from test_cli_auto_skill_triggers.py) ------------

@pytest.fixture
def trigger_turn(tmp_path, monkeypatch):
    from tools import skills_tool

    skills = tmp_path / "skills"
    skill = skills / "market-watch"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: market-watch\ndescription: Market analysis.\n"
        "metadata:\n  hermes:\n    triggers: [market, EUR/USD]\n---\n"
        "Use verified market sources.\n", encoding="utf-8",
    )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills)
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_sig_by_key", {})
    skill_commands.scan_skill_commands()
    shell = SimpleNamespace(
        session_id="trigger-test", agent=SimpleNamespace(run_conversation=MagicMock(return_value={})),
        conversation_history=[], _sudo_password_callback=None, _approval_callback=None,
        _secret_capture_callback=None, _flush_credit_notices=lambda: None,
        _vault_unlock_callback=None, _vault_save_login_callback=None, _vault_code_callback=None,
    )
    return shell, SimpleNamespace(voice_prefix="", stream_callback=None, result=None)


@pytest.mark.parametrize("voice_prefix", ["", "[Voice input] "])
def test_trigger_expansion_preserves_original_turn(trigger_turn, voice_prefix):
    shell, turn = trigger_turn
    turn.voice_prefix = voice_prefix
    message = "What is happening with eur/usd today?"
    prior = {"role": "assistant", "content": "Previous response."}
    staged = {"role": "user", "content": message}
    shell.conversation_history = [prior, staged]

    CLIChatTurnMixin._chat_run_agent(shell, turn, message)

    kwargs = shell.agent.run_conversation.call_args.kwargs
    expanded = kwargs["user_message"]
    assert "Use verified market sources." in expanded
    assert expanded.startswith(voice_prefix + "[IMPORTANT:")
    assert skill_commands.extract_user_instruction_from_skill_message(
        expanded.removeprefix(voice_prefix)
    ) == message
    assert kwargs["persist_user_message"] == message
    assert kwargs["conversation_history"] == [prior]
    assert shell.conversation_history == [prior, staged]


@pytest.mark.parametrize("message", [
    "Tell me a joke.", "This supermarket is busy.", "/market-watch market",
    "", [{"type": "text", "text": "market"}],
])
def test_non_trigger_turns_pass_through(trigger_turn, message):
    shell, turn = trigger_turn
    shell.conversation_history = [{"role": "user", "content": message}]

    CLIChatTurnMixin._chat_run_agent(shell, turn, message)

    kwargs = shell.agent.run_conversation.call_args.kwargs
    assert kwargs["user_message"] == message
    assert kwargs["persist_user_message"] is None


# (g) which turns may match: explicit provenance, never the message text -------------------------

@pytest.mark.parametrize("message,images,system_authored,allowed", [
    ("market update please", None, False, True),                 # typed or spoken by the person
    ("market update please", None, True, False),                 # notification / nudge / goal judge
    ("market update please", ["/tmp/chart.png"], False, False),  # text routing turns an image turn into a string
])
def test_only_user_authored_turns_are_matched(message, images, system_authored, allowed):
    from hermes_cli.cli_chat_turn_mixin import _turn_allows_skill_triggers

    assert _turn_allows_skill_triggers(message, images, system_authored=system_authored) is allowed


def test_a_watch_match_notification_is_queued_with_system_provenance(monkeypatch):
    """Regression: a watch match arrives as a plain string carrying the process's own output, and the
    old ``isinstance(TimelineNotification)`` guard let it load a skill from that output. The queued
    item itself must mark the turn system-authored while its text and display stay unchanged."""
    from hermes_cli import gateway_tools_notify
    from hermes_cli.cli_chat_turn_mixin import _turn_allows_skill_triggers
    from hermes_cli.cli_process_notifications import CLIProcessNotificationsMixin, input_is_system_authored
    from tools import async_delegation
    from tools.process_registry import process_registry
    from tools.process_registry_notifications import SystemNotification, TimelineNotification

    event = {"type": "watch_match", "event_id": "e1", "session_key": "s", "session_id": "p1",
             "pattern": "outlook", "output": "12:00 market outlook is flat"}
    text = "Watch match on process p1 (pattern 'outlook'):\n12:00 market outlook is flat"
    monkeypatch.setattr(process_registry, "drain_notifications", lambda **kw: [(event, text)])
    monkeypatch.setattr(async_delegation, "claim_event_delivery", lambda ev, consumer: "claim")
    monkeypatch.setattr(async_delegation, "complete_event_delivery", lambda ev, claim: None)
    monkeypatch.setattr(gateway_tools_notify, "gateway_store", lambda cli: None)

    class _Shell(CLIProcessNotificationsMixin):
        session_id = "s"
        _session_db = None
        _pending_input = queue.Queue()

        def _background_notifications_suppressed(self):
            return False

    shell = _Shell()
    shell._drain_process_notifications("cli")
    pending = shell._pending_input.get_nowait()

    assert isinstance(pending, SystemNotification), type(pending)
    assert not isinstance(pending, TimelineNotification)  # still painted as a plain row, as before
    assert str(pending) == text
    assert input_is_system_authored(pending) is True
    assert _turn_allows_skill_triggers(str(pending), None, system_authored=True) is False


def test_recovery_and_goal_judge_turns_are_system_authored():
    """Regression: the kanban recovery nudge and the ``-q`` goal-judge prompt both go through
    ``cli.chat`` and are written by Hermes, so a broad trigger ("continue the task") loaded a skill
    on them. They now carry the explicit provenance flag."""
    from agent.kanban_turn_recovery import build_recovery_nudge
    from hermes_cli.cli_chat_turn_mixin import _turn_allows_skill_triggers
    from hermes_cli.cli_single_query import _synthetic_turn

    seen = {}

    class _Chat:
        def chat(self, prompt, **kwargs):
            seen["prompt"], seen["kwargs"] = prompt, kwargs
            return "ok"

    nudge = build_recovery_nudge({"error": "provider stream failure"}, attempt=1, max_attempts=2,
                                 task_id="t-1")
    assert nudge.startswith("[System:")  # system-authored text, matched on content it would fire
    assert _synthetic_turn(_Chat(), nudge) == "ok"
    assert seen["kwargs"] == {"system_authored": True}
    assert _turn_allows_skill_triggers(seen["prompt"], None, system_authored=True) is False


def test_a_background_notification_never_loads_a_skill(trigger_turn):
    from tools.process_registry_notifications import TimelineNotification
    from hermes_cli.cli_chat_turn_mixin import _turn_allows_skill_triggers

    shell, turn = trigger_turn
    note = "[process finished] report: EUR/USD market outlook is flat"
    timeline = TimelineNotification(note, "Process finished", "process")
    assert _turn_allows_skill_triggers(timeline, None) is False

    turn.skill_triggers = False  # what chat() sets for that turn
    shell.conversation_history = [{"role": "user", "content": note}]
    CLIChatTurnMixin._chat_run_agent(shell, turn, note)

    kwargs = shell.agent.run_conversation.call_args.kwargs
    assert kwargs["user_message"] == note
    assert kwargs["persist_user_message"] is None


def test_quiet_one_shot_does_not_match_triggers(trigger_turn):
    """-Q / stream-json one-shots run outside chat(): the documented scope excludes them."""
    from hermes_cli.cli_single_query import _run_quiet_single_query

    class _Sent(Exception):
        pass

    seen = {}

    def run_conversation(user_message, conversation_history, **_kw):
        seen["user_message"] = user_message
        raise _Sent

    cli = SimpleNamespace(session_id="quiet", conversation_history=[],
                          agent=SimpleNamespace(run_conversation=run_conversation))
    with pytest.raises(_Sent):
        _run_quiet_single_query(cli, "What is the EUR/USD market outlook?")

    assert seen["user_message"] == "What is the EUR/USD market outlook?"


def test_no_triggered_skill_means_no_config_read_on_a_plain_turn(skills_dir, monkeypatch):
    _write_skill(skills_dir, "plain", "", "Plain body.")
    skill_commands.scan_skill_commands()
    skill_commands.get_plugin_skill_commands()

    def boom():
        raise AssertionError("a plain turn must not read config when no skill declares triggers")

    monkeypatch.setattr(skill_commands, "_load_skills_config", boom)
    assert skill_commands.find_triggered_skill_command("market outlook please") is None

    _write_skill(skills_dir, "market", "metadata:\n  hermes:\n    triggers: [market outlook]\n", "Market body.")
    skill_commands.scan_skill_commands()
    monkeypatch.setattr(skill_commands, "_load_skills_config", dict)
    assert skill_commands.find_triggered_skill_command("market outlook please")[0] == "/market"
