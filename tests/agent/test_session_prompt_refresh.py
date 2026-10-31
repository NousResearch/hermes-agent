"""``hermes sessions refresh-prompts``: a continuing session adopts updated prompt sources once.

Operators push SOUL.md / skill / AGENTS.md changes into a profile while users keep one long
gateway conversation. The stored system prompt is reused verbatim every turn (prefix cache), so
those changes never reached the session before ``/new``. The verb records a refresh request in the
profile's state.db; the session's next turn rebuilds its prompt once, keeps its history and id,
and persists the new bytes so later turns restore them verbatim.

Everything is real except the model: a real ``AIAgent`` per turn (as the messaging gateway runs
it), the real ``SessionDB`` in a temp HERMES_HOME, the real CLI entrypoint, and a recording
``FakeLLMServer`` whose request bodies are what the model was actually sent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import FakeLLMServer, write_hermes_home

SOUL_V1 = "You are Wren, the operations assistant. Soul revision one."
SOUL_V2 = "You are Wren, the operations assistant. Soul revision two."


def _skill(home: Path, name: str, description: str) -> None:
    skill_dir = home / "skills" / "ops" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nSteps.\n", encoding="utf-8")


def _write_profile(home: Path, base_url: str, project: Path) -> None:
    write_hermes_home(home, base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
    (home / "SOUL.md").write_text(SOUL_V1, encoding="utf-8")
    _skill(home, "deploy-runbook", "Roll out a release. Description revision one.")
    project.mkdir(parents=True, exist_ok=True)
    (project / "AGENTS.md").write_text("Project rules revision one.", encoding="utf-8")


def _update_profile(home: Path, project: Path) -> None:
    """What the operator's update script pulls into the profile."""
    (home / "SOUL.md").write_text(SOUL_V2, encoding="utf-8")
    _skill(home, "deploy-runbook", "Roll out a release. Description revision two.")
    _skill(home, "incident-triage", "Triage a production incident.")
    (project / "AGENTS.md").write_text("Project rules revision two.", encoding="utf-8")


@pytest.fixture
def profile(monkeypatch, tmp_path):
    home = Path(os.environ["HERMES_HOME"])
    project = tmp_path / "project"
    with FakeLLMServer() as srv:
        _write_profile(home, srv.base_url, project)
        monkeypatch.setenv("TERMINAL_CWD", str(project))
        yield srv, home, project


def _agent(srv, home, session_id):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=home / "state.db")
    db.ensure_session(session_id, source="whatsapp")  # the gateway opens the row before the turn
    return AIAgent(provider="custom", base_url=srv.base_url, api_key="sk-fake-e2e", model="fake-model",
                   session_db=db, session_id=session_id, quiet_mode=True, platform="whatsapp")


def _system(request) -> str:
    content = request["messages"][0]["content"]
    assert request["messages"][0]["role"] == "system"
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content)
    return content


def _user_texts(request) -> list[str]:
    out = []
    for message in request["messages"]:
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content)
        out.append(content)
    return out


def _turn(srv, home, session_id, text, history, agent=None):
    """One gateway turn: a fresh AIAgent unless a warm ``agent`` is passed. Returns (history, system)."""
    fresh = agent is None
    agent = agent or _agent(srv, home, session_id)
    try:
        messages = agent.run_conversation(text, conversation_history=history, task_id="t")["messages"]
    finally:
        if fresh:
            agent.close()
    return messages, _system(srv.main_requests()[-1])


def _refresh_prompts(monkeypatch, home: Path, *extra: str) -> None:
    """Run the real CLI entrypoint against ``home`` (what an operator's update script calls)."""
    import hermes_cli.main as main_mod

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "refresh-prompts", *extra])
    try:
        main_mod.main()
    except SystemExit as exc:
        assert exc.code in (0, None), exc.code


def _assert_updated(system: str) -> None:
    assert SOUL_V2 in system and SOUL_V1 not in system
    assert "Description revision two." in system and "Description revision one." not in system
    assert "incident-triage" in system
    assert "Project rules revision two." in system


def _assert_stale(system: str) -> None:
    assert SOUL_V1 in system and SOUL_V2 not in system
    assert "Description revision one." in system
    assert "incident-triage" not in system
    assert "Project rules revision one." in system


def test_without_the_verb_a_continuing_session_keeps_its_stored_prompt(profile):
    srv, home, project = profile
    history, first = _turn(srv, home, "chat-a", "hello", [])
    _assert_stale(first)
    _update_profile(home, project)
    history, second = _turn(srv, home, "chat-a", "still there?", history)
    assert second == first  # main's contract: the stored bytes are replayed until /new


def test_the_verb_makes_the_next_turn_adopt_soul_skills_and_project_files_once(profile, monkeypatch, capsys):
    from run_agent import AIAgent

    srv, home, project = profile
    history, _ = _turn(srv, home, "chat-a", "hello", [])
    history, _ = _turn(srv, home, "chat-a", "how are things?", history)
    _update_profile(home, project)

    _refresh_prompts(monkeypatch, home)
    assert "1 session" in capsys.readouterr().out

    history, refreshed = _turn(srv, home, "chat-a", "anything new?", history)
    _assert_updated(refreshed)
    request = srv.main_requests()[-1]
    assert _user_texts(request)[:3] == ["hello", "how are things?", "anything new?"]  # history intact

    # One break per request: the next fresh agent restores the refreshed bytes without a rebuild.
    builds = []
    original = AIAgent._build_system_prompt
    monkeypatch.setattr(AIAgent, "_build_system_prompt",
                        lambda self, *a, **k: builds.append(1) or original(self, *a, **k))
    history, after = _turn(srv, home, "chat-a", "and now?", history)
    assert after == refreshed and builds == []
    assert len(_user_texts(srv.main_requests()[-1])) == 4

    from hermes_state import SessionDB
    with SessionDB(db_path=home / "state.db") as db:
        assert db.get_session("chat-a")["system_prompt"] == refreshed  # persisted for the next restore


def test_session_flag_refreshes_only_that_session(profile, monkeypatch):
    srv, home, project = profile
    history_a, _ = _turn(srv, home, "chat-a", "hello", [])
    history_b, _ = _turn(srv, home, "chat-b", "hello", [])
    _update_profile(home, project)

    _refresh_prompts(monkeypatch, home, "--session", "chat-a")

    _, system_a = _turn(srv, home, "chat-a", "next", history_a)
    _, system_b = _turn(srv, home, "chat-b", "next", history_b)
    _assert_updated(system_a)
    _assert_stale(system_b)


def test_a_session_started_after_the_request_is_not_rebuilt_again(profile, monkeypatch):
    from run_agent import AIAgent

    srv, home, _project = profile
    _refresh_prompts(monkeypatch, home)
    history, first = _turn(srv, home, "chat-new", "hello", [])
    builds = []
    original = AIAgent._build_system_prompt
    monkeypatch.setattr(AIAgent, "_build_system_prompt",
                        lambda self, *a, **k: builds.append(1) or original(self, *a, **k))
    _, second = _turn(srv, home, "chat-new", "next", history)
    assert second == first and builds == []


def test_warm_cached_agent_drops_its_stale_prompt(profile, monkeypatch):
    srv, home, project = profile
    agent = _agent(srv, home, "chat-warm")
    try:
        history, first = _turn(srv, home, "chat-warm", "hello", [], agent=agent)
        history, second = _turn(srv, home, "chat-warm", "again", history, agent=agent)
        assert second == first
        _update_profile(home, project)
        _refresh_prompts(monkeypatch, home)

        history, refreshed = _turn(srv, home, "chat-warm", "anything new?", history, agent=agent)
        _assert_updated(refreshed)
        assert _user_texts(srv.main_requests()[-1])[:3] == ["hello", "again", "anything new?"]
        history, after = _turn(srv, home, "chat-warm", "and now?", history, agent=agent)
        assert after == refreshed
    finally:
        agent.close()
    # The warm refresh was persisted: an evicted/restarted agent restores the same bytes.
    _, restored = _turn(srv, home, "chat-warm", "after restart", history)
    assert restored == refreshed


def test_refresh_request_is_scoped_to_its_profile(profile, monkeypatch, tmp_path):
    srv, home_a, project = profile
    home_b = tmp_path / "profile-b"
    _write_profile(home_b, srv.base_url, project)

    history_a, _ = _turn(srv, home_a, "chat-a", "hello", [])
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    history_b, _ = _turn(srv, home_b, "chat-b", "hello", [])

    _update_profile(home_a, project)
    _update_profile(home_b, project)
    _refresh_prompts(monkeypatch, home_a)

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    _, system_a = _turn(srv, home_a, "chat-a", "next", history_a)
    _assert_updated(system_a)
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    _, system_b = _turn(srv, home_b, "chat-b", "next", history_b)
    assert SOUL_V1 in system_b and "incident-triage" not in system_b


def test_no_request_means_no_rebuild_and_no_prompt_source_reads(profile, monkeypatch):
    """Never ran the verb: the stored prompt is replayed byte-identically with no source reads."""
    import agent.prompt_builder as pb
    from run_agent import AIAgent

    srv, home, project = profile
    history, first = _turn(srv, home, "chat-a", "hello", [])
    _update_profile(home, project)

    calls = []
    # build_skills_system_prompt is left out: main's per-turn skills delta note reads it already.
    for name in ("load_soul_md", "build_context_files_prompt"):
        original_fn = getattr(pb, name)
        monkeypatch.setattr(pb, name, lambda *a, _n=name, _f=original_fn, **k: calls.append(_n) or _f(*a, **k))
    original = AIAgent._build_system_prompt
    monkeypatch.setattr(AIAgent, "_build_system_prompt",
                        lambda self, *a, **k: calls.append("build") or original(self, *a, **k))
    _, second = _turn(srv, home, "chat-a", "next", history)
    assert second == first
    assert calls == []


def test_refresh_rebuilds_the_skills_index_from_disk_even_when_its_cache_cannot_see_the_edit(profile, monkeypatch):
    """Coarse filesystem timestamps can leave an in-place, same-size SKILL.md edit with an unchanged
    stat signature (see ``utils.RACY_MTIME_WINDOW_NS``); the skills index cache, keyed on those
    versions, would then keep serving the old render. The refresh must rebuild from disk anyway."""
    import agent.prompt_builder as pb

    # A key that never moves on a same-size edit (timestamps and digest frozen).
    monkeypatch.setattr(pb, "file_cache_version", lambda path, st, **_kw: (1, st.st_size, 1, 1))
    srv, home, _project = profile
    history, first = _turn(srv, home, "chat-a", "hello", [])
    assert "Description revision one." in first
    _skill(home, "deploy-runbook", "Roll out a release. Description revision six.")  # same size
    _refresh_prompts(monkeypatch, home)
    _, refreshed = _turn(srv, home, "chat-a", "next", history)
    assert "Description revision six." in refreshed and "Description revision one." not in refreshed


def _tool_description(request, name: str) -> str:
    return next(t["function"]["description"] for t in request["tools"] if t["function"]["name"] == name)


def test_refresh_keeps_the_sessions_tools_pin(profile, monkeypatch):
    """The refresh is about instructions: tools[] stays the array the session already sends, even
    when the agent built for this turn derives a different schema for one of its tools."""
    import model_tools

    srv, home, project = profile
    history, _ = _turn(srv, home, "chat-a", "hello", [])
    pinned = srv.main_requests()[-1]["tools"]

    changed = "Changed after the session pinned its tools."
    monkeypatch.setitem(model_tools._DYNAMIC_SCHEMA_REWRITERS, "clarify",
                        lambda td, _available: {**td, "function": {**td["function"], "description": changed}})
    monkeypatch.setattr(model_tools, "_tool_defs_cache", type(model_tools._tool_defs_cache)())
    _update_profile(home, project)
    _refresh_prompts(monkeypatch, home)

    _, refreshed = _turn(srv, home, "chat-a", "next", history)
    _assert_updated(refreshed)
    assert srv.main_requests()[-1]["tools"] == pinned

    # Control: a new session built by the same process really does carry the changed schema.
    _turn(srv, home, "chat-new", "hello", [])
    assert _tool_description(srv.main_requests()[-1], "clarify") == changed


def test_refresh_prompts_unknown_session_fails(profile, monkeypatch, capsys):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "refresh-prompts", "--session", "nope"])
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 1
    assert "No session matches" in capsys.readouterr().out


def _count_builds(monkeypatch):
    from run_agent import AIAgent
    builds = []
    original = AIAgent._build_system_prompt
    monkeypatch.setattr(AIAgent, "_build_system_prompt",
                        lambda self, *a, **k: builds.append(1) or original(self, *a, **k))
    return builds


def _bot_chat_branch_once(monkeypatch):
    """Take the Bot Chat capability-refresh branch on the next restore only (its real trigger needs
    a Bot Chat capability fingerprint; the branch body is what is under test)."""
    import agent.conversation_loop as cl
    once = []
    monkeypatch.setattr(cl, "_bot_chat_prompt_stale", lambda agent, prompt: not once and once.append(1) is None)
    monkeypatch.setattr(cl, "_refresh_bot_chat_tools", lambda agent: None)
    return once


def test_warm_agent_that_restored_its_prompt_adopts_a_later_refresh(profile, monkeypatch):
    srv, home, project = profile
    history, first = _turn(srv, home, "chat-r", "hello", [])
    warm = _agent(srv, home, "chat-r")  # a gateway/TUI restart, then the agent stays warm
    try:
        history, restored = _turn(srv, home, "chat-r", "again", history, agent=warm)
        assert restored == first
        _update_profile(home, project)
        _refresh_prompts(monkeypatch, home)
        history, refreshed = _turn(srv, home, "chat-r", "new?", history, agent=warm)
        _assert_updated(refreshed)
    finally:
        warm.close()


def test_warm_agent_without_a_request_never_rebuilds(profile, monkeypatch):
    srv, home, project = profile
    warm = _agent(srv, home, "chat-w")
    try:
        history, first = _turn(srv, home, "chat-w", "hello", [], agent=warm)
        _update_profile(home, project)  # files change, the verb is not run
        builds = _count_builds(monkeypatch)
        for text in ("two", "three", "four"):
            history, system = _turn(srv, home, "chat-w", text, history, agent=warm)
            assert system == first
        assert builds == []
    finally:
        warm.close()


def test_warm_agent_rebuilt_by_a_bot_chat_refresh_still_adopts_a_later_refresh(profile, monkeypatch):
    srv, home, project = profile
    history, _ = _turn(srv, home, "chat-bot", "hello", [])
    once = _bot_chat_branch_once(monkeypatch)
    warm = _agent(srv, home, "chat-bot")
    try:
        history, _ = _turn(srv, home, "chat-bot", "again", history, agent=warm)
        assert once == [1]
        _update_profile(home, project)
        _refresh_prompts(monkeypatch, home)
        history, refreshed = _turn(srv, home, "chat-bot", "anything new?", history, agent=warm)
        _assert_updated(refreshed)
    finally:
        warm.close()


def test_a_bot_chat_rebuild_satisfies_a_pending_refresh(profile, monkeypatch):
    srv, home, project = profile
    history, _ = _turn(srv, home, "chat-bot2", "hello", [])
    _update_profile(home, project)
    _refresh_prompts(monkeypatch, home)
    _bot_chat_branch_once(monkeypatch)
    history, rebuilt = _turn(srv, home, "chat-bot2", "again", history)
    _assert_updated(rebuilt)
    builds = _count_builds(monkeypatch)
    history, third = _turn(srv, home, "chat-bot2", "third", history)
    assert third == rebuilt
    assert builds == []


def _rotate(srv, home, session_id):
    """Fill a session, then compact it with rotation (``compression.in_place: false``)."""
    warm = _agent(srv, home, session_id)
    history = []
    try:
        for i in range(14):
            history, _ = _turn(srv, home, session_id, f"turn {i} " + f"lorem ipsum dolor {i} " * 200,
                               history, agent=warm)
        warm.compression_in_place = False
        compressed, _ = warm._compress_context(history, None, force=True)
        assert warm.session_id != session_id
        return warm.session_id, compressed
    finally:
        warm.close()


def test_rotated_compression_keeps_the_refresh_epoch_and_session_flag_targets_the_tip(profile, monkeypatch):
    srv, home, project = profile
    _refresh_prompts(monkeypatch, home)  # a past profile-wide request
    child, history = _rotate(srv, home, "chat-rot")
    builds = _count_builds(monkeypatch)
    history, _ = _turn(srv, home, child, "after eviction", history)
    assert builds == []  # the child satisfies the past request; no extra cache break

    _update_profile(home, project)
    _refresh_prompts(monkeypatch, home, "--session", "chat-rot")  # the conversation's original id
    _, refreshed = _turn(srv, home, child, "next", history)
    _assert_updated(refreshed)
