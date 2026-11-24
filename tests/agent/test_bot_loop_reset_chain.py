"""A human turn can end this session's bot chain and write the reset epoch.

The chain module only calls ``on_reset``. ``note_human_reset`` decides whether
that callback stores an epoch. No conversation loop is booted.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.bot_loop_guard import BotLoopGuardSettings
from tools.bot_mode_probe import BOT_CHAT_TITLE


@pytest.fixture
def bot_chain(monkeypatch):
    from agent import bot_chain

    monkeypatch.setattr(bot_chain, "_CHAINS", {})
    monkeypatch.setattr(bot_chain, "_settings", lambda: BotLoopGuardSettings(max_hops=4))
    return bot_chain


def _agent(session_id="chat", **extra):
    agent = SimpleNamespace(session_id=session_id, _turn_author=None)
    for key, value in extra.items():
        setattr(agent, key, value)
    return agent


def _bot_author():
    return {"id": "bot:alpha", "name": "alpha", "is_bot": True}


def test_human_turn_with_no_chain_calls_on_reset(bot_chain):
    called = []
    agent = _agent()
    bot_chain.on_turn_start(
        agent, {"id": "user", "is_bot": False}, None, "hello",
        on_reset=lambda: called.append("reset"),
    )
    assert called == ["reset"]
    assert "chat" not in bot_chain._CHAINS


def test_human_turn_deletes_existing_chain_and_calls_on_reset(bot_chain):
    agent = _agent()
    bot_chain.on_turn_start(agent, _bot_author(), None, "ping")
    assert bot_chain._CHAINS["chat"]["hops"] >= 1
    called = []
    bot_chain.on_turn_start(agent, None, None, "stop", on_reset=lambda: called.append("reset"))
    assert called == ["reset"]
    assert "chat" not in bot_chain._CHAINS


def test_bot_author_does_not_call_on_reset(bot_chain):
    called = []
    bot_chain.on_turn_start(
        _agent(), _bot_author(), None, "ping",
        on_reset=lambda: called.append("reset"),
    )
    assert called == []
    assert "chat" in bot_chain._CHAINS


def test_machinery_display_kind_does_not_call_on_reset(bot_chain):
    called = []
    bot_chain.on_turn_start(
        _agent(), None, "internal_notification", "wake",
        on_reset=lambda: called.append("reset"),
    )
    assert called == []

    agent = _agent("waking")
    bot_chain.on_turn_start(agent, _bot_author(), None, "ping")
    bot_chain.on_turn_start(
        agent, None, "process_complete", "done",
        on_reset=lambda: called.append("reset"),
    )
    assert called == []
    assert "waking" in bot_chain._CHAINS


def test_guard_disabled_does_not_call_on_reset(bot_chain, monkeypatch):
    monkeypatch.setattr(bot_chain, "_settings", lambda: BotLoopGuardSettings(enabled=False))
    agent = _agent()
    bot_chain._CHAINS["chat"] = {"hops": 2, "sent": 2, "asks": set(), "no_reply_from": None, "updated": 0}
    called = []
    bot_chain.on_turn_start(agent, None, None, "hi", on_reset=lambda: called.append("reset"))
    assert called == []
    assert "chat" in bot_chain._CHAINS


def test_fork_does_not_call_on_reset(bot_chain):
    agent = _agent()
    bot_chain.on_turn_start(agent, _bot_author(), None, "ping")
    agent._turn_origin = "background_review"
    called = []
    bot_chain.on_turn_start(agent, None, None, "hi", on_reset=lambda: called.append("reset"))
    assert called == []
    assert "chat" in bot_chain._CHAINS


def test_raising_on_reset_does_not_propagate(bot_chain, caplog):
    agent = _agent()
    bot_chain.on_turn_start(agent, _bot_author(), None, "ping")

    def boom():
        raise RuntimeError("epoch write failed")

    with caplog.at_level("WARNING", logger="agent.bot_chain"):
        bot_chain.on_turn_start(agent, None, None, "stop", on_reset=boom)
    assert "chat" not in bot_chain._CHAINS
    assert any(r.levelname == "WARNING" for r in caplog.records)


def _install_epoch_sandbox(monkeypatch, tmp_path, *, enabled: object = True):
    """Patch the names ``note_human_reset`` must call. Never the real Hermes root."""
    import gateway.bot_loop_reset as reset

    def _forbid_real_root(*_args, **_kwargs):
        raise AssertionError(
            "get_hermes_root must be the name imported into gateway.bot_loop_reset"
        )

    monkeypatch.setattr("hermes_constants.get_hermes_root", _forbid_real_root)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", _forbid_real_root)
    monkeypatch.setattr(reset, "get_hermes_root", lambda: tmp_path)
    monkeypatch.setattr(reset, "reset_all_on_human", lambda root=None: enabled)

    real_write = reset.write_epoch

    def _write(root, now):
        root = Path(root)
        if root != tmp_path and tmp_path not in root.parents:
            raise AssertionError(f"refusing epoch write outside tmp_path: {root}")
        return real_write(root, now)

    monkeypatch.setattr(reset, "write_epoch", _write)
    return reset


def _bot_chat_agent(**extra):
    agent = SimpleNamespace(
        _session_title_hint=BOT_CHAT_TITLE,
        _session_db=None,
        session_id="s",
    )
    for key, value in extra.items():
        setattr(agent, key, value)
    return agent


def test_note_human_reset_writes_epoch_only_for_bot_chat_when_flag_on(tmp_path, monkeypatch):
    reset = _install_epoch_sandbox(monkeypatch, tmp_path, enabled=True)
    reset.note_human_reset(_bot_chat_agent())
    stored = reset.read_epoch(tmp_path)
    assert isinstance(stored, float) and stored > 0
    assert reset.epoch_path(tmp_path) == tmp_path / reset.EPOCH_NAME
    assert reset.epoch_path(tmp_path).is_file()


def test_flag_off_does_not_write(tmp_path, monkeypatch):
    reset = _install_epoch_sandbox(monkeypatch, tmp_path, enabled=False)
    reset.note_human_reset(_bot_chat_agent())
    assert reset.read_epoch(tmp_path) == 0.0
    assert not reset.epoch_path(tmp_path).exists()


def test_flag_not_exactly_true_does_not_write(tmp_path, monkeypatch):
    reset = _install_epoch_sandbox(monkeypatch, tmp_path, enabled=1)
    reset.note_human_reset(_bot_chat_agent())
    assert not reset.epoch_path(tmp_path).exists()


@pytest.mark.parametrize("title", ["", "   ", "bot chat", "Bot  Chat", "General", None])
def test_other_title_does_not_write(tmp_path, monkeypatch, title):
    reset = _install_epoch_sandbox(monkeypatch, tmp_path, enabled=True)
    reset.note_human_reset(_bot_chat_agent(_session_title_hint=title))
    assert not reset.epoch_path(tmp_path).exists()


def test_empty_hint_uses_session_store_title(tmp_path, monkeypatch):
    reset = _install_epoch_sandbox(monkeypatch, tmp_path, enabled=True)

    class _Sessions:
        def get_title(self, session_id):
            assert session_id == "s"
            return BOT_CHAT_TITLE

    class _Store:
        sessions = _Sessions()

    monkeypatch.setattr("agent.agent_init_session.agent_interface", lambda agent: _Store())
    reset.note_human_reset(_bot_chat_agent(_session_title_hint="", _session_db=object()))
    assert isinstance(reset.read_epoch(tmp_path), float)


def test_empty_hint_uses_session_db_when_store_is_missing(tmp_path, monkeypatch):
    reset = _install_epoch_sandbox(monkeypatch, tmp_path, enabled=True)

    class _DB:
        def get_session_title(self, session_id):
            assert session_id == "s"
            return "Notes"

    monkeypatch.setattr("agent.agent_init_session.agent_interface", lambda agent: None)
    reset.note_human_reset(_bot_chat_agent(_session_title_hint="", _session_db=_DB()))
    assert not reset.epoch_path(tmp_path).exists()


def test_hint_is_used_before_the_session_store(tmp_path, monkeypatch):
    reset = _install_epoch_sandbox(monkeypatch, tmp_path, enabled=True)

    def _boom(_agent):
        raise AssertionError("session store consulted despite a title hint")

    monkeypatch.setattr("agent.agent_init_session.agent_interface", _boom)
    reset.note_human_reset(_bot_chat_agent(_session_title_hint="Notes", _session_db=object()))
    assert not reset.epoch_path(tmp_path).exists()


def test_raising_write_epoch_does_not_propagate(tmp_path, monkeypatch, caplog):
    reset = _install_epoch_sandbox(monkeypatch, tmp_path, enabled=True)

    def _boom(_root, now):
        raise OSError("disk full")

    monkeypatch.setattr(reset, "write_epoch", _boom)
    with caplog.at_level("WARNING", logger="gateway.bot_loop_reset"):
        reset.note_human_reset(_bot_chat_agent())
    assert any(r.levelname == "WARNING" for r in caplog.records)
    assert not reset.epoch_path(tmp_path).exists()


def test_turn_context_passes_note_human_reset_closure():
    source = Path("agent/turn_context.py").read_text(encoding="utf-8")
    assert "note_human_reset" in source
    assert "on_reset=" in source
