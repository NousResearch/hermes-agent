"""``prompt.next_prompt``: the owner runs the suggestion on the session's route and bills it to the session."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from agent import next_prompt_suggester
from agent.aux_accounting import record_aux_usage
from gateway import session_tui_side
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError

HISTORY = [{"role": "user", "content": "run the tests"}, {"role": "assistant", "content": "All green."}]


def _response(text="commit that"):
    return SimpleNamespace(
        model="aux-model", choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=7, total_tokens=127))


@pytest.fixture
def db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-1", source="cli", model="main-model")
    yield db
    db.close()


def _llm(monkeypatch, text="commit that", seen=None):
    # No price lookup (it would reach for the network); usage is what is under test.
    monkeypatch.setattr("agent.usage_pricing.estimate_usage_cost",
                        lambda *a, **k: SimpleNamespace(amount_usd=None))
    def fake(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        response = _response(text)
        record_aux_usage(response, kwargs["task"], provider="openrouter")  # what call_llm does on return
        return response
    monkeypatch.setattr(next_prompt_suggester, "call_llm", fake)


def test_usage_is_recorded_as_next_prompt_against_the_owning_session_in_the_real_store(db, monkeypatch):
    seen = {}
    _llm(monkeypatch, seen=seen)
    authority = SimpleNamespace(db=db)
    ref = SimpleNamespace(session_id="sess-1")
    route = {"model": "live-model", "provider": "live-provider", "session_id": "sess-1"}
    assert session_tui_side.suggest_next_prompt(authority, ref, HISTORY, route) == "commit that"
    assert seen["main_runtime"] == route
    usage = db.auxiliary_usage_by_task("sess-1")
    assert usage["next_prompt"]["input_tokens"] == 120
    assert usage["next_prompt"]["output_tokens"] == 7


def test_none_answer_is_empty_text_and_failure_logs_at_debug(db, monkeypatch, caplog):
    authority, ref = SimpleNamespace(db=db), SimpleNamespace(session_id="sess-1")
    _llm(monkeypatch, text="NONE")
    assert session_tui_side.suggest_next_prompt(authority, ref, HISTORY, {}) == ""

    def boom(**kwargs):
        raise TimeoutError("slow")
    monkeypatch.setattr(next_prompt_suggester, "call_llm", boom)
    with caplog.at_level(logging.DEBUG):
        assert session_tui_side.suggest_next_prompt(authority, ref, HISTORY, {}) == ""
    ours = ("agent.next_prompt_suggester", "gateway.session_tui_side")
    assert not [r for r in caplog.records if r.name in ours and r.levelno >= logging.WARNING]


class _Verb:
    """Drive the real handler with admission and the detached thread stubbed out.

    ``saved`` is the toggle frozen in the session's saved config; ``current`` is what the owning
    profile's config.yaml says now."""

    def __init__(self, monkeypatch, tmp_path, db, *, saved, current, bypass=False):
        self.spawned = []
        (tmp_path / "config.yaml").write_text(f"display:\n  next_prompt_suggestions: {str(current).lower()}\n")
        policy = SimpleNamespace(config=lambda authority: {"display": {"next_prompt_suggestions": saved}},
                                 ignore_user_config=bypass)
        db.db_path = tmp_path / "state.db"
        self.authority = SimpleNamespace(db=db, agent=lambda ref: None)
        self.connection = SimpleNamespace(authority=self.authority)
        monkeypatch.setattr(session_tui_side, "_admit", lambda *a, **k: (None, policy, "", "np_1"))
        monkeypatch.setattr(session_tui_side, "_spawn", lambda *a, **k: self.spawned.append((a, k)) or {"task_id": "np_1"})
        monkeypatch.setattr("gateway.session_local_recovery.transcript_target", lambda authority, ref: None)

    def call(self):
        ref = SimpleNamespace(session_id="sess-1")
        return asyncio.run(session_tui_side.next_prompt(self.connection, ref, {"session_id": "sess-1"}))


def test_verb_refuses_when_the_setting_is_off_and_starts_nothing(db, monkeypatch, tmp_path):
    verb = _Verb(monkeypatch, tmp_path, db, saved=False, current=False)
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        verb.call()
    assert verb.spawned == []


def test_verb_replies_with_a_task_id_and_publishes_next_prompt_complete(db, monkeypatch, tmp_path):
    verb = _Verb(monkeypatch, tmp_path, db, saved=True, current=True)
    assert verb.call() == {"task_id": "np_1"}
    (args, _kwargs), = verb.spawned
    assert args[2:4] == ("np_1", "next_prompt.complete")


def test_an_existing_session_follows_the_current_toggle_not_its_saved_one(db, monkeypatch, tmp_path):
    """The toggle was off when the conversation was created and on now (also after --resume)."""
    verb = _Verb(monkeypatch, tmp_path, db, saved=False, current=True)
    assert verb.call() == {"task_id": "np_1"}
    off_now = _Verb(monkeypatch, tmp_path, db, saved=True, current=False)
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        off_now.call()


def test_a_bypass_session_never_reads_the_profile_and_uses_its_saved_toggle(db, monkeypatch, tmp_path):
    assert _Verb(monkeypatch, tmp_path, db, saved=True, current=False, bypass=True).call() == {"task_id": "np_1"}
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        _Verb(monkeypatch, tmp_path, db, saved=False, current=True, bypass=True).call()


def test_the_verb_is_registered_with_the_other_side_verbs():
    assert "prompt.next_prompt" in session_tui_side.handlers(SimpleNamespace())
