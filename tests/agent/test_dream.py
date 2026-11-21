"""Behaviour contracts for ``hermes dream``: what it may keep, what it must drop, how it undoes itself."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import dream
from tools.memory_tool import MemoryStore

# The store fixture seeds a Hermes home; keep it out of the checkout (see tests/conftest.py).
pytestmark = pytest.mark.usefixtures("authority_cwd_in_tmp")

ENTRY = "The person prefers answers in Spanish and expects them in Spanish."
SPANISH = "Prefiero las respuestas en español, siempre."
ACORDATE = "Y acordate: español, no inglés."
PYTHON3 = "Ahora uso Python 3, olvidate del 2."


def _conv(session_id: str, *lines: tuple[str, str]) -> dream.DreamConversation:
    return dream.DreamConversation(
        session_id=session_id,
        title=session_id,
        turns=tuple(dream.DreamTurn(role=role, text=text) for role, text in lines),
    )


def _proposal(**overrides) -> dream.DreamProposal:
    fields = {
        "target": "user",
        "action": "add",
        "content": ENTRY,
        "from_sessions": (1, 2),
        "evidence": (SPANISH, ACORDATE),
        "confidence": 0.9,
    }
    fields.update(overrides)
    return dream.DreamProposal(**fields)


@pytest.fixture()
def store() -> MemoryStore:
    store = MemoryStore()
    store.load_from_disk()
    return store


@pytest.fixture()
def conversations() -> list[dream.DreamConversation]:
    return [
        _conv("s1", ("user", SPANISH), ("assistant", "Entendido.")),
        _conv("s2", ("user", ACORDATE), ("assistant", "Listo.")),
    ]


def _payload(**overrides) -> str:
    entry = {
        "target": "user",
        "action": "add",
        "content": ENTRY,
        "from": [1, 2],
        "evidence": [SPANISH, ACORDATE],
        "confidence": 0.9,
    }
    entry.update(overrides)
    return json.dumps({"entries": [entry]})


def _patch_conversations(monkeypatch, conversations):
    monkeypatch.setattr(
        dream, "load_conversations", lambda home, *, since_ts, max_sessions: (conversations, [])
    )


# --- what it keeps ------------------------------------------------------------------


def test_a_pattern_two_conversations_show_is_kept(store, conversations):
    assert dream.validate(_proposal(), conversations, store) == ""


def test_accepted_entry_lands_in_memory_and_in_the_journal(monkeypatch, store, conversations):
    _patch_conversations(monkeypatch, conversations)
    report = dream.run_dream(llm=lambda prompt: _payload(), store=store)

    assert [applied["content"] for applied in report["applied"]] == [ENTRY]
    assert ENTRY in store.user_entries
    journal = json.loads(Path(report["journal"]).read_text(encoding="utf-8"))
    assert journal["applied"] == [{"target": "user", "action": "add", "content": ENTRY, "before": None}]


def test_correcting_a_stale_entry_replaces_it(monkeypatch, store, conversations):
    store.add("memory", "The person uses Python 2.")
    stale = [
        _conv("s1", ("user", PYTHON3), ("assistant", "Anotado.")),
        _conv("s2", ("user", "Dale, Python 3 de una vez."), ("assistant", "Hecho.")),
    ]
    _patch_conversations(monkeypatch, stale)
    payload = _payload(
        target="memory",
        action="replace",
        old_text="The person uses Python 2.",
        content="The person uses Python 3.",
        **{"from": [1]},
        evidence=[PYTHON3],
    )

    dream.run_dream(llm=lambda prompt: payload, store=store)

    assert "The person uses Python 3." in store.memory_entries
    assert "The person uses Python 2." not in store.memory_entries


def test_a_second_dream_reports_the_entry_as_already_known(monkeypatch, store, conversations):
    _patch_conversations(monkeypatch, conversations)
    dream.run_dream(llm=lambda prompt: _payload(), store=store)

    second = dream.run_dream(llm=lambda prompt: _payload(), store=store)

    assert second["applied"] == []
    assert second["proposals"][0]["reason"] == "already in memory"
    assert "journal" not in second


# --- what it drops ------------------------------------------------------------------


def test_a_new_entry_needs_two_conversations(store, conversations):
    reason = dream.validate(_proposal(from_sessions=(1,), evidence=(SPANISH,)), conversations, store)
    assert "needs 2" in reason


def test_the_quote_must_really_be_in_the_cited_conversation(store, conversations):
    reason = dream.validate(
        _proposal(evidence=("Me gusta el fútbol los domingos.", ACORDATE)), conversations, store
    )
    assert "quote is not in conversation 1" in reason


def test_machinery_notices_are_not_the_person_s_words():
    rows = [
        {"role": "user", "content": "prefiero respuestas cortas", "display_kind": None},
        {"role": "user", "content": "[IMPORTANT: 3 background results arrived]", "display_kind": "process_complete"},
        {"role": "assistant", "content": "listo", "display_kind": None},
    ]

    turns = dream._turns_from_rows(rows)

    assert [turn.text for turn in turns] == ["prefiero respuestas cortas", "listo"]


def test_an_entry_may_not_invent_a_number_or_a_path(store, conversations):
    reason = dream.validate(
        _proposal(content="The person keeps 3 repos under C:\\work\\lumen."), conversations, store
    )
    assert "invents" in reason
    assert "3" in reason


def test_a_number_the_person_said_is_allowed(store):
    conversations = [
        _conv("s1", ("user", "Trabajo con 3 repos, siempre los mismos.")),
        _conv("s2", ("user", "Los 3 repos del proyecto Lumen me llevan tiempo.")),
    ]
    proposal = _proposal(content="The person works across 3 repositories.", evidence=(
        "Trabajo con 3 repos, siempre los mismos.", "Los 3 repos del proyecto Lumen me llevan tiempo."))
    assert dream.validate(proposal, conversations, store) == ""


def test_a_secret_shaped_entry_is_dropped(store, conversations):
    forged = "api_" + 'key="' + "sk-" + "x" * 24 + '"'
    reason = dream.validate(_proposal(content=forged), conversations, store)
    assert "secret" in reason


def test_old_text_must_name_one_existing_entry(store, conversations):
    reason = dream.validate(
        _proposal(action="replace", old_text="an entry that does not exist"), conversations, store
    )
    assert "old_text" in reason


def test_an_entry_over_the_budget_is_dropped(store, conversations):
    store.add("user", "x" * (store.user_char_limit - 10))
    reason = dream.validate(_proposal(), conversations, store)
    assert "characters" in reason


def test_parse_keeps_good_entries_and_names_bad_ones():
    proposals, notes = dream.parse_proposals(
        "```json\n"
        + json.dumps(
            {
                "entries": [
                    {"target": "user", "action": "add", "content": ENTRY, "from": [1, 2], "evidence": [SPANISH, ACORDATE]},
                    {"target": "elsewhere", "content": "x"},
                    {"target": "memory", "action": "add", "content": ""},
                ]
            }
        )
        + "\n```"
    )
    assert [proposal.content for proposal in proposals] == [ENTRY]
    assert len(notes) == 2


def test_an_answer_without_json_is_an_error():
    with pytest.raises(ValueError):
        dream.parse_proposals("I could not find anything worth keeping.")


# --- dry run, revert -----------------------------------------------------------------


def test_the_budget_is_checked_against_the_whole_batch(monkeypatch, conversations):
    tight = MemoryStore(memory_char_limit=2200, user_char_limit=120)
    tight.load_from_disk()
    second = "The person corrects the agent's language when it drifts."
    payload = json.dumps({"entries": [
        {"target": "user", "action": "add", "content": ENTRY, "from": [1, 2],
         "evidence": [SPANISH, ACORDATE], "confidence": 0.95},
        {"target": "user", "action": "add", "content": second, "from": [1, 2],
         "evidence": [SPANISH, ACORDATE], "confidence": 0.9},
    ]})
    _patch_conversations(monkeypatch, conversations)

    report = dream.run_dream(llm=lambda prompt: payload, store=tight)

    assert [applied["content"] for applied in report["applied"]] == [ENTRY]
    assert tight.user_entries == [ENTRY]
    assert "characters" in report["proposals"][1]["reason"]


def test_dry_run_writes_nothing(monkeypatch, store, conversations):
    _patch_conversations(monkeypatch, conversations)

    report = dream.run_dream(llm=lambda prompt: _payload(), store=store, dry_run=True)

    assert report["applied"] == []
    assert "journal" not in report
    assert ENTRY not in store.user_entries
    assert report["proposals"][0]["accepted"] is True


def test_revert_removes_what_the_dream_added(monkeypatch, store, conversations):
    _patch_conversations(monkeypatch, conversations)
    report = dream.run_dream(llm=lambda prompt: _payload(), store=store)
    assert ENTRY in store.user_entries

    result = dream.revert_journal(report["journal"], store)

    assert result == {"restored": 1, "failures": []}
    assert ENTRY not in store.user_entries


def test_revert_restores_the_entry_a_dream_replaced(monkeypatch, store, conversations):
    store.add("memory", "The person uses Python 2.")
    stale = [
        _conv("s1", ("user", PYTHON3), ("assistant", "Anotado.")),
        _conv("s2", ("user", "Dale, Python 3 de una vez."), ("assistant", "Hecho.")),
    ]
    _patch_conversations(monkeypatch, stale)
    payload = _payload(
        target="memory",
        action="replace",
        old_text="The person uses Python 2.",
        content="The person uses Python 3.",
        **{"from": [1]},
        evidence=[PYTHON3],
    )
    report = dream.run_dream(llm=lambda prompt: payload, store=store)

    dream.revert_journal(report["journal"], store)

    assert "The person uses Python 2." in store.memory_entries
    assert "The person uses Python 3." not in store.memory_entries
