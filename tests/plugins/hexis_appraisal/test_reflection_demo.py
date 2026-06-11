"""Offline cross-session reflection demo — the Plan 03-02 proof (run with -s).

    ./scripts/test.sh hexis_appraisal/tests/test_reflection_demo.py -s

The full cross-lag loop against fake LLMs, through the REAL registered
hooks: session A asserts "Postgres is final" then flatly reverses to
SQLite-everywhere (capture via post_llm_call) -> on_session_start("B")
triggers the debounced reflection pass (bounded deltas through the
apply_deltas funnel) -> session B's appraisal call RECEIVES the persisted
contradiction in its state dump (the carrier works) and its rendered block
carries a REFL-05 trust note. Live proof of the same loop is Plan 03-03.

State is redirected to tmp_path by monkeypatching store.get_db_path (the
deterministic route — get_db_path prefers the host's get_hermes_home,
which may ignore a late env change). The real $HERMES_HOME is untouched.
"""

import json
import sqlite3
import types

import pytest

plugin_llm = pytest.importorskip("agent.plugin_llm")

from .conftest import assert_no_directive_language  # noqa: E402
import hexis_appraisal  # noqa: E402
from hexis_appraisal import appraisal, config, store  # noqa: E402

TOPIC_X_DECLARATION = (
    "We standardize on Postgres for every service — that decision is final."
)
TOPIC_X_REVERSAL = (
    "Forget Postgres entirely — we're moving everything to SQLite tomorrow;"
    " honestly it was always the plan."
)
CONTRADICTION_DESCRIPTION = (
    "user declared the Postgres standard final, then reversed to"
    " SQLite-everywhere within the same span"
)

APPRAISAL_SESSION_A = {
    "salient_observations": [
        {"text": "user is settling database standards", "confidence": 0.8},
    ],
    "gut_reaction": "",
}

REFLECTION_PAYLOAD = {
    "affect": {"summary": "decisive but volatile infrastructure talk",
               "valence_delta": -0.05, "arousal_delta": 0.1,
               "intensity_delta": 0.05},
    "new_concerns": [
        {"text": "database standard unsettled after the reversal",
         "weight": 0.7},
    ],
    "new_contradictions": [
        {"kind": "narrative",
         "description": CONTRADICTION_DESCRIPTION,
         "evidence": "Postgres-final declaration followed by"
                     " SQLite-everywhere reversal in consecutive turns"},
    ],
    "trust_adjustments": [
        {"key": "stated infrastructure decisions", "delta": -0.15},
    ],
}

APPRAISAL_SESSION_B = {
    "salient_observations": [
        {"text": "user is choosing a database for a new service",
         "confidence": 0.85},
    ],
    "contradiction_flags": [
        {"kind": "narrative",
         "text": "persisted reversal resurfaces: Postgres-final became"
                 " SQLite-everywhere, and the database question is open again",
         "confidence": 0.8},
    ],
    "gut_reaction": "",
}


class _Caller:
    def __init__(self, payload):
        self.calls = 0
        self.payload = payload
        self.message_log = []

    def __call__(self, *, messages, model_override=None, **kwargs):
        self.calls += 1
        self.message_log.append(messages)
        return (
            "openai",
            model_override or "fake-cheap-model",
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=json.dumps(self.payload), role="assistant"
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(
                    prompt_tokens=200, completion_tokens=150, total_tokens=350
                ),
                model="fake-cheap-model",
            ),
        )


def _content_of(messages):
    """Flatten message contents (strings or text-block lists) to one str."""
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                else:
                    parts.append(str(block))
        else:
            parts.append(str(content))
    return "\n".join(parts)


def test_reflection_demo(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "hexis_appraisal" / "state.db"
    monkeypatch.setattr(store, "get_db_path", lambda: db_path)
    monkeypatch.setattr(
        config, "get_cfg",
        lambda force_reload=False: {
            "enabled": True, "confidence_threshold": 0.6,
            "deadline_seconds": 8.0, "history_chars": 4000,
            "model": None, "max_tokens": 700,
            "reflection_enabled": True, "reflect_every_n_turns": 5,
            "reflect_max_tokens": 700, "reflect_deadline_seconds": 8.0,
        },
    )

    state = {"caller": _Caller(APPRAISAL_SESSION_A)}

    def _delegating(**kwargs):
        return state["caller"](**kwargs)

    llm = plugin_llm.make_plugin_llm_for_test(
        plugin_id="hexis_appraisal",
        policy=plugin_llm._TrustPolicy(plugin_id="hexis_appraisal"),
        sync_caller=_delegating,
    )
    hexis_appraisal.register(
        types.SimpleNamespace(llm=llm, register_hook=lambda *a, **k: None)
    )
    hexis_appraisal._session_state.update(
        {"session_id": None, "last_msg_norm": None}
    )

    try:
        # ----- session A: declaration, then flat reversal --------------
        hexis_appraisal.on_session_start(session_id="session-a")

        hexis_appraisal.pre_llm_call(
            session_id="session-a", user_message=TOPIC_X_DECLARATION
        )
        hexis_appraisal.post_llm_call(
            session_id="session-a", turn_id="a1",
            user_message=TOPIC_X_DECLARATION,
            assistant_response="Noted — Postgres is the standard for"
                               " services going forward.",
        )
        hexis_appraisal.on_session_end(session_id="session-a", turn_id="a1")

        hexis_appraisal.pre_llm_call(
            session_id="session-a", user_message=TOPIC_X_REVERSAL
        )
        hexis_appraisal.post_llm_call(
            session_id="session-a", turn_id="a2",
            user_message=TOPIC_X_REVERSAL,
            assistant_response="That reverses yesterday's Postgres-final"
                               " decision — flagging the inconsistency.",
        )
        hexis_appraisal.on_session_end(session_id="session-a", turn_id="a2")

        appraisal_a_calls = state["caller"].calls
        assert appraisal_a_calls == 2  # debounce held: NO reflection in-session

        # ----- session boundary: reflection carries the contradiction --
        reflection_caller = _Caller(REFLECTION_PAYLOAD)
        state["caller"] = reflection_caller
        hexis_appraisal.on_session_start(session_id="session-b")
        assert reflection_caller.calls == 1  # the session-change trigger fired

        snap = store.read_snapshot()
        assert any(
            c["description"] == CONTRADICTION_DESCRIPTION
            for c in snap["contradictions"]
        )
        assert snap["trust_scores"]["stated infrastructure decisions"] == (
            pytest.approx(0.35)  # 0.5 baseline - 0.15 (bounded delta)
        )

        # ----- session B: the carried state reaches the appraisal ------
        appraisal_b_caller = _Caller(APPRAISAL_SESSION_B)
        state["caller"] = appraisal_b_caller
        out = hexis_appraisal.pre_llm_call(
            session_id="session-b",
            user_message="Which database should the new ingest service use?",
        )
    finally:
        appraisal._reset_executor_for_tests()
        hexis_appraisal._ctx = None

    # The appraisal LLM's INPUT contained the persisted contradiction:
    # session-A signal reached session-B appraisal input — the carrier works.
    context_b = _content_of(appraisal_b_caller.message_log[0])
    assert CONTRADICTION_DESCRIPTION in context_b
    assert "stated infrastructure decisions" in context_b

    assert isinstance(out, dict) and set(out) == {"context"}
    block = out["context"]
    assert block.startswith("[hexis appraisal]")
    assert "- contradiction (narrative):" in block
    assert "- trust note: low confidence on stated infrastructure" in block
    assert_no_directive_language(block)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        outcomes = [r[0] for r in con.execute(
            "SELECT outcome FROM telemetry ORDER BY id"
        )]
    finally:
        con.close()
    assert outcomes == [
        "reflect_skipped:no_turns",   # session-a start (fresh DB)
        "ok",                         # appraisal a1
        "reflect_skipped:debounce",   # session-a end, 1 unreflected turn
        "ok",                         # appraisal a2
        "reflect_skipped:debounce",   # session-a end, 2 unreflected turns
        "reflect_ok",                 # session-b start: the carrier pass
        "ok",                         # appraisal b1
    ]

    digest = _content_of(reflection_caller.message_log[0])
    print("\n=== HEXIS REFLECTION DEMO ===")
    print("--- reflection digest (session A span), excerpt ---")
    print(digest[digest.index("Conversation span:"):][:700])
    print("--- session B appraisal context, excerpt ---")
    print(context_b[context_b.index("Persisted appraisal state:"):][:700])
    print("--- session B rendered block ---")
    print(block)
    print("=== END DEMO ===")
