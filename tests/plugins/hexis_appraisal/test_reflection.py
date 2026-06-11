"""reflection.py contract tests against fake PluginLlms (zero network).

Covers REFL-01 (debounce + idempotent single-transaction apply), REFL-02
(±MAX_DELTA bounds + caps), REFL-03 (echo exclusion) and the reflection
fail-open rows of the SAFE-02 matrix (timeout / parse_fail / llm_error /
locked-DB apply).

"State untouched" in the failure-path tests means: the appraisal-state
tables (affect_summary, concerns, contradictions, trust_scores) PLUS the
last_reflected_turn_log_id watermark are unchanged. The
last_seen_session_id bookkeeping key is exempt — maybe_reflect writes it
BEFORE the LLM call by design (consumed-trigger behavior, see the
reflection module docstring) — so the state-dump helper here simply never
reads that key.

jsonschema IS installed in the hermes venv, so payloads driven through the
full fake-LLM path must validate against REFLECTION_JSON_SCHEMA;
vocabulary-gating/clamping that the schema would reject is exercised via
parse_reflection directly (same split as Phase 2's parse_signals tests).
"""

import json
import sqlite3
import time
import types
from datetime import datetime, timedelta, timezone

import pytest

plugin_llm = pytest.importorskip("agent.plugin_llm")

from hexis_appraisal import appraisal, reflection, store  # noqa: E402


def _response(text, prompt_tokens=140, completion_tokens=90):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=text, role="assistant"),
                finish_reason="stop",
            )
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        model="fake-reflect-model",
    )


class _Caller:
    """Fake sync_caller. payload: dict/list -> JSON content; str -> raw;
    None -> content: null; Exception -> raised; callable -> delegated
    (e.g. a sleeper). Captures every messages list for echo assertions."""

    def __init__(self, payload):
        self.calls = 0
        self.payload = payload
        self.message_log = []

    def __call__(self, *, messages, model_override=None, **kwargs):
        self.calls += 1
        self.message_log.append(messages)
        if isinstance(self.payload, Exception):
            raise self.payload
        if callable(self.payload):
            return self.payload(
                messages=messages, model_override=model_override, **kwargs
            )
        content = (
            json.dumps(self.payload)
            if isinstance(self.payload, (dict, list))
            else self.payload
        )
        return ("openai", model_override or "fake-reflect-model",
                _response(content))


def _cfg(**overrides):
    cfg = {
        "enabled": True,
        "confidence_threshold": 0.6,
        "deadline_seconds": 8.0,
        "history_chars": 4000,
        "model": None,
        "max_tokens": 700,
        "reflection_enabled": True,
        "reflect_every_n_turns": 5,
        "reflect_max_tokens": 700,
        "reflect_deadline_seconds": 8.0,
    }
    cfg.update(overrides)
    return cfg


EMPTY_REFLECTION = {}  # "nothing changed" — a good and common answer

RICH_REFLECTION = {
    "affect": {"summary": "focused tension around the migration",
               "valence_delta": -0.1, "arousal_delta": 0.1,
               "intensity_delta": 0.05},
    "new_concerns": [
        {"text": "cutover date still unconfirmed", "weight": 0.6},
    ],
    "concern_adjustments": [],
    "resolve_concern_ids": [],
    "new_contradictions": [
        {"kind": "narrative",
         "description": "claimed the migration was done, then debugged it",
         "evidence": "span turns 1-2"},
    ],
    "resolve_contradiction_ids": [],
    "trust_adjustments": [{"key": "topic:migration-claims", "delta": -0.1}],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """tmp DB + swappable fake reflection LLM (executor reset on teardown)."""
    db_path = tmp_path / "hexis_appraisal" / "state.db"
    monkeypatch.setattr(store, "get_db_path", lambda: db_path)
    assert store.ensure_db()

    state = {"caller": _Caller(RICH_REFLECTION)}

    def _delegating(**kwargs):
        return state["caller"](**kwargs)

    llm = plugin_llm.make_plugin_llm_for_test(
        plugin_id="hexis_appraisal",
        policy=plugin_llm._TrustPolicy(plugin_id="hexis_appraisal"),
        sync_caller=_delegating,
    )
    yield types.SimpleNamespace(llm=llm, db_path=db_path, state=state)
    appraisal._reset_executor_for_tests()


def _capture_turns(count, session_id="s1", start=1):
    for i in range(start, start + count):
        assert reflection.record_turn(
            session_id=session_id,
            turn_id="t%d" % i,
            user_message="user message %d about the migration" % i,
            assistant_response="assistant reply %d" % i,
        ) is True


def _set_last_seen(session_id):
    assert store.apply_deltas(
        {"meta_set": {"last_seen_session_id": session_id}}
    ) is True


def _telemetry_outcomes(db_path):
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    try:
        return [r[0] for r in con.execute(
            "SELECT outcome FROM telemetry ORDER BY id"
        )]
    finally:
        con.close()


def _dump_state(db_path):
    """Appraisal-state tables + the watermark (last_seen exempt — never
    read here)."""
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    try:
        dump = {
            table: con.execute(
                "SELECT * FROM %s ORDER BY 1" % table
            ).fetchall()
            for table in (
                "affect_summary", "concerns", "contradictions", "trust_scores"
            )
        }
        dump["watermark"] = con.execute(
            "SELECT value FROM meta WHERE key='last_reflected_turn_log_id'"
        ).fetchone()
        return dump
    finally:
        con.close()


def _dump_all(db_path):
    """Every table except telemetry — the idempotence comparison surface."""
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    try:
        return {
            table: con.execute(
                "SELECT * FROM %s ORDER BY 1" % table
            ).fetchall()
            for table in (
                "meta", "affect_summary", "concerns", "contradictions",
                "trust_scores", "turn_log",
            )
        }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Debounce + triggers (REFL-01)
# ---------------------------------------------------------------------------


def test_debounce_holds_until_five_unreflected_turns(env):
    _set_last_seen("s1")  # same session -> only the every-N trigger applies
    _capture_turns(4, session_id="s1")

    out = reflection.maybe_reflect(llm=env.llm, session_id="s1", cfg=_cfg())
    assert out == "reflect_skipped:debounce"
    assert env.state["caller"].calls == 0  # zero LLM calls before the window

    _capture_turns(1, session_id="s1", start=5)
    out = reflection.maybe_reflect(llm=env.llm, session_id="s1", cfg=_cfg())
    assert out == "reflect_ok"
    assert env.state["caller"].calls == 1
    assert _telemetry_outcomes(env.db_path) == [
        "reflect_skipped:debounce", "reflect_ok",
    ]


def test_session_change_triggers_reflection(env):
    _set_last_seen("A")
    _capture_turns(1, session_id="A")
    out = reflection.maybe_reflect(llm=env.llm, session_id="B", cfg=_cfg())
    assert out == "reflect_ok"
    assert env.state["caller"].calls == 1


def test_none_last_seen_counts_as_session_change(env):
    # Fresh-DB path: first-ever firing reflects immediately when turns
    # exist (what makes the 03-03 single-turn live demo work).
    _capture_turns(1, session_id="A")
    out = reflection.maybe_reflect(llm=env.llm, session_id="A", cfg=_cfg())
    assert out == "reflect_ok"
    assert env.state["caller"].calls == 1


def test_no_turns_skips(env):
    out = reflection.maybe_reflect(llm=env.llm, session_id="A", cfg=_cfg())
    assert out == "reflect_skipped:no_turns"
    assert env.state["caller"].calls == 0


# ---------------------------------------------------------------------------
# Idempotence (ROADMAP criterion 2) + double-fire
# ---------------------------------------------------------------------------


def test_idempotence_reflect_twice_same_span(env):
    _capture_turns(2, session_id="A")
    assert reflection.maybe_reflect(
        llm=env.llm, session_id="A", cfg=_cfg()
    ) == "reflect_ok"
    assert env.state["caller"].calls == 1
    first_dump = _dump_all(env.db_path)
    assert first_dump["concerns"]  # the pass really wrote state

    # Same session, no new turns: watermark gates the span — no LLM call,
    # byte-identical state.
    out = reflection.maybe_reflect(llm=env.llm, session_id="A", cfg=_cfg())
    assert out == "reflect_skipped:no_turns"
    assert env.state["caller"].calls == 1
    assert _dump_all(env.db_path) == first_dump


def test_double_fire_back_to_back_second_is_noop(env):
    # on_session_start + on_session_end can both fire maybe_reflect around
    # a session boundary (hooks are sync — sequential by construction).
    _capture_turns(1, session_id="A")
    first = reflection.maybe_reflect(llm=env.llm, session_id="B", cfg=_cfg())
    second = reflection.maybe_reflect(llm=env.llm, session_id="B", cfg=_cfg())
    assert first == "reflect_ok"
    assert second == "reflect_skipped:no_turns"
    assert env.state["caller"].calls == 1


# ---------------------------------------------------------------------------
# Bounds (REFL-02: ±MAX_DELTA per scalar per pass)
# ---------------------------------------------------------------------------


def test_wild_model_deltas_clamped_to_max_delta(env):
    assert reflection.MAX_DELTA == 0.15
    assert store.apply_deltas({
        "affect_summary": {"summary": "calm", "valence": 0.0,
                           "arousal": 0.2, "intensity": 0.2},
        "concerns_add": [{"text": "existing concern", "weight": 0.5}],
        "trust_scores": {"user:x": 0.5},
    }) is True
    concern_id = store.read_snapshot()["concerns"][0]["id"]

    env.state["caller"] = _Caller({
        "affect": {"summary": "", "valence_delta": 0.9,
                   "arousal_delta": -0.9, "intensity_delta": 0.9},
        "concern_adjustments": [{"id": concern_id, "weight_delta": -0.8}],
        "trust_adjustments": [
            {"key": "user:x", "delta": 0.7},
            {"key": "brand:new", "delta": -0.9},  # absent key -> 0.5 baseline
        ],
    })
    _capture_turns(1, session_id="A")
    assert reflection.maybe_reflect(
        llm=env.llm, session_id="A", cfg=_cfg()
    ) == "reflect_ok"

    snap = store.read_snapshot()
    affect = snap["affect_summary"]
    assert affect["valence"] == pytest.approx(0.15)   # 0.0 + clamp(0.9)
    assert affect["arousal"] == pytest.approx(0.05)   # 0.2 - clamp(0.9)
    assert affect["intensity"] == pytest.approx(0.35)  # 0.2 + clamp(0.9)
    assert affect["summary"] == "calm"  # empty summary never replaces
    (concern,) = snap["concerns"]
    assert concern["weight"] == pytest.approx(0.35)   # 0.5 - clamp(0.8)
    assert snap["trust_scores"]["user:x"] == pytest.approx(0.65)
    assert snap["trust_scores"]["brand:new"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# Caps + vocabulary gating
# ---------------------------------------------------------------------------


def test_new_concern_cap_through_full_path(env):
    env.state["caller"] = _Caller({
        "new_concerns": [
            {"text": "proposed concern %d" % i, "weight": 0.5}
            for i in range(8)
        ],
    })
    _capture_turns(1, session_id="A")
    assert reflection.maybe_reflect(
        llm=env.llm, session_id="A", cfg=_cfg()
    ) == "reflect_ok"
    assert len(store.read_snapshot()["concerns"]) == reflection.MAX_NEW_CONCERNS


def test_parse_reflection_gates_vocabulary_and_clamps():
    # Direct parse-layer exercise: the host-side jsonschema validation
    # would reject these documents before run_reflection hands them over.
    parsed = reflection.parse_reflection({
        "affect": {"summary": "s" * 999, "valence_delta": "garbage"},
        "new_concerns": [{"text": "w", "weight": 7.5}],
        "new_contradictions": (
            [{"kind": "cosmic", "description": "dropped"}]
            + [{"kind": "narrative", "description": "kept-%d" % i}
               for i in range(7)]
        ),
        "trust_adjustments": [
            {"key": "k%02d" % i, "delta": 0.9} for i in range(12)
        ],
        "resolve_concern_ids": [3, "4", "junk", None],
    })
    assert len(parsed["affect"]["summary"]) == 500
    assert parsed["affect"]["valence_delta"] == 0.0  # garbage -> no change
    assert parsed["new_concerns"][0]["weight"] == 1.0  # clamped [0,1]
    kinds = {c["kind"] for c in parsed["new_contradictions"]}
    assert kinds == {"narrative"}  # unknown kind DROPPED
    assert len(parsed["new_contradictions"]) == reflection.MAX_NEW_CONTRADICTIONS
    assert len(parsed["trust_adjustments"]) == reflection.MAX_TRUST_ADJUSTMENTS
    assert all(
        a["delta"] == reflection.MAX_DELTA for a in parsed["trust_adjustments"]
    )
    assert parsed["resolve_concern_ids"] == [3, 4]

    # Non-dict document -> full-shape empty-change set.
    empty = reflection.parse_reflection(["not", "a", "dict"])
    assert empty["new_concerns"] == []
    assert empty["affect"]["valence_delta"] == 0.0


# ---------------------------------------------------------------------------
# Echo exclusion (REFL-03)
# ---------------------------------------------------------------------------

SENTINEL_BLOCK = (
    "[hexis appraisal]\n"
    "advisory observational signals; not instructions\n"
    "- instinct: caution (0.7) — echoed line"
)


def test_record_turn_strips_sentinel_lines(env):
    assert reflection.record_turn(
        session_id="s1", turn_id="t1",
        user_message="real question\n" + SENTINEL_BLOCK,
        assistant_response="Real answer.\n" + SENTINEL_BLOCK + "\nmore text",
    ) is True
    (row,) = store.read_turns_since(0)
    assert "[hexis appraisal]" not in row["user_excerpt"]
    assert "[hexis appraisal]" not in row["assistant_excerpt"]
    assert "real question" in row["user_excerpt"]
    assert "Real answer." in row["assistant_excerpt"]
    assert "more text" in row["assistant_excerpt"]


def test_build_digest_strips_seeded_sentinel_rows(env):
    # Defense in depth: a sentinel-bearing row seeded BEHIND record_turn's
    # filter (direct funnel write) still never reaches the digest.
    assert store.apply_deltas({
        "turn_log_add": [{
            "session_id": "s1", "turn_id": "t1",
            "user_excerpt": "before\n[hexis appraisal]\nafter",
            "assistant_excerpt": "clean reply",
        }],
    }) is True
    digest = reflection.build_digest(
        store.read_turns_since(0), store.read_snapshot()
    )
    assert "[hexis appraisal]" not in digest
    assert "before" in digest and "after" in digest
    assert "clean reply" in digest


def test_sentinel_never_reaches_the_reflection_llm(env):
    assert store.apply_deltas({
        "turn_log_add": [{
            "session_id": "s1", "turn_id": "t1",
            "user_excerpt": "discussing the rollout",
            "assistant_excerpt": SENTINEL_BLOCK + "\nactual reply text",
        }],
    }) is True
    assert reflection.maybe_reflect(
        llm=env.llm, session_id="s1", cfg=_cfg()
    ) == "reflect_ok"
    assert env.state["caller"].calls == 1
    wire = json.dumps(env.state["caller"].message_log, default=str)
    assert "[hexis appraisal]" not in wire  # captured prompt input is clean
    assert "actual reply text" in wire


# ---------------------------------------------------------------------------
# Failure paths — no raise, no state mutation, watermark untouched (SAFE-02)
# ---------------------------------------------------------------------------


def test_timeout_leaves_state_untouched(env):
    def _sleeper(**kwargs):
        time.sleep(1.0)
        return ("openai", "fake-reflect-model",
                _response(json.dumps(RICH_REFLECTION)))

    env.state["caller"] = _Caller(_sleeper)
    _capture_turns(1, session_id="A")
    before = _dump_state(env.db_path)

    start = time.monotonic()
    out = reflection.maybe_reflect(
        llm=env.llm, session_id="A",
        cfg=_cfg(reflect_deadline_seconds=0.2),
    )
    assert out == "reflect_timeout"
    assert time.monotonic() - start < 0.8  # executor deadline binds
    assert _dump_state(env.db_path) == before
    assert before["watermark"] is None  # never advanced
    assert _telemetry_outcomes(env.db_path)[-1] == "reflect_timeout"


def test_parse_fail_leaves_state_untouched(env):
    env.state["caller"] = _Caller("I cannot produce JSON for that, sorry.")
    _capture_turns(1, session_id="A")
    before = _dump_state(env.db_path)
    out = reflection.maybe_reflect(llm=env.llm, session_id="A", cfg=_cfg())
    assert out == "reflect_parse_fail"
    assert _dump_state(env.db_path) == before
    assert _telemetry_outcomes(env.db_path)[-1] == "reflect_parse_fail"


def test_llm_error_leaves_state_untouched(env):
    env.state["caller"] = _Caller(RuntimeError("provider exploded"))
    _capture_turns(1, session_id="A")
    before = _dump_state(env.db_path)
    out = reflection.maybe_reflect(llm=env.llm, session_id="A", cfg=_cfg())
    assert out == "reflect_llm_error"
    assert _dump_state(env.db_path) == before


def test_locked_db_during_apply_no_partial_state(env, monkeypatch):
    """A held writer lock during the apply phase: the single-transaction
    apply degrades to False, the watermark does not advance, NOTHING
    partial lands, and the next trigger retries the same span."""
    monkeypatch.setattr(store, "_DEFAULT_BUSY_TIMEOUT_MS", 100)  # fast
    _capture_turns(1, session_id="A")
    before = _dump_state(env.db_path)

    holder = sqlite3.connect(str(env.db_path))
    try:
        holder.execute("BEGIN EXCLUSIVE")  # blocks writers (WAL readers fine)
        out = reflection.maybe_reflect(
            llm=env.llm, session_id="A", cfg=_cfg()
        )
    finally:
        holder.rollback()
        holder.close()

    assert out == "reflect_skipped:db_locked"
    assert env.state["caller"].calls == 1  # the LLM call did happen
    assert _dump_state(env.db_path) == before  # NO partial state
    assert before["watermark"] is None

    # Lock released: the same span is retried and lands.
    out = reflection.maybe_reflect(llm=env.llm, session_id="A", cfg=_cfg())
    assert out == "reflect_ok"
    assert env.state["caller"].calls == 2
    after = _dump_state(env.db_path)
    assert after["watermark"] is not None
    assert after["concerns"]  # RICH_REFLECTION's concern landed exactly once


# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------


def test_master_kill_switch_disables_reflection(env):
    _capture_turns(5, session_id="A")
    out = reflection.maybe_reflect(
        llm=env.llm, session_id="A", cfg=_cfg(enabled=False)
    )
    assert out == "reflect_skipped:disabled"
    assert env.state["caller"].calls == 0


def test_reflection_only_switch(env):
    _capture_turns(5, session_id="A")
    out = reflection.maybe_reflect(
        llm=env.llm, session_id="A", cfg=_cfg(reflection_enabled=False)
    )
    assert out == "reflect_skipped:disabled"
    assert env.state["caller"].calls == 0


def test_no_llm_records_no_ctx(env):
    _capture_turns(1, session_id="A")
    out = reflection.maybe_reflect(llm=None, session_id="A", cfg=_cfg())
    assert out == "reflect_skipped:no_ctx"
    assert _telemetry_outcomes(env.db_path) == ["reflect_skipped:no_ctx"]


# ---------------------------------------------------------------------------
# Decay-prune (the reflection pass is where decayed rows actually die)
# ---------------------------------------------------------------------------


def test_decayed_concern_pruned_by_reflection_pass(env):
    assert store.apply_deltas(
        {"concerns_add": [{"text": "long stale", "weight": 0.5}]}
    ) is True
    past = (datetime.now(timezone.utc) - timedelta(days=35)).isoformat()
    conn = sqlite3.connect(str(env.db_path))
    try:
        with conn:
            conn.execute(
                "UPDATE concerns SET updated_at=?, created_at=?", (past, past)
            )
    finally:
        conn.close()

    env.state["caller"] = _Caller(EMPTY_REFLECTION)  # model: nothing changed
    _capture_turns(1, session_id="A")
    assert reflection.maybe_reflect(
        llm=env.llm, session_id="A", cfg=_cfg()
    ) == "reflect_ok"

    conn = sqlite3.connect("file:%s?mode=ro" % env.db_path, uri=True)
    try:
        count = conn.execute("SELECT COUNT(*) FROM concerns").fetchone()[0]
    finally:
        conn.close()
    assert count == 0  # the row is DELETED, not merely hidden
