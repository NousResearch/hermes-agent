"""Dangling-promise guard (bounded continuation tracking): agent.promise_guard + its stop gate."""

import sqlite3
from types import SimpleNamespace

import pytest

from agent import promise_guard as pg
from agent.promise_guard import (
    DanglingPromiseConfig, WakeState, build_nudge, evaluate_final_reply, find_dangling_claims, rewrite_reply)

# Evidence: session 20261106_182844_ade685 (desktop, default). Each ended a turn; nothing ran.
MSG_513565 = (
    "**I checked the file: this is still partial, despite the “done” alert.** It contains **26 "
    "unverified records**, with several unchanged generic citations. #11 is not complete, and no memory "
    "cleanup is justified.\n\nI recorded that finding on the existing card and stopped the same-method retry "
    "cycle. Nothing was changed or deleted.\n\n**Doing now:** default is taking over `t_21905319` directly, "
    "starting with small, saved comparisons rather than sending another worker to repeat the survey.\n")
MSG_513581 = (
    "**This services batch is usable review evidence.** I verified it contains the exact 30 intended records, "
    "without duplicates, and its totals match: **10 same, 9 partial, 3 different, and 8 candidate gaps.**\n\n"
    "**Nothing was applied or deleted.** “Candidate gap” means absent from the inspected sources, not "
    "approved for adoption.\n\n**Doing now:** I’m consolidating these findings into the #11 review while "
    "taking over workflow card `t_21905319`; coding’s `t_02bb37b6` retains #10’s activation verification.\n")
MSG_513671 = (
    "**You mean the execution failures, not another status recap. Understood.** Repeatedly retrying workers "
    "and narrating unfinished work was my failure.\n\nI’ve now landed **three actual source comparisons "
    "myself** in the existing #11 report, rather than commissioning another survey. In parallel, coding’s "
    "`t_a11feac1` is recording the verified #10 changes and reconciling its remaining scope on GitHub.\n\n"
    "I will preserve your explicit #12 hold and the existing approval gates. I won’t ask you to repeat "
    "approvals or call partial reports finished.\n\n**Doing now:** I own #11’s remaining comparisons "
    "directly on `t_21905319`; coding owns #10’s resolution. Next updates will contain landed results or "
    "a concrete blocker, not another promise that work is “ready.”\n")
# Legitimate card dispatches from the same session.
MSG_513687 = (
    "**Approved: drop those three proposals and close #10**, preserving the record of the changes actually "
    "implemented.\n\n**Doing now:** coding’s existing `t_a11feac1` is queued to record your decision, close "
    "#10, and update #18. No further approval needed.\n")
MSG_513010 = ("`t_21905319` is queued for its remaining 25 comparisons. No action needed from you; I’ll report "
              "the result, not each ready-state transition.")
MSG_513728 = (
    "**I’ve now retrieved all 562 full preference records successfully, with zero retrieval errors.**\n\n"
    "**Doing now:** four parallel comparisons cover the entire captured set under my ownership of `t_21905319`.")

DONE = {"t_21905319": "done", "t_02bb37b6": "done", "t_a11feac1": "running"}


def _claims(text, **wake):
    return find_dangling_claims(text, WakeState(**wake))


# ---- the evidence fires, the dispatches do not -----------------------------------------------------

def test_513565_fires_on_taking_over_a_done_card():
    claims = _claims(MSG_513565, card_status=DONE)
    assert [c.cards for c in claims] == [("t_21905319",)]
    assert "t_21905319 is done" in claims[0].reason


def test_513581_fires_on_every_uncovered_clause():
    claims = _claims(MSG_513581, card_status=DONE)
    kinds = [(c.kind, c.cards) for c in claims]
    assert ("self", ()) in kinds                      # "I'm consolidating these findings"
    assert ("self", ("t_21905319",)) in kinds         # "taking over workflow card t_21905319"
    assert ("delegated", ("t_02bb37b6",)) in kinds    # coding's done card "retains" the work


def test_513671_fires_although_another_profiles_card_was_running():
    # t_a11feac1 (coding) was running and subscribed: it wakes the session, but it does not do #11.
    claims = _claims(MSG_513671, card_status=DONE, session_cards_active=True)
    assert len(claims) == 1 and claims[0].cards == ("t_21905319",)
    assert "own" in claims[0].clause and "directly" in claims[0].clause


def test_queued_card_dispatch_does_not_fire():
    assert not _claims(MSG_513687, card_status={"t_a11feac1": "ready"}, session_cards_active=True)


def test_report_back_with_a_queued_card_does_not_fire():
    assert not _claims(MSG_513010, card_status={"t_21905319": "ready"}, session_cards_active=True)


def test_report_back_with_nothing_to_wake_the_session_fires():
    assert _claims("I'll report the result when it is done.")


def test_own_live_delegation_covers_self_work():
    assert _claims(MSG_513728, card_status=DONE)
    assert not _claims(MSG_513728, card_status=DONE, live_delegation=True)
    assert not _claims("Doing now: consolidating the results myself.", live_process=True)


def test_undelivered_card_notification_counts_as_a_wake():
    text = "**Doing now:** coding’s `t_d3ac52f2` is checking #11’s acceptance requirements."
    assert _claims(text, card_status={"t_d3ac52f2": "done"})
    assert not _claims(text, card_status={"t_d3ac52f2": "done"}, pending_notice=frozenset({"t_d3ac52f2"}))


def test_card_not_found_on_any_board_is_not_accused():
    assert not _claims("**Doing now:** default is taking over `t_deadbeef` directly.", card_status={"t_deadbeef": None})


# ---- not promises -----------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Want me to fix it now?",
    "Should I take over the card?",
    "I'll fix it once you approve.",
    "I'll run the migration after your go-ahead.",
    "Reply and I'll finish those.",
    "Tell me which one and I'll start.",
    "I won't ask you to repeat approvals.",
    "I will not delete anything.",
    "I will preserve your explicit #12 hold.",
    "I checked the file and verified the totals.",
    "I've now landed three comparisons myself.",
    "Going forward I'll write cards without the PR gate.",
    "If the integration cannot handle them, I'll report that explicitly.",
    "I'll keep that in mind.",
    "Here's what I'll do, step by step, below.",
    "MacPorts does not automatically take over Homebrew packages.",
    "Second mistake: I said \"I'll check both cards when they finish.\"",
    "```\nI'm deploying now\n```",
    "> I'm consolidating these findings",
    "**Doing now:** nothing. Monitoring is not yet confirmed active.",
    "**Doing now:** the visibility card `t_66942efa` stays parked until the repair is deployed.",
    "Dear Mr. Smith,\n\nI'll work harder than any employee you've had.\n\nSincerely,\nAndrew",
    "",
])
def test_not_a_dangling_promise(text):
    assert not _claims(text)


@pytest.mark.parametrize("text", [
    "I'm working on this directly now.",
    "Doing now: consolidating the findings into the review.",
    "I'll finish the remaining comparisons myself.",
    "Working on it: running the full suite and opening the PR.",
    "Got it, and now I'm fixing what they were stuck on.",
])
def test_dangling_promise_with_nothing_running(text):
    assert _claims(text)


def test_report_kind_waits_on_any_wake_source():
    text = "When coding finishes, I'll check the skill counts on a few profiles myself."
    claims = _claims(text)
    assert claims and claims[0].kind == "report"
    assert not _claims(text, session_cards_active=True)


def test_profile_name_is_the_session_only_with_a_self_marker():
    worker_card = "Default is checking the backup ledger on `t_7e519eb0`."
    assert _claims(worker_card, card_status={"t_7e519eb0": "done"})[0].kind == "delegated"
    assert not _claims(worker_card, card_status={"t_7e519eb0": "running"})
    own = find_dangling_claims("homelab is taking over `t_7e519eb0` directly.", WakeState(card_status={"t_7e519eb0": "done"}),
                               profile="homelab")
    assert own and own[0].kind == "self"


# ---- nudge and rewrite text ---------------------------------------------------------------------------

def test_nudge_quotes_the_claim_and_offers_both_exits():
    note = build_nudge(_claims(MSG_513671, card_status=DONE))
    assert "I own 11’s remaining comparisons directly on t_21905319" in note or "t_21905319" in note
    assert "do the work now" in note and "you stopped" in note


def test_rewrite_drops_the_claim_and_states_the_stop():
    claims = _claims(MSG_513565, card_status=DONE)
    out = rewrite_reply(MSG_513565, claims)
    assert "Doing now" not in out and "taking over" not in out.split("**Stopped here.**")[0]
    assert out.startswith("**I checked the file")
    assert "**Stopped here.**" in out and "Send `continue`" in out
    assert not find_dangling_claims(out, WakeState())  # the rewrite is not itself a promise


# ---- evaluate_final_reply: config, cap, platforms ----------------------------------------------------

def _agent(platform="desktop", nudges=0, cfg=None, **extra):
    guard = SimpleNamespace(config=SimpleNamespace(dangling_promise=cfg or DanglingPromiseConfig()))
    return SimpleNamespace(platform=platform, _dangling_promise_nudges=nudges, _tool_guardrails=guard,
                           session_id="s1", _interrupt_requested=False, **extra)


def _wake(*_):
    return WakeState(card_status=DONE)


def test_first_stop_nudges_then_rewrites_when_attended(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    first = evaluate_final_reply(_agent(), MSG_513671, wake_fn=_wake)
    assert first.action == "nudge" and "dangling promise" in first.nudge
    second = evaluate_final_reply(_agent(nudges=1), MSG_513671, wake_fn=_wake)
    assert second.action == "rewrite" and "**Stopped here.**" in second.rewritten


@pytest.mark.parametrize("platform", ["telegram", "discord", "desktop", "cli", "tui"])
def test_rewrite_on_platforms_a_person_reads(platform, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert evaluate_final_reply(_agent(platform, nudges=1), MSG_513565, wake_fn=_wake).action == "rewrite"


def test_unattended_platform_nudges_but_never_rewrites(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert evaluate_final_reply(_agent("cron"), MSG_513565, wake_fn=_wake).action == "nudge"
    assert evaluate_final_reply(_agent("cron", nudges=1), MSG_513565, wake_fn=_wake) is None


def test_disabled_or_kanban_worker_or_interrupted_does_nothing(monkeypatch):
    off = DanglingPromiseConfig.from_mapping({"enabled": False})
    assert evaluate_final_reply(_agent(cfg=off), MSG_513565, wake_fn=_wake) is None
    interrupted = _agent()
    interrupted._interrupt_requested = True
    assert evaluate_final_reply(interrupted, MSG_513565, wake_fn=_wake) is None
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    assert evaluate_final_reply(_agent(), MSG_513565, wake_fn=_wake) is None


def test_rewrite_can_be_turned_off_and_nudges_raised(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    cfg = DanglingPromiseConfig.from_mapping({"max_nudges": 2, "rewrite_when_attended": False})
    assert evaluate_final_reply(_agent(cfg=cfg, nudges=1), MSG_513565, wake_fn=_wake).action == "nudge"
    assert evaluate_final_reply(_agent(cfg=cfg, nudges=2), MSG_513565, wake_fn=_wake) is None


def test_wake_lookup_failure_lets_the_reply_through(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    def boom(*_):
        raise sqlite3.OperationalError("locked")
    assert evaluate_final_reply(_agent(), MSG_513565, wake_fn=boom) is None


def test_no_candidate_claim_skips_the_lookup(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    called = []
    assert evaluate_final_reply(_agent(), "Done: #10 is closed.", wake_fn=lambda *a: called.append(a)) is None
    assert not called


def test_config_from_mapping_and_guardrail_config():
    from agent.tool_guardrails import ToolCallGuardrailConfig
    assert DanglingPromiseConfig.from_mapping(None) == DanglingPromiseConfig()
    assert DanglingPromiseConfig.from_mapping({"max_nudges": "x", "enabled": "no"}) == DanglingPromiseConfig()
    cfg = ToolCallGuardrailConfig.from_mapping({"dangling_promise": {"max_nudges": 0}}, platform="desktop")
    assert cfg.dangling_promise.max_nudges == 0 and cfg.dangling_promise.enabled


# ---- wake-state lookups against a real kanban schema ----------------------------------------------------

def _kanban(tmp_path):
    db = tmp_path / "kanban.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT NOT NULL);
        CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, kind TEXT, created_at INTEGER);
        CREATE TABLE kanban_notify_subs (task_id TEXT, platform TEXT, chat_id TEXT, last_event_id INTEGER DEFAULT 0);
        INSERT INTO tasks VALUES ('t_aaaaaaaa', 'done'), ('t_bbbbbbbb', 'running'), ('t_cccccccc', 'done');
        INSERT INTO task_events (task_id, kind, created_at) VALUES ('t_aaaaaaaa', 'completed', 1);   -- id 1
        INSERT INTO task_events (task_id, kind, created_at) VALUES ('t_cccccccc', 'completed', 2);   -- id 2
        INSERT INTO kanban_notify_subs VALUES ('t_aaaaaaaa', 'tui', 's1', 1);  -- delivered
        INSERT INTO kanban_notify_subs VALUES ('t_cccccccc', 'tui', 's1', 0);  -- not yet delivered
        INSERT INTO kanban_notify_subs VALUES ('t_bbbbbbbb', 'tui', 's2', 0);
    """)
    conn.commit()
    other = tmp_path / "kanban" / "boards" / "homelab"
    other.mkdir(parents=True)
    oc = sqlite3.connect(other / "kanban.db")
    oc.executescript("CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT); INSERT INTO tasks VALUES ('t_dddddddd', 'ready');")
    oc.commit()
    oc.close()
    return lambda: sqlite3.connect(db)


def test_kanban_wake_reads_status_subscriptions_and_pending_notices(tmp_path):
    connect = _kanban(tmp_path)
    status, active, pending = pg._kanban_wake("s1", ["t_aaaaaaaa", "t_bbbbbbbb", "t_dddddddd", "t_eeeeeeee"],
                                              connect=connect)
    assert status == {"t_aaaaaaaa": "done", "t_bbbbbbbb": "running", "t_dddddddd": "ready", "t_eeeeeeee": None}
    assert active is False                      # s1's subscribed cards are all done
    assert pending == frozenset({"t_cccccccc"})  # its completion has not reached s1 yet
    _, active2, _ = pg._kanban_wake("s2", [], connect=connect)
    assert active2 is True


# ---- the stop gate --------------------------------------------------------------------------------------

class _GateAgent(SimpleNamespace):
    def _emit_interim_assistant_message(self, msg):
        self.interims.append(msg)

    def _flush_messages_to_session_db(self, messages, history):
        pass

    def _interim_content_was_streamed(self, text):
        return False


@pytest.fixture
def gate(monkeypatch):
    from agent import turn_stop_gates as tsg
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    for name in ("_late_lsp_verdicts", "_verify_on_stop_nudge", "_kanban_stop_nudge"):
        monkeypatch.setattr(tsg, name, lambda *a, **k: None)
    monkeypatch.setattr(tsg, "_pre_verify_nudge", lambda *a, **k: None)
    monkeypatch.setattr(pg, "collect_wake_state", _wake)
    monkeypatch.setattr(pg, "_profile_for", lambda agent: "default")
    return tsg


def _run_gate(tsg, agent, text):
    final_msg = {"role": "assistant", "content": text}
    messages = [{"role": "user", "content": "done?"}]
    verdict = tsg.apply_stop_gates(agent, final_msg, final_response=text, messages=messages, conversation_history=[],
                                   pending_verification_response=None, pending_verification_response_previewed=False)
    return verdict, final_msg, messages


def test_gate_continues_the_turn_once_with_a_flagged_nudge(gate):
    agent = _GateAgent(**vars(_agent()), interims=[])
    verdict, final_msg, messages = _run_gate(gate, agent, MSG_513671)
    assert verdict.continue_turn and verdict.final_response is None
    assert verdict.pending_verification_response == MSG_513671
    assert messages[-2] is final_msg and final_msg["finish_reason"] == "verification_required"
    assert messages[-1]["role"] == "user" and messages[-1]["_dangling_promise_synthetic"]
    assert agent._dangling_promise_nudges == 1 and agent.interims == [final_msg]


def test_gate_rewrites_after_the_cap(gate):
    agent = _GateAgent(**vars(_agent(nudges=1)), interims=[])
    verdict, final_msg, messages = _run_gate(gate, agent, MSG_513671)
    assert not verdict.continue_turn
    assert "**Stopped here.**" in verdict.final_response and final_msg["content"] == verdict.final_response
    assert len(messages) == 1


def test_gate_passes_a_legitimate_dispatch(gate, monkeypatch):
    monkeypatch.setattr(pg, "collect_wake_state", lambda *a: WakeState(card_status={"t_a11feac1": "ready"},
                                                                       session_cards_active=True))
    agent = _GateAgent(**vars(_agent()), interims=[])
    verdict, final_msg, _ = _run_gate(gate, agent, MSG_513687)
    assert not verdict.continue_turn and verdict.final_response == MSG_513687


def test_nudge_row_is_scaffolding_everywhere():
    from agent.conversation_compression import _SYNTHETIC_USER_FLAGS
    from agent.session_persistence import _EPHEMERAL_SCAFFOLDING_FLAGS
    from agent.turn_context import _PER_TURN_RESET_STATE
    from agent.turn_finalizer import _VERIFICATION_CONTINUATION_FLAGS
    for flags in (_SYNTHETIC_USER_FLAGS, _EPHEMERAL_SCAFFOLDING_FLAGS, _VERIFICATION_CONTINUATION_FLAGS):
        assert "_dangling_promise_synthetic" in flags
    assert ("_dangling_promise_nudges", 0) in _PER_TURN_RESET_STATE
