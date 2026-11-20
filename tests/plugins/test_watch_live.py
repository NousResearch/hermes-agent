"""Tests for the live loop: input track privacy and decider restraint.

Two contracts here, both of which fail silently in production if broken:

* the input track is a keylogger unless its four rules hold, and a leak is
  invisible to the user;
* the decider's value is restraint, and a nagging loop still "works".

Pure functions only — no OS hooks, no network. The LLM is injected.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(module_name: str):
    path = REPO_ROOT / "plugins" / "watch" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_watch_{module_name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inp():
    return _load("inputs")


@pytest.fixture(scope="module")
def dec():
    return _load("decider")


def _events(inp, rows):
    return [inp.InputEvent(at=at, symbol=sym, app=app) for at, sym, app in rows]


# ══ Input track: privacy ══════════════════════════════════════════════════

def test_default_allowlist_contains_no_letters(inp):
    """The core guarantee: a mis-started capture cannot reconstruct typing.

    If a single letter is on the default allowlist, a recording that ran by
    accident becomes a partial transcript of whatever the user wrote.
    """
    letters = set("abcdefghijklmnopqrstuvwxyz")
    for symbol in inp.GAME_KEYS:
        assert symbol not in letters, f"{symbol!r} is a bare letter on the default allowlist"


def test_letters_are_rejected_even_with_modifiers(inp):
    """`shift+q` must fail for the same reason `q` does — the base key decides."""
    assert not inp.is_allowed("q")
    assert not inp.is_allowed("shift+q")
    assert not inp.is_allowed("ctrl+alt+e")


def test_action_bar_and_transport_keys_are_allowed(inp):
    for symbol in ("1", "5", "0", "-", "f1", "f12", "shift+3", "ctrl+2", "m1", "wheelup", "space"):
        assert inp.is_allowed(symbol), f"{symbol!r} should be recordable"


def test_movement_keys_require_explicit_opt_in(inp):
    """WASD is four letters; recording them is opt-in, never the default."""
    assert not inp.is_allowed("w")
    assert inp.is_allowed("w", allow_movement=True)


def test_events_outside_the_watched_app_are_dropped(inp):
    """Focus gating: alt-tab to a password manager and recording stops itself."""
    events = _events(inp, [(0.0, "1", "WoW"), (1.0, "2", "1Password"), (2.0, "3", "WoW")])
    kept = inp.filter_events(events, watched_app="WoW")
    assert [e.symbol for e in kept] == ["1", "3"]


def test_text_mode_suppresses_everything_until_closed(inp):
    """The chat box is where private text lives, and it opens with a key we see."""
    events = _events(
        inp,
        [
            (0.0, "1", "WoW"),
            (1.0, "enter", "WoW"),   # opens chat
            (1.5, "5", "WoW"),       # would be a keystroke inside the message
            (2.0, "f4", "WoW"),
            (3.0, "enter", "WoW"),   # closes chat
            (4.0, "2", "WoW"),
        ],
    )
    kept = inp.filter_events(events, watched_app="WoW")
    assert [e.symbol for e in kept] == ["1", "2"]


def test_slash_opens_text_mode_too(inp):
    """`/` starts a command in most games — same exposure as chat."""
    events = _events(inp, [(0.0, "/", "WoW"), (1.0, "3", "WoW"), (2.0, "esc", "WoW"), (3.0, "4", "WoW")])
    assert [e.symbol for e in inp.filter_events(events, watched_app="WoW")] == ["4"]


def test_leaving_the_app_clears_text_mode(inp):
    """Otherwise an unclosed chat mutes the rest of the session permanently."""
    events = _events(
        inp,
        [(0.0, "enter", "WoW"), (1.0, "5", "Chrome"), (2.0, "1", "WoW")],
    )
    assert [e.symbol for e in inp.filter_events(events, watched_app="WoW")] == ["1"]


def test_filtering_never_returns_rejected_events(inp):
    """There is no 'dropped keys' channel — keeping one would defeat the point."""
    events = _events(inp, [(0.0, "s", "WoW"), (1.0, "e", "WoW"), (2.0, "c", "WoW")])
    assert inp.filter_events(events, watched_app="WoW") == []


def test_movement_is_summarized_as_a_count_not_a_direction(inp):
    """Opt-in movement informs pacing without tracing where the user went."""
    events = _events(inp, [(0.0, "w", "WoW"), (0.5, "a", "WoW"), (1.0, "1", "WoW")])
    summary = inp.summarize_inputs(events, duration_s=60.0)
    assert summary["movement_events"] == 2
    assert summary["actions"] == 1
    assert "w" not in str(summary["top"])


# ══ Input track: resolution and encoding ══════════════════════════════════

def test_gaps_expose_timing_the_frames_cannot(inp):
    """The reason this track exists: 1 fps cannot see a 1.5s cooldown."""
    events = _events(inp, [(0.0, "1", ""), (1.5, "2", ""), (3.0, "3", "")])
    assert inp.gaps(events) == [1.5, 1.5]


def test_encoding_is_relative_to_the_window(inp):
    """Absolute timestamps in a 10s window are noise the model has to subtract."""
    events = _events(inp, [(1841.2, "1", ""), (1842.7, "2", "")])
    encoded = inp.encode_events(events, since=1841.2)
    assert "1@0.0" in encoded
    assert "2@1.5" in encoded
    assert "1841" not in encoded


def test_repeated_presses_collapse(inp):
    """Nine spams of one ability is one fact, not nine rows crowding the window."""
    events = _events(inp, [(float(i) * 0.2, "1", "") for i in range(9)])
    assert inp.encode_events(events) == "1@0.0x9"


def test_distinct_symbols_do_not_collapse(inp):
    events = _events(inp, [(0.0, "1", ""), (0.2, "1", ""), (0.4, "2", "")])
    encoded = inp.encode_events(events)
    assert encoded.count("@") == 2
    assert "1@0.0x2" in encoded and "2@0.4" in encoded


def test_modifier_order_is_canonical(inp):
    """ctrl+shift+3 and shift+ctrl+3 are one keybind, not two abilities."""
    assert inp.compose("3", ["shift", "ctrl"]) == inp.compose("3", ["ctrl", "shift"])


def test_apm_and_idle_come_from_actions_only(inp):
    events = _events(inp, [(0.0, "1", ""), (10.0, "2", ""), (11.0, "w", "")])
    summary = inp.summarize_inputs(events, duration_s=60.0)
    assert summary["apm"] == 2.0
    assert summary["longest_idle"] == 10.0


def test_input_block_is_scaled_by_clip_speed(inp):
    """Same rule as the window timeline: the model only has clip time to check."""
    events = _events(inp, [(20.0, "1", "")])
    assert "@10.0" in inp.render_input_block(events, duration_s=40.0, speed=2.0)


def test_empty_input_track_renders_nothing(inp):
    assert inp.render_input_block([]) == ""


def test_activity_since_counts_presses_after_a_mark(inp):
    events = _events(inp, [(1.0, "1", ""), (5.0, "2", ""), (9.0, "3", "")])
    assert inp.activity_since(events, 4.0) == 2


def test_window_selects_the_rolling_span(inp):
    events = _events(inp, [(0.0, "1", ""), (55.0, "2", ""), (59.0, "3", "")])
    assert len(inp.window(events, now=60.0, span=10.0)) == 2


# ══ Novelty: change, not activity ═════════════════════════════════════════

def test_steady_play_scores_low_novelty(inp):
    """The bug this exists to fix: in a game, activity is the steady state.

    A player pressing the same rotation at the same rate is not news. Measured
    before this gate existed: 90% of ticks called the model, for 5 utterances.
    """
    baseline = _events(inp, [(float(i) * 1.5, ["1", "2", "3"][i % 3], "") for i in range(20)])
    recent = _events(inp, [(30.0 + float(i) * 1.5, ["1", "2", "3"][i % 3], "") for i in range(7)])
    assert inp.novelty_score(recent, baseline) < 0.34


def test_switching_keys_scores_high_novelty(inp):
    """A different rotation is a genuine behaviour change."""
    baseline = _events(inp, [(float(i) * 1.5, ["1", "2", "3"][i % 3], "") for i in range(20)])
    recent = _events(inp, [(30.0 + float(i) * 1.5, ["f5", "f6", "m4"][i % 3], "") for i in range(7)])
    assert inp.novelty_score(recent, baseline) >= 0.34


def test_a_burst_scores_high_novelty(inp):
    """Same keys, four times the rate — a burst window opening."""
    baseline = _events(inp, [(float(i) * 1.5, "1", "") for i in range(20)])
    recent = _events(inp, [(30.0 + float(i) * 0.3, "1", "") for i in range(10)])
    assert inp.novelty_score(recent, baseline) >= 0.34


def test_stopping_scores_high_novelty(inp):
    """A sudden stop is often the interesting event — died, or gave up."""
    baseline = _events(inp, [(float(i) * 1.5, "1", "") for i in range(20)])
    assert inp.novelty_score([], baseline) >= 0.34


def test_first_window_of_a_session_is_novel(inp):
    """With no baseline, everything is new."""
    assert inp.novelty_score(_events(inp, [(0.0, "1", "")]), []) == 1.0


def test_novelty_is_bounded(inp):
    baseline = _events(inp, [(float(i) * 1.5, "1", "") for i in range(5)])
    recent = _events(inp, [(10.0 + float(i) * 0.01, "f9", "") for i in range(50)])
    assert 0.0 <= inp.novelty_score(recent, baseline) <= 1.0


def test_key_mix_is_a_distribution(inp):
    events = _events(inp, [(0.0, "1", ""), (1.0, "1", ""), (2.0, "2", "")])
    mix = inp.key_mix(events)
    assert mix["1"] == pytest.approx(2 / 3)
    assert sum(mix.values()) == pytest.approx(1.0)
    assert inp.key_mix([]) == {}


def _sig(salience, *, name="screen", can_speak=True, block=""):
    """Build a calibrated Signal, the way a real source would."""
    from plugins.watch.signals import Signal
    return Signal(name=name, salience=salience, block=block, can_speak=can_speak)


# ══ Decider: silence detection ════════════════════════════════════════════

@pytest.mark.parametrize(
    "answer",
    ["NO REPLY", "no reply", "  NO REPLY  ", '"NO REPLY"', "NO REPLY.", "", "   "],
)
def test_silence_is_recognized_generously(dec, answer):
    """Shipping the literal 'NO REPLY' to a user as coaching is the worst case."""
    assert dec.is_silence(answer)


def test_real_feedback_is_not_mistaken_for_silence(dec):
    assert not dec.is_silence("You clipped that cooldown by half a second.")


# ══ Decider: repetition ═══════════════════════════════════════════════════

def test_rephrasings_count_as_repetition(dec):
    """The actual failure mode: same note, new words, every few seconds."""
    history = ["Watch your timing on the transition."]
    assert dec.is_repetition("Watch the timing on that transition.", history)


def test_genuinely_new_feedback_passes(dec):
    history = ["Watch your timing on the transition."]
    assert not dec.is_repetition("You are standing in the fire.", history)


def test_shared_long_opening_counts_as_repetition(dec):
    """MMDuet2's COMMON_PREFIX_THRES: a model that starts every line the same."""
    a = "Your rotation is drifting out of sync with the cooldown window and you should"
    b = "Your rotation is drifting out of sync with the cooldown window but the pull was fine"
    assert dec.is_repetition(b, [a])


def test_repetition_ignores_case_and_punctuation(dec):
    assert dec.is_repetition("watch your timing", ["Watch your timing!"])


# ══ Decider: the policy ═══════════════════════════════════════════════════
#
# Every test here drives `decide` through the SIGNAL contract: sources report a
# calibrated 0..1 salience and the decider compares one number to one
# threshold. That collapse is the point — before it, three sources reported
# three incompatible scales into one `novelty` parameter, and frame movement
# never reached a keypress-shaped threshold (159 gated sweeps, one utterance).


def _ask(text):
    return lambda _s, _u: text


def _quiet():
    """A tick where no source saw anything."""
    return [_sig(0.0)]


def _busy(salience=0.9, **kw):
    """A tick where the screen clearly moved."""
    return [_sig(salience, **kw)]


def test_quiet_ticks_never_call_the_model(dec):
    """An idle session must cost nothing — this is the whole free gate."""
    state = dec.DeciderState()
    calls = []

    def spy(system, user):
        calls.append(user)
        return "something"

    result = dec.decide(state, _quiet(), at=1.0, brief="rotation", ask=spy)
    assert not result.spoke
    assert result.reason == "quiet"
    assert result.called_model is False
    assert calls == []


def test_one_loud_source_is_enough_to_look(dec):
    """Max, not mean: a quiet source must not drown out the one that saw it.

    A cutscene has no keypresses; a steady rotation has no screen change. Either
    alone is a reason to look.
    """
    state = dec.DeciderState()
    signals = [_sig(0.0, name="keys"), _sig(0.9, name="screen")]
    assert dec.decide(state, signals, at=1.0, brief="", ask=_ask("You died.")).spoke


def test_salience_below_the_threshold_is_free(dec):
    state = dec.DeciderState()
    calls = []

    def spy(system, user):
        calls.append(1)
        return "something"

    result = dec.decide(state, [_sig(0.1)], at=5.0, brief="", ask=spy)
    assert result.reason == "quiet"
    assert calls == []


def test_refractory_blocks_before_spending_a_call(dec):
    """A nagging loop must also be a cheap loop."""
    state = dec.DeciderState()
    calls = []

    def spy(system, user):
        calls.append(1)
        return "First note."

    policy = dec.Policy(refractory=12.0, call_cooldown=0.0)
    assert dec.decide(state, _busy(), at=10.0, brief="", ask=spy, policy=policy).spoke

    second = dec.decide(state, _busy(), at=12.0, brief="", ask=spy, policy=policy)
    assert not second.spoke
    assert second.reason == "refractory"
    assert second.called_model is False
    assert len(calls) == 1


def test_speaking_is_allowed_again_after_the_refractory_window(dec):
    state = dec.DeciderState()
    dec.decide(state, _busy(), at=10.0, brief="", ask=_ask("First note."))
    later = dec.decide(state, _busy(), at=40.0, brief="", ask=_ask("A different observation."))
    assert later.spoke


def test_model_silence_is_respected(dec):
    state = dec.DeciderState()
    result = dec.decide(state, _busy(), at=5.0, brief="", ask=_ask("NO REPLY"))
    assert not result.spoke
    assert result.reason == "model_silent"
    assert result.called_model is True


def test_a_repeat_is_suppressed_even_when_the_model_chose_to_speak(dec):
    """The model can decide to speak and still be repeating itself."""
    state = dec.DeciderState()
    dec.decide(state, _busy(), at=0.0, brief="", ask=_ask("Watch your timing on the transition."))
    result = dec.decide(
        state,
        _busy(),
        at=60.0,
        brief="",
        ask=_ask("Watch the timing on that transition."),
        policy=dec.Policy(refractory=1.0),
    )
    assert not result.spoke
    assert result.reason == "repetition"


def test_history_is_bounded(dec):
    """An unbounded history would grow the decider prompt without limit."""
    state = dec.DeciderState()
    for i in range(20):
        state.remember(f"Distinct observation number {i}.", at=float(i))
    assert len(state.history) <= dec.DEFAULT_HISTORY


def test_every_tick_is_logged_for_offline_replay(dec):
    """The trace is what makes a brief tunable without re-recording."""
    state = dec.DeciderState()
    dec.decide(state, _quiet(), at=1.0, brief="", ask=_ask("x"))
    dec.decide(state, _busy(), at=20.0, brief="", ask=_ask("NO REPLY"))
    dec.decide(state, _busy(), at=40.0, brief="", ask=_ask("A real note."))

    assert len(state.log) == 3
    stats = dec.replay_stats(state.log)
    assert stats["ticks"] == 3
    assert stats["model_calls"] == 2
    assert stats["spoke"] == 1
    assert stats["reasons"]["quiet"] == 1


def test_the_log_records_which_source_woke_the_loop(dec):
    """Replay needs to know whether it was the screen or the keys."""
    state = dec.DeciderState()
    dec.decide(
        state,
        [_sig(0.1, name="keys"), _sig(0.9, name="screen")],
        at=1.0,
        brief="",
        ask=_ask("A note."),
    )
    assert state.log[-1]["signals"] == {"keys": 0.1, "screen": 0.9}


def test_replay_stats_on_an_empty_log_do_not_divide_by_zero(dec):
    assert dec.replay_stats([])["call_rate"] == 0.0


# ══ Decider: call economy (the measured regression) ═══════════════════════

def test_a_suppressed_repeat_still_backs_off_the_next_call(dec):
    """The 90%-call-rate bug: suppression left no trace, so the loop re-asked.

    A repeat is suppressed AFTER the model call, so it never touched
    ``last_spoke_at`` and refractory never engaged — the loop paid for an answer
    it threw away, every single tick. Measured: 550 calls to ship 5 utterances
    over ten minutes.
    """
    state = dec.DeciderState()
    calls = []

    def spy(system, user):
        calls.append(1)
        return "Watch your timing on the transition."

    policy = dec.Policy(refractory=1.0)
    dec.decide(state, _busy(), at=0.0, brief="", ask=spy, policy=policy)
    assert len(calls) == 1

    result = dec.decide(state, _busy(), at=1.0, brief="", ask=spy, policy=policy)
    assert result.reason == "call_cooldown"
    assert result.called_model is False
    assert len(calls) == 1


def test_model_silence_also_starts_the_call_cooldown(dec):
    """'Nothing to add' covers the seconds after it too."""
    state = dec.DeciderState()
    calls = []

    def spy(system, user):
        calls.append(1)
        return "NO REPLY"

    dec.decide(state, _busy(), at=0.0, brief="", ask=spy)
    second = dec.decide(state, _busy(), at=2.0, brief="", ask=spy)
    assert second.reason == "call_cooldown"
    assert len(calls) == 1


def test_call_cooldown_expires(dec):
    state = dec.DeciderState()
    calls = []

    def spy(system, user):
        calls.append(1)
        return "NO REPLY"

    policy = dec.Policy(call_cooldown=8.0)
    dec.decide(state, _busy(), at=0.0, brief="", ask=spy, policy=policy)
    dec.decide(state, _busy(), at=9.0, brief="", ask=spy, policy=policy)
    assert len(calls) == 2


def test_a_busy_session_stays_cheap_end_to_end(dec):
    """Regression guard on the whole policy, not one gate.

    Simulates the shape that produced the bug: a model that always wants to
    speak and mostly repeats itself, against continuous activity.
    """
    state = dec.DeciderState()
    nag = "Watch your timing on the transition."
    spoken = []
    for tick in range(600):
        result = dec.decide(
            state,
            _busy(0.9 if tick % 60 == 0 else 0.1),
            at=float(tick),
            brief="rotation",
            ask=_ask(nag if tick % 7 else f"Distinct note {tick}."),
        )
        if result.spoke:
            spoken.append(result.text)

    stats = dec.replay_stats(state.log)
    assert stats["call_rate"] < 0.2, f"call rate {stats['call_rate']} — the loop is re-asking"
    assert len(spoken) == len(set(spoken)), "a duplicate reached the user"


# ══ Decider: the hold gate ════════════════════════════════════════════════

def test_an_utterance_produced_at_a_bad_moment_is_held_not_dropped(dec):
    """Dropping it means the loop is silent exactly when it had something."""
    state = dec.DeciderState()
    result = dec.decide(
        state,
        _busy(can_speak=False),
        at=5.0,
        brief="timing",
        ask=_ask("You are rushing the second bar."),
    )
    assert not result.spoke
    assert result.reason == "held"
    assert result.deferred is True
    assert state.pending == "You are rushing the second bar."


def test_a_held_utterance_is_delivered_at_the_next_gap(dec):
    state = dec.DeciderState()
    dec.decide(
        state, _busy(can_speak=False), at=5.0, brief="timing",
        ask=_ask("You are rushing the second bar."),
    )
    delivered = dec.decide(state, _quiet(), at=9.0, brief="timing")
    assert delivered.spoke
    assert delivered.text == "You are rushing the second bar."
    assert delivered.deferred is True
    assert state.pending is None


def test_delivering_a_held_utterance_costs_no_model_call(dec):
    """It was already paid for and already checked for repetition."""
    state = dec.DeciderState()
    calls = []

    def spy(_s, _u):
        calls.append(1)
        return "A note about your timing."

    dec.decide(state, _busy(can_speak=False), at=5.0, brief="", ask=spy)
    delivered = dec.decide(state, _busy(), at=20.0, brief="", ask=spy)
    assert delivered.spoke
    assert delivered.called_model is False
    assert len(calls) == 1


def test_a_newer_observation_replaces_a_stale_held_one(dec):
    """Feedback about a passage two phrases ago is worse than none.

    Both ticks sit inside the hold timeout, so the first is still waiting when
    the second arrives — that is the case this replacement is for. Past the
    timeout the first would have been delivered instead, which is a different
    (and also correct) behaviour.
    """
    state = dec.DeciderState()
    policy = dec.Policy(hold_timeout=60.0, call_cooldown=1.0)
    dec.decide(state, _busy(can_speak=False), at=5.0, brief="", ask=_ask("First note."), policy=policy)
    dec.decide(
        state, _busy(can_speak=False), at=20.0, brief="",
        ask=_ask("Second, newer note."), policy=policy,
    )
    assert state.pending == "Second, newer note."


def test_one_source_saying_wait_is_enough_to_hold(dec):
    """Unanimous, not majority: being wrong the other way ruins the take."""
    state = dec.DeciderState()
    signals = [_sig(0.9, name="screen"), _sig(0.5, name="notes", can_speak=False)]
    result = dec.decide(state, signals, at=5.0, brief="", ask=_ask("Your left hand is dragging."))
    assert result.reason == "held"


def test_holding_does_not_apply_when_speaking_is_safe(dec):
    state = dec.DeciderState()
    result = dec.decide(state, _busy(), at=5.0, brief="", ask=_ask("A note."))
    assert result.spoke
    assert result.deferred is False
    assert state.pending is None


# ══ Decider: the prompt ═══════════════════════════════════════════════════

def test_prompt_carries_prior_utterances_so_the_model_can_avoid_repeating(dec):
    state = dec.DeciderState()
    state.remember("You clipped the cooldown.", at=1.0)
    prompt = dec.build_decider_prompt("rotation", recent_utterances=list(state.history))
    assert "You clipped the cooldown." in prompt


def test_prompt_states_the_no_reply_contract(dec):
    assert "NO REPLY" in dec.build_decider_prompt("anything")
    assert "NO REPLY" in dec.SYSTEM_PROMPT


def test_system_prompt_prefers_silence_and_forbids_repeating(dec):
    lowered = dec.SYSTEM_PROMPT.lower()
    assert "prefer silence" in lowered
    assert "since your last reply" in lowered
    assert "never repeat" in lowered


def test_brief_comes_last_so_it_is_the_live_instruction(dec):
    """Burying the brief above the evidence is how a decider drifts generic."""
    prompt = dec.build_decider_prompt("my cooldown drift", input_block="Input track: 1@0.0")
    assert prompt.index("Input track") < prompt.index("my cooldown drift")


def test_input_track_is_offered_to_the_model_when_present(dec):
    prompt = dec.build_decider_prompt("rotation", input_block="Input track: 1@0.0 2@1.5")
    assert "1@0.0" in prompt


def test_no_brief_still_produces_a_usable_instruction(dec):
    assert "notable" in dec.build_decider_prompt("")


def test_a_held_utterance_is_eventually_delivered_even_without_a_gap(dec):
    """The measured mute: continuous combat has no lull, so the hold never ended.

    75 utterances held and ZERO delivered across a ten-minute session — a
    failure that reads exactly like a working quiet loop. Interrupting slightly
    is strictly better than never speaking.
    """
    state = dec.DeciderState()
    policy = dec.Policy(hold_timeout=20.0, refractory=1.0, call_cooldown=1.0)

    first = dec.decide(
        state, _busy(can_speak=False), at=0.0, brief="", ask=_ask("You are clipping cooldowns."),
        policy=policy,
    )
    assert first.reason == "held"

    # Still no safe moment, but the deadline has passed.
    late = dec.decide(state, _busy(can_speak=False), at=25.0, brief="", policy=policy)
    assert late.spoke
    assert late.deferred is True
    assert state.pending is None


def test_a_gap_still_delivers_sooner_than_the_timeout(dec):
    """The timeout is a backstop, not the mechanism."""
    state = dec.DeciderState()
    policy = dec.Policy(hold_timeout=20.0)
    dec.decide(state, _busy(can_speak=False), at=0.0, brief="", ask=_ask("A note."), policy=policy)
    delivered = dec.decide(state, _busy(can_speak=True), at=3.0, brief="", policy=policy)
    assert delivered.spoke
    assert delivered.at == 3.0


def test_replacing_a_held_utterance_does_not_restart_its_deadline(dec):
    """The other half of the permanent-mute bug.

    Newer text should replace stale text, but if the wait clock restarts with
    it, a session that never goes quiet refreshes its own deadline every tick
    and the hold never times out. Measured: 75 held, 0 delivered.
    """
    state = dec.DeciderState()
    policy = dec.Policy(hold_timeout=20.0, call_cooldown=1.0, refractory=1.0)

    for tick in range(0, 19, 2):
        dec.decide(
            state, _busy(can_speak=False), at=float(tick), brief="",
            ask=_ask(f"Observation at {tick}."), policy=policy,
        )
    assert state.pending_since == 0.0, "the wait clock must date from the FIRST hold"

    late = dec.decide(state, _busy(can_speak=False), at=21.0, brief="", policy=policy)
    assert late.spoke, "the deadline must fire even while the session stays busy"
