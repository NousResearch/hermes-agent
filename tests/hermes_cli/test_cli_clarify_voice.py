"""Spoken clarify answers: the read-aloud sentence, the shared scorer, and the panel loop.

The loop tests drive a fake CLI (no prompt_toolkit, no audio device) so the behaviour under
test is the panel's own: what gets read, what a transcript resolves to, that a misheard answer
locks nothing, and that every exit hands the mic back.
"""

import json
import threading
import time

import pytest


@pytest.fixture(autouse=True)
def _clean_i18n_cache():
    # agent.i18n caches catalogs per (home, lang) and the resolved language globally; a stale entry
    # from another test's isolated home makes t() resolve against THAT home (tripping the real-home
    # guard in a hang, not a clean failure). Start and end every test from a cold cache.
    from agent.i18n import reset_language_cache
    reset_language_cache()
    yield
    reset_language_cache()


@pytest.fixture(autouse=True)
def _pinned_stop_phrases(monkeypatch):
    # is_voice_stop_phrase reads voice.stop_phrases from config; pin it deterministically.
    import tools.voice_mode_transcript as vmt
    monkeypatch.setattr(vmt, "_load_voice_stop_phrases", lambda: ("we are done", "finish it"))


# --- render_clarify_speech ------------------------------------------------

def test_renders_question_and_numbered_options_without_the_recommended_suffix():
    from hermes_cli.cli_clarify_voice import render_clarify_speech

    speech = render_clarify_speech(
        "Which database?", ["Postgres (Recommended)", "SQLite"], False,
        choice_label="{index}. {text}")
    assert speech.startswith("Which database? ")
    assert "1. Postgres" in speech and "(Recommended)" not in speech
    assert "2. SQLite" in speech


def test_multi_question_headers_and_multiselect_other_cue():
    from hermes_cli.cli_clarify_voice import render_clarify_speech

    speech = render_clarify_speech(
        "Pick targets", ["staging", "prod"], True, index=2, total=3,
        choice_label="{index}. {text}", question_label="Question {index} of {total}.",
        other_label="You can also name your own answer.")
    assert speech.startswith("Question 2 of 3. Pick targets")
    assert speech.endswith("You can also name your own answer.")


def test_open_ended_question_reads_only_its_text():
    from hermes_cli.cli_clarify_voice import render_clarify_speech

    assert render_clarify_speech("What should it be called?", None, False) == "What should it be called?"


# --- score_answer: the same matching a typed reply gets -------------------

@pytest.mark.parametrize("spoken, expected", [
    ("postgres", "Postgres (Recommended)"),
    ("Postgres", "Postgres (Recommended)"),
    ("sqlite", "SQLite"),
    ("2", "SQLite"),
    ("1", "Postgres (Recommended)"),
])
def test_single_choice_scores_labels_and_numbers_like_typed_text(spoken, expected):
    from hermes_cli.cli_clarify_voice import score_answer

    value, reason = score_answer(spoken, ["Postgres (Recommended)", "SQLite"], False, awaiting_text=False)
    assert value == expected and reason is None


def test_out_of_range_number_is_not_a_silent_neighbour_pick():
    from hermes_cli.cli_clarify_voice import score_answer

    value, reason = score_answer("9", ["Postgres", "SQLite"], False, awaiting_text=False)
    assert value is None and reason == "invalid_selection"


def test_multi_select_returns_a_json_array_and_rejects_a_bad_token():
    from hermes_cli.cli_clarify_voice import score_answer

    value, reason = score_answer("1, prod", ["staging", "prod"], True, awaiting_text=False)
    assert value is not None and json.loads(value) == ["staging", "prod"] and reason is None
    value, reason = score_answer("staging, nope", ["staging", "prod"], True, awaiting_text=False)
    assert value is None and reason == "invalid_selection"


def test_free_text_mode_accepts_the_words_and_reports_prose_otherwise():
    from hermes_cli.cli_clarify_voice import score_answer

    assert score_answer("call it atlas", ["a", "b"], False, awaiting_text=True)[0] == "call it atlas"
    value, reason = score_answer("that one over there", ["a", "b"], False, awaiting_text=False)
    assert value is None and reason == "prose"


def test_open_ended_takes_any_transcript():
    from hermes_cli.cli_clarify_voice import score_answer

    assert score_answer("whatever you think", None, False, awaiting_text=False)[0] == "whatever you think"


# --- the loop -------------------------------------------------------------

class _FakeRecorder:
    def __init__(self):
        self.cancelled = False

    def start(self, on_silence_stop=None):
        if on_silence_stop is not None:
            threading.Timer(0.01, on_silence_stop).start()

    def stop(self):
        return None

    def cancel(self):
        self.cancelled = True

    def shutdown(self):
        pass


class _FakeCLI:
    """The panel's callbacks, with _clarify_state cleared once the single question is answered."""

    def __init__(self, choices):
        self._voice_mode = True
        self._voice_recording = False
        self._clarify_freetext = False
        self._clarify_multi_base = None
        self._clarify_prefill = ""
        self._voice_tts_done = threading.Event()
        self._voice_tts_done.set()
        self._voice_lock = threading.RLock()
        self._clarify_state = {
            "questions": [{"qid": "q1", "question": "Which database?", "choices": choices, "multi_select": False}],
            "answers": {}, "answer_meta": {}, "active": 0, "choices": list(choices), "selected": 0,
            "multi_select": False, "selected_indices": None, "response_queue": None, "voice": None}
        self.locked, self.rejected, self.cancelled = [], [], 0

    def _voice_stt_model(self):
        return None


@pytest.fixture
def trimmed(monkeypatch):
    """Run the loop with the audio and recorder plumbing stubbed to the transcript list.

    An empty list means the panel closed: leaving the question open would make the loop wait
    for more speech (correctly), which is not what a test with a fixed script wants.
    """
    from hermes_cli import cli_clarify_voice as mod

    def _install(cli, transcripts):
        pending = list(transcripts)

        def _next(self, transcribe_recording):
            if not pending:
                cli._clarify_state = None
                return None
            return pending.pop(0)

        monkeypatch.setattr(mod.SpokenClarifyPanel, "_one_utterance", _next)
        monkeypatch.setattr(mod.SpokenClarifyPanel, "_ensure_recorder", lambda self, create: True)
        monkeypatch.setattr(mod.SpokenClarifyPanel, "_wait_for_silence", lambda self: True)
        return pending

    return _install


def _panel(cli, spoken, locked_kwargs=None):
    from hermes_cli.cli_clarify_voice import SpokenClarifyPanel

    def _on_answer(value, own_text=False, picks=None):
        if locked_kwargs is not None:
            locked_kwargs.append({"own_text": own_text, "picks": picks})
        cli.locked.append(value)
        cli._clarify_state = None

    panel = SpokenClarifyPanel(
        cli, speak=spoken.append,
        on_answer=_on_answer,
        on_cancel=lambda: (setattr(cli, "cancelled", cli.cancelled + 1), setattr(cli, "_clarify_state", None)),
        on_reject=cli.rejected.append)
    panel._recorder = _FakeRecorder()
    return panel


def test_spoken_answer_is_read_then_locked(trimmed):
    cli = _FakeCLI(["Postgres", "SQLite"])
    trimmed(cli, ["sqlite"])
    spoken = []
    _panel(cli, spoken)._run()
    assert spoken and "Which database?" in spoken[0]
    assert cli.locked == ["SQLite"]


def test_stop_phrase_cancels_instead_of_answering(trimmed):
    cli = _FakeCLI(["Postgres", "SQLite"])
    trimmed(cli, ["we are done"])
    _panel(cli, [])._run()
    assert cli.cancelled == 1 and cli.locked == []


def test_misheard_answer_locks_nothing_and_is_reported_once(trimmed):
    cli = _FakeCLI(["Postgres", "SQLite"])
    trimmed(cli, ["that one you mentioned", "still wrong"])
    _panel(cli, [])._run()
    assert cli.locked == []
    assert cli.rejected == ["prose"]  # once for the same question, not per attempt


def test_question_is_not_re_read_on_every_attempt(trimmed):
    cli = _FakeCLI(["Postgres", "SQLite"])
    trimmed(cli, ["nonsense", "sqlite"])
    spoken = []
    _panel(cli, spoken)._run()
    assert len(spoken) == 1  # the retry captured again without re-reading the question
    assert cli.locked == ["SQLite"]


def test_stop_hands_the_mic_back_and_is_idempotent():
    from hermes_cli.cli_clarify_voice import SpokenClarifyPanel

    cli = _FakeCLI(["a", "b"])
    panel = SpokenClarifyPanel(cli, speak=lambda _s: None, on_answer=lambda _v: None,
                               on_cancel=lambda: None, on_reject=lambda _r: None)
    rec = _FakeRecorder()
    panel._recorder = rec
    panel.stop()
    panel.stop()
    assert rec.cancelled and panel._recorder is None


def test_waits_out_a_playing_reply_then_captures():
    """The settle gate reads the CLI's TTS flag, never the speak callback (a bound method)."""
    from hermes_cli.cli_clarify_voice import SpokenClarifyPanel

    cli = _FakeCLI(["a"])
    panel = SpokenClarifyPanel(cli, speak=len, on_answer=lambda _v: None,
                               on_cancel=lambda: None, on_reject=lambda _r: None)

    def _release():
        threading.Timer(0.15, cli._voice_tts_done.set).start()

    cli._voice_tts_done.clear()  # "still speaking"
    _release()
    start = time.monotonic()
    assert panel._wait_for_silence() is True
    assert time.monotonic() - start >= 0.1, "it captured while the reply was still playing"


@pytest.mark.parametrize("said", ["five", "5", "option five", "number five", "fifth", "five please"])
def test_a_spoken_number_picks_that_option(said):
    """STT rarely emits a bare digit; "option five" is what actually arrives for a spoken pick."""
    from hermes_cli.cli_clarify_voice import score_answer

    choices = ["Apple", "Banana", "Mango", "Orange", "Kiwi"]
    assert score_answer(said, choices, False, awaiting_text=False) == ("Kiwi", None)


@pytest.mark.parametrize("said", ["two", "kiwi please", "Kiwi"])
def test_a_label_still_wins_over_a_number(said):
    from hermes_cli.cli_clarify_voice import score_answer

    choices = ["Apple", "Banana", "Mango", "Orange", "Kiwi"]
    value, reason = score_answer(said, choices, False, awaiting_text=False)
    assert reason is None and value in choices


def test_a_spoken_number_is_one_pick_for_multi_select():
    from hermes_cli.cli_clarify_voice import score_answer

    value, reason = score_answer("five", ["Apple", "Banana", "Mango", "Orange", "Kiwi"], True,
                                 awaiting_text=False)
    assert reason is None and value is not None and json.loads(value) == ["Kiwi"]


def test_a_stale_freetext_flag_does_not_open_the_next_question_to_free_text(trimmed):
    """Regression: a leftover _clarify_freetext made a one-option question accept any words."""
    cli = _FakeCLI(["Blue"])
    cli._clarify_freetext = True  # left over from an earlier question's Other edit
    trimmed(cli, ["green"])
    _panel(cli, [])._run()
    assert cli.locked == [] and cli.rejected, "an unmatched phrase was locked as the answer"


def test_an_unmatched_phrase_on_a_choice_prompt_is_never_taken_as_the_answer():
    """A mis-heard phrase must not become a custom answer: only an option/number/Other resolves."""
    from hermes_cli.cli_clarify_voice import score_answer

    value, reason = score_answer("it's snowy", ["Sunny", "Rainy"], False, awaiting_text=False)
    assert value is None and reason is not None
    value, reason = score_answer("it's snowy", ["Sunny", "Rainy"], True, awaiting_text=False)
    assert value is None and reason is not None


def test_the_own_answer_prompt_still_accepts_any_words():
    from hermes_cli.cli_clarify_voice import score_answer

    assert score_answer("it's snowy", ["Sunny", "Rainy"], False, awaiting_text=True) == ("it's snowy", None)


@pytest.mark.parametrize("said", ["other", "the other option", "other please", "custom"])
def test_saying_other_opens_free_text_instead_of_locking_a_word(said):
    """The panel's Other row is implicit: Enter on it types an answer. Speech must do the same."""
    from hermes_cli.cli_clarify_voice import OTHER_SENTINEL, score_answer

    value, reason = score_answer(said, ["Sunny", "Rainy"], False, awaiting_text=False)
    assert (value, reason) == (OTHER_SENTINEL, None)


def test_other_then_the_typed_answer_locks_as_other(trimmed):
    """End to end through the loop: "other" prompts once, the next utterance becomes the answer."""
    cli = _FakeCLI(["Sunny", "Rainy"])
    trimmed(cli, ["other", "snowy"])
    spoken = []
    _panel(cli, spoken)._run()
    assert cli.locked == ["snowy"]
    assert any("instead" in s for s in spoken), "it never asked for the user's own answer"


def test_a_stop_phrase_at_the_own_answer_prompt_cancels(trimmed):
    cli = _FakeCLI(["Sunny", "Rainy"])
    trimmed(cli, ["other", "we are done"])
    _panel(cli, [])._run()
    assert cli.locked == [] and cli.cancelled


def test_other_in_the_same_breath_as_labels_keeps_them(trimmed):
    """Live 2026-11-14: "teal, other" then a custom answer must keep Teal, not replace it."""
    from hermes_cli.cli_clarify_voice import OTHER_WITH_CHOICES_SENTINEL, score_answer

    picks: list[str] = []
    assert score_answer("teal, other", ["Teal", "Navy", "White"], True, awaiting_text=False,
                        picks=picks) == (OTHER_WITH_CHOICES_SENTINEL, None)
    assert picks == ["Teal"]

    cli = _FakeCLI(["Teal", "Navy", "White"])
    cli._clarify_state["questions"][0]["multi_select"] = True
    trimmed(cli, ["teal and other", "black"])
    kwargs: list[dict] = []
    _panel(cli, [], locked_kwargs=kwargs)._run()
    assert cli.locked == ["black"]
    assert kwargs[0]["own_text"] is True and kwargs[0]["picks"] == ["Teal"]


def test_a_wrong_answer_like_word_is_refused_not_picked():
    from hermes_cli.cli_clarify_voice import score_answer

    value, reason = score_answer("teal, other, hmm", ["Teal", "Navy"], True, awaiting_text=False)
    assert value is None and reason is not None


def test_a_lowercase_clause_still_matches_its_label():
    """STT lowercases; "teal and other" must find "Teal" or the whole reply is refused."""
    from hermes_cli.cli_clarify_voice import OTHER_WITH_CHOICES_SENTINEL, score_answer

    picks: list[str] = []
    assert score_answer("teal and other", ["Teal", "Navy"], True, awaiting_text=False,
                        picks=picks) == (OTHER_WITH_CHOICES_SENTINEL, None)
    assert picks == ["Teal"]


def test_multi_select_lists_still_score_as_before():
    from hermes_cli.cli_clarify_voice import score_answer

    assert score_answer("navy, white", ["Teal", "Navy", "White"], True,
                        awaiting_text=False)[0] == '["Navy", "White"]'


def test_an_out_of_range_number_locks_nothing():
    from hermes_cli.cli_clarify_voice import score_answer

    value, reason = score_answer("nine", ["Apple", "Banana"], False, awaiting_text=False)
    assert value is None and reason is not None


def test_a_refused_capture_is_retried_before_the_mic_is_given_up(monkeypatch):
    """CoreAudio err -50 on start: retry rather than disabling the microphone for the question."""
    from hermes_cli import cli_clarify_voice as mod

    cli = _FakeCLI(["Blue"])
    attempts = {"n": 0}

    class _FlakyRecorder:
        def start(self, on_silence_stop=None):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("PaMacCore (AUHAL) err=-50")
            cli._clarify_state = None  # the question was answered while it retried
            if on_silence_stop:
                on_silence_stop()

        def stop(self):
            return None

        def cancel(self):
            pass

        def shutdown(self):
            pass

    panel = _panel(cli, [])
    panel._recorder = _FlakyRecorder()
    monkeypatch.setattr(mod, "RECORDER_OPEN_BACKOFF_SECONDS", 0)
    assert panel._one_utterance(lambda *a, **k: {"success": True, "transcript": "blue"}) is None
    assert attempts["n"] == 3, "it did not retry a refused capture"


def test_a_missing_tts_flag_does_not_block_capture():
    from hermes_cli.cli_clarify_voice import SpokenClarifyPanel

    cli = _FakeCLI(["a"])
    del cli._voice_tts_done
    panel = SpokenClarifyPanel(cli, speak=len, on_answer=lambda _v: None,
                               on_cancel=lambda: None, on_reject=lambda _r: None)
    assert panel._wait_for_silence() is True


def test_speech_callback_is_only_called_with_a_sentence(trimmed):
    """Regression: the listener died on its first question because it asked the callback for state."""
    cli = _FakeCLI(["Postgres", "SQLite"])
    trimmed(cli, ["sqlite"])
    spoken = []
    _panel(cli, spoken)._run()
    assert spoken and isinstance(spoken[0], str) and "Which database?" in spoken[0]


def test_no_listener_without_voice_mode():
    """Keyboard-only sessions keep the panel exactly as it was: nothing opens the mic."""
    from hermes_cli.cli_clarify_voice import is_live

    cli = _FakeCLI(["a"])
    cli._voice_mode = False
    assert is_live(cli) is False


# --- panel painting ---------------------------------------------------------
# The panel repaints itself every second from `_poll_modal_queue` via `_paint_now`, while voice
# notices print through `_cprint`, which erases the renderer and redraws at an absolute cursor
# position. Whichever lands second wins, so a notice near a tick left the panel half-drawn or a row
# low. A notice must therefore print first and repaint explicitly, and the reject path must be able
# to repaint at all (it was a `@staticmethod`, so it could not).


class _FakeCliModule:
    """Stands in for the `cli` module: `_clarify_voice_notice` imports print helpers lazily."""

    _DIM = _RST = ""

    def __init__(self, order, sink):
        self._order, self._sink = order, sink

    def _cprint(self, text):
        self._order.append("print")
        self._sink.append(text)


def _notice_cli(order):
    from hermes_cli.cli_voice_mixin import CLIVoiceMixin

    class _FakeVoiceCli(CLIVoiceMixin):
        def __init__(self):
            self._paint_now = lambda: order.append("paint")

    return CLIVoiceMixin, _FakeVoiceCli()


def test_notice_prints_the_row_before_repainting_the_panel(monkeypatch):
    import sys

    order, printed = [], []
    monkeypatch.setitem(sys.modules, "cli", _FakeCliModule(order, printed))
    mixin, cli = _notice_cli(order)

    mixin._clarify_voice_notice(cli, "Sorry, I didn't catch an option there.")

    assert order == ["print", "paint"], f"panel repainted before the row landed on screen: {order}"
    assert "didn't catch an option" in printed[0]


def test_reject_is_an_instance_method_so_it_can_repaint(monkeypatch):
    import sys

    order, printed = [], []
    monkeypatch.setitem(sys.modules, "cli", _FakeCliModule(order, printed))
    mixin, cli = _notice_cli(order)

    assert not isinstance(mixin.__dict__["_clarify_voice_reject"], staticmethod), \
        "a staticmethod cannot reach _paint_now, so the panel stays broken after a refusal"
    mixin._clarify_voice_reject(cli, "no option named that")

    assert order == ["print", "paint"], f"refusal notice did not repaint the panel: {order}"


def test_a_refusal_alone_never_prints_without_repainting(monkeypatch):
    """Guards the original defect directly: a bare `_cprint` with no paint left the panel broken."""
    import sys

    order, _ = [], []
    monkeypatch.setitem(sys.modules, "cli", _FakeCliModule(order, _))
    mixin, cli = _notice_cli(order)

    mixin._clarify_voice_reject(cli, "x")
    assert order.count("print") == order.count("paint"), \
        f"a printed row was left without a repaint: {order}"


def test_immediate_invalidate_paints_and_the_default_stays_throttled():
    import types

    from hermes_cli.cli_voice_mixin import CLIVoiceMixin

    painted, invalidated = [], []
    cli = types.SimpleNamespace(
        _paint_now=lambda: painted.append(1),
        _app=types.SimpleNamespace(invalidate=lambda: invalidated.append(1)),
    )

    CLIVoiceMixin._voice_invalidate(cli, immediate=True)
    assert painted and not invalidated, "immediate=True did not paint synchronously"

    CLIVoiceMixin._voice_invalidate(cli)
    assert invalidated and len(painted) == 1, \
        "the default path must stay on the throttled invalidate (the audio level loop needs it)"
