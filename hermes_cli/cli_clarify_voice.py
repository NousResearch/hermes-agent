"""Spoken answers for the CLI's clarify panel (voice mode only).

While a clarify question is on screen the agent thread is blocked, the mic is idle and the
keyboard owns the panel. This module gives that panel a voice side so a spoken answer resolves
the *same* state the bindings write — ``_clarify_batch_lock`` — instead of a parallel scoring
path. One daemon thread per clarify request; it never paints and never clears the state it was
handed, so an answered question, a timeout or a ``request.cancel`` tears the panel down exactly
as it does for the keyboard.

One utterance is resolved in this order:

1. an exact ``voice.stop_phrases`` match cancels the whole clarify and ends voice mode — the
   user's words are never swallowed as an option pick;
2. free-text mode ("Other", or an open-ended question) takes the words as typed text would;
3. otherwise the transcript is scored by ``tools.clarify_gateway`` — a bare number, an exact
   label (the ``(Recommended)`` suffix ignored), or a comma list for multi-select;
4. anything else is a misheard answer: say so once and keep listening for the same question.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time

logger = logging.getLogger(__name__)

#: Cap on one answer capture. The panel's own clarify timeout bounds the whole wait; this only
#: stops a recorder whose silence callback never fires from holding the mic forever.
ANSWER_CAPTURE_MAX_SECONDS = 25.0

#: Mic released between attempts, so a noisy room cannot spin the loop.
ANSWER_ATTEMPT_GAP_SECONDS = 0.3

#: How long to let a still-playing reply finish before capturing, so the question Hermes is
#: speaking is not recorded back as its own answer.
TTS_SETTLE_MAX_SECONDS = 180.0

#: CoreAudio can refuse a new input stream for a moment after a playback stream closed (AUHAL -50
#: / -574). The panel keeps one stream open for its whole life, and retries briefly when it must open.
RECORDER_OPEN_ATTEMPTS = 3
RECORDER_OPEN_BACKOFF_SECONDS = 0.35

#: ``score_answer`` value meaning "the words name the Other row": the caller asks one more utterance
#: for the free-text answer. A plain string, so it can never collide with a real choice label.
OTHER_SENTINEL = "\x00other"

#: ``score_answer`` value meaning "the words name a real option *and* the Other row" ("teal, other"):
#: the caller asks one more utterance and merges it onto the labels the first pass picked.
OTHER_WITH_CHOICES_SENTINEL = "\x00other+"

#: Split a spoken answer into its clauses: the separators STT emits, plus "and"/"then" filler.
_CLAUSE_SPLIT = re.compile(r"[,;]|\band\b|\bthen\b|\bplus\b", re.IGNORECASE)


def split_clauses(text: str) -> list[str]:
    """``"teal, other"`` -> ``["teal", "other"]``; unusable fragments dropped."""
    return [part.strip(" .!?") for part in _CLAUSE_SPLIT.split(text or "") if part.strip(" .!?")]



def render_clarify_speech(question: str, choices, multi_select: bool, *,
                          index: int = 1, total: int = 1,
                          choice_label: str = "{index}. {text}",
                          question_label: str = "Question {index} of {total}.",
                          other_label: str = "") -> str:
    """The sentence read aloud for one question: its position, its text, then its options.

    ``choices`` are the raw labels (the first still carries the ``(Recommended)`` suffix); that
    suffix is stripped — it is a hint for a human eye, not something to hear. Multi-select gets
    the "Other" cue appended; "Other" is not offered for single choice, because naming a
    free-text answer is exactly what the mic is already for.
    """
    from tools.clarify_tool import strip_recommended

    labels = [strip_recommended(str(c)).strip() for c in choices or ()]
    if not labels:
        return question_text(question, index=index, total=total, question_label=question_label)
    spoken = ", ".join(choice_label.format(index=i, text=label) for i, label in enumerate(labels, 1))
    return f"{question_text(question, index=index, total=total, question_label=question_label)} {spoken}. {other_label}".strip()


def question_text(question: str, *, index: int = 1, total: int = 1,
                  question_label: str = "Question {index} of {total}.") -> str:
    """Just the question — spoken first, so listening can start while the options read out."""
    text = question.strip()
    if total > 1:
        text = f"{question_label.format(index=index, total=total)} {text}"
    return text


def active_question(state: dict) -> dict | None:
    """The question the panel is pointing at, or None for a stale index."""
    questions = state.get("questions") or []
    index = state.get("active", 0)
    if isinstance(index, int) and 0 <= index < len(questions) and isinstance(questions[index], dict):
        return questions[index]
    return None


def _spoken_number(text: str) -> int | None:
    """``"five"`` / ``"option five"`` / ``"number 5"`` / ``"fifth"`` as 0-based option indexes.

    Speech-to-text rarely emits a bare digit for a short answer ("option five" is the normal
    transcription), so a number-only selection has to be understood in words too. Returns None
    when the text is not a spoken index — a label like "Kiwi" is left to the label matcher.
    """
    words = text.casefold().replace(",", " ").split()
    while words and words[0] in _INDEX_LEAD_WORDS:
        words = words[1:]
    # "five please" / "the fifth one" are how people actually speak; trailing filler is noise,
    # but a second *number* ("five six") is ambiguous for single-select and falls through.
    while words and words[-1] in _INDEX_LEAD_WORDS:
        words = words[:-1]
    if not words or len(words) > 1:
        return None
    token = words[0]
    if token.isdigit():
        return int(token) - 1
    if token in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[token]
    cardinal = _CARDINAL_WORDS.get(token)
    return None if cardinal is None else cardinal - 1


_INDEX_LEAD_WORDS = frozenset({"option", "number", "choice", "the", "please", "thanks"})
_CARDINAL_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_ORDINAL_WORDS = {
    "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "sixth": 5, "seventh": 6,
    "eighth": 7, "ninth": 8, "tenth": 9, "eleventh": 10, "twelfth": 11,
}


def _match_choice_label(text: str, labels: list[str]) -> str | None:
    """Exact label match, case- and "(Recommended)"-insensitive; None when no label matches."""
    from tools.clarify_gateway import _match_label

    return _match_label(text, labels)


def strip_filler(text: str) -> str:
    """Drop the lead/trail filler words STT attaches around a short answer ("option five please")."""
    words = [w for w in text.casefold().replace(",", " ").split() if w not in _INDEX_LEAD_WORDS]
    return " ".join(words)


def is_other_label(label: str) -> bool:
    """The implicit "Other" row the panel appends after the real choices.

    The panel never sends this label as a choice — pressing Enter on it switches to free text and
    submits what was typed — so a spoken "other" must open the same free-text route instead of
    locking the word itself. Matched loosely: the row is rendered with a hint ("Other (type your
    own)"), and STT adds nothing but a leading article.
    """
    words = strip_filler(label).split()
    return bool(words) and words[0] in {"other", "others", "custom"} and len(words) <= 3


def _label_for(clause: str, labels: list[str]) -> str | None:
    """One clause as a real choice label (spoken number or label match), None if neither."""
    index = _spoken_number(clause)
    if index is not None and 0 <= index < len(labels):
        return labels[index].strip()
    return _case_insensitive_match(clause, labels)


def _case_insensitive_match(text: str, labels: list[str]) -> str | None:
    """``"teal"`` -> ``"Teal"``: the gateway matcher alone is exact-case, clauses are not.

    On a choice prompt ``score_answer`` passes the *raw* transcript, so an exact-case match is not
    what a speaker can reliably produce; a clause split purely to pick labels should be lenient.
    """
    wanted = text.strip().strip(" .!?").casefold()
    for label in labels:
        if label.strip().casefold() == wanted:
            return label.strip()
    return None


def _score_multi_clauses(text: str, labels: list[str]):
    """Split a multi-select answer into clauses and score each one.

    Returns ``(value, picks)`` where ``value`` is a JSON array when every clause is a label/number,
    ``OTHER_SENTINEL`` when "other" is the only clause, or ``OTHER_WITH_CHOICES_SENTINEL`` when it
    names options *and* "other" ("teal, other") — ``picks`` then carries those options so the
    follow-up answer merges onto them. None when the clauses are not a clean selection: the caller
    falls through to the single-pass matcher, so prose is never reshaped into picks.
    """
    clauses = split_clauses(text)
    if len(clauses) < 2:
        return None
    picks: list[str] = []
    wants_other = False
    for clause in clauses:
        if is_other_label(clause) or clause.casefold() in _OTHER_WORDS:
            wants_other = True
            continue
        label = _label_for(clause, labels)
        if label is None:
            return None
        if label not in picks:
            picks.append(label)
    if not picks:
        return (OTHER_SENTINEL, []) if wants_other else None
    if wants_other:
        return OTHER_WITH_CHOICES_SENTINEL, picks
    return json.dumps(picks, ensure_ascii=False), []


#: The words that name the implicit "Other" row when spoken as their own clause.
_OTHER_WORDS = frozenset({"other", "others", "custom", "custom option"})


def score_answer(text: str, choices, multi_select: bool, *, awaiting_text: bool, picks: list | None = None):
    """``(value, None)`` or ``(None, reason)`` for one transcript, via the gateway's matcher.

    On a choice prompt only an option, a spoken number or the "Other" row resolves; unmatched words
    are rejected rather than taken as free text, because a mis-heard phrase must never become the
    answer (the user can then say "other" and dictate). A throwaway ``_ClarifyEntry`` is built so
    the remaining shapes are scored by the gateway's typed-answer matcher; nothing is registered
    and no module-level index is touched. ``value`` is a
    choice label, ``OTHER_SENTINEL`` when the words name the implicit "Other" row (the caller then
    takes a follow-up utterance as the free-text answer), a JSON array string for multi-select,
    the raw text in free-text mode, or None for a skip. ``reason`` is ``"invalid_selection"``
    (selection-shaped but unrecognised: say it again) or ``"prose"`` (a real sentence that matched
    nothing).
    """
    from tools.clarify_gateway import _ClarifyEntry, _coerce_text_response_detailed

    labels: list[str] | None = [str(c) for c in choices] if choices else None
    if labels is not None and not awaiting_text:
        # Strict on a choice prompt: only an option, a number or the "Other" row resolves. Anything
        # else is rejected, so a mis-heard phrase can never be locked as an answer of its own.
        # "kiwi please" and "five" both arrive from STT with filler attached; stripping it once lets
        # the label matcher win on a real label and the index reader win on a number.
        cleaned = strip_filler(text) or text
        if multi_select:
            # Split on the raw transcript: strip_filler drops the commas that separate clauses.
            mixed = _score_multi_clauses(text, labels)
            if mixed is not None:
                value, picked = mixed
                if picks is not None:
                    picks.extend(picked)
                return value, None
        if is_other_label(cleaned) or "other" in labels:
            # Say "other" in words, or pick a row literally named "Other": both open free text.
            return OTHER_SENTINEL, None
        spoken_index = _spoken_number(cleaned)
        if not multi_select:
            matched = _match_choice_label(cleaned, labels)
            if matched is not None:
                return matched, None
        if spoken_index is not None and 0 <= spoken_index < len(labels):
            if multi_select:
                return json.dumps([labels[spoken_index].strip()], ensure_ascii=False), None
            return labels[spoken_index].strip(), None
        if multi_select:
            # "1, prod" and "staging, nope" are selections: the gateway parses the list, a bad token
            # rejecting the whole reply, exactly as a typed one.
            entry = _ClarifyEntry(
                clarify_id="spoken-answer", session_key="spoken-answer", question="",
                choices=labels, multi_select=True, awaiting_text=False)
            return _coerce_text_response_detailed(entry, text)
        selection_shaped = _spoken_number(text) is not None or any(c.isdigit() for c in text)
        return None, "invalid_selection" if selection_shaped else "prose"

    entry = _ClarifyEntry(
        clarify_id="spoken-answer",
        session_key="spoken-answer",
        question="",
        choices=labels,
        multi_select=bool(multi_select) and bool(choices) and not awaiting_text,
        awaiting_text=bool(awaiting_text) or not choices)
    return _coerce_text_response_detailed(entry, text)


def is_live(cli) -> bool:
    """Voice mode still on and a panel still open: the only condition the loop keeps running on."""
    return bool(getattr(cli, "_voice_mode", False)) and getattr(cli, "_clarify_state", None) is not None


class SpokenClarifyPanel:
    """One clarify request's voice side: read the panel aloud, capture answers, report them.

    ``on_answer(value, own_text=..., picks=...)`` locks an answer through the panel's own
    ``_clarify_batch_lock``;
    ``on_cancel()`` withdraws the panel and ends voice mode; ``on_reject(reason)`` is told when a
    transcript matched nothing so it can say so once. All three run on the listener thread.
    """

    def __init__(self, cli, *, speak, on_answer, on_cancel, on_reject) -> None:
        self._cli = cli
        self._speak = speak
        self._on_answer = on_answer
        self._on_cancel = on_cancel
        self._on_reject = on_reject
        self._recorder = None
        self._thread: threading.Thread | None = None
        self._spoken: dict[str, str] = {}  # qid -> the sentence already read for it
        self._rejected_qid: str | None = None
        self._mic_claimed = False  # True only while this panel holds the CLI's mic flag
        self._freetext_at_read = False  # the CLI's freetext flag when the current question was read

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Start the listener once; a no-op when voice mode is off or a panel is already closed."""
        from agent.memory_provider import spawn_context_thread

        if self._thread is not None or not is_live(self._cli):
            return
        self._thread = spawn_context_thread(self._run, name="clarify-voice")
        self._thread.start()

    def stop(self) -> None:
        """Hand the mic back; idempotent, called on every panel-teardown path.

        The recorder is cancelled here (cheap) and shut down on a short-lived daemon thread: this
        runs on the agent thread, and a teardown triggered by ``request.cancel`` arrives on the
        gateway connection's thread, neither of which may block on the audio backend.
        """
        from agent.memory_provider import spawn_context_thread

        self._thread = None
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return
        self._teardown_recorder(recorder)
        self._release_claim()
        spawn_context_thread(self._shutdown_recorder, name="clarify-voice-shutdown",
                             args=(recorder,)).start()

    def speaking(self) -> bool:
        """True while the reply/question TTS is still playing (the CLI's own flag)."""
        tts_done = getattr(self._cli, "_voice_tts_done", None)
        return tts_done is not None and not tts_done.is_set()

    # --- the loop ---------------------------------------------------------
    def _run(self) -> None:
        """Thread body: the loop ends on its own; the recorder it still holds is released here."""
        try:
            self._loop()
        except Exception:  # health: allow BLE001 -- a dead listener must never take the panel with it; logged with its traceback
            logger.exception("clarify voice listener crashed; keyboard answering is unaffected")
        finally:
            recorder, self._recorder = self._recorder, None
            if recorder is not None:
                self._shutdown_recorder(recorder)
                self._release_claim()

    def _loop(self) -> None:
        from tools.voice_mode import create_audio_recorder, transcribe_recording
        from tools.voice_mode_transcript import is_voice_stop_phrase

        while is_live(self._cli):
            entry = active_question(self._cli._clarify_state)
            if entry is None or not self._read_question(entry):
                return
            # The read blocks until the question has been heard (or the panel closed): capturing
            # while TTS plays would record Hermes' own words as the answer.
            if not self._wait_for_silence():
                return
            if not self._ensure_recorder(create_audio_recorder):
                return
            transcript = self._one_utterance(transcribe_recording)
            if not transcript:  # None: panel closed / capture failed; "": heard nothing
                continue
            if is_voice_stop_phrase(transcript):
                self._on_cancel()
                return
            live_state = self._cli._clarify_state
            live_entry = active_question(live_state) if live_state else None
            if live_entry is None:
                return
            # Free text is honoured only for an open-ended question, or when "Other" was switched on
            # *during this question* (the keyboard's Other key). A flag left over from an earlier
            # question must not turn the next choice prompt into a free-text prompt.
            flag = bool(getattr(self._cli, "_clarify_freetext", False))
            awaiting = not live_entry["choices"] or (flag and flag != self._freetext_at_read)
            picks: list[str] = []
            value, reason = score_answer(
                transcript,
                live_entry["choices"],
                bool(live_entry["multi_select"]),
                awaiting_text=awaiting,
                picks=picks)
            own_text = False
            if value in (OTHER_SENTINEL, OTHER_WITH_CHOICES_SENTINEL):
                # "teal, other": the labels picked in the same sentence survive the follow-up
                # prompt, or the custom answer replaces them instead of merging (2026-11-14 bug).
                own_text = True
                value = self._ask_own_answer(transcribe_recording, is_voice_stop_phrase)
                if value is None:
                    return
                if value == "":
                    continue  # nothing heard: the panel is still open, ask again on this pass
            if value is not None:
                self._spoken.pop(live_entry["qid"], None)
                self._rejected_qid = None
                self._on_answer(value, own_text=own_text, picks=picks or None)
                continue
            # Not an answer: say so once per question and keep the question open.
            if self._rejected_qid != live_entry["qid"]:
                self._rejected_qid = live_entry["qid"]
                self._on_reject(reason)

    def _ask_own_answer(self, transcribe_recording, is_voice_stop_phrase) -> str | None:
        """The words named the implicit "Other" row: take one more utterance as the answer.

        Returns the typed-answer text (which the caller locks as "Other"), ``""`` when nothing was
        heard so the caller can prompt again, or None when the panel closed or the stop phrase was
        spoken — the same cancel the first utterance gets.
        """
        from agent.i18n import t
        from tools.voice_mode import create_audio_recorder

        self._speak(t("cli.clarify.speech_own_answer"))
        if not self._wait_for_silence() or not self._ensure_recorder(create_audio_recorder):
            return None
        while is_live(self._cli):
            transcript = self._one_utterance(transcribe_recording)
            if transcript is None:
                return None
            if not transcript:
                return ""
            if is_voice_stop_phrase(transcript):
                self._on_cancel()
                return None
            value, reason = score_answer(transcript, None, False, awaiting_text=True)
            if value is not None:
                return value
            self._on_reject(reason)
            return ""
        return None

    # --- speech -----------------------------------------------------------
    def _read_question(self, entry: dict) -> bool:
        """Read ``entry`` aloud unless it was already read; True when the loop may continue."""
        speech = self._speech_for(entry)
        if self._spoken.get(entry["qid"]) == speech:
            return True
        self._spoken[entry["qid"]] = speech
        self._freetext_at_read = bool(getattr(self._cli, "_clarify_freetext", False))
        self._speak(speech)
        return is_live(self._cli)

    def _speech_for(self, entry: dict) -> str:
        from agent.i18n import t

        choices = entry.get("choices") or []
        order = [q.get("qid") for q in self._cli._clarify_state.get("questions") or []]
        position = order.index(entry["qid"]) + 1 if entry["qid"] in order else 1
        speech = render_clarify_speech(
            entry.get("question") or "",
            choices,
            bool(entry.get("multi_select")),
            index=position,
            total=len(order) or 1,
            choice_label=t("cli.clarify.speech_choice"),
            question_label=t("cli.clarify.speech_question"),
            other_label="")  # the cue below comes last, after the options, and is what opens the mic
        if not choices:
            return f"{speech} {t('cli.clarify.speech_open')}"
        if not entry.get("multi_select"):
            # "Other" is reached by saying it, the same words the cue names.
            return f"{speech} {t('cli.clarify.speech_free_text')}"
        return f"{speech} {t('cli.clarify.speech_other')}"

    def _wait_for_silence(self) -> bool:
        """Let a reply that is still playing finish; False if the panel closed meanwhile.

        The CLI's own TTS flag is the signal (``speaking()``): asking the speak callback would
        assume state a bound method does not carry.
        """
        deadline = time.monotonic() + TTS_SETTLE_MAX_SECONDS
        while is_live(self._cli):
            if not self.speaking() or time.monotonic() > deadline:
                return True
            time.sleep(0.1)
        return False

    # --- capture ----------------------------------------------------------
    def _ensure_recorder(self, create_audio_recorder) -> bool:
        """One input stream for the whole panel: reopened only after a failed capture."""
        if self._recorder is not None:
            return True
        with self._cli._voice_lock:
            # Claim the mic flag the way the turn path does, so neither side opens a second stream
            # and the recorder can be handed back on every teardown path.
            if getattr(self._cli, "_voice_recording", False):
                return False  # the turn path owns the mic; try again on the next pass
            for attempt in range(RECORDER_OPEN_ATTEMPTS):
                try:
                    self._recorder = create_audio_recorder()
                    break
                except Exception:  # health: allow BLE001 -- CoreAudio can refuse a stream while the device settles; retried below, then reported
                    if attempt == RECORDER_OPEN_ATTEMPTS - 1:
                        logger.exception("clarify voice: could not open the microphone; keyboard answering still works")
                        return False
                    time.sleep(RECORDER_OPEN_BACKOFF_SECONDS)
            self._cli._voice_recording = True
            self._mic_claimed = True
        return True

    def _release_claim(self) -> None:
        """Hand back the mic flag this panel claimed (idempotent; safe off-thread)."""
        with self._cli._voice_lock:
            if self._mic_claimed:
                self._mic_claimed = False
                self._cli._voice_recording = False

    def _one_utterance(self, transcribe_recording) -> str | None:
        """Record until the silence auto-stop (or the cap), then transcribe.

        Returns the transcript (``""`` when nothing was heard or STT failed, so the loop keeps
        listening) or None when the capture itself failed or the panel closed.
        """
        recorder = self._recorder
        if recorder is None:
            return None
        done = threading.Event()
        # CoreAudio can refuse the stream for a moment while the player releases the device
        # ("PaMacCore (AUHAL) ... err=-50"): retry the start before giving up on the microphone.
        for attempt in range(RECORDER_OPEN_ATTEMPTS):
            try:
                recorder.start(on_silence_stop=done.set)
                break
            except Exception:  # health: allow BLE001 -- a transient device error is retried below; the last one is logged
                if attempt == RECORDER_OPEN_ATTEMPTS - 1:
                    logger.exception("clarify voice: capture start failed after retries; keyboard answering still works")
                    self._drop_recorder()
                    return ""
                done.clear()
                time.sleep(RECORDER_OPEN_BACKOFF_SECONDS)
        else:  # pragma: no cover - the loop always breaks or returns
            return ""
        started = time.monotonic()
        while not done.wait(0.2):
            if not is_live(self._cli) or time.monotonic() - started > ANSWER_CAPTURE_MAX_SECONDS:
                break
        try:
            wav_path = recorder.stop()
        except Exception:  # health: allow BLE001 -- a failing stop leaves no audio to score; recorder dropped below
            logger.exception("clarify voice: capture stop failed; microphone answering disabled for this question")
            self._drop_recorder()
            return None
        if not is_live(self._cli):
            return None
        if not wav_path:
            return ""
        result = transcribe_recording(wav_path, model=self._cli._voice_stt_model())
        if not isinstance(result, dict) or not result.get("success"):
            return ""
        return (result.get("transcript") or "").strip()

    def _drop_recorder(self) -> None:
        """Release a recorder whose stream failed, so the next pass builds a fresh one."""
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            self._shutdown_recorder(recorder)
            self._release_claim()

    @staticmethod
    def _teardown_recorder(recorder) -> None:
        """Stop capture now so the mic is free; a failing cancel is logged, never raised."""
        try:
            recorder.cancel()
        except Exception:  # health: allow BLE001 -- a recorder already stopped or shut down refuses cancel; logged below
            logger.debug("clarify voice: recorder cancel failed at teardown", exc_info=True)

    @staticmethod
    def _shutdown_recorder(recorder) -> None:
        """Release the audio stream; the backend may already be gone, which is not a failure."""
        try:
            recorder.shutdown()
        except Exception:  # health: allow BLE001 -- teardown of a dead device must not raise; logged below
            logger.debug("clarify voice: recorder shutdown failed", exc_info=True)
