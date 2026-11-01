"""The decider — speak, or say nothing.

The whole live feature is this one decision, and the research says the hard part
is not perception but restraint. MMDuet2's RL reward (Apache-2.0,
``rl/verl/recipe/proactive/reward_function.py``) weights
``REPETITION_WEIGHT=3`` equal to ``PAUC_WEIGHT=3`` for correctness, and
penalizes speaking-when-nothing-was-happening at ``OUTSPAN_WEIGHT=2``. So a
coach that is right but repetitive scores no better than one that is wrong.
Those weights are the design brief for this module.

Three stages, cheapest first, and only the last one costs money:

1. **Novelty** — free, local. No keypresses and no pixel change means nothing
   happened. Nothing happened needs no model call.
2. **Refractory** — free, local. Having just spoken, the bar rises for a while.
   Hysteresis in the same shape ``hud-game-overlay.ts`` already uses for the
   game-overlay treatment: entering needs a high bar, staying needs less.
3. **Ask** — one call, ``NO REPLY`` or a sentence, with the last utterances in
   the prompt so the model can see what it already said.

Then the candidate is checked for repetition against recent utterances before it
reaches the user: near-duplicate text is dropped even though the model chose to
speak, because "watch your timing" every four seconds is worse than silence.

The prompt contract is MMDuet2's, adapted: the model outputs the literal string
``NO REPLY`` when it has nothing to add. That needs no trained head, no local
weights, and works against any chat-completions endpoint.

Pure functions plus one dataclass of state. The LLM call is injected, so the
entire policy is testable without a network.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from plugins.watch.signals import Signal, blocks, can_speak, salience_of, summarize

#: The literal the model emits to stay silent. Uppercase and exact so a
#: near-miss ("no reply.") is caught by `is_silence` rather than shipped to the
#: user as commentary.
NO_REPLY = "NO REPLY"

#: How many past utterances the model sees. Enough to recognize its own recent
#: points; small enough that the prompt stays cheap. MMDuet2 compares against
#: all previous replies in the span; a rolling K is the streaming equivalent.
DEFAULT_HISTORY = 6

#: Seconds of enforced quiet after speaking. Not a rate limit for cost — it is
#: the difference between a coach and a nag.
DEFAULT_REFRACTORY = 12.0

#: Minimum seconds between MODEL CALLS, whatever the outcome. Separate from the
#: refractory window because the expensive failure is not talking too much, it
#: is *asking* too much: a call answered with silence or a repeat still covers
#: the seconds around it, and re-asking a second later pays again for the same
#: answer. Measured before this existed: 561 calls to produce 4 utterances.
DEFAULT_CALL_COOLDOWN = 8.0

#: How long a held utterance waits for a safe moment before being delivered
#: anyway. The hold gate exists so feedback lands in a gap rather than over the
#: top of a performance — but a gap is not guaranteed to arrive. Continuous
#: combat has no lull; a player looping a passage may not rest for minutes.
#: Measured without this: 75 held utterances and ZERO delivered across a
#: ten-minute session, a permanent mute that looks exactly like a working quiet
#: loop. Interrupting slightly is strictly better than never speaking.
HOLD_TIMEOUT = 20.0

#: Salience below this is not worth a model call. ONE threshold, on ONE scale,
#: because every source calibrates itself before reporting (see
#: ``signals.calibrate``). This replaced a per-source threshold the caller had
#: to match by hand — the arrangement that produced 159 gated filter sweeps
#: because frame change and keypress novelty were different units wearing the
#: same name.
MIN_SALIENCE = 0.25

#: Similarity above which a new utterance counts as a repeat of a recent one.
#: 0.72 catches rephrasings ("watch your timing" / "your timing is off") while
#: letting genuinely new notes through.
DEFAULT_SIMILARITY = 0.72

#: MMDuet2 treats a shared opening of >= 50 chars as repetition outright
#: (`COMMON_PREFIX_THRES`). Cheap, and catches the failure where a model starts
#: every line the same way.
COMMON_PREFIX_CHARS = 50


def is_silence(text: str) -> bool:
    """Whether the model chose not to speak.

    Generous on purpose: a model that emits ``no reply``, wraps it in quotes, or
    adds a period has still said it has nothing to add, and shipping that string
    to the user as if it were feedback is the worst possible outcome. An empty
    or whitespace answer counts as silence too.
    """
    stripped = re.sub(r"[^a-z ]", "", (text or "").strip().lower()).strip()
    return stripped in {"", "no reply", "noreply", "no reply needed"}


def normalize(text: str) -> str:
    """Lowercased, punctuation-free form used for similarity only."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split())


def similarity(a: str, b: str) -> float:
    """0..1 similarity between two utterances.

    ``difflib`` rather than embeddings: this runs on every candidate, must not
    add latency to a live loop, and needs no model. It catches rephrasing, which
    is the actual failure mode; true paraphrase detection would want the
    LLM-judge prompt MMDuet2 uses for offline scoring.
    """
    left, right = normalize(a), normalize(b)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def is_repetition(
    candidate: str,
    history: list[str],
    *,
    threshold: float = DEFAULT_SIMILARITY,
    prefix_chars: int = COMMON_PREFIX_CHARS,
) -> bool:
    """Whether ``candidate`` is something we effectively already said."""
    for previous in history:
        if similarity(candidate, previous) >= threshold:
            return True
        # MMDuet2's cheap structural check, on normalized text so casing and
        # punctuation can't defeat it.
        shared = _common_prefix(normalize(candidate), normalize(previous))
        if len(shared) >= prefix_chars:
            return True
    return False


def _common_prefix(a: str, b: str) -> str:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return a[:index]


# ── Prompting ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are watching a live session and giving the person short, specific "
    "coaching. You see recent frames, an input track (keys with timestamps), "
    "and what you have already said.\n"
    "\n"
    "Your responses should cover only what has happened SINCE YOUR LAST REPLY. "
    'If there is nothing new and worth saying right now, output exactly "NO REPLY".\n'
    "\n"
    "Rules:\n"
    "- Prefer silence. Most moments do not need a comment.\n"
    '- Never repeat a point you have already made. Output "NO REPLY" instead.\n'
    "- One sentence. Name the specific thing you saw, not general advice.\n"
    "- The input track is higher resolution than the frames: use it for timing, "
    "rhythm, and rotation questions.\n"
    "- If the person asked you to watch for something specific, only speak "
    "about that."
)


def build_decider_prompt(
    brief: str,
    *,
    input_block: str = "",
    window_seconds: float = 10.0,
    recent_utterances: Optional[list[str]] = None,
    elapsed: float = 0.0,
) -> str:
    """The text half of one decision.

    The brief goes LAST rather than first: it is the instruction the model
    should be holding when it answers, and burying it above the evidence is how
    a decider drifts back to generic commentary.
    """
    parts = [f"Session time: {elapsed:.0f}s. Window: last {window_seconds:.0f}s."]

    if input_block:
        parts.append(input_block)

    said = recent_utterances or []
    if said:
        parts.append(
            "You have already said:\n" + "\n".join(f"- {line}" for line in said)
        )
    else:
        parts.append("You have not said anything yet.")

    parts.append(
        f"Watch for: {brief.strip()}" if brief.strip()
        else "Watch for anything notable about how they are performing."
    )
    parts.append('Reply with one sentence, or exactly "NO REPLY".')

    return "\n\n".join(parts)


# ── Policy ────────────────────────────────────────────────────────────────

@dataclass
class DeciderState:
    """Rolling state for one live session."""

    history: list[str] = field(default_factory=list)
    last_spoke_at: Optional[float] = None
    #: When the model was last CALLED, regardless of outcome. Distinct from
    #: ``last_spoke_at`` and load-bearing: a call that produced silence or a
    #: repeat still established that this moment is covered, so re-asking one
    #: second later buys nothing. Without this the loop re-asks every tick
    #: forever — measured at 561 calls for 4 utterances over ten minutes, a 93%
    #: call rate, because suppressed repeats left no trace to back off from.
    last_call_at: Optional[float] = None
    #: An utterance produced during a bad moment to interrupt, waiting for a
    #: gap. Only ever one: if a second observation arrives before the first is
    #: delivered, the newer one is what matters — stale feedback about a passage
    #: two phrases ago is worse than none.
    pending: Optional[str] = None
    #: When the pending utterance started waiting, so the hold can time out
    #: rather than becoming a permanent mute in a session that never goes quiet.
    pending_since: Optional[float] = None
    #: Every decision, for offline replay. The reason a live coach can be tuned
    #: at all: record the trace once, re-run policies against it forever.
    log: list[dict] = field(default_factory=list)

    def remember(self, text: str, at: float, limit: int = DEFAULT_HISTORY) -> None:
        self.history.append(text)
        del self.history[:-limit]
        self.last_spoke_at = at

    def note_call(self, at: float) -> None:
        self.last_call_at = at


@dataclass(frozen=True)
class Decision:
    """What the loop concluded for one tick."""

    spoke: bool
    #: 'novelty' | 'unchanged' | 'refractory' | 'call_cooldown' | 'held'
    #: | 'model_silent' | 'repetition' | 'spoke'
    reason: str
    at: float
    text: str = ""
    #: True when a model call was made — the only expensive outcome.
    called_model: bool = False
    #: Set when the utterance was produced but withheld for a safe moment.
    #: The caller delivers it when ``can_speak_now`` turns true; dropping it
    #: instead would mean the one useful observation is the one nobody hears.
    deferred: bool = False


@dataclass(frozen=True)
class Policy:
    """The tuning knobs, in one object.

    Collected here because ``decide`` had grown to fourteen parameters, which is
    a function that has not found its abstraction. These are the numbers a
    replay sweeps to tune a session; everything else ``decide`` needs is either
    state or a signal.
    """

    refractory: float = DEFAULT_REFRACTORY
    call_cooldown: float = DEFAULT_CALL_COOLDOWN
    hold_timeout: float = HOLD_TIMEOUT
    min_salience: float = MIN_SALIENCE
    similarity: float = DEFAULT_SIMILARITY
    history: int = DEFAULT_HISTORY
    window_seconds: float = 10.0


def decide(
    state: DeciderState,
    signals: Sequence[Signal],
    *,
    at: float,
    brief: str,
    ask: Optional[Callable[[str, str], str]] = None,
    policy: Optional[Policy] = None,
) -> Decision:
    """Run one tick and return what happened.

    Takes SIGNALS, not raw measurements. Each source calibrated its own
    salience onto 0..1 before arriving here, so this function compares one
    number against one threshold and knows nothing about Hamming distances, key
    distributions or BPM. Adding a source later cannot break the gate by
    arriving in different units.

    ``ask(system_prompt, user_prompt) -> str`` is injected; in production it is
    an ``auxiliary_client`` call pinned to a vision-capable model, and in tests
    a stub.

    Four free gates then one paid call. The free ones exist because of measured
    failures, not caution:

    * **salience** — a simulated WoW session called the model on 90% of ticks
      when the gate was "is something happening". In a game something is always
      happening; only CHANGE is news.
    * **refractory** — having just spoken, stay quiet.
    * **call cooldown** — a call answered "nothing to add" covers the seconds
      after it. Without this, suppressed repeats left no trace and the loop
      re-asked every tick: 550 calls to ship 5 utterances.
    * **hold** — an utterance produced mid-phrase is kept, not dropped. Talking
      over a musician ruins the take being judged.
    """
    rules = policy or Policy()

    def record(decision: Decision) -> Decision:
        state.log.append(
            {
                "at": round(decision.at, 2),
                "reason": decision.reason,
                "spoke": decision.spoke,
                "called_model": decision.called_model,
                "text": decision.text,
                "deferred": decision.deferred,
                "signals": summarize(signals),
            }
        )
        return decision

    safe = can_speak(signals)

    # 0. A held utterance outranks everything: already paid for, already checked
    #    for repetition, only waiting for a gap. It is delivered when a gap
    #    arrives OR when it has waited long enough — a session that never goes
    #    quiet (continuous combat, a looped passage) otherwise holds forever,
    #    which reads as a working quiet loop and is actually a permanent mute.
    if state.pending is not None:
        waited = (
            at - state.pending_since if state.pending_since is not None else 0.0
        )
        if safe or waited >= rules.hold_timeout:
            text = state.pending
            state.pending = None
            state.pending_since = None
            state.remember(text, at, limit=rules.history)
            return record(
                Decision(spoke=True, reason="spoke", at=at, text=text, deferred=True)
            )

    # 1. Salience — free, and the gate that matters under load.
    score = salience_of(signals)
    if score < rules.min_salience:
        return record(Decision(spoke=False, reason="quiet", at=at))

    # 2. Refractory — free.
    if state.last_spoke_at is not None and at - state.last_spoke_at < rules.refractory:
        return record(Decision(spoke=False, reason="refractory", at=at))

    # 3. Call cooldown — free.
    if state.last_call_at is not None and at - state.last_call_at < rules.call_cooldown:
        return record(Decision(spoke=False, reason="call_cooldown", at=at))

    if ask is None:
        return record(Decision(spoke=False, reason="quiet", at=at))

    # 4. Ask.
    prompt = build_decider_prompt(
        brief,
        input_block=blocks(signals),
        window_seconds=rules.window_seconds,
        recent_utterances=list(state.history),
        elapsed=at,
    )
    state.note_call(at)
    answer = (ask(SYSTEM_PROMPT, prompt) or "").strip()

    if is_silence(answer):
        return record(
            Decision(spoke=False, reason="model_silent", at=at, called_model=True)
        )

    # 5. Repetition — after the call, because the model can choose to speak and
    #    still be repeating itself. MMDuet2 weights this as heavily as being
    #    right, so a duplicate is suppressed rather than delivered.
    if is_repetition(answer, state.history, threshold=rules.similarity):
        return record(
            Decision(
                spoke=False,
                reason="repetition",
                at=at,
                text=answer,
                called_model=True,
            )
        )

    # 6. Hold, if this is a bad moment to interrupt. Checked LAST so the
    #    expensive work is not wasted.
    if not safe:
        # Replacing the TEXT is right — newer feedback beats stale — but the
        # wait clock must not restart, or a session that never goes quiet keeps
        # refreshing its own deadline and the hold never times out. That was the
        # second half of the permanent-mute bug: 75 held, 0 delivered, because
        # every tick pushed the deadline back a tick.
        state.pending = answer
        if state.pending_since is None:
            state.pending_since = at
        return record(
            Decision(
                spoke=False,
                reason="held",
                at=at,
                text=answer,
                called_model=True,
                deferred=True,
            )
        )

    state.remember(answer, at, limit=rules.history)
    return record(
        Decision(spoke=True, reason="spoke", at=at, text=answer, called_model=True)
    )


def replay_stats(log: list[dict]) -> dict:
    """Aggregate a decision log — the readout for tuning a brief offline."""
    total = len(log)
    spoke = sum(1 for row in log if row["spoke"])
    calls = sum(1 for row in log if row["called_model"])
    reasons: dict[str, int] = {}
    for row in log:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1

    return {
        "ticks": total,
        "spoke": spoke,
        "model_calls": calls,
        "call_rate": round(calls / total, 3) if total else 0.0,
        "speak_rate": round(spoke / total, 3) if total else 0.0,
        "suppressed_as_repetition": reasons.get("repetition", 0),
        "reasons": reasons,
    }
