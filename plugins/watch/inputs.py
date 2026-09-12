"""The input track — what the user pressed, and when.

This is the highest-resolution signal in the whole plugin and the cheapest. A
provider samples video at 1 fps, so a frame CANNOT resolve a 1.5 s game cooldown
or a sixteenth note; a keypress timestamp at millisecond resolution can. For
"how was my rotation" or "was I on the beat", the input track is the primary
evidence and the frames are context. Encoded compactly it also costs ~5 tokens
an event against 66-258 for a frame.

It is also a keylogger, so the privacy model is the design rather than a
disclaimer. Four hard rules, each enforced here and covered by a test:

1. ALLOWLIST, never denylist. Only symbols on an explicit list are recorded.
   The default list is game/instrument shaped — digits, function keys,
   modifiers, mouse buttons — and contains NO letters, so ordinary typing
   cannot be reconstructed even from a capture that ran by mistake.
2. FOCUS-GATED. Events are dropped unless the watched app is frontmost. Alt-tab
   to a password manager and recording stops on its own; this is the strongest
   control here because it needs no per-key judgment.
3. TEXT-MODE SUPPRESSION. Opening a chat/console (Enter, ``/``, ``T``) mutes the
   track until it closes. In a game the chat box is where the private text is,
   and it is entered through keys we can see.
4. NO CONTENT, EVER. An event carries a symbol and a time. There is no
   character buffer, no word assembly, and nothing here can produce a string
   the user typed.

Pure functions over event lists — no hooks, no OS calls. The platform capture
lives in a provider (see ``recorder``), which keeps these rules testable without
a keyboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type hints only
    from plugins.watch.signals import Signal

# ── Allowlists ────────────────────────────────────────────────────────────
#
# Deliberately letter-free. Action bars, transport controls and instrument pads
# live on digits and function keys; letters are where sentences live.

DIGIT_KEYS = frozenset("1234567890-=")
FUNCTION_KEYS = frozenset(f"f{i}" for i in range(1, 13))
MODIFIER_KEYS = frozenset({"shift", "ctrl", "alt"})
MOUSE_BUTTONS = frozenset({"m1", "m2", "m3", "m4", "m5", "wheelup", "wheeldown"})
CONTROL_KEYS = frozenset({"tab", "space", "esc"})

#: The default: everything an action bar or a transport uses, nothing that
#: spells. Modifier+digit combos ("shift+3") are allowed as composed symbols.
GAME_KEYS = DIGIT_KEYS | FUNCTION_KEYS | MODIFIER_KEYS | MOUSE_BUTTONS | CONTROL_KEYS

#: Opt-in only, and aggregated rather than logged per-press (see
#: ``summarize_inputs``). WASD is four letters — recording them individually is
#: how a keylogger starts, so the track reports "moved" not "w,a,s,d".
MOVEMENT_KEYS = frozenset({"w", "a", "s", "d"})

#: Keys that OPEN a text field in most games. Everything after one of these is
#: assumed private until the field closes.
TEXT_MODE_OPENERS = frozenset({"enter", "return", "/", "t"})
TEXT_MODE_CLOSERS = frozenset({"enter", "return", "esc"})


@dataclass(frozen=True)
class InputEvent:
    """One press.

    Attributes:
        at: Seconds since recording started, ideally sub-second — the whole
            point of this track is resolution the frames don't have.
        symbol: An allowlisted key name, possibly modifier-composed
            (``"shift+3"``). Never a character the user typed into a field.
        app: The frontmost app when it happened, for focus gating.
    """

    at: float
    symbol: str
    app: str = ""


def compose(symbol: str, modifiers: Iterable[str] = ()) -> str:
    """Canonical ``mod+mod+key`` symbol, modifiers in a stable order.

    Stable order matters: ``ctrl+shift+3`` and ``shift+ctrl+3`` are the same
    keybind, and if they encode differently the model sees two abilities where
    the user has one.
    """
    order = [m for m in ("ctrl", "alt", "shift") if m in set(modifiers)]
    return "+".join(order + [symbol])


def is_allowed(symbol: str, *, allow_movement: bool = False) -> bool:
    """Whether a symbol may be recorded at all.

    The base key is checked against the allowlist AFTER stripping modifiers, so
    ``shift+q`` is rejected for the same reason ``q`` is.
    """
    base = symbol.rsplit("+", 1)[-1].lower()
    if base in GAME_KEYS:
        return True
    if allow_movement and base in MOVEMENT_KEYS:
        return True
    return False


def filter_events(
    events: Iterable[InputEvent],
    *,
    watched_app: Optional[str] = None,
    allow_movement: bool = False,
) -> list[InputEvent]:
    """Apply every privacy rule, in order, and return what survives.

    Order is deliberate: focus gating first (cheapest and broadest), then text
    mode (a stateful sweep), then the allowlist (per-symbol). A dropped event is
    dropped silently — there is no "rejected events" list, because keeping one
    would be a record of the keys we promised not to record.
    """
    kept: list[InputEvent] = []
    in_text_mode = False

    for event in sorted(events, key=lambda e: e.at):
        symbol = event.symbol.lower()
        base = symbol.rsplit("+", 1)[-1]

        # 2. Focus gate. An unnamed watched_app means "record anywhere", which
        #    is only ever set explicitly by a caller that knows what it wants.
        if watched_app is not None and event.app and event.app != watched_app:
            # Leaving the app also closes any text field we thought was open —
            # the next return to it starts clean rather than muted forever.
            in_text_mode = False
            continue

        # 3. Text-mode suppression. Closers are evaluated BEFORE openers so
        #    Enter (which is both) ends a field rather than immediately
        #    reopening one.
        if in_text_mode:
            if base in TEXT_MODE_CLOSERS:
                in_text_mode = False
            continue

        if base in TEXT_MODE_OPENERS:
            in_text_mode = True
            continue

        # 1. Allowlist.
        if not is_allowed(symbol, allow_movement=allow_movement):
            continue

        kept.append(event)

    return kept


# ── Encoding for the model ────────────────────────────────────────────────

def encode_events(
    events: Iterable[InputEvent],
    *,
    since: float = 0.0,
    collapse_repeats: bool = True,
) -> str:
    """Compact ``symbol@offset`` line for a rolling window.

    Offsets are relative to ``since`` so a 10-second window reads ``3@0.0
    1@1.5`` rather than ``3@1841.2`` — the model is being asked about the last
    few seconds, and absolute timestamps there are noise it has to subtract.

    Repeats collapse to ``1@0.0x3``: spamming one ability nine times is one fact
    about the player, not nine, and nine rows crowd out the rest of the window.
    """
    rows: list[str] = []
    ordered = sorted(events, key=lambda e: e.at)

    for event in ordered:
        stamp = f"{event.symbol}@{event.at - since:.1f}"
        if collapse_repeats and rows:
            previous, _, count = rows[-1].partition("x")
            prev_symbol = previous.split("@")[0]
            if prev_symbol == event.symbol:
                rows[-1] = f"{previous}x{(int(count) if count else 1) + 1}"
                continue
        rows.append(stamp)

    return " ".join(rows)


def gaps(events: Iterable[InputEvent]) -> list[float]:
    """Inter-press intervals — the actual signal for rotation or rhythm.

    A game's cooldown floor and a musician's tempo are both "how long between
    actions", which is exactly what 1 fps sampling destroys and this preserves.
    """
    times = sorted(e.at for e in events)
    return [round(b - a, 3) for a, b in zip(times, times[1:])]


def summarize_inputs(
    events: Iterable[InputEvent],
    duration_s: float,
    *,
    movement_keys: Iterable[str] = MOVEMENT_KEYS,
) -> dict:
    """Derived stats, which are often more useful to the decider than raw rows.

    "APM 42, longest idle 3.2 s, most-used shift+3" is a handful of tokens and
    answers "was anything happening" directly, where the raw window makes the
    model count. Movement is reported as a COUNT only — never as which
    direction — so the opt-in movement keys can inform pacing without becoming
    a trace of where the user went.
    """
    kept = list(events)
    move_set = {m.lower() for m in movement_keys}

    actions = [e for e in kept if e.symbol.rsplit("+", 1)[-1].lower() not in move_set]
    moves = len(kept) - len(actions)

    counts: dict[str, int] = {}
    for event in actions:
        counts[event.symbol] = counts.get(event.symbol, 0) + 1

    intervals = gaps(actions)
    minutes = duration_s / 60 if duration_s > 0 else 0

    return {
        "actions": len(actions),
        "apm": round(len(actions) / minutes, 1) if minutes else 0.0,
        "movement_events": moves,
        "longest_idle": max(intervals) if intervals else 0.0,
        "median_gap": sorted(intervals)[len(intervals) // 2] if intervals else 0.0,
        "top": sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5],
    }


def window(
    events: Iterable[InputEvent],
    now: float,
    span: float,
) -> list[InputEvent]:
    """Events within the last ``span`` seconds — the decider's rolling view."""
    return [e for e in events if now - span <= e.at <= now]


def activity_since(events: Iterable[InputEvent], since: float) -> int:
    """How many presses happened after ``since``.

    The free half of the gate: no keys and no pixel change means nothing
    happened, and nothing happened needs no model call.
    """
    return sum(1 for e in events if e.at > since)


def key_mix(events: Iterable[InputEvent]) -> dict[str, float]:
    """Normalized distribution of symbols — what the player is doing, as a shape."""
    counts: dict[str, float] = {}
    total = 0
    for event in events:
        counts[event.symbol] = counts.get(event.symbol, 0.0) + 1
        total += 1
    if not total:
        return {}
    return {symbol: count / total for symbol, count in counts.items()}


def novelty_score(recent: Iterable[InputEvent], baseline: Iterable[InputEvent]) -> float:
    """0..1: how much has behaviour CHANGED, versus merely continued.

    This is the gate that matters under load, and the distinction the first
    implementation got wrong. Raw activity is useless in a game because the
    steady state IS activity — a keypress every 1.5 s for an hour. Asking "is
    something happening" answers yes on every tick and the loop spends money
    every tick (measured: 93% call rate). The question worth asking is whether
    what they are doing now differs from what they were doing.

    Two components, maxed rather than averaged so either kind of change is
    enough to wake the decider:

    * **mix shift** — total-variation distance between the recent key
      distribution and the baseline's. Swapping from one rotation to another,
      or from abilities to movement, scores high.
    * **rate shift** — relative change in presses per second. A burst or a
      sudden stop both matter; a stop is often the interesting one (died,
      stopped playing, lost the beat).

    An empty baseline scores 1.0: the first window of a session is entirely new.
    """
    recent_list, baseline_list = list(recent), list(baseline)
    if not baseline_list:
        return 1.0
    if not recent_list:
        return 1.0 if baseline_list else 0.0

    recent_mix, baseline_mix = key_mix(recent_list), key_mix(baseline_list)
    symbols = set(recent_mix) | set(baseline_mix)
    # Total variation distance: half the L1 gap between two distributions,
    # which lands in 0..1 and needs no tuning constant.
    mix_shift = sum(
        abs(recent_mix.get(s, 0.0) - baseline_mix.get(s, 0.0)) for s in symbols
    ) / 2

    recent_span = max(e.at for e in recent_list) - min(e.at for e in recent_list)
    baseline_span = max(e.at for e in baseline_list) - min(e.at for e in baseline_list)
    recent_rate = len(recent_list) / recent_span if recent_span > 0 else float(len(recent_list))
    baseline_rate = (
        len(baseline_list) / baseline_span if baseline_span > 0 else float(len(baseline_list))
    )
    if baseline_rate <= 0:
        rate_shift = 1.0
    else:
        rate_shift = min(1.0, abs(recent_rate - baseline_rate) / baseline_rate)

    return round(max(mix_shift, rate_shift), 3)


def render_input_block(
    events: Iterable[InputEvent],
    *,
    since: float = 0.0,
    duration_s: float = 0.0,
    speed: float = 1.0,
) -> str:
    """The text block handed to the model alongside the frames.

    ``speed`` divides the offsets for the same reason the window timeline's
    does: on a retimed clip, clip time is the only clock the model can check
    against what it sees.
    """
    kept = list(events)
    if not kept:
        return ""

    scaled = [InputEvent(at=e.at / speed, symbol=e.symbol, app=e.app) for e in kept]
    stats = summarize_inputs(kept, duration_s or 0.0)
    top = ", ".join(f"{sym}x{n}" for sym, n in stats["top"]) or "none"

    return (
        "Input track (clip time, key@seconds):\n"
        f"{encode_events(scaled, since=since / speed if speed else since)}\n"
        f"APM {stats['apm']}, longest idle {stats['longest_idle']:.1f}s, "
        f"median gap {stats['median_gap']:.2f}s, most used: {top}"
    )


# ── The signal contract ───────────────────────────────────────────────────

#: Salience calibration for the input track. ``novelty_score`` is a
#: distribution/rate distance: steady rotation play sits under 0.15, a genuine
#: switch of abilities or a rate burst runs 0.4+. The floor sits above the
#: wobble that ordinary play produces so a steady rotation never wakes the loop.
SALIENCE_FLOOR = 0.18
SALIENCE_CEILING = 0.55

#: A lull this long means the person has stopped acting — a safe moment to
#: speak. Shorter than a musical rest because a game's pauses are shorter and
#: an interruption costs less.
QUIET_SECONDS = 2.5


def keys_signal(
    events: Iterable[InputEvent],
    *,
    now: float,
    window_seconds: float = 10.0,
    baseline_seconds: float = 30.0,
    duration_s: float = 0.0,
) -> "Signal":
    """This track's opinion about the current moment, as a calibrated Signal.

    Enrichment, not the mechanism: the screen already shows what the person is
    doing. What this adds is RESOLUTION — a 1 fps frame cannot resolve a 1.5 s
    cooldown or a sixteenth note, and a keypress timestamp can.
    """
    from plugins.watch.signals import Signal, calibrate

    kept = list(events)
    recent = window(kept, now=now, span=window_seconds)
    baseline = [
        e for e in kept if now - baseline_seconds <= e.at < now - window_seconds
    ]
    raw = novelty_score(recent, baseline)
    quiet = activity_since(kept, now - QUIET_SECONDS) == 0

    return Signal(
        name="keys",
        salience=calibrate(raw, SALIENCE_FLOOR, SALIENCE_CEILING),
        block=render_input_block(recent, since=now - window_seconds, duration_s=duration_s),
        can_speak=quiet,
        detail={"raw_novelty": raw, "events": len(recent)},
    )
