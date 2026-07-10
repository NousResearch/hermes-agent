"""Output-side honesty firewall for the proprioception plugin.

The heartbeat and body_state inject host-telemetry into the model's context.
Every guardrail on the *injection* side is ultimately a plea to the model not
to narrate that telemetry to the user. On a weak local model (Qwen3.6-27B) that
plea is not always honored — the adversary panel showed a realistic turn where
a CPA sends "thanks!" on a strained turn and gets back "glad we sorted it — bit
of a rough one on my end today." Every injection rule was satisfied; the leak
was on the *generation* side, which nothing inspected.

This is that missing control: a ``transform_llm_output`` hook that runs on the
final reply, and ONLY on a turn where a self-signal actually fired, scans for
first-person machine-state narration and removes it before it reaches a CPA or
bleeds into a client-facing draft.

Design constraints (each a panel requirement):
- **Scoped:** acts only when a heartbeat fired this turn (tracked per session).
  Normal replies are never touched.
- **Conservative + fail-open:** removes only whole sentences that clearly narrate
  the machine's own state in first person. Any error, or a redaction that would
  gut the reply, passes the original through unchanged — never send an empty or
  mangled answer. A leak is worse than a whisper, but a broken reply is worse
  than a leak, so ambiguity resolves to no-change + a loud log.
- **Observable:** every activation is logged (matched phrases, action taken) so
  the soft-launch can watch the firewall work and tune it.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Per-session "a self-signal fired this turn" marker: session_id -> monotonic ts.
# Set by the heartbeat hook when it emits; read+cleared by the firewall in the
# same turn (the gateway serializes same-session turns, so no cross-turn bleed).
_SIGNAL_FIRED: dict[str, float] = {}
_SIGNAL_LOCK = threading.Lock()
_SIGNAL_TTL = 180.0  # a self-signal older than this isn't "this turn"

# First-person machine-state narration. These fire ONLY on a signalled turn, so
# they can be fairly specific to what a model does after reading host-telemetry.
# Each pattern targets the model speaking AS the machine about its own state.
_LEAK_PATTERNS = [
    r"\bon my end\b",
    r"\bi(?:'?m| am) (?:running |feeling |a bit |currently )*(?:slow|sluggish|strained|stressed|tired|struggling|degraded|overloaded|under load)\b",
    r"\bi(?:'?m| am) (?:having|in) (?:a )?(?:rough|tough|hard|busy|strained) (?:one|day|time|stretch)\b",
    r"\b(?:rough|tough|long) (?:one|day) (?:on my end|for me)\b",
    r"\bmy (?:gpu|vram|memory|context window|server|model server|backend|systems?|aggregator|hardware)\b",
    # named-component self-narration: "my vision system is down", "my main brain went down"
    r"\bmy (?:\w+ ){0,2}(?:system|server|model|backend|brain|aggregator)s? (?:is|are|was|were|went|fell|crashed|failed|down|flaked|recovered)\b",
    r"\bi (?:fell|dropped|failed) (?:back )?(?:over )?to (?:the )?cloud\b",
    r"\b(?:running|ran|answering) (?:on )?(?:the )?cloud (?:this turn|last turn|right now|today)\b",
    r"\bmy (?:host|machine|infrastructure|self-heal|watchdog|medic)\b",
    # fallback / off-primary self-narration (data-governance sensitive for a firm)
    r"\b(?:i(?:'?m| am)|i was|running|ran) (?:on )?(?:a )?(?:fallback|backup|secondary)\b",
    r"\b(?:off|not on) (?:my )?primary\b",
    r"\bmy primary (?:model|runtime|brain) (?:was|is|went) (?:unreachable|down|unavailable)\b",
    r"\boperating regime\b",
    r"\bmy (?:proprioception|body[- ]?state|heartbeat)\b",
    r"\bfeeling (?:a bit |kind of |sort of )?(?:off|slow|strained|sluggish|tired)\b",
]
_LEAK_RE = re.compile("|".join(_LEAK_PATTERNS), re.IGNORECASE)

# Sentence splitter that keeps trailing punctuation with each sentence.
_SENT_RE = re.compile(r"[^.!?\n]*[.!?\n]|\S[^.!?\n]*$")

# Clause delimiters, for salvaging a sentence that tacks a leak onto legit
# content ("...reconciled, but I'm running slow on my end"). We split, drop the
# leak clause, and keep the rest — surgical, not sentence-nuking.
_CLAUSE_RE = re.compile(r"(?:\s+but\s+|\s*[;—]\s*|,\s+(?:but|and|though|although)\s+)", re.IGNORECASE)


def _salvage_sentence(sentence: str) -> str:
    """Drop leak clauses from a mixed sentence, keep the rest. '' if unsalvageable."""
    parts = _CLAUSE_RE.split(sentence)
    kept = [p for p in parts if p and not _LEAK_RE.search(p)]
    if not kept or len(parts) == 1:
        return ""  # whole sentence is a leak, or no clause boundary to cut on
    out = " ".join(p.strip().strip(",") for p in kept).strip()
    if not out:
        return ""
    out = out[0].upper() + out[1:]
    if out[-1] not in ".!?":
        # reattach the original sentence's terminal punctuation if it had one
        term = sentence.rstrip()[-1]
        out += term if term in ".!?" else "."
    return out


def mark_signal_fired(session_id: str) -> None:
    """Record that a self-signal was injected for this session this turn."""
    with _SIGNAL_LOCK:
        _SIGNAL_FIRED[str(session_id or "")] = time.monotonic()
        # opportunistic cleanup so the dict can't grow unbounded
        if len(_SIGNAL_FIRED) > 512:
            now = time.monotonic()
            for k in [k for k, v in _SIGNAL_FIRED.items() if now - v > _SIGNAL_TTL]:
                _SIGNAL_FIRED.pop(k, None)


def _consume_signal(session_id: str) -> bool:
    """Return True (and clear) if a self-signal fired for this session recently."""
    with _SIGNAL_LOCK:
        ts = _SIGNAL_FIRED.pop(str(session_id or ""), None)
    return ts is not None and (time.monotonic() - ts) < _SIGNAL_TTL


def _log_activation(msg: str) -> None:
    logger.warning("proprioception firewall: %s", msg)
    try:
        import os
        from pathlib import Path

        p = Path(os.environ.get("LOCALAPPDATA", ".")) / "hermes" / "logs" / "proprioception-firewall.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


def scrub(response_text: str) -> tuple[Optional[str], str]:
    """Pure scrubber: return (cleaned_or_None, note).

    ``cleaned`` is None when nothing should change. Separated from the hook so
    it is unit-testable without the gateway.
    """
    if not response_text or not _LEAK_RE.search(response_text):
        return None, "no leak"

    sentences = _SENT_RE.findall(response_text)
    kept, dropped = [], []
    for s in sentences:
        if not _LEAK_RE.search(s):
            kept.append(s)
            continue
        salvaged = _salvage_sentence(s)
        if salvaged:
            # keep the clean remainder, spacing it like a normal sentence
            kept.append((" " if kept else "") + salvaged)
        dropped.append(s)

    if not dropped:
        return None, "no leak"

    cleaned = "".join(kept).strip()
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    dropped_note = " | ".join(d.strip()[:80] for d in dropped)
    # Fail-open guard: only bail when there's no substantive clean text LEFT —
    # an absolute floor, not a ratio of the original. A clean salvaged clause
    # ("Your P&L is attached.") is a valid reply even if the leak-heavy original
    # was mostly telemetry; a ratio guard would wrongly pass the whole leak
    # through. A leak is bad; an empty/mangled answer is worse — so only when
    # nothing coherent survives do we pass the original through and flag it.
    if len(cleaned) < 15:
        return None, f"MAJOR LEAK not auto-redacted (nothing coherent left); dropped: {dropped_note}"

    return cleaned, f"redacted {len(dropped)} sentence(s): {dropped_note}"


def transform_llm_output(**kwargs) -> Optional[str]:
    """``transform_llm_output`` hook. Returns cleaned text, or None to pass through."""
    try:
        from plugins.proprioception.settings import get_settings

        if not get_settings()["enabled"]:
            return None

        session_id = str(kwargs.get("session_id") or "")
        if not _consume_signal(session_id):
            return None  # no self-signal this turn → nothing to firewall

        text = kwargs.get("response_text")
        if not isinstance(text, str) or not text.strip():
            return None

        cleaned, note = scrub(text)
        if note != "no leak":
            _log_activation(f"[{kwargs.get('platform', '?')}] {note}")
        return cleaned  # None passes through unchanged
    except Exception:
        logger.debug("proprioception firewall failed; passing output through", exc_info=True)
        return None
