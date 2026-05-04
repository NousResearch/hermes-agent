"""Multilingual conservative stop-phrase matcher for busy-session routing.

Goal: detect literal halt intent in mid-stream user messages with near-zero
false-positive rate. Used by gateway busy-session router to decide whether
a follow-up message means "stop the running agent" vs "incorporate this
into the running stream as a steer."

Match conditions (ALL must hold):
- Normalized text length <= MAX_STOP_LEN (longer messages always steer)
- Word-boundary exact match against language tables below (case-insensitive)
- Optional trailing "!", ".", "。" tolerated

Universal triggers (matched regardless of language):
- empty message, whitespace-only
- lone "/"
- "/stop", "/cancel", "/halt", "/abort"

This is intentionally narrow. "we should stop including Bob" does NOT match
because length > MAX_STOP_LEN. "stopover" does NOT match because of word-
boundary semantics in our normalize() routine.
"""

from typing import Optional

# Per-language frozensets of literal halt phrases. All <= 30 chars normalized.
#
# Selection bias is conservative: we deliberately exclude common function
# words that double as halts ("para" in Spanish/Portuguese is also the
# preposition "for"; "basta" in Italian is also the filler "enough"; etc.)
# Native-speaker review encouraged via the issue tracker — the table is
# meant to be extended/corrected over time. False-positives here are more
# damaging than missed-positives: a missed stop just steers (the agent
# can still respond), but a false-positive interrupts unrelated work.
STOP_PHRASES: dict[str, frozenset[str]] = {
    "en": frozenset({
        "stop", "wait", "halt", "cancel", "abort", "pause",
        "please stop", "please wait",
    }),
    # Spanish: dropped "para" (also the preposition "for") in favor of more
    # imperative forms. "alto" is universally recognized as the stop-sign word.
    "es": frozenset({
        "alto", "espera", "detén", "deten", "cancela", "pausa",
        "por favor para",  # safe in this multi-word form
    }),
    "fr": frozenset({
        "stop", "arrête", "arrete", "attends", "pause", "annule",
    }),
    "de": frozenset({
        "stopp", "halt", "warte", "pause", "abbrechen",
    }),
    # Portuguese: prefer "pare" (verb imperative) over "para" (also "for").
    "pt": frozenset({
        "pare", "espera", "pausa", "cancela",
    }),
    # Italian: dropped "basta" (also a filler "enough") in favor of imperatives.
    "it": frozenset({
        "ferma", "aspetta", "pausa", "annulla",
    }),
    "nl": frozenset({
        "stop", "wacht", "pauze", "annuleer",
    }),
    "ja": frozenset({
        "止まれ", "止まって", "待って", "ストップ", "中断",
    }),
    "zh-Hans": frozenset({
        "停", "停下", "等等", "暂停", "取消",
    }),
    "zh-Hant": frozenset({
        "停", "停下", "等等", "暫停", "取消",
    }),
    # Korean: dropped "잠깐" (often a filler "just a moment") in favor of
    # explicit halt verbs.
    "ko": frozenset({
        "멈춰", "정지", "일시정지", "중단",
    }),
    "ru": frozenset({
        "стой", "погоди", "стоп", "пауза", "отмена",
    }),
    "ar": frozenset({
        "قف", "انتظر", "توقف", "إلغاء",
    }),
    "hi": frozenset({
        "रुको", "ठहरो", "बंद करो",
    }),
    "tr": frozenset({
        "dur", "bekle", "iptal",
    }),
    "pl": frozenset({
        "stop", "czekaj", "pauza", "anuluj",
    }),
}

# Slash-command and panic-button shortcuts (universal, language-independent).
SLASH_STOP_COMMANDS: frozenset[str] = frozenset({
    "/", "/stop", "/cancel", "/halt", "/abort",
})

# Flattened union for O(1) lookup.
_ALL_STOP_PHRASES: frozenset[str] = frozenset().union(*STOP_PHRASES.values())

# Conservative length cap. Longer messages always steer (never match as stop).
MAX_STOP_LEN: int = 30

# Trailing punctuation we tolerate before stripping.
_TRAILING_PUNCT = ("!", ".", "。", "！", "．")


def normalize(text: Optional[str]) -> str:
    """Lowercase, strip whitespace, drop trailing punctuation.

    Used for word-boundary exact matching against the phrase tables.
    """
    if not text:
        return ""
    n = text.strip().lower()
    # Strip trailing punctuation iteratively (e.g. "stop!." -> "stop")
    changed = True
    while changed:
        changed = False
        for p in _TRAILING_PUNCT:
            if n.endswith(p):
                n = n[: -len(p)].rstrip()
                changed = True
    return n


def matches_stop_phrase(text: Optional[str]) -> Optional[str]:
    """Return the matched language code if `text` is a stop signal, else None.

    Return values:
    - "universal" — empty / whitespace-only message
    - "slash"     — slash-command shortcut ("/", "/stop", etc.)
    - "<lang>"    — matched a phrase in language <lang> (e.g. "en", "es", "ja")
    - None        — does not match (proceed with default routing)
    """
    if text is None:
        return "universal"
    raw = text.strip()
    if not raw:
        return "universal"
    if raw in SLASH_STOP_COMMANDS:
        return "slash"
    n = normalize(text)
    if not n or len(n) > MAX_STOP_LEN:
        return None
    if n not in _ALL_STOP_PHRASES:
        return None
    # Found in flattened set — identify which language.
    for lang, phrases in STOP_PHRASES.items():
        if n in phrases:
            return lang
    return None


__all__ = [
    "MAX_STOP_LEN",
    "SLASH_STOP_COMMANDS",
    "STOP_PHRASES",
    "matches_stop_phrase",
    "normalize",
]
