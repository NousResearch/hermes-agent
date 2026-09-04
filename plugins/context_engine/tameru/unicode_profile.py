"""Bounded Unicode profiling for industrial compaction.

The module analyses logical text order only. Matching may use a normalised
shadow, while callers retain and return original substrings.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterator

_BIDI_CONTROLS = frozenset(
    {
        "\u061c",  # Arabic letter mark
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u202a",  # embeddings / overrides / pop directional format
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",  # isolates
        "\u2067",
        "\u2068",
        "\u2069",
        "\ufeff",  # BOM / zero-width no-break space
    }
)
_BIDI_OVERRIDES = frozenset({"\u202d", "\u202e"})
_ZWJ = "\u200d"
_ZWNJ = "\u200c"
_VARIATION_RANGES = ((0xFE00, 0xFE0F), (0xE0100, 0xE01EF))
_EMOJI_MODIFIER_RANGE = (0x1F3FB, 0x1F3FF)
_REGIONAL_RANGE = (0x1F1E6, 0x1F1FF)
_TAG_RANGE = (0xE0020, 0xE007F)
_VERTICAL_RE = re.compile(
    r"writing-mode\s*:\s*(?:vertical-(?:rl|lr)|sideways-(?:rl|lr))",
    re.IGNORECASE,
)
_WORD_JOINERS = frozenset("_-./:@+")
_NO_SPACE_SCRIPTS = frozenset({"han", "kana", "hangul", "thai", "lao", "khmer", "myanmar"})


@dataclass(frozen=True)
class UnicodeProfile:
    direction: str
    scripts: frozenset[str]
    grapheme_count: int
    bidi_controls: int
    bidi_overrides: int
    vertical_hint: bool
    malformed_surrogates: bool

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["scripts"] = sorted(self.scripts)
        return data


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= codepoint <= end for start, end in ranges)


def _is_variation_selector(char: str) -> bool:
    return _in_ranges(ord(char), _VARIATION_RANGES)


def _is_emoji_modifier(char: str) -> bool:
    return _EMOJI_MODIFIER_RANGE[0] <= ord(char) <= _EMOJI_MODIFIER_RANGE[1]


def _is_regional_indicator(char: str) -> bool:
    return _REGIONAL_RANGE[0] <= ord(char) <= _REGIONAL_RANGE[1]


def _is_extend(char: str) -> bool:
    category = unicodedata.category(char)
    codepoint = ord(char)
    return (
        category in {"Mn", "Mc", "Me"}
        or _is_variation_selector(char)
        or _is_emoji_modifier(char)
        or codepoint == 0x20E3
        or _TAG_RANGE[0] <= codepoint <= _TAG_RANGE[1]
    )


def _is_virama(char: str) -> bool:
    name = unicodedata.name(char, "")
    return "VIRAMA" in name or "HALANT" in name


def iter_graphemes(text: str) -> Iterator[str]:
    """Yield deterministic approximate extended grapheme clusters.

    This is a documented UAX #29 tailoring suitable for stdlib-only matching;
    it is intentionally conservative around combining and join sequences.
    """
    current = ""
    regional_count = 0
    for char in str(text or ""):
        if not current:
            current = char
            regional_count = 1 if _is_regional_indicator(char) else 0
            continue
        previous = current[-1]
        append = (
            _is_extend(char)
            or char in {_ZWJ, _ZWNJ}
            or previous in {_ZWJ, _ZWNJ}
            or (_is_virama(previous) and unicodedata.category(char).startswith("L"))
            or (
                _is_regional_indicator(char)
                and _is_regional_indicator(previous)
                and regional_count % 2 == 1
            )
        )
        if append:
            current += char
            if _is_regional_indicator(char):
                regional_count += 1
            continue
        yield current
        current = char
        regional_count = 1 if _is_regional_indicator(char) else 0
    if current:
        yield current


def graphemes(text: str) -> list[str]:
    return list(iter_graphemes(text))


def matching_shadow(text: str) -> str:
    """Return case-folded NFKC text with presentation-only bidi controls removed."""
    normalised = unicodedata.normalize("NFKC", str(text or ""))
    return "".join(char for char in normalised if char not in _BIDI_CONTROLS).casefold()


def unicode_safety_counts(text: str) -> tuple[int, int, bool]:
    """Return bidi controls, overrides, and malformed-surrogate presence."""
    value = str(text or "")
    if value.isascii():
        return 0, 0, False
    controls = 0
    overrides = 0
    malformed = False
    for char in value:
        malformed = malformed or unicodedata.category(char) == "Cs"
        if char in _BIDI_CONTROLS:
            controls += 1
            overrides += int(char in _BIDI_OVERRIDES)
    return controls, overrides, malformed


def script_of(char: str) -> str | None:
    """Return a stable script family for a representative character."""
    if not char:
        return None
    codepoint = ord(char)
    ranges = (
        ("hebrew", 0x0590, 0x05FF),
        ("arabic", 0x0600, 0x08FF),
        ("devanagari", 0x0900, 0x097F),
        ("bengali", 0x0980, 0x09FF),
        ("gurmukhi", 0x0A00, 0x0A7F),
        ("gujarati", 0x0A80, 0x0AFF),
        ("odia", 0x0B00, 0x0B7F),
        ("tamil", 0x0B80, 0x0BFF),
        ("telugu", 0x0C00, 0x0C7F),
        ("kannada", 0x0C80, 0x0CFF),
        ("malayalam", 0x0D00, 0x0D7F),
        ("sinhala", 0x0D80, 0x0DFF),
        ("thai", 0x0E00, 0x0E7F),
        ("lao", 0x0E80, 0x0EFF),
        ("tibetan", 0x0F00, 0x0FFF),
        ("myanmar", 0x1000, 0x109F),
        ("georgian", 0x10A0, 0x10FF),
        ("ethiopic", 0x1200, 0x137F),
        ("khmer", 0x1780, 0x17FF),
        ("mongolian", 0x1800, 0x18AF),
        ("han", 0x3400, 0x9FFF),
        ("kana", 0x3040, 0x30FF),
        ("hangul", 0xAC00, 0xD7AF),
        ("armenian", 0x0530, 0x058F),
        ("greek", 0x0370, 0x03FF),
        ("cyrillic", 0x0400, 0x052F),
    )
    for name, start, end in ranges:
        if start <= codepoint <= end:
            return name
    unicode_name = unicodedata.name(char, "")
    for name, marker in (
        ("latin", "LATIN"),
        ("greek", "GREEK"),
        ("cyrillic", "CYRILLIC"),
        ("arabic", "ARABIC"),
        ("hebrew", "HEBREW"),
    ):
        if marker in unicode_name:
            return name
    return "other" if unicodedata.category(char).startswith("L") else None


def _representative(cluster: str) -> str:
    for char in cluster:
        if unicodedata.category(char)[0] in {"L", "N"}:
            return char
    return cluster[0] if cluster else ""


def _flush_run(run: list[str], units: list[str]) -> None:
    if not run:
        return
    representative = _representative(run[0])
    script = script_of(representative)
    joined = "".join(run)
    if script in _NO_SPACE_SCRIPTS:
        if len(joined) <= 64:
            units.append(joined)
        if len(run) == 1 and run[0].strip():
            units.append(run[0])
        for width in (2, 3, 4):
            if len(run) < width:
                continue
            units.extend("".join(run[index : index + width]) for index in range(len(run) - width + 1))
    elif joined:
        units.append(joined)
    run.clear()


def search_units(text: str) -> list[str]:
    """Return ordered, deduplicated Unicode-aware units for query matching."""
    shadow = matching_shadow(text)
    units: list[str] = []
    run: list[str] = []
    for cluster in iter_graphemes(shadow):
        representative = _representative(cluster)
        category = unicodedata.category(representative) if representative else ""
        if category[:1] in {"L", "N", "M"} or all(char in _WORD_JOINERS for char in cluster):
            run.append(cluster)
        else:
            _flush_run(run, units)
    _flush_run(run, units)
    seen: set[str] = set()
    ordered: list[str] = []
    for unit in units:
        cleaned = unit.strip("_-./:@+")
        value = unit if cleaned else ""
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def token_units(text: str) -> list[str]:
    """Return non-overlapping logical units for deterministic token estimates."""
    units: list[str] = []
    run: list[str] = []
    run_script: str | None = None

    def flush() -> None:
        nonlocal run_script
        if run:
            units.append("".join(run))
            run.clear()
        run_script = None

    for cluster in iter_graphemes(matching_shadow(text)):
        if not cluster or cluster.isspace():
            flush()
            continue
        representative = _representative(cluster)
        category = unicodedata.category(representative) if representative else ""
        script = script_of(representative)
        if category[:1] in {"L", "N", "M"}:
            if script in _NO_SPACE_SCRIPTS:
                flush()
                units.append(cluster)
                continue
            if run and script and run_script and script != run_script:
                flush()
            run.append(cluster)
            run_script = run_script or script
            continue
        if all(char in _WORD_JOINERS for char in cluster) and run:
            run.append(cluster)
            continue
        flush()
        if any(unicodedata.category(char) == "So" for char in cluster):
            units.append(cluster)
    flush()
    return units


def _looks_vertical_ocr(text: str) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) < 6:
        return False
    short = sum(1 for line in lines[:128] if len(graphemes(line)) <= 2)
    return short / min(len(lines), 128) >= 0.8


def profile_text(text: str) -> UnicodeProfile:
    value = str(text or "")
    if value.isascii():
        has_latin = any(char.isalpha() for char in value)
        lines = [line.strip() for line in value.splitlines() if line.strip()][:128]
        vertical = bool(_VERTICAL_RE.search(value)) or (
            len(lines) >= 6
            and sum(len(line) <= 2 for line in lines) / len(lines) >= 0.8
        )
        return UnicodeProfile(
            direction="ltr" if has_latin else "neutral",
            scripts=frozenset({"latin"}) if has_latin else frozenset(),
            grapheme_count=len(value),
            bidi_controls=0,
            bidi_overrides=0,
            vertical_hint=vertical,
            malformed_surrogates=False,
        )
    scripts: set[str] = set()
    ltr = 0
    rtl = 0
    controls = 0
    overrides = 0
    malformed = False
    for char in value:
        category = unicodedata.category(char)
        malformed = malformed or category == "Cs"
        if char in _BIDI_CONTROLS:
            controls += 1
            overrides += int(char in _BIDI_OVERRIDES)
        bidi = unicodedata.bidirectional(char)
        if bidi == "L":
            ltr += 1
        elif bidi in {"R", "AL"}:
            rtl += 1
        script = script_of(char)
        if script:
            scripts.add(script)
    if ltr and rtl:
        direction = "mixed"
    elif rtl:
        direction = "rtl"
    elif ltr:
        direction = "ltr"
    else:
        direction = "neutral"
    return UnicodeProfile(
        direction=direction,
        scripts=frozenset(scripts),
        grapheme_count=sum(1 for _ in iter_graphemes(value)),
        bidi_controls=controls,
        bidi_overrides=overrides,
        vertical_hint=bool(_VERTICAL_RE.search(value)) or _looks_vertical_ocr(value),
        malformed_surrogates=malformed,
    )
