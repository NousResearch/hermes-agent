r"""`\w` / `\d` are exact under every RE2 Unicode generation the table knows, not just this build's.

One google-re2 version answers ``\p{..}`` from RE2's bundled Unicode 15.1 tables (the wheels) or,
in a libre2 built with RE2_USE_ICU (Alpine's, so musl), from ICU's newer data. ``compile()``
translates without the engine, so the translation must be right for each known generation. The
engine-backed proofs (``test_tables_current``, ``test_word_digit_space_are_exact_on_every_code_point``)
cover the build that runs them; here each OTHER generation is simulated from its table entry.
"""
import unicodedata
from bisect import bisect_right

import pytest

from hermes_re import _engine, _tables, _translate_class

TABLE = _tables.TABLES.get(unicodedata.unidata_version)
PY_SETS = {"w": lambda ch: ch.isalnum() or ch == "_", "d": str.isdecimal}


def _codes(pairs) -> set:
    return {cp for lo, hi in pairs for cp in range(lo, hi + 1)}


def _points() -> list:
    """ASCII, every code point in any generation's deltas, and their neighbours."""
    found = set(range(128))
    for entry in TABLE["RE2"].values():
        for key in ("W_ADD", "W_SUB", "D_ADD", "D_SUB"):
            for lo, hi in entry[key]:
                found |= set(range(max(lo - 2, 0), min(hi + 3, 0x110000)))
    return sorted(cp for cp in found if not 0xD800 <= cp <= 0xDFFF)


def _cell(cp: int, delta: dict) -> int:
    """The cell (0 L, 1 Nd, 2 Nl/No, 3 other) RE2 of one generation (its deltas as code point
    sets) puts ``cp`` in."""
    ch = chr(cp)
    digit = (ch.isdecimal() and cp not in delta["D_ADD"]) or cp in delta["D_SUB"]
    word = cp != 95 and ((PY_SETS["w"](ch) and cp not in delta["W_ADD"]) or cp in delta["W_SUB"])
    if digit:
        return 1
    if not word:
        return 3
    return 2 if unicodedata.category(ch) in ("Nl", "No", "Nd") else 0


def _member(pairs) -> callable:
    lows = [lo for lo, _ in pairs]
    return lambda cp: (index := bisect_right(lows, cp) - 1) >= 0 and cp <= pairs[index][1]


def _wrong(cells: dict, entry: dict) -> list:
    """Code points where the translated cell masks disagree with Python under ``entry``'s RE2."""
    delta = {key: _codes(entry[key]) for key in ("W_ADD", "W_SUB", "D_ADD", "D_SUB")}
    care = [_member(pairs) for pairs in cells["care"]]
    masks = {letter: [_member(pairs) for pairs in cells[letter]] for letter in PY_SETS}
    wrong = []
    for cp in _points():
        cell = _cell(cp, delta)
        for letter, python in PY_SETS.items():
            if not care[cell](cp) or masks[letter][cell](cp) != python(chr(cp)):
                wrong.append((letter, hex(cp)))
    return wrong


def _cells_for(monkeypatch, table: dict) -> dict:
    monkeypatch.setattr(_translate_class, "_TABLES", {**_tables.TABLES, unicodedata.unidata_version: table})
    monkeypatch.setattr(_translate_class, "_unicode", None)
    return _translate_class._unicode_cells()


def test_the_generation_is_the_last_version_whose_sentinels_and_all_earlier_ones_match():
    every = {ch for _, chars in _engine.SENTINELS for ch in chars}
    assert _engine.generation_of(every.__contains__) == _engine.SENTINELS[-1][0]
    assert _engine.generation_of(lambda ch: False) == _engine.OLDEST
    upto_15_1 = {ch for version, chars in _engine.SENTINELS[:2] for ch in chars}
    assert _engine.generation_of(upto_15_1.__contains__) == "15.1"
    # a later version's letters without an earlier one's digits: the earlier generation
    assert _engine.generation_of((upto_15_1 | {"\U00016EA0", "\U00011DE0"}).__contains__) == "15.1"
    assert _engine.generation_of((every - {"\U00010D40"}).__contains__) == "15.1"


def test_this_build_has_a_known_generation_for_this_interpreter():
    assert TABLE is not None, f"no table for Unicode {unicodedata.unidata_version}"
    generation = _engine.unicode_generation()
    assert generation in TABLE["RE2"], (
        f"no entry for RE2 Unicode generation {generation}: run scripts/gen_hermes_re_tables.py with this build")


@pytest.mark.skipif(TABLE is None, reason="no table for this interpreter's Unicode version")
def test_the_translation_is_exact_under_every_known_generation(monkeypatch):
    cells = _cells_for(monkeypatch, TABLE)
    for generation, entry in TABLE["RE2"].items():
        assert not _wrong(cells, entry), generation


@pytest.mark.skipif(TABLE is None, reason="no table for this interpreter's Unicode version")
def test_a_translation_missing_a_generation_is_wrong_under_it(monkeypatch):
    # The mutation the per-generation entries exist for: tables for one RE2 build only.
    differing = [name for name, entry in TABLE["RE2"].items()
                 if any(entry[key] for key in ("W_ADD", "W_SUB", "D_ADD", "D_SUB"))]
    if not differing:
        pytest.skip("every known generation agrees with this interpreter")
    for generation in differing:
        others = {name: entry for name, entry in TABLE["RE2"].items() if name != generation}
        empty = {"W_ADD": (), "W_SUB": (), "D_ADD": (), "D_SUB": ()}
        cells = _cells_for(monkeypatch, {**TABLE, "RE2": others or {"none": empty}})
        assert _wrong(cells, TABLE["RE2"][generation]), generation
