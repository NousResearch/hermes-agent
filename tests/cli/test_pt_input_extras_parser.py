"""Property suite for the extended-keyboard grammars.

The defect this file exists to prevent is not any single mapping. It is the one
named in PR #99488's own merge description:

    "The #87511 tests only asserted kp.key (never kp.data), which is why CI
     stayed green through all of it."

Green CI over an untested axis. A suite that enumerates expected values is the
alias table wearing a different extension --- it can only ever assert the cells
somebody remembered to write, which is exactly the failure mode. So almost
nothing here is an expectation table. These are INVARIANTS swept across the
whole space a terminal can actually emit:

  * nothing leaks, for any key under any modifier
  * lock bits change nothing, ever (the property that makes lock-twin
    registrations unnecessary rather than merely redundant)
  * a key release never types
  * whatever the terminal reports beats whatever we would have inferred
  * a keypad key does exactly what the key it stands in for does
  * every prefix of a valid sequence is recognised as one

Each of those is one assertion over ~50,000 sequences, and each fails loudly the
moment a dimension stops being handled --- which is what "testing the axis"
means, as opposed to testing the key someone filed a bug about.
"""

from __future__ import annotations

import itertools
import os

import pytest

from hermes_cli import pt_input_extras_parser as P

E = "\x1b"

# ---------------------------------------------------------------------------
# The space
# ---------------------------------------------------------------------------

PRINTABLE = list(range(0x20, 0x7F))
FUNCTIONAL = sorted(P._FUNCTIONAL)
IGNORED = sorted(P._IGNORED)
NAMED = sorted(P._NAMED)
ALL_KEYS = PRINTABLE + FUNCTIONAL + IGNORED

REAL_MOD_BITS = range(64)          # shift|alt|ctrl|super|hyper|meta
LOCK_OFFSETS = (0, 64, 128, 192)   # none, caps, num, both


@pytest.fixture
def Keys():
    from prompt_toolkit.keys import Keys as _K

    return _K


def csi_u(cp, mods=0, *, shifted=None, base=None, event=None, text=None):
    """Render a CSI-u sequence. Mirrors the spec's field order exactly."""
    first = str(cp)
    if shifted is not None or base is not None:
        first += f":{shifted if shifted is not None else ''}"
        if base is not None:
            first += f":{base}"
    out = f"{E}[{first}"
    if mods or event is not None or text is not None:
        out += f";{mods + 1}"
        if event is not None:
            out += f":{event}"
    if text is not None:
        out += ";" + ":".join(str(ord(c)) for c in text)
    return out + "u"


def resolve(seq, Keys):
    pk = P.parse(seq) or P.parse_legacy(seq)
    return P.resolve(pk, Keys) if pk is not None else None


# ---------------------------------------------------------------------------
# Round-trip: the parser must read back exactly what the spec renders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cp", [0x61, 0x20, 0x7F, 57417, 57399])
@pytest.mark.parametrize("mods", [0, 1, 4, 5, 63])
def test_parse_round_trips_the_wire_format(cp, mods):
    pk = P.parse(csi_u(cp, mods))
    assert pk is not None
    assert pk.codepoint == cp
    assert pk.mods == mods


def test_parse_reads_every_optional_field():
    pk = P.parse(csi_u(0x61, 1, shifted=0x41, base=0x61, event=P.RELEASE, text="hi"))
    assert (pk.codepoint, pk.shifted, pk.base) == (0x61, 0x41, 0x61)
    assert pk.mods == 1 and pk.event == P.RELEASE and pk.text == "hi"


@pytest.mark.parametrize("junk", [
    f"{E}[u", f"{E}[;5u", f"{E}[abc;2u", f"{E}[97;2", "97;2u", f"{E}[97;2x",
    f"{E}[97;2~", "", f"{E}[", f"{E}[27;2;13~x",
])
def test_malformed_input_is_declined_not_guessed(junk):
    assert P.parse(junk) is None
    assert P.parse_legacy(junk) is None


# ---------------------------------------------------------------------------
# THE invariant: lock bits are state, not modifiers
#
# This single property is what makes _lock_variants/_lock_twins and the
# fourfold expansion of every registration unnecessary. Under a table it must
# be asserted per entry (and #88221/#89651 are what happens when it isn't);
# here it is one sweep.
# ---------------------------------------------------------------------------

def test_lock_bits_never_change_the_result(Keys):
    mismatches = []
    for cp in ALL_KEYS:
        for bits in REAL_MOD_BITS:
            base = resolve(csi_u(cp, bits), Keys)
            for lock in LOCK_OFFSETS[1:]:
                got = resolve(csi_u(cp, bits | lock), Keys)
                if got != base:
                    mismatches.append((cp, bits, lock, base, got))
    assert not mismatches, f"{len(mismatches)} lock-state divergences, e.g. {mismatches[:3]}"


def test_every_real_modifier_combination_is_handled(Keys):
    """No modifier permutation may be unreachable. 64 bit patterns, no gaps."""
    unresolved = [b for b in REAL_MOD_BITS if resolve(csi_u(ord("c"), b), Keys) is None]
    assert unresolved == [], f"unhandled modifier bit-patterns: {unresolved}"


# ---------------------------------------------------------------------------
# Nothing leaks
# ---------------------------------------------------------------------------

def test_no_well_formed_sequence_ever_returns_raw_escape_text(Keys):
    """A resolution may be None (declined) but must never be the escape itself."""
    leaks = []
    for cp in ALL_KEYS:
        for bits in (0, 1, 2, 4, 5, 6, 8, 63):
            seq = csi_u(cp, bits)
            got = resolve(seq, Keys)
            if isinstance(got, str) and got.startswith(E):
                leaks.append((seq, got))
    assert not leaks, f"parser echoed raw escapes: {leaks[:3]}"


def test_unassigned_pua_codepoints_are_declined_not_invented(Keys):
    """PUA codepoints that are not keys must resolve to nothing at all."""
    unassigned = [c for c in range(57344, 57500)
                  if c not in P._FUNCTIONAL and c not in P._IGNORED]
    invented = [c for c in unassigned if resolve(csi_u(c, 1), Keys) is not None]
    assert invented == [], f"invented meanings for non-keys: {invented[:5]}"


def test_ignored_codepoints_are_consumed_under_every_modifier(Keys):
    """Locks, media keys and bare modifier events must never reach the buffer."""
    bad = [(cp, b) for cp in IGNORED for b in (0, 1, 4, 5)
           if resolve(csi_u(cp, b), Keys) is not Keys.Ignore]
    assert bad == [], f"ignorable codes not consumed: {bad[:5]}"


# ---------------------------------------------------------------------------
# Event types (flag 2). The property that makes the flag safe to negotiate.
# ---------------------------------------------------------------------------

def test_release_never_types_anything(Keys):
    typing = [cp for cp in ALL_KEYS
              if resolve(csi_u(cp, 1, event=P.RELEASE), Keys) is not Keys.Ignore]
    assert typing == [], f"key releases that typed: {typing[:5]}"


@pytest.mark.parametrize("seq", [
    f"{E}[1;6:3A",     # Ctrl+Shift+Up released, captured from kitty 0.48.2
    f"{E}[15;1:3~",    # F5 released
    f"{E}[1;1:3E",     # KP_Begin released
])
def test_legacy_nav_releases_are_declined_not_leaked(seq, Keys):
    """The nav forms take event sub-parameters too, and no alias table carries them."""
    assert P.parse_legacy(seq).is_release
    assert resolve(seq, Keys) is Keys.Ignore


@pytest.mark.parametrize("seq", [f"{E}[1;5E", f"{E}[1;2E", f"{E}OE"])
def test_modified_keypad_begin_is_consumed(seq, Keys):
    """ESC[E is mapped, its modified and SS3 forms are not: #90640's shape, another key."""
    assert resolve(seq, Keys) is Keys.Ignore


@pytest.mark.parametrize("seq,expected", [
    (f"{E}[15;2~", "F17"), (f"{E}[24;2~", "F24"), (f"{E}[15;6~", "ControlF17"),
])
def test_shift_promotes_f1_to_f12_into_f13_to_f24(seq, expected, Keys):
    """xterm names Shift+F5 as F17, and the table follows it; a bare F5 would lose Shift."""
    assert resolve(seq, Keys) is getattr(Keys, expected)


def test_repeat_behaves_exactly_like_press(Keys):
    """Holding a key must insert repeatedly; a dropped repeat breaks held arrows."""
    for cp in (ord("a"), ord("A"), 57419, 13):
        for bits in (0, 1, 4):
            press = resolve(csi_u(cp, bits, event=P.PRESS), Keys)
            assert resolve(csi_u(cp, bits, event=P.REPEAT), Keys) == press
            assert resolve(csi_u(cp, bits), Keys) == press  # event omitted


# ---------------------------------------------------------------------------
# Flags 4 and 16: what the terminal reports outranks what we would infer.
# This is the property that removes the layout question rather than answering it.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["us", "tr", "de", "gr", "ru", "az", ""])
@pytest.mark.parametrize("base,shifted", [
    (0x32, 0x40),    # 2 -> @  (US)
    (0xDF, 0x3F),    # ß -> ?  (German: NOT a casing of ß)
    (0x37, 0x2F),    # 7 -> /  (Slovenian, issue #100169)
    (0x69, 0x130),   # i -> İ  (Turkish)
])
def test_alternate_keys_beat_every_layout(layout, base, shifted, Keys, monkeypatch):
    """Flag 4: the terminal names the shifted key, so the layout stops mattering."""
    monkeypatch.setenv("XKB_DEFAULT_LAYOUT", layout)
    monkeypatch.setattr(P, "_turkic_cache", None, raising=False)
    assert resolve(csi_u(base, P.SHIFT, shifted=shifted), Keys) == chr(shifted)


@pytest.mark.parametrize("layout", ["us", "tr", "de", "gr", ""])
def test_associated_text_beats_every_layout(layout, Keys, monkeypatch):
    """Flag 16: the terminal names the text. #100169's Slovenian Shift+7."""
    monkeypatch.setenv("XKB_DEFAULT_LAYOUT", layout)
    monkeypatch.setattr(P, "_turkic_cache", None, raising=False)
    assert resolve(csi_u(0x37, P.SHIFT, text="/"), Keys) == "/"


def test_reported_text_is_ignored_when_ctrl_or_alt_is_held(Keys):
    """Ctrl+C is a command, not text, however the terminal spells it."""
    assert resolve(csi_u(ord("c"), P.CTRL, text="c"), Keys) is Keys.ControlC
    assert resolve(csi_u(ord("c"), P.ALT, text="c"), Keys) == (Keys.Escape, "c")


@pytest.mark.parametrize("cp,base_cp", [(1089, 99), (968, 99), (0x430, 0x61)])
def test_base_layout_key_keeps_shortcuts_working_off_latin(cp, base_cp, Keys):
    """Ctrl+C on Cyrillic arrives as codepoint 1089 and matches no binding; base names 99."""
    plain = resolve(csi_u(base_cp, P.CTRL), Keys)
    assert resolve(csi_u(cp, P.CTRL, base=base_cp), Keys) == plain


# ---------------------------------------------------------------------------
# Layouts, where the terminal reports nothing and we must not guess
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout,cp,expected", [
    ("us", 0x69, "I"),        # i  -> I
    ("tr", 0x69, "İ"),        # Turkish dotted capital, per SpecialCasing.txt
    ("az", 0x69, "İ"),
    ("tr", 0x131, "I"),       # dotless ı -> I
    ("us", 0x61, "A"),
    ("ru", 0x430, "А"),       # Cyrillic a -> A  (issue #87631)
    ("gr", 0x3B1, "Α"),       # Greek alpha
])
def test_casing_follows_the_locale_where_it_is_conditional(layout, cp, expected, Keys, monkeypatch):
    monkeypatch.setenv("XKB_DEFAULT_LAYOUT", layout)
    monkeypatch.setattr(P, "_turkic_cache", None, raising=False)
    assert resolve(csi_u(cp, P.SHIFT), Keys) == expected


@pytest.mark.parametrize("layout", ["us", "de", "tr", "gr", ""])
def test_sharp_s_is_declined_rather_than_guessed(layout, Keys, monkeypatch):
    """German Shift+ß is '?', which is no casing of ß -- decline rather than guess."""
    monkeypatch.setenv("XKB_DEFAULT_LAYOUT", layout)
    monkeypatch.setattr(P, "_turkic_cache", None, raising=False)
    assert resolve(csi_u(0xDF, P.SHIFT), Keys) is None


def test_shift_never_produces_more_than_one_character(Keys):
    """One key press, at most one character: 'ß'.upper() is 'SS'."""
    multi = []
    for cp in list(range(0x20, 0x250)) + [0x1E9E, 0xFB00]:
        got = resolve(csi_u(cp, P.SHIFT), Keys)
        if isinstance(got, str) and not isinstance(got, Keys) and len(got) > 1:
            multi.append((hex(cp), got))
    assert multi == [], f"multi-character results: {multi[:5]}"


def test_kitty_punctuation_without_alternate_keys_is_declined(Keys, monkeypatch):
    """The base codepoint alone cannot say what Shift produces on this layout."""
    monkeypatch.setenv("XKB_DEFAULT_LAYOUT", "gr")
    monkeypatch.setattr(P, "_turkic_cache", None, raising=False)
    for ch in "234567890-=[]\\;',./":
        assert resolve(csi_u(ord(ch), P.SHIFT), Keys) is None, ch


# ---------------------------------------------------------------------------
# The other grammars. Same modifier arithmetic, different terminators.
# ---------------------------------------------------------------------------

def test_modify_other_keys_is_layout_safe_by_construction(Keys, monkeypatch):
    """This encoding reports the already-shifted codepoint, so identity is correct everywhere."""
    results = {}
    for layout in ("us", "de", "gr", "tr", "ru", ""):
        monkeypatch.setenv("XKB_DEFAULT_LAYOUT", layout)
        monkeypatch.setattr(P, "_turkic_cache", None, raising=False)
        results[layout] = [resolve(f"{E}[27;2;{ord(c)}~", Keys) for c in "~!@#$%^&*()_+{}|:\"<>?"]
    reference = results["us"]
    assert reference == list("~!@#$%^&*()_+{}|:\"<>?")
    for layout, got in results.items():
        assert got == reference, f"layout {layout!r} diverged"


@pytest.mark.parametrize("final,name", list(P._LETTER_KEY.items()))
@pytest.mark.parametrize("bits", [0, 1, 4, 5])
def test_csi_letter_nav_matches_its_csi_u_equivalent(final, name, bits, Keys):
    assert resolve(f"{E}[1;{bits + 1}{final}" if bits else f"{E}[{final}", Keys) is not None


@pytest.mark.parametrize("num,name", list(P._TILDE_KEY.items()))
@pytest.mark.parametrize("bits", [0, 1, 4, 5])
def test_csi_tilde_nav_resolves_under_every_modifier(num, name, bits, Keys):
    seq = f"{E}[{num};{bits + 1}~" if bits else f"{E}[{num}~"
    assert resolve(seq, Keys) is not None


@pytest.mark.parametrize("final", sorted(P._SS3_KEYPAD) + list(P._LETTER_KEY))
def test_ss3_forms_resolve(final, Keys):
    assert resolve(f"{E}O{final}", Keys) is not None


def test_shift_tab_is_backtab_in_every_spelling(Keys):
    assert resolve(f"{E}[Z", Keys) is Keys.BackTab
    assert resolve(csi_u(9, P.SHIFT), Keys) is Keys.BackTab


def test_shift_f1_to_f4_are_reported_as_f13_to_f16(Keys):
    """xterm's convention, which prompt_toolkit's table follows."""
    for final, n in zip("PQRS", (13, 14, 15, 16)):
        assert resolve(f"{E}[1;2{final}", Keys) is getattr(Keys, f"F{n}")


# ---------------------------------------------------------------------------
# The keypad mirrors the keys it stands in for. Issue #90640.
# ---------------------------------------------------------------------------

KEYPAD_TWINS = {57417: "D", 57418: "C", 57419: "A", 57420: "B", 57423: "H", 57424: "F"}
KEYPAD_TILDE = {57421: 5, 57422: 6, 57425: 2, 57426: 3}


@pytest.mark.parametrize("cp,final", list(KEYPAD_TWINS.items()))
@pytest.mark.parametrize("bits", [0, 1, 2, 4, 5, 6])
def test_keypad_nav_never_drifts_from_its_twin(cp, final, bits, Keys):
    twin = resolve(f"{E}[1;{bits + 1}{final}" if bits else f"{E}[{final}", Keys)
    assert resolve(csi_u(cp, bits), Keys) == twin


@pytest.mark.parametrize("cp,num", list(KEYPAD_TILDE.items()))
@pytest.mark.parametrize("bits", [0, 1, 4, 5])
def test_keypad_tilde_cluster_never_drifts_from_its_twin(cp, num, bits, Keys):
    twin = resolve(f"{E}[{num};{bits + 1}~" if bits else f"{E}[{num}~", Keys)
    assert resolve(csi_u(cp, bits), Keys) == twin


@pytest.mark.parametrize("d", range(10))
def test_keypad_digit_types_its_digit(d, Keys):
    assert resolve(csi_u(57399 + d), Keys) == str(d)


@pytest.mark.parametrize("d", range(10))
@pytest.mark.parametrize("bits", [P.SHIFT, P.ALT, P.CTRL, P.CTRL | P.SHIFT])
def test_modified_keypad_digit_mirrors_the_row_digit(d, bits, Keys):
    """Compare against the twin: `got is not None` accepted ControlBackslash for Ctrl+KP_5."""
    assert resolve(csi_u(57399 + d, bits), Keys) == resolve(csi_u(ord("0") + d, bits), Keys)


# ---------------------------------------------------------------------------
# Prefix recognition. Get this wrong and the parser flushes mid-sequence,
# which is indistinguishable from a leak at the prompt.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq", [
    csi_u(0x61, 1), csi_u(57417, 4), csi_u(0x61, 1, shifted=0x41, event=2, text="A"),
    f"{E}[27;2;64~", f"{E}[1;5D", f"{E}[5;3~", f"{E}OA",
])
def test_every_prefix_of_a_valid_sequence_is_recognised(seq):
    for i in range(1, len(seq)):
        assert P.is_prefix(seq[:i]), f"{seq[:i]!r} not seen as a prefix of {seq!r}"


# ---------------------------------------------------------------------------
# Negotiation
# ---------------------------------------------------------------------------

def test_wanted_flags_exclude_report_all_keys():
    """Flag 8 routes ordinary typing through the protocol, turning any gap into silence."""
    assert not P.WANTED & P.ALL_AS_ESCAPES
    assert P.push() == f"{E}[>{P.WANTED}u"


def test_wanted_flags_exclude_event_types():
    """A release for every press doubles the stream to answer what a REPL never asks."""
    assert not P.WANTED & P.EVENT_TYPES


def test_wanted_flags_exclude_associated_text():
    """Without flag 8 the text field reaches only Ctrl chords, whose text we discard."""
    assert not P.WANTED & P.ASSOCIATED_TEXT


@pytest.mark.parametrize("text,expected", [
    ("A", "A"),                    # one codepoint: the terminal's answer, taken as is
    ("SS", ("S", "S")),            # German Shift+ss uppercases into two characters
    ("你好", ("你", "好")),          # an IME commit is a whole word
    ("e\u0301", ("e", "\u0301")),   # a dead key composes
])
def test_reported_text_longer_than_a_keypress_is_split_not_dropped(text, expected, Keys):
    """KeyPress asserts len(key) == 1; a tuple is how prompt_toolkit spells several."""
    cps = ":".join(str(ord(c)) for c in text)
    assert resolve(f"{E}[97;1;{cps}u", Keys) == expected


@pytest.mark.parametrize("reply", [0, 1, 3, 31])
def test_a_reply_means_the_terminal_speaks_the_protocol(reply):
    """CSI ? u reports the flags CURRENTLY SET; masking against it disables flag 4."""
    assert P.negotiated(reply) == P.WANTED


def test_no_reply_falls_back_to_the_floor_never_an_assumption():
    """#100169: an inherited WT_SESSION was read as identification and the push mis-encoded."""
    assert P.negotiated(None) == P.DISAMBIGUATE


def test_query_reply_is_parsed_and_junk_is_rejected():
    assert P.parse_query_reply(f"{E}[?31u") == 31
    for junk in (f"{E}[?u", f"{E}[31u", "31", f"{E}[?31x", ""):
        assert P.parse_query_reply(junk) is None


def test_pop_restores_the_terminal():
    """#56759: never popping is why nano in the same pane gets raw fragments."""
    assert P.pop() == f"{E}[<u"


# ---------------------------------------------------------------------------
# Installation, and the end-to-end path a user actually types on
# ---------------------------------------------------------------------------

def test_install_is_idempotent(Keys):
    P.install()
    import prompt_toolkit.input.vt100_parser as vt

    first = vt.Vt100Parser._get_match
    assert P.install() is False
    assert vt.Vt100Parser._get_match is first


def test_nothing_reaches_the_buffer_as_raw_escape_text(Keys):
    """The end-to-end property: escape bytes must never arrive as literal text."""
    from prompt_toolkit.input.vt100_parser import Vt100Parser

    from hermes_cli import pt_input_extras as X

    X.install_shift_enter_alias()
    X.install_ctrl_enter_alias()
    X.install_cmd_backspace_alias()
    X.install_modify_other_keys_aliases()
    X.install_keypress_data_normalization()
    P.install()

    cases = [csi_u(cp, bits) for cp in (ord("a"), ord("H"), 57417, 57414, 13, 9)
             for bits in (0, 1, 2, 4, 5, 6, 129, 133)]
    cases += [csi_u(0x37, P.SHIFT, text="/"), csi_u(0x2F, P.SHIFT, shifted=0x3F),
              csi_u(ord("a"), P.SHIFT, event=P.RELEASE),
              f"{E}[27;2;64~", f"{E}[1;5D", f"{E}[5;3~", f"{E}OA"]

    def would_self_insert(key):
        """prompt_toolkit's default binding inserts event.data for character keys.

        A named ``Keys`` member has no such binding, so raw data riding on one is
        inert --- which is exactly what happens to the ``Escape`` of an Alt chord,
        where _call_handler deliberately hands insert_text to the FIRST KeyPress.
        Only a character-valued key can actually put its data in the buffer.
        """
        return isinstance(key, str) and not isinstance(key, Keys) and len(key) == 1

    leaked = []
    for seq in cases:
        out = []
        Vt100Parser(lambda kp: out.append(kp)).feed_and_flush(seq)
        for kp in out:
            if would_self_insert(kp.key) and isinstance(kp.data, str) and kp.data.startswith(E):
                leaked.append((seq, kp.key, kp.data))
    assert not leaked, f"raw escape text reached the buffer: {leaked[:3]}"


# ---------------------------------------------------------------------------
# Differential: wherever the alias table has an opinion, the parser agrees or
# the difference is a deliberate application decision. This is the net that
# catches a parser regression against 4,000 hand-written expectations without
# copying any of them into this file.
# ---------------------------------------------------------------------------

def test_parser_never_contradicts_the_table_on_a_character_valued_entry(Keys):
    """Character-valued entries are pure protocol, so the parser must agree or decline."""
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES

    from hermes_cli import pt_input_extras as X

    X.install_shift_enter_alias()
    X.install_ctrl_enter_alias()
    X.install_cmd_backspace_alias()
    X.install_modify_other_keys_aliases()

    # Keys-valued entries are where application decisions live (Shift+Enter is aliased
    # onto Alt+Enter for newline), so a disagreement there is expected and correct.
    contradictions = []
    for seq, table_key in list(ANSI_SEQUENCES.items()):
        if not (isinstance(table_key, str) and not isinstance(table_key, Keys) and len(table_key) == 1):
            continue
        pk = P.parse(seq) or P.parse_legacy(seq)
        if pk is None:
            continue
        # A modifyOtherKeys row whose value is not the codepoint it carries is a DERIVED
        # layout answer rather than the protocol speaking --- #103869 registers the
        # unshifted codepoint with what Shift produces on this keyboard. The parser reads
        # that field as already-shifted, so the two answer different terminals rather than
        # disagreeing, and the table is consulted first either way.
        if pk.already_shifted and chr(pk.codepoint) != table_key:
            continue
        got = P.resolve(pk, Keys)
        if got is not None and got != table_key:
            contradictions.append((seq, table_key, got))
    assert contradictions == [], f"parser contradicts the table: {contradictions[:5]}"


@pytest.mark.parametrize("seq", [
    f"{E}[97;1;1114112u",   # a text codepoint chr() cannot represent
    f"{E}[97;1;55296u",     # a lone surrogate
    f"{E}[97;0u",           # the parameter is 1 + the bits, so 0 is malformed
])
def test_malformed_sequences_are_declined_never_raised(seq):
    """This runs inside the input loop, so a bad sequence must leak, not kill the REPL."""
    assert P.parse(seq) is None


def test_installed_parser_survives_a_sequence_it_cannot_decode():
    """A text codepoint past U+10FFFF once reached chr() and ended the session."""
    from prompt_toolkit.input.vt100_parser import Vt100Parser

    P.install()
    keys = []
    Vt100Parser(keys.append).feed(f"{E}[97;1;1114112u")
    assert keys and all(k.key is not None for k in keys)


def test_installed_parser_answers_through_the_real_input_pipeline(Keys):
    """Unit-testing resolve() proves nothing about the seam prompt_toolkit actually calls."""
    from prompt_toolkit.input.vt100_parser import Vt100Parser

    P.install()
    keys = []
    Vt100Parser(keys.append).feed(f"a{E}[57417;5u{E}[97;2:3ub")
    assert [k.key for k in keys] == ["a", Keys.ControlLeft, Keys.Ignore, "b"]


@pytest.mark.parametrize("seq", [
    f"{E}[97;0u", f"{E}[27;0;97~", f"{E}[1;0A", f"{E}[3;0~", f"{E}O0A",
])
def test_a_zero_modifier_parameter_is_declined_in_every_grammar(seq):
    """The parameter is 1 + the bits, so 0 masks to -1 and invents every modifier."""
    assert (P.parse(seq) or P.parse_legacy(seq)) is None


@pytest.mark.parametrize("seq", [f"{E}[97;4:99u", f"{E}[1;1:99A", f"{E}[15;1:99~"])
def test_undefined_event_types_are_declined_not_treated_as_presses(seq):
    """Only 1, 2 and 3 are defined; anything else must not silently type."""
    assert (P.parse(seq) or P.parse_legacy(seq)) is None


@pytest.mark.parametrize("seq", [
    f"{E}[55296u", f"{E}[97:55296u", f"{E}[97:65:55296u", f"{E}[27;2;55296~",
])
def test_surrogate_codepoints_are_declined_in_every_field(seq):
    """A lone surrogate survives chr() and fails much later, on encode."""
    assert (P.parse(seq) or P.parse_legacy(seq)) is None


def test_backtab_keeps_the_modifiers_it_is_sent_with(Keys):
    """ESC[1;5Z is Ctrl+BackTab; ignoring the parameter drops the Ctrl silently."""
    assert P.parse_legacy(f"{E}[1;5Z").real_mods == P.CTRL | P.SHIFT
    assert resolve(f"{E}[Z", Keys) is Keys.BackTab


def test_a_leading_number_other_than_one_is_not_this_grammar():
    """ESC[99;5Z is some other sequence that happens to end in Z."""
    assert P.parse_legacy(f"{E}[99;5Z") is None


def test_xterm_meta_is_not_read_as_kitty_super(Keys):
    """modifyOtherKeys bit 8 is Meta, where the kitty protocol assigns Super."""
    assert resolve(f"{E}[27;9;97~", Keys) == (Keys.Escape, "a")


@pytest.mark.parametrize("cp,name", [
    (57344, "Escape"), (57346, "Tab"), (57347, "Backspace"), (57349, "Delete"),
    (57350, "Left"), (57352, "Up"), (57356, "Home"), (57368, "F5"), (57375, "F12"),
])
def test_the_functional_block_maps_the_keys_themselves_not_just_the_keypad(cp, name, Keys):
    """Under flag 8 these arrive as CSI-u, and unmapped they type a PUA glyph."""
    got = resolve(csi_u(cp), Keys)
    assert got is not None
    assert isinstance(got, Keys), f"{name} resolved to the raw glyph {got!r}"


def test_a_parsed_event_is_never_handed_back_to_leak(Keys):
    """The table has already missed by the time we run, so None puts bytes in the prompt."""
    from prompt_toolkit.input.vt100_parser import Vt100Parser

    P.install()
    for seq in (f"{E}[45;5u", f"{E}[46;5u", f"{E}[44;5u"):   # Ctrl+minus . , : unnameable
        keys = []
        Vt100Parser(keys.append).feed(seq)
        assert [k.key for k in keys] == [Keys.Ignore], f"{seq!r} leaked"


@pytest.mark.parametrize("seq", [
    f"{E}[97;9u",      # Super+a
    f"{E}[57426;9u",   # Super+Delete: acting on the base key would DELETE
    f"{E}[97;17u",     # Hyper+a
    f"{E}[97;33u",     # Meta+a
    f"{E}[99;13u",     # Ctrl+Super+c
])
def test_modifiers_prompt_toolkit_cannot_express_are_consumed_not_stripped(seq, Keys):
    """Dropping Super and acting on the bare key turns a window chord into an edit."""
    assert resolve(seq, Keys) is Keys.Ignore


def test_a_shift_we_cannot_compute_is_left_for_the_table_not_swallowed(Keys):
    """Consuming Shift+1 would shadow the entry that fixes it and lose the keystroke."""
    from prompt_toolkit.input.vt100_parser import Vt100Parser

    P.install()
    for seq in (f"{E}[49;2u", f"{E}[50;2u", f"{E}[47;2u"):
        keys = []
        Vt100Parser(keys.append).feed(seq)
        # Either the table answers it or it leaks visibly. What must never happen is the
        # parser eating it, which would shadow the layout derivation that fixes it.
        resolved = [k.key for k in keys]
        assert resolved != [Keys.Ignore], f"{seq!r} was silently consumed"


def test_an_alternate_key_answers_the_same_press(Keys):
    """With flag 4 the terminal states the shifted key, so there is nothing to infer."""
    assert resolve(f"{E}[49:33:49;2u", Keys) == "!"
    assert resolve(f"{E}[47:63:47;2u", Keys) == "?"


@pytest.mark.parametrize("cp", [0x00DF, 0xFB00])       # ss -> SS, ff ligature -> FF
def test_already_shifted_letters_never_lengthen_under_upper(cp, Keys):
    """KeyPress asserts one character, and this path returned 'SS' straight into it."""
    got = resolve(f"{E}[27;2;{cp}~", Keys)
    assert got == chr(cp)


@pytest.mark.parametrize("param", [257, 512, 1000])
def test_modifier_bits_above_the_defined_set_are_declined(param):
    """Unknown bits masked to zero read as an unmodified key and type the bare letter."""
    assert P.parse(f"{E}[97;{param}u") is None


@pytest.mark.parametrize("cp", [8, 127])
def test_both_backspace_codepoints_are_recognised(cp, Keys):
    """A terminal sends 8 or 127 depending on its backspace mode; only 127 was named."""
    assert resolve(csi_u(cp), Keys) is Keys.ControlH
