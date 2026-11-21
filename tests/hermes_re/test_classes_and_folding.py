"""Shorthand classes, bracket classes and IGNORECASE folding with stdlib meaning (fixed tables)."""
import re as std
import unicodedata

import hermes_re as hre

# A sample of code points where RE2's own classes or folding differ from Python's, plus ordinary ones.
SAMPLE = sorted({*range(0x250), 0x345, 0x390, 0x399, 0x3A9, 0x3B9, 0x3C2, 0x3C3, 0x3F4, 0x660, 0x663, 0x1680, 0x1C89, 0x1C8A,
                 0x1E9E, 0x1FBE, *range(0x2000, 0x2030), 0x205F, 0x2126, 0x212A, 0x212B, 0x3000, 0x65E5, 0xA7CB, 0xA7DC, 0xFB05,
                 0xFB06, 0xFF11, 0xFF21, 0x10400, 0x10428, 0x10D50, 0x10D70, 0x1D7CE, 0x1E5F0, 0x1F600, 0xE0041, 0x10FFFF})
CLASSES = [r"\s", r"\S", r"\d", r"\D", r"\w", r"\W", r"[\s\S]", r"[^\S\n]", r"[\W_]", r"[^\W\d_]", r"[^\S \t]",
           r"[^\W\dugUCDhpRrtT]", r"[<\S]", r"[^\W_]", r"[a-z]", r"[A-Za-z0-9_]", r"[i]", r"[^i]", r"[\d\s]", r"[^\s\S]",
           r"[\w.-]", r"[^\D5]", r"[\x00-\x7f]", r"[\u00e0-\u00ff]", r"[\]\w]", r"[]a]", r"[\b]", r"[a\-z]", r"[k]", r"[\u017f]",
           r"[[:alpha:]]", r"."]
FLAG_SETS = (0, 256, 2, 258)


def test_classes_agree_with_stdlib_on_the_sample_code_points():
    for flags in FLAG_SETS:
        for source in CLASSES:
            mine, ref = hre.compile(source, flags), std.compile(source, flags)
            wrong = [hex(cp) for cp in SAMPLE if (mine.fullmatch(chr(cp)) is None) != (ref.fullmatch(chr(cp)) is None)]
            assert not wrong, f"{source} flags={flags}: differs at {wrong[:8]}"


def test_literal_folding_agrees_with_stdlib_on_the_sample():
    cased = [cp for cp in SAMPLE if chr(cp).lower() != chr(cp) or chr(cp).upper() != chr(cp)]
    text = "".join(map(chr, SAMPLE))
    for flags in (2, 258):
        for cp in cased:
            source = std.escape(chr(cp))
            assert hre.findall(hre.escape(chr(cp)), text, flags) == std.findall(source, text, flags), (hex(cp), flags)


def test_dotted_and_dotless_i_and_kelvin_and_long_s():
    assert hre.fullmatch("i", "\u0130", hre.I) and hre.fullmatch("i", "\u0131", hre.I) and hre.fullmatch("\u0130", "I", hre.I)
    assert hre.fullmatch("k", "\u212a", hre.I) and hre.fullmatch("s", "\u017f", hre.I)
    assert not hre.fullmatch("k", "\u212a", hre.I | hre.A) and not hre.fullmatch("s", "\u017f", hre.I | hre.A)
    assert hre.search("api_key", "AP\u0131_KEY", hre.I) is not None


def test_scoped_inline_flags_switch_folding():
    for source, text in ((r"(?i:a)b", "Ab"), (r"(?i:a)b", "AB"), (r"a(?i:b)c", "aBc"), (r"(?i)a(?-i:b)", "AB"),
                         (r"(?i)a(?-i:b)", "Ab"), (r"(?i)[a-c]+", "aBC"), (r"(?i)\x69", "\u0130"), (r"(?i)\151", "I")):
        assert bool(hre.fullmatch(source, text)) == bool(std.fullmatch(source, text)), (source, text)


def test_escapes_are_read_as_stdlib_reads_them():
    cases = ((r"\x41\u00e9\U0001f600", "A\u00e9\U0001f600"), (r"\0", "\0"), (r"\012", "\n"), (r"\101", "A"), (r"\Aa\Z", "a"),
             ("\\\u00e9", "\u00e9"), (r"[\x41-\x43]+", "ABC"), (r"a\Z", "a\n"), (r"\t\n\r\f\v\a", "\t\n\r\f\v\a"))
    for source, text in cases:
        assert bool(hre.fullmatch(source, text)) == bool(std.fullmatch(source, text)), source


def test_the_generated_unicode_table_knows_this_interpreter():
    from hermes_re import _tables
    assert unicodedata.unidata_version in _tables.TABLES, (
        f"no delta table for Unicode {unicodedata.unidata_version}: run gen_tables.py on this interpreter")


def test_the_generated_unicode_table_is_exact_for_this_interpreter_and_engine():
    # The table corrects RE2's `[\p{L}\p{N}_]` / `\p{Nd}` to Python's `\w` / `\d`. Recompute the delta from the
    # running interpreter (str predicates) and the installed engine, and compare with the shipped entry.
    from re2 import _re2
    from hermes_re import _tables
    points = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    text = "".join(map(chr, points))
    data = text.encode("utf-8")

    def engine_set(source):
        options = _re2.RE2.Options()
        options.log_errors = False
        program = _re2.RE2(source.encode(), options)
        found, pos, anchor = set(), 0, _re2.RE2.Anchor.UNANCHORED
        while True:
            start, end = program.Match(anchor, data, pos, len(data))[0]
            if start < 0:
                return found
            found.add(ord(data[start:end].decode("utf-8")))
            pos = end

    def ranges(values):
        out = []
        for cp in sorted(values):
            if out and out[-1][1] == cp - 1:
                out[-1][1] = cp
            else:
                out.append([cp, cp])
        return [tuple(pair) for pair in out]

    word = {ord(ch) for ch in text if ch.isalnum() or ch == "_"}
    digit = {ord(ch) for ch in text if ch.isdecimal()}
    space = {ord(ch) for ch in text if ch.isspace()}
    engine_word, engine_digit = engine_set(r"[\p{L}\p{N}_]"), engine_set(r"\p{Nd}")
    shipped = _tables.TABLES[unicodedata.unidata_version]
    assert ranges(word - engine_word) == list(shipped["W_ADD"])
    assert ranges(engine_word - word) == list(shipped["W_SUB"])
    assert ranges(digit - engine_digit) == list(shipped["D_ADD"])
    assert ranges(engine_digit - digit) == list(shipped["D_SUB"])
    assert ranges(space) == list(shipped["SPACE"])


def test_word_digit_space_are_exact_on_every_code_point():
    # The end-to-end proof for the three bare shorthands (about one second).
    text = "".join(chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF)
    for source in (r"\w", r"\d", r"\s"):
        assert hre.findall(source, text) == std.findall(source, text), source
