"""load_env() parity with the pinned python-dotenv (==1.2.2).

#76544's root cause was two hand-rolled .env loaders diverging on inline
comments and quoting. Both loaders now parse through python-dotenv
(``dotenv_values(..., interpolate=False)``), so these tests pin the
*contract* the loaders must keep: inline comments stripped, quoting and
escapes honored, malformed lines dropped, ``export`` prefixes accepted,
no ``${VAR}`` expansion, and agreement with dotenv itself on a corpus of
edge shapes plus a deterministic fuzz sample.
"""

import random
import tempfile
from pathlib import Path

from dotenv import dotenv_values

from hermes_cli import config as hermes_config


def _load(tmp_path, monkeypatch, text):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text(text, encoding="utf-8")
    hermes_config.invalidate_env_cache()
    return hermes_config.load_env()


def test_load_env_strips_inline_comment_on_base_url(tmp_path, monkeypatch):
    # The #76544 report path: a commented base URL must resolve without the
    # comment text corrupting it (that corruption caused an opaque 404).
    env_vars = _load(
        tmp_path,
        monkeypatch,
        "MINIMAX_BASE_URL=https://api.minimax.io/anthropic  # official\n"
        "MINIMAX_API_KEY=sk-cp-test\n",
    )
    assert env_vars["MINIMAX_BASE_URL"] == "https://api.minimax.io/anthropic"
    assert env_vars["MINIMAX_API_KEY"] == "sk-cp-test"


def test_load_env_does_not_interpolate(tmp_path, monkeypatch):
    # Critical regression for the dotenv switch: dotenv interpolates by
    # default, but Hermes never expands .env values. ``interpolate=False``
    # must stay on or ``${HOME}`` would silently become the home path.
    env_vars = _load(tmp_path, monkeypatch, "PATH_REF=${HOME}\nLITERAL=$$HOME\n")
    assert env_vars["PATH_REF"] == "${HOME}"
    assert env_vars["LITERAL"] == "$$HOME"


def test_load_env_tolerates_invalid_utf8(tmp_path, monkeypatch):
    # load_env reads with errors="replace" so a partially-corrupted .env
    # (e.g. a truncated copy-paste) keeps loading instead of raising
    # UnicodeDecodeError and taking the whole loader down.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_bytes(b"GOOD=value\nBROKEN=abc\xffdef\n")
    hermes_config.invalidate_env_cache()

    env_vars = hermes_config.load_env()

    assert env_vars["GOOD"] == "value"
    assert env_vars["BROKEN"] == "abc\ufffddef"


def test_load_env_skips_unparseable_lines(tmp_path, monkeypatch):
    # Unclosed quote -> dotenv error binding -> line dropped (key absent),
    # same contract the old parser had.
    env_vars = _load(
        tmp_path,
        monkeypatch,
        'BROKEN="unclosed\n'
        "OK=value\n"
        "BARE_KEY\n"  # no ``=`` -> None value -> dropped
        "\n"
        "# full comment\n",
    )
    assert "BROKEN" not in env_vars
    assert "BARE_KEY" not in env_vars
    assert env_vars["OK"] == "value"


def test_load_env_handles_mixed_comment_shapes(tmp_path, monkeypatch):
    env_vars = _load(
        tmp_path,
        monkeypatch,
        "PLAIN=value\n"
        "COMMENTED=value  # trailing note\n"
        "HASH=value#notacomment\n"
        'QUOTED="a # b"\n'
        'ESCAPED="a\\"b"  # note\n'
        'SINGLE=\'a\\\'b\'\n'
        'TABBED=sk-abc\t# note\n'
        "EMPTY=\n"
        "SPACES  =   padded   \n",
    )
    assert env_vars["PLAIN"] == "value"
    assert env_vars["COMMENTED"] == "value"
    assert env_vars["HASH"] == "value#notacomment"
    assert env_vars["QUOTED"] == "a # b"
    assert env_vars["ESCAPED"] == 'a"b'
    assert env_vars["SINGLE"] == "a'b"
    assert env_vars["TABBED"] == "sk-abc"
    assert env_vars["EMPTY"] == ""
    assert env_vars["SPACES"] == "padded"


def test_load_env_handles_export_prefix(tmp_path, monkeypatch):
    # ``export KEY=...`` (bash-compatible form, #6659) parses as KEY, and
    # plain lines keep working alongside it.
    env_vars = _load(
        tmp_path,
        monkeypatch,
        "export API_KEY=sk-export\n"
        "NORMAL=value\n"
        "export  DOUBLE_SPACE=ok\n",
    )
    assert env_vars["API_KEY"] == "sk-export"
    assert env_vars["NORMAL"] == "value"
    assert env_vars["DOUBLE_SPACE"] == "ok"
    assert "export" not in env_vars


def test_load_env_last_duplicate_wins(tmp_path, monkeypatch):
    env_vars = _load(
        tmp_path,
        monkeypatch,
        "DUP=first\nDUP=second\n",
    )
    assert env_vars["DUP"] == "second"


def test_load_env_handles_multiline_quoted_value(tmp_path, monkeypatch):
    # dotenv supports quoted values spanning physical lines; the old
    # per-line parser could not. Both loaders share the same parser now.
    env_vars = _load(
        tmp_path,
        monkeypatch,
        'MULTI="line1\nline2"\nSINGLE=ok\n',
    )
    assert env_vars["MULTI"] == "line1\nline2"
    assert env_vars["SINGLE"] == "ok"


def test_load_env_preserves_multiline_value_whitespace(tmp_path, monkeypatch):
    # Every byte inside a cross-line quoted value is value content: the
    # opening line's trailing whitespace, the continuation lines' leading/
    # trailing whitespace, and their #-markers are NOT comments. Any
    # per-line normalization would corrupt the value (fuzz regression:
    # seed 76544 cases K24/K358/K393).
    env_vars = _load(
        tmp_path,
        monkeypatch,
        "MULTI='a  \nb  # not a comment\nc'\n",
    )
    assert env_vars["MULTI"] == "a  \nb  # not a comment\nc"


# ---------------------------------------------------------------------------
# Differential parity against the pinned python-dotenv (==1.2.2)
# ---------------------------------------------------------------------------
#
# The corpus and fuzz below compare load_env() output with dotenv's own
# parser on the SAME file. They guard the wiring (sanitizer + interpolate
# off + None filtering), not dotenv's internals.

_CORPUS = [
    "https://api.minimax.io/anthropic  # official",
    "sk-abc\t# note",
    "https://x.test/path#frag",
    "abc#def",
    '"a # b"',
    "'a # b'",
    '"a # b"  # note',
    'abc" # note',
    "abc' # note",
    '"unclosed',
    "'unclosed",
    '"a"x',
    "'a'x",
    '"a"#note',
    '"a"\t#note',
    '"a\\nb"',
    '"a\\tb"',
    '"a\\qb"',
    '"a\\\'b"',
    '"a\\\\b"',
    '"esc\\"q" # note',
    "'a' # c",
    "'a\\'b'",
    "'a\\\\b'",
    "'a\\nb'",
    "plain",
    "value # trailing note",
    "",
    "   ",
    "\\n",
    "12345",
    '"  spaced  "',
    '  "a"  ',
    "  spaced  ",
    "plain ",
    "  a  # c",
    '"a" x"y"',
    '"a""b"',
    '"a" # x"y"',
    '"a\\"',
    '"a\\"x"',
    '"a\\rb"',
    '"a\\\\""',
    '"a\\\\"',
    '"a\\\\" # c',
    '"a\\\\"x',
    '"x\\\\"y\\\\"',
    "'\\\\'  # c",
    "'x\\\\' # c",
    "'\\\\' # c x",
    "'\\\\'x",
]


def test_corpus_parity_with_python_dotenv(tmp_path, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        env_path = Path(d) / "t.env"
        for raw in _CORPUS:
            env_path.write_text("K=" + raw + "\n", encoding="utf-8")
            expected = dotenv_values(str(env_path), interpolate=False).get("K")
            env_vars = _load(tmp_path, monkeypatch, "K=" + raw + "\n")
            ours = env_vars.get("K")
            if expected is None:
                assert "K" not in env_vars, f"expected line drop for {raw!r}, got {ours!r}"
            else:
                assert ours == expected, f"{raw!r}: ours {ours!r}, dotenv {expected!r}"


def test_fuzz_parity_with_python_dotenv(tmp_path, monkeypatch):
    # Deterministic sample of generated edge shapes; whole-file dict
    # comparison against dotenv on the identical text.
    random.seed(76544)
    parts = [
        "a", "b", "c", " ", "\t", "#", "# note", '"', "'", "\\", '\\"', "x",
        ".", "/", "\\t", "\\r", "\\n", "sk-abc", "  ", '"a"  # z',
    ]
    lines = []
    for i in range(400):
        n = random.randint(1, 4)
        v = "".join(random.choice(parts) for _ in range(n))
        if random.random() < 0.45:
            v = '"' + v
        if random.random() < 0.45:
            v = "'" + v
        if random.random() < 0.35:
            v = v + "  # c"
        lines.append(f"K{i}={v}")
    text = "\n".join(lines) + "\n"

    env_vars = _load(tmp_path, monkeypatch, text)
    expected = dotenv_values(stream=type("S", (), {"read": lambda s: text})(), interpolate=False)
    assert env_vars == expected


def test_writer_reader_round_trip(tmp_path, monkeypatch):
    # Anything _quote_env_value writes must parse back to the original value
    # (the writer emits only double quotes and backslash escapes).
    from hermes_cli.config import _quote_env_value

    values = ["plain", "a b", "a # b", 'qu"ote', "back\\slash", "tab\there", ""]
    text = "".join(f"K{i}={_quote_env_value(v)}\n" for i, v in enumerate(values))
    env_vars = _load(tmp_path, monkeypatch, text)
    for i, v in enumerate(values):
        assert env_vars[f"K{i}"] == v
