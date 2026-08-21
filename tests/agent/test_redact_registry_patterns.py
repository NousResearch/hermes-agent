"""Tests for the registry-fed exact-value redaction pass.

The pass loads an optional user pattern file (``HERMES_REDACT_PATTERNS``,
default ``~/.hermes/state/redaction/redact_patterns.json``) and
masks exact registered secrets plus registered ``KEY=value`` forms whose
keys the built-in keyword families do not recognize (e.g. ``PIN=1234``).

Mask styles (D188): absent/unknown -> length-driven default (2 visible head +
2 visible tail chars, floor 12 — shorter values fully masked); ``"full"`` ->
nothing visible; ``{"head", "tail", "floor"}`` -> custom. Fail-safe: a
missing, unreadable, or broken file degrades to a no-op (a broken file keeps
the last-good pattern set active). File-content redaction uses a non-reusable
sentinel (issue #35519 semantics) so masked values can never be written back
over the real file.
"""

import json

import pytest

from agent.redact import redact_sensitive_text


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Ensure HERMES_REDACT_SECRETS is not disabled by prior test imports."""
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


@pytest.fixture
def patterns_file(tmp_path, monkeypatch):
    """Point HERMES_REDACT_PATTERNS at a throwaway file and reset the cache."""
    p = tmp_path / "patterns.json"
    monkeypatch.setenv("HERMES_REDACT_PATTERNS", str(p))
    monkeypatch.setattr(
        "agent.redact._PATTERNS_CACHE",
        {"mtime": None, "lit_re": None, "key_re": None,
         "lit_masks": {}, "default_mask": None, "broken": False},
    )
    return p


def _write(p, literals, keys=None):
    p.write_text(json.dumps({
        "literals": literals,
        "key_patterns": {k: True for k in (keys or [])},
    }))


def test_exact_literal_partially_masked(patterns_file):
    _write(patterns_file, ["SUPERSECRET" + "TESTVALUE12345"], [])
    out = redact_sensitive_text("the token SUPERSECRETTESTVALUE12345 appears")
    assert "SUPERSECRETTESTVALUE12345" not in out
    assert "SU...45" in out  # 2+2 default style, head/tail preserved


def test_short_literal_fully_masked(patterns_file):
    _write(patterns_file, ["TINY9"], [])
    out = redact_sensitive_text("value TINY9 here")
    assert "TINY9" not in out
    assert "TI...Y9" not in out
    assert "***" in out


def test_full_mask_style(patterns_file):
    _write(patterns_file, [{"value": "FULLSECRET" + "VALUE123", "mask": "full"}], [])
    out = redact_sensitive_text("the FULLSECRETVALUE123 is here")
    assert "FULLSECRETVALUE123" not in out
    assert "FU" not in out
    assert "***" in out


def test_custom_mask_style(patterns_file):
    _write(patterns_file, [{"value": "CUSTOMSECRETVAL", "mask": {"head": 4, "tail": 4, "floor": 12}}], [])
    out = redact_sensitive_text("the CUSTOMSECRETVAL value")
    assert "CUSTOMSECRETVAL" not in out
    assert "CUST...TVAL" in out


def test_invalid_mask_style_falls_back_to_default(patterns_file):
    _write(patterns_file, [
        {"value": "BADSTYLESECRET", "mask": {"head": "x"}},
        {"value": "UNKNOWNSTYLESECRET", "mask": "nope"},
    ], [])
    out = redact_sensitive_text("BADSTYLESECRET and UNKNOWNSTYLESECRET here")
    assert "BADSTYLESECRET" not in out and "UNKNOWNSTYLESECRET" not in out
    assert "BA...ET" in out and "UN...ET" in out


def test_key_pattern_masked_keeps_separator(patterns_file):
    _write(patterns_file, [], ["PIN", "GITHUB_PAT"])
    out = redact_sensitive_text("PIN=1234 and GITHUB_PAT=xYzQwEr" + "12345AbCd")
    assert "PIN=1234" not in out
    assert "PIN=***" in out
    assert "xYzQwEr12345AbCd" not in out
    assert "GITHUB_PAT=xY...Cd" in out  # long value partially masked


def test_key_pattern_json_forms_masked(patterns_file):
    """External-audit F1: JSON "KEY": "value" and 'KEY': 'value' forms."""
    _write(patterns_file, [], ["x_passphrase"])
    for text in (
        '{"x_passphrase": "hunter2secret99"}',
        '"x_passphrase": "hunter2secret99"',
        "'x_passphrase': 'hunter2secret99'",
    ):
        out = redact_sensitive_text(text)
        assert "hunter2secret99" not in out
        assert "hu...99" in out  # 2+2 partial


def test_key_pattern_quoted_multiple_words_masked(patterns_file):
    """External-audit F1: quoted values containing spaces."""
    _write(patterns_file, [], ["x_passphrase"])
    out = redact_sensitive_text('x_passphrase="hello world secret"')
    assert "hello world secret" not in out
    assert "he...et" in out


def test_key_pattern_preserves_separator_whitespace(patterns_file):
    _write(patterns_file, [], ["x_passphrase"])
    out = redact_sensitive_text("x_passphrase = 1234")
    assert "1234" not in out
    assert "x_passphrase = ***" in out


def test_key_pattern_empty_value_passthrough(patterns_file):
    _write(patterns_file, [], ["x_passphrase"])
    for text in ('x_passphrase=""', "x_passphrase=''",
                 "x_passphrase=", "x_passphrase:"):
        assert redact_sensitive_text(text) == text


def test_cache_keyed_by_path(tmp_path, monkeypatch):
    """External-audit F3: cache must not serve a stale set after
    HERMES_REDACT_PATTERNS switches to a different file with the same mtime."""
    import os
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"literals": ["ALPHA" + "SECRETVALUE1"]}))
    b.write_text(json.dumps({"literals": ["BETA" + "SECRETVALUE2"]}))
    st = os.stat(a).st_mtime_ns
    os.utime(b, ns=(st, st))  # force identical mtimes
    monkeypatch.setattr(
        "agent.redact._PATTERNS_CACHE",
        {"mtime": None, "path": None, "lit_re": None, "key_re": None,
         "lit_masks": {}, "default_mask": None, "broken": False},
    )
    monkeypatch.setenv("HERMES_REDACT_PATTERNS", str(a))
    assert "ALPHASECRETVALUE1" not in redact_sensitive_text("ALPHASECRETVALUE1")
    monkeypatch.setenv("HERMES_REDACT_PATTERNS", str(b))
    out = redact_sensitive_text("ALPHASECRETVALUE1 and BETASECRETVALUE2")
    assert "BETASECRETVALUE2" not in out  # B loaded despite identical mtime
    assert "ALPHASECRETVALUE1" in out     # A's set no longer applies


def test_file_read_uses_nonreusable_sentinel(patterns_file):
    _write(patterns_file, ["SUPERSECRET" + "TESTVALUE12345"], [])
    out = redact_sensitive_text("value SUPERSECRETTESTVALUE12345", file_read=True)
    assert "SUPERSECRETTESTVALUE12345" not in out
    assert out != "value ***"


def test_longest_first_prefix_overlap(patterns_file):
    _write(patterns_file, ["MYPREFIXabc", "MYPREFIXabcdefgh123456"], [])
    out = redact_sensitive_text("x MYPREFIXabcdefgh123456 y")
    assert "MYPREFIXabcdefgh123456" not in out
    assert "defgh123456" not in out  # shorter alternative must not win


def test_short_text_still_masked(patterns_file):
    _write(patterns_file, [], ["PIN"])
    out = redact_sensitive_text("PIN=12")  # 6 chars — no len<8 early return
    assert "PIN=12" not in out
    assert "PIN=***" in out


def test_broken_file_keeps_last_good_set(patterns_file):
    _write(patterns_file, ["OLD" + "SECRETVALUE1"], [])
    assert "OLDSECRETVALUE1" not in redact_sensitive_text("OLDSECRETVALUE1")
    patterns_file.write_text("{ this is not json")
    out = redact_sensitive_text("still OLDSECRETVALUE1 here")
    assert "OLDSECRETVALUE1" not in out  # no unmasked gap while broken
    _write(patterns_file, ["NEW" + "SECRETVALUE2"], [])
    out = redact_sensitive_text("NEWSECRETVALUE2 and OLDSECRETVALUE1")
    assert "NEWSECRETVALUE2" not in out  # auto-recovery on repair
    assert "OLDSECRETVALUE1" in out      # stale set dropped


def test_missing_file_is_fail_safe_noop(monkeypatch):
    monkeypatch.setenv("HERMES_REDACT_PATTERNS", "/nonexistent/patterns.json")
    monkeypatch.setattr(
        "agent.redact._PATTERNS_CACHE",
        {"mtime": None, "lit_re": None, "key_re": None,
         "lit_masks": {}, "default_mask": None, "broken": False},
    )
    t = "plain text and a built-in sk-" + "abcdef1234567890"
    out = redact_sensitive_text(t)
    assert "sk-" + "abcdef1234567890" not in out  # built-in families still run
    assert "plain text" in out


def test_rotation_picked_up_via_mtime(patterns_file):
    _write(patterns_file, ["OLD" + "SECRETVALUE1"], [])
    assert "OLDSECRETVALUE1" not in redact_sensitive_text("OLDSECRETVALUE1")
    _write(patterns_file, ["ROTATED" + "SECRETVALUE2"], [])
    out = redact_sensitive_text("fresh ROTATEDSECRETVALUE2")
    assert "ROTATEDSECRETVALUE2" not in out


def test_default_path_resolves_via_hermes_home(tmp_path, monkeypatch):
    """Default path follows $HERMES_HOME (profiles / relocated installs),
    never a hardcoded ~/.hermes."""
    cf = tmp_path / "state" / "redaction"
    cf.mkdir(parents=True)
    (cf / "redact_patterns.json").write_text(json.dumps(
        {"literals": ["PROFILEHOME" + "SECRETVALUE1"]}))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REDACT_PATTERNS", raising=False)
    monkeypatch.setattr(
        "agent.redact._PATTERNS_CACHE",
        {"mtime": None, "lit_re": None, "key_re": None,
         "lit_masks": {}, "default_mask": None, "broken": False},
    )
    out = redact_sensitive_text("token PROFILEHOMESECRETVALUE1 here")
    assert "PROFILEHOMESECRETVALUE1" not in out
