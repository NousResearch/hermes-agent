"""Tests for the encoding-direction footgun rules in
``scripts/check-windows-footguns.py``.

Two rules enforce the repo encoding policy (reads ``utf-8-sig``, writes
``utf-8``):

- ``read with encoding='utf-8' (BOM-intolerant — use 'utf-8-sig')`` —
  Windows tooling (PowerShell Set-Content/Out-File, some editors)
  BOM-prefixes files it touches; a plain-utf-8 read hands ``'\\ufeff{...'``
  to ``json.load``, which fails with "Expecting value". Live case: PR #3 —
  a BOM'd ``install-stamp.json`` made ``read_build_info`` demote the tree
  to ``unknown``.
- ``write with encoding='utf-8-sig' (emits a BOM)`` — the inverse: writing
  with ``utf-8-sig`` emits the exact bytes the read rule exists to
  tolerate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINTER_PATH = REPO_ROOT / "scripts" / "check-windows-footguns.py"

READ_RULE = "read with encoding='utf-8' (BOM-intolerant — use 'utf-8-sig')"
WRITE_RULE = "write with encoding='utf-8-sig' (emits a BOM)"


def _load_linter_module():
    """Import the linter script as a module (it's not a package).

    Register the module in sys.modules BEFORE exec_module so that
    ``@dataclass`` can resolve ``cls.__module__`` (CPython 3.11+).
    """
    spec = importlib.util.spec_from_file_location(
        "check_windows_footguns", LINTER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_windows_footguns"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def linter():
    return _load_linter_module()


def _find_footgun(linter, name: str):
    for fg in linter.FOOTGUNS:
        if fg.name == name:
            return fg
    pytest.fail(f"Footgun rule '{name}' not found in FOOTGUNS")


def _scan_line(linter, line: str, footgun_name: str) -> bool:
    """Return True if the given line triggers the named footgun rule.

    Replicates the relevant checks from scan_file(): suppression marker,
    guard hints, comment stripping, then pattern + post_filter — so the
    test exercises the real detection path.
    """
    fg = _find_footgun(linter, footgun_name)
    if linter.SUPPRESS_MARKER.search(line):
        return False
    if any(hint in line for hint in linter.GUARD_HINTS):
        return False
    code = linter._strip_code(line)
    if not code.strip():
        return False
    match = fg.pattern.search(code)
    if not match:
        return False
    if fg.post_filter is not None:
        try:
            if not fg.post_filter(match, line):
                return False
        except (IndexError, AttributeError):
            return False
    return True


# ---------------------------------------------------------------------------
# READ rule — plain utf-8 on read-shaped lines SHOULD be flagged
# ---------------------------------------------------------------------------


class TestReadRuleDetection:
    def test_flags_read_text_plain_utf8(self, linter):
        line = '    data = path.read_text(encoding="utf-8")'
        assert _scan_line(linter, line, READ_RULE)

    def test_flags_open_read_mode_plain_utf8(self, linter):
        line = "    with open(path, 'r', encoding='utf-8') as f:"
        assert _scan_line(linter, line, READ_RULE)

    def test_flags_open_omitted_mode_plain_utf8(self, linter):
        # open() defaults to mode 'r' — still a read.
        line = "    with open(path, encoding='utf-8') as f:"
        assert _scan_line(linter, line, READ_RULE)

    def test_flags_fdopen_read_mode_plain_utf8(self, linter):
        line = "    f = os.fdopen(fd, 'r', encoding='utf-8')"
        assert _scan_line(linter, line, READ_RULE)

    def test_flags_underscore_spelling(self, linter):
        line = '    data = path.read_text(encoding="utf_8")'
        assert _scan_line(linter, line, READ_RULE)

    def test_flags_mode_keyword_read(self, linter):
        line = "    with open(path, mode='r', encoding='utf-8') as f:"
        assert _scan_line(linter, line, READ_RULE)


class TestReadRuleNegatives:
    def test_does_not_flag_utf8_sig_read(self, linter):
        # The policy-compliant form. The pattern must stop at the closing
        # quote of 'utf-8' and never match 'utf-8-sig'.
        line = '    data = path.read_text(encoding="utf-8-sig")'
        assert not _scan_line(linter, line, READ_RULE)

    def test_does_not_flag_write_text_plain_utf8(self, linter):
        # Writes with plain utf-8 are the CORRECT form.
        line = '    path.write_text(data, encoding="utf-8")'
        assert not _scan_line(linter, line, READ_RULE)

    def test_does_not_flag_open_write_mode_plain_utf8(self, linter):
        line = "    with open(path, 'w', encoding='utf-8') as f:"
        assert not _scan_line(linter, line, READ_RULE)

    def test_does_not_flag_append_mode(self, linter):
        line = "    with open(path, 'a', encoding='utf-8') as f:"
        assert not _scan_line(linter, line, READ_RULE)

    def test_does_not_flag_plus_mode(self, linter):
        # 'r+' can write — treat as write-capable, not a read.
        line = "    with open(path, 'r+', encoding='utf-8') as f:"
        assert not _scan_line(linter, line, READ_RULE)

    def test_does_not_flag_subprocess_encoding(self, linter):
        # Not a file read; the rule stays quiet on unclassifiable lines.
        line = "    subprocess.run(cmd, text=True, encoding='utf-8')"
        assert not _scan_line(linter, line, READ_RULE)

    def test_does_not_flag_suppressed_line(self, linter):
        line = (
            "    data = path.read_text(encoding='utf-8')"
            "  # windows-footgun: ok — file is repo-owned, never BOM'd"
        )
        assert not _scan_line(linter, line, READ_RULE)

    def test_does_not_flag_comment_only_line(self, linter):
        line = "    # use path.read_text(encoding='utf-8') here"
        assert not _scan_line(linter, line, READ_RULE)


# ---------------------------------------------------------------------------
# WRITE rule — utf-8-sig on write-shaped lines SHOULD be flagged
# ---------------------------------------------------------------------------


class TestWriteRuleDetection:
    def test_flags_write_text_utf8_sig(self, linter):
        line = '    path.write_text(data, encoding="utf-8-sig")'
        assert _scan_line(linter, line, WRITE_RULE)

    def test_flags_open_write_mode_utf8_sig(self, linter):
        line = "    with open(path, 'w', encoding='utf-8-sig') as f:"
        assert _scan_line(linter, line, WRITE_RULE)

    def test_flags_append_mode_utf8_sig(self, linter):
        line = "    with open(path, 'a', encoding='utf-8-sig') as f:"
        assert _scan_line(linter, line, WRITE_RULE)

    def test_flags_fdopen_write_mode_utf8_sig(self, linter):
        line = "    f = os.fdopen(fd, 'w', encoding='utf-8-sig')"
        assert _scan_line(linter, line, WRITE_RULE)

    def test_flags_path_open_append_utf8_sig(self, linter):
        # Path.open puts the mode in the FIRST argument — the original
        # classifier only understood the builtin second-arg shape, so the
        # live sweep rewrote 18 append/write sites to utf-8-sig (BOM per
        # append in jsonl logs; caught by test_lifecycle_ledger).
        line = '    with path.open("a", encoding="utf-8-sig") as fh:'
        assert _scan_line(linter, line, WRITE_RULE)

    def test_flags_path_open_plus_mode_utf8_sig(self, linter):
        # r+/a+ can write at position 0 — write-shaped.
        line = '    lock = path.open("a+", encoding="utf-8-sig")'
        assert _scan_line(linter, line, WRITE_RULE)

    def test_flags_underscore_spelling(self, linter):
        line = '    path.write_text(data, encoding="utf_8_sig")'
        assert _scan_line(linter, line, WRITE_RULE)


class TestWriteRuleNegatives:
    def test_does_not_flag_utf8_sig_read(self, linter):
        # Reads with utf-8-sig are the CORRECT form.
        line = '    data = path.read_text(encoding="utf-8-sig")'
        assert not _scan_line(linter, line, WRITE_RULE)

    def test_does_not_flag_open_read_mode_utf8_sig(self, linter):
        line = "    with open(path, 'r', encoding='utf-8-sig') as f:"
        assert not _scan_line(linter, line, WRITE_RULE)

    def test_does_not_flag_open_omitted_mode_utf8_sig(self, linter):
        # Omitted mode defaults to 'r' — a read.
        line = "    with open(path, encoding='utf-8-sig') as f:"
        assert not _scan_line(linter, line, WRITE_RULE)

    def test_does_not_flag_plain_utf8_write(self, linter):
        line = '    path.write_text(data, encoding="utf-8")'
        assert not _scan_line(linter, line, WRITE_RULE)

    def test_does_not_flag_unclassifiable_line(self, linter):
        # No file-opening call — conservative default is no flag.
        line = "    codec = 'utf-8-sig'"
        assert not _scan_line(linter, line, WRITE_RULE)

    def test_does_not_flag_nested_call_first_arg(self, linter):
        # Regression: the first live sweep false-positived this line from
        # _startup_fast.py — the old mode regex swallowed the nested
        # ``join(`` paren and captured "install-stamp.json" as the mode
        # (its 'a' made it look write-shaped). Omitted mode = read.
        line = (
            '    with open(os.path.join(root, "install-stamp.json"), '
            'encoding="utf-8-sig") as handle:'
        )
        assert not _scan_line(linter, line, WRITE_RULE)

    def test_does_not_flag_path_open_read_mode_utf8_sig(self, linter):
        line = '    with path.open("r", encoding="utf-8-sig") as fh:'
        assert not _scan_line(linter, line, WRITE_RULE)

    def test_read_rule_does_not_rewrite_path_open_append(self, linter):
        # The inverse guard on the READ rule: an append-mode Path.open with
        # plain utf-8 is CORRECT and must never be flagged for conversion.
        line = '    with path.open("a", encoding="utf-8") as fh:'
        assert not _scan_line(linter, line, READ_RULE)

    def test_read_rule_skips_method_open_without_mode(self, linter):
        # `.open(` with no recognizable mode is unclassified — the sweep
        # must never rewrite on a guess about the receiver.
        line = '    with something.open(encoding="utf-8") as fh:'
        assert not _scan_line(linter, line, READ_RULE)

    def test_does_not_flag_suppressed_line(self, linter):
        line = (
            "    path.write_text(data, encoding='utf-8-sig')"
            "  # windows-footgun: ok — downstream consumer requires a BOM"
        )
        assert not _scan_line(linter, line, WRITE_RULE)


# ---------------------------------------------------------------------------
# Suppression marker cannot be a bare comment mention (the real invariant —
# a comment that merely NAMES the marker text elsewhere must not exempt).
# ---------------------------------------------------------------------------


class TestShapeHelpers:
    def test_read_shaped_read_text(self, linter):
        assert linter._is_read_shaped('x = p.read_text(encoding="utf-8")')

    def test_read_shaped_rejects_write_text(self, linter):
        assert not linter._is_read_shaped('p.write_text(d, encoding="utf-8")')

    def test_write_shaped_x_mode(self, linter):
        assert linter._is_write_shaped("open(p, 'x', encoding='utf-8-sig')")

    def test_write_shaped_rejects_no_call(self, linter):
        assert not linter._is_write_shaped("enc = 'utf-8-sig'")

    def test_read_shaped_rejects_no_call(self, linter):
        # No open/fdopen/read_text on the line at all.
        assert not linter._is_read_shaped("enc = 'utf-8'")
