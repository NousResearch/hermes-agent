"""Tests for multi-line ``read_text``/``write_text`` detection in
``scripts/check-windows-footguns.py``.

The line-based scanner deliberately skips any call whose closing paren is not
on the matched line: ``encoding=`` may sit on a continuation line, and flagging
that would be a false positive. The consequence is a blind spot for calls that
genuinely lack ``encoding=`` and happen to wrap::

    path.write_text(json.dumps({
        "k": "v",
    }))

That shape is common in tests that seed JSON fixtures, and it read as clean for
as long as the gate was line-oriented — PR #95486 fixed 18 findings the gate
reported and left 6 of this shape behind, all in files the gate had just
declared clean.

``_wrapped_encoding_calls`` closes the gap with an AST pass, which answers
exactly rather than heuristically: the outer call's keywords are on the node.
The line scanner is untouched, so its false-positive guarantees still hold.

See issue #37423 and the #71014 / read_text campaign.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINTER_PATH = REPO_ROOT / "scripts" / "check-windows-footguns.py"


def _load_linter_module():
    """Import the linter script as a module (it's not a package).

    Register the module in sys.modules BEFORE exec_module so that
    ``@dataclass`` can resolve ``cls.__module__`` (CPython 3.11+ requirement).
    """
    spec = importlib.util.spec_from_file_location("check_windows_footguns", LINTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_windows_footguns"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def linter():
    return _load_linter_module()


def _linenos(linter, source: str) -> list[int]:
    return [lineno for lineno, _name in linter._wrapped_encoding_calls(source)]


# --- calls that MUST be flagged -------------------------------------------


def test_wrapped_write_text_without_encoding_is_flagged(linter):
    """The exact shape PR #95486 left behind: json.dumps wrapping the args."""
    source = (
        "import json\n"
        "def f(d):\n"
        '    (d / "a.json").write_text(json.dumps({\n'
        '        "k": "v",\n'
        "    }))\n"
    )
    assert _linenos(linter, source) == [3]


def test_wrapped_read_text_without_encoding_is_flagged(linter):
    source = "def f(p):\n    return p.read_text(\n    )\n"
    assert _linenos(linter, source) == [2]


# --- calls that MUST NOT be flagged ---------------------------------------


def test_encoding_on_a_continuation_line_is_not_flagged(linter):
    """The false positive the line scanner avoids by skipping wrapped calls.

    The AST pass must not reintroduce it.
    """
    source = (
        "import json\n"
        "def f(d):\n"
        '    (d / "a.json").write_text(\n'
        '        json.dumps({"k": "v"}),\n'
        '        encoding="utf-8",\n'
        "    )\n"
    )
    assert _linenos(linter, source) == []


def test_binary_mode_open_is_not_flagged(linter):
    """Binary mode takes no encoding= — demanding one would be a TypeError."""
    source = 'def f(p):\n    return open(\n        p,\n        "rb",\n    )\n'  # windows-footgun: ok
    assert _linenos(linter, source) == []


def test_os_open_is_not_flagged(linter):
    """``os.open`` is the POSIX fd-level call and rejects ``encoding=``.

    This is the atomic-write prelude used across hermes_cli/auth.py,
    tools/mcp_oauth.py and the platform plugins: os.open() for the fd, then
    os.fdopen(..., encoding="utf-8") for the text wrapper. Flagging the
    os.open() half would demand a kwarg that raises TypeError.
    """
    source = (
        "import os\n"
        "import stat\n"
        "def f(p):\n"
        "    fd = os.open(\n"
        "        str(p),\n"
        "        os.O_WRONLY | os.O_CREAT,\n"
        "        stat.S_IRUSR,\n"
        "    )\n"
        '    with os.fdopen(fd, "w", encoding="utf-8") as fh:\n'
        '        fh.write("x")\n'
    )
    assert _linenos(linter, source) == []


def test_attribute_open_is_left_to_the_line_rule(linter):
    """``x.open(...)`` receivers are ambiguous from the AST alone.

    urllib openers, zipfile, tarfile and mocks all expose ``.open()`` and do
    not take ``encoding=``; scripts/ci/live_comment.py has exactly this shape.
    """
    source = (
        "import urllib.request\n"
        "def f(url, opener):\n"
        "    return opener.open(urllib.request.Request(url, headers={\n"
        '        "Accept": "application/json",\n'
        "    }), timeout=30)\n"
    )
    assert _linenos(linter, source) == []


def test_kwargs_splat_is_trusted(linter):
    """``**kw`` may carry encoding — same trust the regex rule extends."""
    source = "def f(p, **kw):\n    return p.read_text(\n        **kw,\n    )\n"
    assert _linenos(linter, source) == []


def test_single_line_calls_are_left_to_the_line_scanner(linter):
    """No double-reporting: the AST pass only owns calls that wrap."""
    source = 'def f(p):\n    return p.read_text()\n'
    assert _linenos(linter, source) == []


def test_unparseable_source_returns_nothing(linter):
    """A syntactically broken file must not raise — the line scanner covers it."""
    assert linter._wrapped_encoding_calls("def broken(:\n") == []


# --- end-to-end through scan_file -----------------------------------------


def test_scan_file_reports_wrapped_call(linter, tmp_path):
    """The finding reaches scan_file output, not just the helper."""
    f = tmp_path / "seed.py"
    f.write_text(
        "import json\n"
        "def f(d):\n"
        '    (d / "a.json").write_text(json.dumps({\n'
        '        "k": "v",\n'
        "    }))\n",
        encoding="utf-8",
    )
    findings = linter.scan_file(f, linter.FOOTGUNS)
    assert [lineno for lineno, _line, _fg in findings] == [3]
    assert "read_text" in findings[0][2].name


def test_scan_file_honours_suppression_marker(linter, tmp_path):
    """``# windows-footgun: ok`` on the opening line still suppresses."""
    f = tmp_path / "seed.py"
    f.write_text(
        "import json\n"
        "def f(d):\n"
        '    (d / "a.json").write_text(json.dumps({  # windows-footgun: ok\n'
        '        "k": "v",\n'
        "    }))\n",
        encoding="utf-8",
    )
    assert linter.scan_file(f, linter.FOOTGUNS) == []


def test_scan_file_does_not_double_report(linter, tmp_path):
    """A single-line bare call is reported exactly once."""
    f = tmp_path / "seed.py"
    f.write_text("def f(p):\n    return p.read_text()\n", encoding="utf-8")
    findings = linter.scan_file(f, linter.FOOTGUNS)
    assert len(findings) == 1
