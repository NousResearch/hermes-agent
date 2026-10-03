"""Unit + self-scan coverage for ``scripts/check_inert_tuple_assertions.py``.

Mirrors ``tests/scripts/test_windows_footguns_full_repo_scan.py``: exercise the
detector's decision function directly on both the shapes it must flag and the
shapes it must leave alone, then run the real checker over the whole repo and
require a clean exit — so a regression is caught by a normal pytest run, not
only by the CI step that invokes the script.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_inert_tuple_assertions.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("_check_inert_tuple_assertions", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


@pytest.mark.parametrize(
    "src",
    [
        # The four shapes that actually shipped in this tree.
        'sync.assert_not_awaited(), "a reconnect must not sync"',
        'store._save.assert_called_once_with(), (\n    "msg"\n)',
        'clear.assert_called(), "msg"',
        'holder[0].close.assert_called_once(), ("a", "b")',
        # Sibling shapes with the same defect.
        'assertSomething(x), "msg"',
        'm.assert_called_once, "msg"',
        'x == 1, "msg"',
        'not x, "msg"',
    ],
)
def test_detector_fires_on_each_inert_shape(src):
    """The detector must actually fire, not vacuously pass."""
    assert checker.inert_tuple_assertions(src, "fake.py")


@pytest.mark.parametrize(
    "src",
    [
        # The correct idioms — must NOT be flagged.
        "m.assert_called_once()",
        'assert m.called, "msg"',
        'assert x == 1, "msg"',
        "m.assert_called_once_with(1, 2)",
        # A genuine side-effect tuple statement that is not an assertion shape.
        'foo(), "msg"',
        "a, b",
        "(cache.clear(), locks.clear())",
        # A tuple that is assigned or returned is not discarded.
        'pair = (m.assert_called(), "msg")',
        'def f():\n    return m.assert_called(), "msg"',
    ],
)
def test_detector_allows_correct_shapes(src):
    assert not checker.inert_tuple_assertions(src, "fake.py")


def test_marker_suppresses_on_the_line_and_the_line_above():
    flagged = 'm.assert_called(), "msg"'
    assert checker.inert_tuple_assertions(flagged, "fake.py")
    assert not checker.inert_tuple_assertions(
        'm.assert_called(), "msg"  # inert-tuple: ok — deliberate\n', "fake.py"
    )
    assert not checker.inert_tuple_assertions(
        '# inert-tuple: ok — deliberate\nm.assert_called(), "msg"\n', "fake.py"
    )


def test_full_repo_scan_has_no_inert_tuple_assertions():
    """Run the real checker against the whole repo and require a clean exit."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"inert tuple assertion check failed:\n{result.stdout}\n{result.stderr}"
    )


def test_empty_scan_is_a_failure_not_a_pass(tmp_path):
    """A scan that finds zero files is vacuous and must fail loudly."""
    assert checker.main([str(tmp_path)]) == 1
