"""Regression tests for bug #39706 — ensure_uv() ValueError on update boundary.

Why: ``hermes update`` runs the call site from the *old*, already-imported
``hermes_cli.main`` against the *freshly pulled* ``managed_uv`` module.
When ``ensure_uv()``'s return arity flips (e.g. single-str → 2-tuple or back),
the mismatch causes a crash before deps can be installed:

  - ``uv_bin, fresh = ensure_uv()`` with a plain-str return walks the string's
    characters and raises ``ValueError: too many values to unpack (expected 2)``
  - ``uv_bin, fresh = ensure_uv()`` with a ``None`` return raises
    ``TypeError: cannot unpack non-iterable NoneType``

The fix is ``_UvResult`` — a ``str`` subclass that answers both conventions.

Test strategy:
  - 2-target unpack of ensure_uv() does not raise
  - Single-assignment caller gets a falsy-on-absent, truthy-on-present value
  - _UvResult.__iter__ yields exactly 2 items
  - Windows path still returns a plain str/None (subprocess-safe)
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestUvResultUpdateBoundaryContract:
    """_UvResult must be unpackable as (path, fresh) AND usable as a plain str."""

    def test_two_target_unpack_does_not_raise_when_uv_present(self, tmp_path):
        """Why: old in-memory call sites do ``a, b = ensure_uv()``.
        What: unpacking succeeds and yields (str_path, bool).
        Test: create a fake uv binary; call ensure_uv(); unpack into a, b.
        """
        from hermes_cli.managed_uv import _UvResult

        fake_uv = tmp_path / "uv"
        fake_uv.write_text("#!/bin/sh\n")
        fake_uv.chmod(0o755)

        result = _UvResult(str(fake_uv), False)
        a, b = result  # must not raise ValueError

        assert a == str(fake_uv)
        assert b is False

    def test_two_target_unpack_does_not_raise_when_uv_absent(self):
        """Why: the failure path (None) historically raised TypeError on unpack.
        What: _UvResult(None) is falsy and unpacks cleanly as (None, False).
        Test: unpack _UvResult(None); assert first element is None, second False.
        """
        from hermes_cli.managed_uv import _UvResult

        result = _UvResult(None, False)
        a, b = result  # must not raise TypeError

        assert a is None
        assert b is False

    def test_iter_yields_exactly_two_items(self, tmp_path):
        """Why: ValueError comes from too many items; __iter__ must yield 2.
        What: list(_UvResult(...)) has length 2.
        Test: _UvResult with a valid path; assert len(list(...)) == 2.
        """
        from hermes_cli.managed_uv import _UvResult

        fake_uv = tmp_path / "uv"
        fake_uv.write_text("#!/bin/sh\n")
        result = _UvResult(str(fake_uv), True)

        items = list(result)
        assert len(items) == 2, (
            f"Expected 2 items from __iter__, got {len(items)}: {items!r}"
        )

    def test_single_assignment_truthy_when_present(self, tmp_path):
        """Why: single callers check ``if not uv_bin:``; must behave as plain str.
        What: _UvResult behaves as truthy str when path is set.
        Test: assert bool(_UvResult(str(path))) is True.
        """
        from hermes_cli.managed_uv import _UvResult

        fake_uv = tmp_path / "uv"
        fake_uv.write_text("#!/bin/sh\n")
        result = _UvResult(str(fake_uv))

        assert result, f"_UvResult should be truthy when path is set, got: {result!r}"

    def test_single_assignment_falsy_when_absent(self):
        """Why: ``if not uv_bin:`` must still work on failure path.
        What: _UvResult(None) is falsy (empty string).
        Test: assert not _UvResult(None).
        """
        from hermes_cli.managed_uv import _UvResult

        result = _UvResult(None)
        assert not result, f"_UvResult(None) should be falsy, got: {result!r}"


class TestEnsureUvUpdateBoundary:
    """ensure_uv() must return a value safe for both old and new call sites."""

    def test_ensure_uv_two_target_unpack_succeeds(self, tmp_path):
        """Why: the primary crash was ``a, b = ensure_uv()`` on an old in-memory
        call site against a freshly-pulled module.
        What: ensure_uv() returns something that unpacks to exactly 2 values.
        Test: stub resolve_uv to return a fake path; call ensure_uv(); unpack.
        """
        from hermes_cli import managed_uv

        fake_uv = str(tmp_path / "uv")
        Path(fake_uv).write_text("#!/bin/sh\n")

        with patch.object(managed_uv, "resolve_uv", return_value=fake_uv):
            uv_bin, fresh_bootstrap = managed_uv.ensure_uv()  # must not raise

        assert uv_bin == fake_uv
        assert fresh_bootstrap is False

    def test_ensure_uv_single_assignment_truthy(self, tmp_path):
        """Why: new call sites use ``uv_bin = ensure_uv()`` and ``if not uv_bin``.
        What: ensure_uv() is truthy and equals the path when uv is available.
        Test: stub resolve_uv; assert bool(ensure_uv()) is True.
        """
        from hermes_cli import managed_uv

        fake_uv = str(tmp_path / "uv")
        Path(fake_uv).write_text("#!/bin/sh\n")

        with patch.object(managed_uv, "resolve_uv", return_value=fake_uv):
            result = managed_uv.ensure_uv()

        assert result, f"ensure_uv() should be truthy when uv exists, got {result!r}"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only _UvResult wrapper")
    def test_ensure_uv_returns_uv_result_on_posix(self, tmp_path):
        """Why: the POSIX path must return _UvResult, not a plain tuple.
        What: ensure_uv() returns an instance of _UvResult on POSIX.
        Test: stub resolve_uv; assert isinstance(ensure_uv(), _UvResult).
        """
        from hermes_cli import managed_uv
        from hermes_cli.managed_uv import _UvResult

        fake_uv = str(tmp_path / "uv")
        Path(fake_uv).write_text("#!/bin/sh\n")

        with patch.object(managed_uv, "resolve_uv", return_value=fake_uv):
            result = managed_uv.ensure_uv()

        assert isinstance(result, _UvResult), (
            f"ensure_uv() should return _UvResult on POSIX, got {type(result)}"
        )

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only plain str path")
    def test_ensure_uv_returns_plain_str_on_windows(self, tmp_path):
        """Why: Windows subprocess.list2cmdline iterates argv entries per-character;
        the _UvResult.__iter__ override would break the command line.
        What: ensure_uv() returns a plain str (not _UvResult) on Windows.
        Test: stub resolve_uv; assert type(ensure_uv()) is str (not _UvResult).
        """
        from hermes_cli import managed_uv
        from hermes_cli.managed_uv import _UvResult

        fake_uv = str(tmp_path / "uv.exe")
        Path(fake_uv).write_text("@echo off\n")

        with patch.object(managed_uv, "resolve_uv", return_value=fake_uv):
            result = managed_uv.ensure_uv()

        assert type(result) is str and not isinstance(result, _UvResult), (
            "ensure_uv() should return plain str on Windows, "
            f"got {type(result)}: {result!r}"
        )

    def test_ensure_uv_none_path_unpacks_without_type_error(self):
        """Why: the failure path (uv not installed, install fails) returned None;
        ``a, b = None`` raises TypeError.  _UvResult(None) must not raise.
        What: even when uv is absent, ensure_uv() unpacks to (None, False).
        Test: stub _ensure_uv_path to return None; unpack ensure_uv().
        """
        from hermes_cli import managed_uv

        with (
            patch.object(managed_uv, "_ensure_uv_path", return_value=None),
            patch.object(managed_uv, "platform") as mock_platform,
        ):
            mock_platform.system.return_value = "Linux"
            # Should not raise TypeError
            uv_bin, fresh_bootstrap = managed_uv.ensure_uv()

        assert uv_bin is None
        assert fresh_bootstrap is False


class TestUvResultIterFootgunIsIntentional:
    """Document that _UvResult.__iter__ yields a 2-tuple, NOT str characters.

    Why: a future caller doing ``[*uv_bin]`` or ``for ch in uv_bin`` would
    silently get 2 items ``(path|None, False)`` instead of the individual
    characters of the path string.  This class pins that behaviour so any
    future change that accidentally makes iteration yield characters (thus
    breaking the update-boundary compat contract) is caught immediately.

    These tests exist to document the INTENTIONAL footgun.  Do not remove or
    weaken them — they are the regression guard for issue #39706.
    """

    def test_list_spread_yields_two_tuple_not_characters(self, tmp_path):
        """Why: [*uv_bin] should yield (path, False), not path characters.
        What: spreading a _UvResult into a list gives exactly 2 elements.
        Test: create _UvResult with a multi-char path; assert list has 2 items
        and the items are the path string and False (not individual chars).
        """
        from hermes_cli.managed_uv import _UvResult

        fake_path = str(tmp_path / "uv")
        result = _UvResult(fake_path, False)

        spread = [*result]
        assert len(spread) == 2, (
            f"[*_UvResult] must yield exactly 2 items (the dual-mode contract); "
            f"got {len(spread)}: {spread!r}"
        )
        assert spread[0] == fake_path, (
            f"First item must be the path string, got {spread[0]!r}"
        )
        assert spread[1] is False, (
            f"Second item must be fresh_bootstrap=False, got {spread[1]!r}"
        )

    def test_result_is_still_usable_as_path_string(self, tmp_path):
        """Why: the str contract must survive the __iter__ override — callers
        that treat the result as a path string must keep working.
        What: str(), os.fspath-style usage, and truthiness all work correctly.
        Test: assert str(result) equals the path; assert the result is truthy
        when non-empty; assert it is falsy when the path is absent (None).
        """
        import os
        from hermes_cli.managed_uv import _UvResult

        fake_path = str(tmp_path / "uv")
        result = _UvResult(fake_path, False)

        # str() must return the path, not something from __iter__
        assert str(result) == fake_path, (
            f"str(_UvResult) must be the path; got {str(result)!r}"
        )
        # os.fspath delegates to __fspath__ which falls back to str for str subclasses
        assert os.fspath(result) == fake_path, (
            f"os.fspath(_UvResult) must be the path; got {os.fspath(result)!r}"
        )
        # Truthiness: non-empty path → truthy
        assert result, "non-empty _UvResult must be truthy"
        # Truthiness: absent path → falsy
        absent = _UvResult(None, False)
        assert not absent, "_UvResult(None) must be falsy"

    def test_runtime_invariant_assertion_fires_on_broken_subclass(self):
        """Why: the assert in __new__ must catch any future change that breaks
        the 2-item contract before it reaches a call site.
        What: a subclass that overrides __iter__ to yield the wrong number of
        items must raise AssertionError at construction time.
        Test: define a bad subclass; assert that constructing it raises.
        """
        from hermes_cli.managed_uv import _UvResult

        class _BrokenResult(_UvResult):
            def __iter__(self):  # type: ignore[override]
                # Accidentally yields 3 items instead of 2 — contract violation
                return iter(("path", False, "extra"))

        with pytest.raises(AssertionError, match="exactly 2 items"):
            _BrokenResult("/usr/bin/uv", False)
