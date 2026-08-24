"""Tests for the remembered results in `hermes_home_key`.

`Path.resolve()` is a filesystem call. `hermes_home_key` sits under
`ToolRegistry.current_scope_key()`, which runs on every registry lookup, so
before the results were remembered the registry paid a syscall per lookup.

The value this returns must not change, so most of these tests compare
against the plain uncached calculation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import hermes_constants as hc


def _uncached(path=None) -> str:
    """The calculation as it was before results were remembered."""
    candidate = Path(path) if path is not None else hc.get_hermes_home()
    return os.path.normcase(str(candidate.expanduser().resolve(strict=False)))


@pytest.fixture(autouse=True)
def _clear_cache():
    hc.reset_hermes_home_key_cache()
    yield
    hc.reset_hermes_home_key_cache()


class TestSameAnswerAsBefore:
    """Remembering a result must not change what comes back."""

    @pytest.mark.parametrize("case", ["real", "missing", "trailing_sep", "dot_dot"])
    def test_matches_the_uncached_calculation(self, tmp_path, case):
        (tmp_path / "real").mkdir()
        target = {
            "real": str(tmp_path / "real"),
            "missing": str(tmp_path / "not_there"),
            "trailing_sep": str(tmp_path / "real") + os.sep,
            "dot_dot": str(tmp_path / "real" / ".." / "real"),
        }[case]
        assert hc.hermes_home_key(target) == _uncached(target)

    def test_matches_for_the_default_home(self):
        assert hc.hermes_home_key() == _uncached()

    def test_matches_for_a_tilde_path(self):
        assert hc.hermes_home_key("~") == _uncached("~")

    def test_accepts_a_path_object(self, tmp_path):
        (tmp_path / "real").mkdir()
        assert hc.hermes_home_key(tmp_path / "real") == _uncached(tmp_path / "real")

    def test_second_call_returns_the_same_string(self, tmp_path):
        (tmp_path / "real").mkdir()
        first = hc.hermes_home_key(str(tmp_path / "real"))
        second = hc.hermes_home_key(str(tmp_path / "real"))
        assert first == second == _uncached(str(tmp_path / "real"))


class TestWhatGetsRemembered:
    def test_an_existing_path_is_remembered(self, tmp_path):
        (tmp_path / "real").mkdir()
        hc.hermes_home_key(str(tmp_path / "real"))
        assert len(hc._HOME_KEY_CACHE) == 1

    def test_a_missing_path_is_not_remembered(self, tmp_path):
        # The answer can change once the directory is created, for example
        # when part of the path turns out to be a link, so it must not stick.
        hc.hermes_home_key(str(tmp_path / "not_there"))
        assert hc._HOME_KEY_CACHE == {}

    def test_a_path_created_later_picks_up_the_real_answer(self, tmp_path):
        later = tmp_path / "later"
        before = hc.hermes_home_key(str(later))
        later.mkdir()
        after = hc.hermes_home_key(str(later))
        assert after == _uncached(str(later))
        assert hc._HOME_KEY_CACHE == {str(later): after}
        # On a plain directory both answers agree anyway. The point is that
        # the first one was never stored.
        assert before == after

    def test_different_paths_get_their_own_entries(self, tmp_path):
        for name in ("a", "b", "c"):
            (tmp_path / name).mkdir()
            hc.hermes_home_key(str(tmp_path / name))
        assert len(hc._HOME_KEY_CACHE) == 3

    def test_reset_clears_everything(self, tmp_path):
        (tmp_path / "real").mkdir()
        hc.hermes_home_key(str(tmp_path / "real"))
        assert hc._HOME_KEY_CACHE
        hc.reset_hermes_home_key_cache()
        assert hc._HOME_KEY_CACHE == {}


class TestHomeChanges:
    def test_pointing_hermes_home_somewhere_else_gives_a_new_key(
        self, tmp_path, monkeypatch,
    ):
        # A different home is a different input path, so it lands on its own
        # entry rather than reusing the first one.
        first = tmp_path / "home_one"
        second = tmp_path / "home_two"
        first.mkdir()
        second.mkdir()

        monkeypatch.setenv("HERMES_HOME", str(first))
        key_one = hc.hermes_home_key()
        monkeypatch.setenv("HERMES_HOME", str(second))
        key_two = hc.hermes_home_key()

        assert key_one != key_two
        assert key_one == _uncached(str(first))
        assert key_two == _uncached(str(second))


class TestSymlinks:
    def test_a_link_resolves_to_its_target(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("this platform or account cannot create symlinks")
        assert hc.hermes_home_key(str(link)) == _uncached(str(link))
        assert hc.hermes_home_key(str(link)) == hc.hermes_home_key(str(target))


class TestRegistryLookupsDoNotHitTheDisk:
    def test_scope_key_resolves_the_path_once(self, monkeypatch):
        # The reason this cache exists. ToolRegistry.current_scope_key() runs
        # on every registry lookup, so it must not resolve the home path on
        # the filesystem every time.
        from tools.registry import registry

        calls = {"n": 0}
        real_resolve = Path.resolve

        def counting_resolve(self, *a, **kw):
            calls["n"] += 1
            return real_resolve(self, *a, **kw)

        monkeypatch.setattr(Path, "resolve", counting_resolve)

        registry.current_scope_key()
        after_first = calls["n"]
        for _ in range(50):
            registry.current_scope_key()

        assert calls["n"] == after_first, (
            f"current_scope_key resolved the path on the filesystem "
            f"{calls['n'] - after_first} extra times across 50 calls"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
