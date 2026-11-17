"""Launch-home identity must not pay a Path.resolve() syscall per comparison.

The gateway calls agent.secret_scope._is_process_home on the event loop for every admitted
turn (gateway/session_ingress.py::_launch_body_if_launch_owner); an uncached resolve there is a
filesystem round-trip on the loop — the second stall site reported in #145022. The comparison
goes through hermes_constants.hermes_home_key, whose resolved-key cache the repo already trusts
for process-wide profile identity (registry overlays, MCP discovery)."""

import pathlib

import pytest

from agent.secret_scope import _is_process_home
from hermes_constants import reset_hermes_home_key_cache


@pytest.fixture
def resolve_calls(monkeypatch):
    """Record every Path.resolve() call made while the fixture is active."""
    calls = []
    real = pathlib.Path.resolve

    def counting(self, *args, **kwargs):
        calls.append(str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", counting)
    return calls


def _launch_home(tmp_path, monkeypatch, name="launch"):
    home = tmp_path / name
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_hermes_home_key_cache()
    return home


def test_warm_launch_home_comparison_pays_no_syscalls(tmp_path, monkeypatch, resolve_calls):
    """After the key cache is warm, a repeated launch-home check is a pure dict hit: the
    per-turn gateway path must never reach the filesystem again."""
    home = _launch_home(tmp_path, monkeypatch)
    assert _is_process_home(home) is True  # cold: resolves populate the resolved-key cache
    resolve_calls.clear()
    for _ in range(3):
        assert _is_process_home(home) is True
    assert resolve_calls == []


def test_symlinked_launch_home_keeps_symlink_identity(tmp_path, monkeypatch):
    """The cached key must not weaken the symlink semantics resolve() provided: a home reached
    through a symlink is still the process's own home."""
    real_home = tmp_path / "real"
    real_home.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    reset_hermes_home_key_cache()
    assert _is_process_home(real_home) is True
    assert _is_process_home(link) is True


def test_foreign_and_missing_homes_are_not_the_process_home(tmp_path, monkeypatch):
    """A satellite profile's home is never the process home, and a home that does not exist yet
    resolves leniently without raising (the behavior raw strict=False resolve gave before)."""
    _launch_home(tmp_path, monkeypatch)
    satellite = tmp_path / "satellite"
    satellite.mkdir()
    assert _is_process_home(satellite) is False
    assert _is_process_home(tmp_path / "ghost") is False
