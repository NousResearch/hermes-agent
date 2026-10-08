"""Concurrent ``.env`` writers must not clobber each other (#77187).

``save_env_value`` / ``remove_env_value`` do a read-modify-write cycle on
``~/.hermes/.env``. Without a lock serializing the whole cycle, two concurrent
writers both snapshot the file and the loser's atomic replace silently drops
the winner's key.

The race window is widened deterministically through the ``_read_env_lines``
module seam: a two-party barrier makes both writers snapshot before either
renames (red on an unlocked base). On a serialized path the second writer cannot
even read until the first writer's whole cycle completes, so the barrier times
out, breaks, and both writers proceed one-after-the-other (green).
"""

from __future__ import annotations

import threading
import unittest.mock as mock

import pytest

from hermes_cli import config as config_mod

# Barrier timeout: on the serialized (fixed) path the lock holder waits out this
# window for a second reader that cannot arrive (blocked on the lock). Short
# enough to keep the suite fast, long enough that a genuinely concurrent second
# reader on the broken path trips the barrier before it expires.
_BARRIER_WAIT_SECONDS = 1.0
_JOIN_TIMEOUT = 20.0


def _env_keys(env_path) -> set:
    keys = set()
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        name, sep, _ = stripped.partition("=")
        if sep:
            keys.add(name.strip())
    return keys


@pytest.fixture
def race_env_home(tmp_path, monkeypatch):
    """A temp HERMES_HOME whose .env carries a pre-existing BASE key."""
    env_path = tmp_path / ".env"
    env_path.write_text("BASE_KEY=base\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for leaked in ("RACE_A", "RACE_B"):
        monkeypatch.delenv(leaked, raising=False)
    return env_path


def _race_two_writers(writer_a, writer_b):
    """Run two ``.env`` writers concurrently, both entering the read phase together.

    The seam barriers both threads AFTER their read: on an unlocked cycle both
    are guaranteed to snapshot the same pre-write file before either renames, so
    the last rename deterministically drops the other writer's key. On a locked
    cycle the second reader only runs after the first writer finishes, the
    barrier times out (BrokenBarrierError, swallowed) and writers serialize.
    """
    barrier = threading.Barrier(2)
    original_read = config_mod._read_env_lines
    errors: list = []

    def read_then_barrier(env_path_arg):
        lines = original_read(env_path_arg)
        try:
            barrier.wait(timeout=_BARRIER_WAIT_SECONDS)
        except threading.BrokenBarrierError:
            pass  # serialized path: the other writer is behind the file lock
        return lines

    def run(fn):
        try:
            fn()
        except BaseException as exc:  # surfaced via the errors assert below
            errors.append(exc)

    with mock.patch.object(config_mod, "_read_env_lines", side_effect=read_then_barrier):
        threads = [
            threading.Thread(target=run, args=(writer_a,), name="env-writer-a"),
            threading.Thread(target=run, args=(writer_b,), name="env-writer-b"),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(_JOIN_TIMEOUT)

    assert not errors, f"writer threads raised: {errors!r}"
    assert not any(t.is_alive() for t in threads), "writer threads timed out"


def test_concurrent_save_env_value_writers_do_not_lose_keys(race_env_home):
    """Two threads behind a barrier save different keys; ``.env`` must end with
    BASE plus BOTH new keys — the loser's write must not be clobbered by the
    winner's stale snapshot (#77187)."""
    env_path = race_env_home
    _race_two_writers(
        lambda: config_mod.save_env_value("RACE_A", "a"),
        lambda: config_mod.save_env_value("RACE_B", "b"),
    )
    keys = _env_keys(env_path)
    assert keys == {"BASE_KEY", "RACE_A", "RACE_B"}, (
        f"lost a write to the unlocked read-modify-write race: {keys}")


def test_concurrent_remove_and_save_do_not_lose_keys(race_env_home):
    """A remove racing a save of a DIFFERENT key must neither lose the saved key
    nor resurrect the removed one: the remove's stale snapshot (which still
    contains the removed key) must never be the one that lands."""
    env_path = race_env_home
    env_path.write_text("BASE_KEY=base\nDOOMED_KEY=old\n", encoding="utf-8")
    _race_two_writers(
        lambda: config_mod.remove_env_value("DOOMED_KEY"),
        lambda: config_mod.save_env_value("RACE_B", "b"),
    )
    keys = _env_keys(env_path)
    assert keys == {"BASE_KEY", "RACE_B"}, (
        f"remove/save race produced a torn .env: {keys}")
