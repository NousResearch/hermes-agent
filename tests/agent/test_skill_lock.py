"""Process-level contracts for the skill-library locking protocol.

Two layers are covered here:

* the lock primitives, on both backends — the sentinel (Windows) backend is
  forced via ``use_lock_backend`` so it runs on ordinary POSIX CI instead of
  only on a Windows runner;
* the real mutation entry points (``skill_manage``, ``sync_skills``), in
  ``tests/tools/test_skill_lock_integration.py`` — a protocol that only holds
  for the primitives is not the property this change claims.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

from agent.skill_lock import BACKEND_FLOCK, BACKEND_SENTINEL


def _attempt_lock(
    home: str, skill: str | None, exclusive_namespace: bool, queue, backend: str = BACKEND_FLOCK
) -> None:
    """Child-process helper; must remain module-level for spawn platforms."""
    os.environ["HERMES_HOME"] = home
    from agent.skill_lock import skill_write_lock, skills_namespace_lock, use_lock_backend

    try:
        with use_lock_backend(backend):
            with skills_namespace_lock(exclusive=exclusive_namespace, timeout=0.25):
                if skill is None:
                    queue.put("acquired")
                else:
                    with skill_write_lock(Path(home) / "skills" / skill, timeout=0.25):
                        queue.put("acquired")
    except TimeoutError:
        queue.put("timed-out")


@pytest.mark.skipif(os.name == "nt", reason="flock backend is POSIX-only; sentinel covered below")
def test_same_skill_is_exclusive_but_independent_skills_are_concurrent(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    skills = home / "skills"
    (skills / "alpha").mkdir(parents=True)
    (skills / "beta").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.skill_lock import skill_write_lock, skills_namespace_lock

    context = multiprocessing.get_context("spawn")
    with skills_namespace_lock(exclusive=False):
        with skill_write_lock(skills / "alpha"):
            queue = context.Queue()
            blocked = context.Process(
                target=_attempt_lock, args=(str(home), "alpha", False, queue)
            )
            allowed = context.Process(
                target=_attempt_lock, args=(str(home), "beta", False, queue)
            )
            blocked.start()
            allowed.start()
            outcomes = {queue.get(timeout=5), queue.get(timeout=5)}
            blocked.join(timeout=5)
            allowed.join(timeout=5)

    assert outcomes == {"acquired", "timed-out"}
    assert blocked.exitcode == 0
    assert allowed.exitcode == 0


@pytest.mark.skipif(os.name == "nt", reason="flock backend is POSIX-only; sentinel covered below")
def test_structural_lock_waits_for_inflight_content_transaction(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "skills" / "alpha").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.skill_lock import skills_namespace_lock

    context = multiprocessing.get_context("spawn")
    with skills_namespace_lock(exclusive=False):
        queue = context.Queue()
        writer = context.Process(
            target=_attempt_lock, args=(str(home), None, True, queue)
        )
        writer.start()
        assert queue.get(timeout=5) == "timed-out"
        writer.join(timeout=5)
    assert writer.exitcode == 0


# ---------------------------------------------------------------------------
# Sentinel backend — the Windows fallback, exercised on every platform.
# ---------------------------------------------------------------------------


def test_sentinel_backend_serializes_across_processes(tmp_path, monkeypatch):
    """The fallback still excludes a second process holding the namespace."""
    home = tmp_path / "hermes"
    (home / "skills" / "alpha").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.skill_lock import skills_namespace_lock, use_lock_backend

    context = multiprocessing.get_context("spawn")
    with use_lock_backend(BACKEND_SENTINEL):
        with skills_namespace_lock():
            queue = context.Queue()
            writer = context.Process(
                target=_attempt_lock,
                args=(str(home), None, True, queue, BACKEND_SENTINEL),
            )
            writer.start()
            assert queue.get(timeout=5) == "timed-out"
            writer.join(timeout=5)
    assert writer.exitcode == 0


def test_sentinel_backend_serializes_independent_skills(tmp_path, monkeypatch):
    """Documented platform difference: no per-skill granularity on the fallback.

    The flock backend lets an unrelated skill proceed (see the first test);
    the sentinel cannot, because it has exactly one lockable object. Pinning
    the behaviour keeps a future "optimisation" from quietly reintroducing
    per-skill sentinels inside skill directories, which the bundled-sync
    content hashes would read as user modifications.
    """
    home = tmp_path / "hermes"
    skills = home / "skills"
    (skills / "alpha").mkdir(parents=True)
    (skills / "beta").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.skill_lock import skill_write_lock, skills_namespace_lock, use_lock_backend

    context = multiprocessing.get_context("spawn")
    with use_lock_backend(BACKEND_SENTINEL):
        with skills_namespace_lock(exclusive=False):
            with skill_write_lock(skills / "alpha"):
                queue = context.Queue()
                other = context.Process(
                    target=_attempt_lock,
                    args=(str(home), "beta", False, queue, BACKEND_SENTINEL),
                )
                other.start()
                assert queue.get(timeout=5) == "timed-out"
                other.join(timeout=5)
    assert other.exitcode == 0


def test_sentinel_backend_nesting_does_not_self_deadlock(tmp_path, monkeypatch):
    """shared namespace → per-skill write lock must not block on itself.

    A Windows byte-range lock conflicts with the same process re-taking it on
    a second handle, and this nesting is the ordinary agent write path. Without
    the in-process guard, every ``skill_manage`` edit on Windows would stall
    until the lock timeout and then fail.
    """
    home = tmp_path / "hermes"
    skills = home / "skills"
    (skills / "alpha").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.skill_lock import skill_write_lock, skills_namespace_lock, use_lock_backend

    with use_lock_backend(BACKEND_SENTINEL):
        with skills_namespace_lock(exclusive=False, timeout=1.0):
            with skill_write_lock(skills / "alpha", timeout=1.0):
                pass


def test_sentinel_file_lives_outside_the_skill_tree(tmp_path, monkeypatch):
    """The sentinel must never appear inside ``skills/``.

    A lock file under a skill directory would be hashed by the bundled sync
    and reported as a user modification, permanently freezing that skill's
    updates.
    """
    home = tmp_path / "hermes"
    skills = home / "skills"
    (skills / "alpha").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.skill_lock import skills_namespace_lock, use_lock_backend

    with use_lock_backend(BACKEND_SENTINEL):
        with skills_namespace_lock():
            pass

    assert (home / "locks" / "skills.lock").exists()
    assert not list(skills.rglob("*.lock"))


def test_shared_namespace_lock_cannot_be_upgraded(tmp_path, monkeypatch):
    """Upgrading shared → exclusive is a deadlock; it must fail loudly."""
    home = tmp_path / "hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.skill_lock import skills_namespace_lock

    with skills_namespace_lock(exclusive=False):
        with pytest.raises(RuntimeError, match="cannot upgrade"):
            with skills_namespace_lock(exclusive=True):
                pass


def test_nested_structural_transaction_is_reentrant(tmp_path, monkeypatch):
    """``skill_manage`` delete → ``archive_skill`` nests two structural writers.

    Both take the exclusive namespace lock; the inner one must be a no-op
    rather than a second acquisition, which flock would block on forever.
    """
    home = tmp_path / "hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.skill_lock import namespace_write_locked, skills_namespace_lock

    @namespace_write_locked
    def inner():
        return "ok"

    with skills_namespace_lock(exclusive=True, timeout=1.0):
        assert inner() == "ok"


def test_lock_timeout_comes_from_config(tmp_path, monkeypatch):
    """``skills.lock_timeout`` is the user-facing knob (AGENTS.md: no new env vars)."""
    import agent.skill_lock as skill_lock

    monkeypatch.delenv(skill_lock._INTERNAL_TIMEOUT_ENV, raising=False)
    monkeypatch.setattr(skill_lock, "_configured_timeout", lambda: 45.0)
    assert skill_lock._resolve_timeout(None, skill_lock.DEFAULT_TIMEOUT) == 45.0

    # An explicit caller argument still wins over configuration.
    assert skill_lock._resolve_timeout(9.0, skill_lock.DEFAULT_TIMEOUT) == 9.0

    # The opportunistic waits ignore it: a generous configured wait must not
    # turn "defer and retry" into "stall startup".
    assert (
        skill_lock._resolve_timeout(
            None, skill_lock.OPPORTUNISTIC_TIMEOUT, configurable=False
        )
        == skill_lock.OPPORTUNISTIC_TIMEOUT
    )


def test_config_lock_timeout_is_read_through_cfg_get(tmp_path, monkeypatch):
    """The real config path resolves, not just the patched seam."""
    import agent.skill_lock as skill_lock

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"skills": {"lock_timeout": 12.5}},
    )
    assert skill_lock._configured_timeout() == 12.5

    # A section that isn't a mapping, or a missing key, must not raise.
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"skills": "nope"})
    assert skill_lock._configured_timeout() is None
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    assert skill_lock._configured_timeout() is None


def test_unusable_timeout_values_fall_back(monkeypatch):
    """Garbage and non-positive waits are ignored rather than applied."""
    import agent.skill_lock as skill_lock

    assert skill_lock._coerce_timeout("not-a-number", "src") is None
    assert skill_lock._coerce_timeout("0", "src") is None
    assert skill_lock._coerce_timeout(-1, "src") is None
    assert skill_lock._coerce_timeout(None, "src") is None
    assert skill_lock._coerce_timeout("", "src") is None
    assert skill_lock._coerce_timeout("2.5", "src") == 2.5


def test_internal_env_bridge_outranks_config(monkeypatch):
    """The bridge exists for bootstrap and spawned subprocesses only.

    It is not documented in cli-config.yaml.example and must stay internal;
    this pins the precedence so a future change doesn't quietly make it the
    de facto user-facing setting.
    """
    import agent.skill_lock as skill_lock

    monkeypatch.setenv(skill_lock._INTERNAL_TIMEOUT_ENV, "0.25")
    monkeypatch.setattr(skill_lock, "_configured_timeout", lambda: 45.0)
    assert skill_lock._resolve_timeout(None, skill_lock.DEFAULT_TIMEOUT) == 0.25


def test_unknown_backend_is_rejected():
    from agent.skill_lock import use_lock_backend

    with pytest.raises(ValueError):
        with use_lock_backend("nope"):
            pass
