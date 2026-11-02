"""Cron: a fire landing while PM is between dependency generations is held, not failed.

``pm.environments.activate_dependencies`` refuses to boot an external worker when this
install is store-managed and has no committed generation — correct, because an inherited
``sys.path`` is not a committed one — and the worker then exits BEFORE its ownership
acknowledgement, which ``run_one_job`` books as a failed run plus a crash notice. ``pm``
unlinks the committed record while it rebuilds, so a fire inside that window is a missed
slot, not a broken job: the same install commits a generation seconds later (measured
2026-11-02 on a store-managed install: five jobs failed 13:10:51–13:16:12 while facts.json
was absent, and the rebuilt generation was committed at 13:16:12).

The hold copies the headless gateway gate's contract — no run is recorded, the due instant
does not move — and is bounded by ``GRACE_SECONDS`` so a genuinely broken install still
reaches the loud failure path (#123401). A venv / wheel / Nix install has no committed
generation either and must keep dispatching: only PM's own refusal predicate decides.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from cron import scheduler_worker_env as env_mod
from pm.environments import NoCommittedEnvironment

JOBS = [{"id": "5c1e66eb1f55"}, {"id": "134c2b28506a"}]


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def refused(monkeypatch) -> dict:
    """Capture the gate's refusal stamp instead of writing the job store."""
    seen: dict = {}

    def fake(ids, detail):
        seen["ids"] = list(ids)
        seen["detail"] = detail

    monkeypatch.setattr("cron.scheduler_gateway_gate._note_fire_refused", fake)
    return seen


def _store_managed_uncommitted(monkeypatch) -> None:
    """The production shape: no committed generation AND a store-managed interpreter."""

    def refuse(_root):
        raise NoCommittedEnvironment(
            "no dependency environment is committed for this install"
        )

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr("pm.environments._require_own_dependencies", refuse)


def _set_committed(monkeypatch, value) -> None:
    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: value)


def _defer(
    home: Path, *, now: float | None = None, grace: float = env_mod.GRACE_SECONDS
):
    return env_mod.defer_while_dependencies_uncommitted(
        JOBS,
        home=home,
        project_root=home.parent / "repo",
        grace_seconds=grace,
        now=time.time() if now is None else now,
    )


def test_holds_due_jobs_while_the_install_is_between_generations(
    monkeypatch, home, refused
):
    _store_managed_uncommitted(monkeypatch)

    assert _defer(home) == []
    assert refused["ids"] == [job["id"] for job in JOBS]
    assert "dependency environment is being rebuilt" in refused["detail"]
    assert env_mod._defer_marker(home).is_file()


def test_the_hold_does_not_slide_within_the_grace_window(monkeypatch, home, refused):
    """A second tick must not restart the clock, or the deferral never expires."""
    _store_managed_uncommitted(monkeypatch)
    start = time.time()

    assert _defer(home, now=start) == []
    first = env_mod._defer_marker(home).read_text(encoding="utf-8")
    assert _defer(home, now=start + 120) == []
    assert env_mod._defer_marker(home).read_text(encoding="utf-8") == first


def test_grace_expiry_restores_the_loud_failure_path(monkeypatch, home, refused):
    """Past the window an uncommitted install is treated as broken, not as rebuilding."""
    _store_managed_uncommitted(monkeypatch)
    now = time.time()
    marker = env_mod._defer_marker(home)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(now - env_mod.GRACE_SECONDS - 1), encoding="utf-8")

    assert _defer(home, now=now) == JOBS
    assert refused == {}


def test_committed_environment_dispatches_and_clears_a_stale_marker(
    monkeypatch, home, refused
):
    _set_committed(monkeypatch, home.parent / "env")
    marker = env_mod._defer_marker(home)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("1.0", encoding="utf-8")

    assert _defer(home) == JOBS
    assert not marker.exists()
    assert refused == {}


def test_a_venv_install_with_no_generation_still_dispatches(monkeypatch, home, refused):
    """No committed generation is normal for a venv / wheel / Nix install: PM never refuses."""
    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr("pm.environments._require_own_dependencies", lambda _root: None)

    assert _defer(home) == JOBS
    assert refused == {}


def test_no_due_jobs_never_probes_the_install(monkeypatch, home):
    def boom(_root):
        raise AssertionError("committed_venv must not be consulted on an idle tick")

    monkeypatch.setattr("pm.environments.committed_venv", boom)

    assert env_mod.defer_while_dependencies_uncommitted([], home=home) == []


def test_the_hold_can_be_switched_off(monkeypatch, home, refused):
    _store_managed_uncommitted(monkeypatch)
    monkeypatch.setattr(env_mod, "_defer_enabled", lambda cfg=None: False)

    assert _defer(home) == JOBS
    assert refused == {}


def test_an_unreadable_record_dispatches_rather_than_hiding_behind_the_hold(
    monkeypatch, home
):
    """A runtime error is not evidence of a rebuild: fail loudly instead of deferring."""

    def raises(_root):
        raise RuntimeError("invalid dependency environment record")

    monkeypatch.setattr("pm.environments.committed_venv", raises)

    assert _defer(home) == JOBS
