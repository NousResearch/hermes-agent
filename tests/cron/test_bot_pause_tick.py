"""A paused Bot Mode bot's scheduled jobs do not fire, and its resume does not burst the runs it
missed: each due slot is consumed while paused. The pause is the bot's own (profile.yaml), so a
sibling profile ticked by the same process keeps firing."""
from datetime import timedelta

import pytest

import hermes_yaml as yaml


def _set_paused(home, paused):
    meta = {"ui_meta": {"hermes-bots": {"title": home.name, "paused": paused}}}
    (home / "profile.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")


def _overdue_job(home, minutes=3):
    from cron.jobs import _hermes_now, create_job, load_jobs, save_jobs
    from cron.scheduler_provider import _profile_cron_scope

    with _profile_cron_scope(home):
        job = create_job(prompt="report", schedule="every 5m")
        jobs = load_jobs()
        for row in jobs:
            row["next_run_at"] = (_hermes_now() - timedelta(minutes=minutes)).isoformat()
        save_jobs(jobs)
    return job["id"]


def _tick(home):
    from cron import scheduler
    from cron.jobs import load_jobs
    from cron.scheduler_provider import _profile_cron_scope

    with _profile_cron_scope(home):
        scheduler.tick(verbose=False)
        return load_jobs()


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    from cron import scheduler

    root = tmp_path / ".hermes"
    paused, active = root / "profiles" / "scribe", root / "profiles" / "coder"
    for home in (paused, active):
        home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    fired = []
    monkeypatch.setattr(scheduler, "_process_due_job", lambda job, *a, **k: fired.append(job["id"]) or True)
    return paused, active, fired


def test_a_paused_bot_skips_its_due_run_and_its_sibling_still_fires(fleet):
    from cron.jobs import _hermes_now, _parse_aware

    paused, active, fired = fleet
    _set_paused(paused, True)
    _set_paused(active, False)
    held, live = _overdue_job(paused), _overdue_job(active)

    (job,) = _tick(paused)
    _tick(active)

    assert fired == [live]
    # The missed slot is consumed, not left overdue for the resume to fire.
    assert job["id"] == held and _parse_aware(job["next_run_at"]) > _hermes_now()


def test_resume_does_not_burst_the_runs_missed_while_paused(fleet):
    paused, _active, fired = fleet
    _set_paused(paused, True)
    _overdue_job(paused)
    _tick(paused)
    _tick(paused)

    _set_paused(paused, False)
    _tick(paused)

    assert fired == []


def test_a_pause_no_tick_saw_still_skips_the_slot_that_fell_due_inside_it(fleet, monkeypatch):
    """Pause, a slot falls due, resume, all before the next tick (or with the ticker down)."""
    import tui_gateway.server as srv
    from cron.jobs import _hermes_now, _parse_aware

    paused, _active, fired = fleet
    monkeypatch.setenv("HOME", str(paused.parent.parent.parent))  # profiles resolve under ~/.hermes

    def configure(flag):
        params = {"name": "scribe", "ui_meta": {"hermes-bots": {"title": "scribe", "paused": flag}}}
        return srv._methods["profiles.configure"]("c", params)["result"]["applied"]

    pause = configure(True)
    held = _overdue_job(paused, minutes=0)  # due now: inside the pause
    configure(False)
    (job,) = _tick(paused)

    assert pause["bot_pause_enforced"] is True  # the Desktop's proof this gateway enforces the pause
    assert fired == [] and job["id"] == held and _parse_aware(job["next_run_at"]) > _hermes_now()


def test_a_resumed_bot_holds_a_raw_string_schedule_slot_from_its_pause(fleet, monkeypatch):
    """Programmatic callers and older stores hand the tick the raw schedule string; the resume
    check classifies it as the scheduler does instead of crashing the tick."""
    from cron import scheduler
    from cron.jobs import _hermes_now
    from tools.bot_mode_pause import record_pause_change

    paused, _active, fired = fleet
    record_pause_change(paused, True)
    slot = _hermes_now().isoformat()
    record_pause_change(paused, False)
    later = (_hermes_now() + timedelta(seconds=1)).isoformat()
    legacy = [{"id": "in-pause", "schedule": "every 5m", "next_run_at": slot, "enabled": True},
              {"id": "after", "schedule": "every 5m", "next_run_at": later, "enabled": True}]
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: list(legacy))
    _tick(paused)

    assert fired == ["after"]


def test_a_tick_between_the_resume_writes_never_runs_a_slot_from_the_pause(fleet, monkeypatch):
    """The resume writes the flag and the pause record separately; a tick landing between them
    (or after a failed second write) must not see an unpaused bot whose pause is not on record."""
    import tui_gateway.server as srv
    import utils

    paused, _active, fired = fleet
    monkeypatch.setenv("HOME", str(paused.parent.parent.parent))

    def configure(flag):
        params = {"name": "scribe", "ui_meta": {"hermes-bots": {"title": "scribe", "paused": flag}}}
        srv._methods["profiles.configure"]("c", params)

    configure(True)
    _overdue_job(paused, minutes=0)  # due now: inside the pause
    for name in ("atomic_json_write", "atomic_yaml_write"):
        real = getattr(utils, name)

        def write_then_tick(path, *a, _real=real, **k):
            _real(path, *a, **k)
            if path.name in {"profile.yaml", "bot_pause_windows.json"}:
                _tick(paused)

        monkeypatch.setattr(utils, name, write_then_tick)
    configure(False)

    assert fired == []


def test_a_slot_due_after_the_resume_still_runs_when_an_earlier_one_fell_in_the_pause(fleet, monkeypatch):
    """Ticker down across a short pause: the slot inside it is consumed, but the one that fell due
    after the resume fires, as any missed slot does on the first tick back."""
    from types import SimpleNamespace

    from cron.jobs import _hermes_now, create_job, load_jobs, save_jobs
    from cron.scheduler_provider import _profile_cron_scope
    from tools import bot_mode_pause

    paused, _active, fired = fleet
    now = _hermes_now()
    for flag, ago in ((True, 110), (False, 80)):  # paused 110s ago, resumed 80s ago
        with monkeypatch.context() as m:
            m.setattr(bot_mode_pause, "time", SimpleNamespace(time=lambda ago=ago: now.timestamp() - ago))
            bot_mode_pause.record_pause_change(paused, flag)
    with _profile_cron_scope(paused):
        job = create_job(prompt="report", schedule="every 1m")
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (now - timedelta(seconds=100)).isoformat()  # inside the pause
        save_jobs(jobs)

    _tick(paused)

    assert fired == [job["id"]]  # its next slot (resume + 1m, 20s ago) was never paused
