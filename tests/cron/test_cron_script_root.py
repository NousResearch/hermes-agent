"""Which directory a cron script must live in follows the store, not the process's import-time home.

Cron is per-profile: a job lives in the store of the home it was created under, and that home's
scheduler is the only thing that runs its script. The create-time check and the ``cron doctor``
check each used to compute that directory on their own — the doctor from the import-time
``CRON_DIR`` constant, the create-time check from the ambient ``HERMES_HOME`` — so a profile scope
applied after ``cron.jobs`` was imported (a re-pointed ``HERMES_HOME``, an explicit
``use_cron_store()`` scope) left both naming the seat's scripts dir. Following the message then put
the script where no scheduler would ever look for it: the job registers clean and fails admission on
its first tick.

These tests pin the two checks to the store's home and assert the rejection names that one
directory plus the accepted shape.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def seat_store(tmp_path, monkeypatch):
    """Cron storage re-pointed at a seat home, as an embedder or a test harness would."""
    cron_dir = tmp_path / "seat" / "cron"
    cron_dir.mkdir(parents=True)
    monkeypatch.setattr("cron.jobs.CRON_DIR", cron_dir)
    monkeypatch.setattr("cron.jobs.JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", cron_dir / "output")
    return tmp_path / "seat"


@pytest.fixture()
def profile_home(tmp_path):
    profile = tmp_path / "profiles" / "coder"
    (profile / "scripts").mkdir(parents=True)
    return profile


def test_doctor_check_names_the_stores_scripts_dir(seat_store, profile_home):
    from cron.jobs import use_cron_store
    from hermes_cli.cron import _scripts_dir_for_cron

    with use_cron_store(profile_home):
        assert _scripts_dir_for_cron() == profile_home / "scripts"


def test_doctor_rejection_names_the_stores_scripts_dir_and_the_accepted_shape(
    seat_store, profile_home
):
    from cron.jobs import use_cron_store
    from hermes_cli.cron import _script_health_issue

    with use_cron_store(profile_home):
        issue = _script_health_issue("/etc/passwd")

    assert issue is not None
    assert str(profile_home / "scripts") in issue
    assert str(seat_store / "scripts") not in issue
    assert "filename" in issue


def test_create_time_check_agrees_with_the_doctor_check(seat_store, profile_home):
    from cron.jobs import use_cron_store
    from hermes_cli.cron import _scripts_dir_for_cron
    from tools.cronjob_job_args import _validate_cron_script_path

    with use_cron_store(profile_home):
        expected = str(_scripts_dir_for_cron())
        error = _validate_cron_script_path("monitor.py")

    assert error is not None
    assert expected in error
    assert str(seat_store / "scripts") not in error


def test_only_the_stores_scripts_dir_admits_a_bare_filename(seat_store, profile_home):
    from cron.jobs import use_cron_store
    from tools.cronjob_job_args import _validate_cron_script_path

    seat_scripts = seat_store / "scripts"
    seat_scripts.mkdir(parents=True)
    (seat_scripts / "watch.py").write_text("print('seat')\n")

    with use_cron_store(profile_home):
        assert _validate_cron_script_path("watch.py") is not None
        (profile_home / "scripts" / "watch.py").write_text("print('profile')\n")
        assert _validate_cron_script_path("watch.py") is None


def test_home_repointed_after_import_moves_the_named_dir_with_the_store(tmp_path, monkeypatch):
    profile = tmp_path / "profiles" / "coder"
    (profile / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    from hermes_cli.cron import _script_health_issue, _scripts_dir_for_cron
    from tools.cronjob_job_args import _validate_cron_script_path

    expected = str(profile / "scripts")
    create_error = _validate_cron_script_path("monitor.py")
    doctor_issue = _script_health_issue("/etc/passwd")
    assert str(_scripts_dir_for_cron()) == expected
    assert create_error is not None and expected in create_error
    assert doctor_issue is not None and expected in doctor_issue
