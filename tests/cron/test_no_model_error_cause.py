"""A cron job that resolves no model must name the cause it actually observed.

``_load_cron_job_config`` downgrades any ``config.yaml`` failure to a WARNING
and carries on with defaults. The fail-fast raise below then asserted
``config.yaml model.default missing or empty`` unconditionally, so an operator
whose file was merely unreadable (permission denied, bad YAML) was sent to
inspect config fields instead of the file's ownership and permissions.
"""

from __future__ import annotations


def _message(**kwargs):
    from cron.scheduler import _no_model_configured_message

    base = {
        "job_name": "Morning Briefing",
        "job_id": "abc123",
        "job_model": None,
        "env_model": "",
    }
    base.update(kwargs)
    return _no_model_configured_message(**base)


def test_unreadable_config_is_not_reported_as_a_missing_field():
    """The bug: a permission error was reported as 'model.default missing'."""
    msg = _message(
        config_load_error="PermissionError: [Errno 13] Permission denied: "
        "'/home/svc/.hermes/config.yaml'"
    )

    assert "missing or empty" not in msg, (
        "an unreadable config was still described as a missing field"
    )
    assert "could not be read" in msg
    assert "Permission denied" in msg


def test_unreadable_config_points_at_the_file_not_the_field():
    """The remedy must follow the cause, or the message is still misleading."""
    msg = _message(config_load_error="PermissionError: [Errno 13] Permission denied")

    assert "ownership/permissions" in msg
    # `hermes model <name>` writes to the file that cannot be read.
    assert "hermes model <name>" not in msg


def test_readable_config_keeps_the_original_diagnosis():
    """Guard: the genuine missing-field case must read as it did before."""
    msg = _message()

    assert "config.yaml model.default missing or empty" in msg
    assert "hermes model <name>" in msg
    assert "could not be read" not in msg


def test_message_still_reports_the_other_two_sources():
    for err in (None, "OSError: boom"):
        msg = _message(job_model="", env_model="", config_load_error=err)
        assert "job.model=''" in msg
        assert "HERMES_MODEL=''" in msg
        assert "Morning Briefing" in msg


def test_the_suggested_command_is_the_current_one():
    """The remedy must be runnable as printed, in both branches.

    Pinned deliberately: the CLI moved from ``cronjob action=update`` to
    ``hermes cron edit``, and a stale command in an error message is its own
    small bug.
    """
    for err in (None, "OSError: boom"):
        msg = _message(job_id="deadbeef", config_load_error=err)
        assert "hermes cron edit deadbeef --model <name>" in msg
        assert "cronjob action=update" not in msg


def test_end_to_end_unreadable_config_names_the_file(tmp_path, monkeypatch):
    """Behavioural probe: drive the real loader with a config it cannot read.

    Imported inside the test so the module still collects when the helper is
    absent, which is what makes the revert failure behavioural rather than an
    ImportError.
    """
    import pytest

    from cron import scheduler as sched

    (tmp_path / "config.yaml").write_text("model:\n  default: gpt-4o\n", encoding="utf-8")
    monkeypatch.setattr(sched, "_get_hermes_home", lambda: tmp_path)

    def _boom(_path):
        raise PermissionError(13, "Permission denied", str(tmp_path / "config.yaml"))

    monkeypatch.setattr(
        "hermes_cli.config.read_user_config_raw", _boom, raising=False
    )
    monkeypatch.delenv("HERMES_MODEL", raising=False)

    with pytest.raises(RuntimeError) as exc:
        sched._load_cron_job_config({"id": "abc123"}, "abc123", "Morning Briefing")

    msg = str(exc.value)
    assert "could not be read" in msg, (
        "the unreadable config was still reported as a missing field: " + msg
    )
    assert "Permission denied" in msg
    assert "missing or empty" not in msg
