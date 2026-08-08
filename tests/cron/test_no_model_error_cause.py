"""A cron job that resolves no model must name the cause it actually observed.

When ``config.yaml`` fails to load (permission denied, missing, unparseable),
the loader downgrades it to a WARNING and carries on with defaults. The
fail-fast error below then claimed ``config.yaml model.default missing or
empty`` unconditionally, so an operator whose file was merely unreadable was
sent to inspect config fields instead of the file's ownership and permissions.
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
    # 'hermes model <name>' cannot help while the file is unreadable.
    assert "hermes model <name>" not in msg


def test_readable_config_keeps_the_original_diagnosis():
    """Guard: the genuine missing-field case must read exactly as before."""
    msg = _message()

    assert "config.yaml model.default missing or empty" in msg
    assert "hermes model <name>" in msg
    assert "could not be read" not in msg


def test_message_still_reports_the_other_two_sources():
    """Both non-config sources stay in the message for either cause."""
    for err in (None, "OSError: boom"):
        msg = _message(job_model="", env_model="", config_load_error=err)
        assert "job.model=''" in msg
        assert "HERMES_MODEL=''" in msg
        assert "Morning Briefing" in msg


def test_job_id_is_carried_into_the_remedy_command():
    """The suggested command must be runnable as printed, for either cause."""
    for err in (None, "OSError: boom"):
        msg = _message(job_id="deadbeef", config_load_error=err)
        assert "job_id=deadbeef" in msg
