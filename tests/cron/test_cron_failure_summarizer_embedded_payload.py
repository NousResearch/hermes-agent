"""Regression tests for #99988: cron failure summarizer must not classify
embedded payload text (script source, headers) as provider failures.

OSError/WinError messages embed the entire "filename" on a single line, so
first-line-only matching alone is insufficient — skip provider branches when
the first line carries an OS error signature ([Errno N] or [WinError N]).
"""

from cron.scheduler import _summarize_cron_failure_for_delivery


def _script_with_auth_header() -> str:
    return (
        "#!/bin/bash\n"
        "TOKEN=$(curl ...)\n"
        'MANIFEST=$(curl -s -H "Authorization: Bearer $TOKEN" ...)\n'
    )


def test_errno63_embedded_authorization_not_labeled_provider_auth():
    script = _script_with_auth_header()
    # Linux capture form: str(OSError) embeds the filename on one line (#99988).
    error = (
        "OSError: [Errno 63] File name too long: "
        f"'{script.replace(chr(10), chr(92) + 'n')}'"
    )
    assert len(error.splitlines()) == 1

    job = {"name": "monitor-job", "no_agent": False}
    msg = _summarize_cron_failure_for_delivery(job, error)

    assert "provider authentication error" not in msg.lower()
    assert "monitor-job" in msg


def test_winerror206_embedded_authorization_not_labeled_provider_auth():
    script = _script_with_auth_header().replace("\n", "\\n")
    error = (
        "OSError: [WinError 206] The filename or extension is too long: "
        f"'{script}'"
    )
    assert len(error.splitlines()) == 1

    job = {"name": "monitor-job", "no_agent": False}
    msg = _summarize_cron_failure_for_delivery(job, error)

    assert "provider authentication error" not in msg.lower()
    assert "WinError 206" in msg or "filename" in msg.lower()


def test_flipping_authorization_to_xuthorization_does_not_change_classification():
    script = _script_with_auth_header().replace("\n", "\\n")
    script_x = script.replace("Authorization", "Xuthorization")
    error_a = f"OSError: [Errno 63] File name too long: '{script_x}'"
    error_b = f"OSError: [Errno 63] File name too long: '{script}'"

    job = {"name": "monitor-job", "no_agent": False}
    msg_a = _summarize_cron_failure_for_delivery(job, error_a)
    msg_b = _summarize_cron_failure_for_delivery(job, error_b)
    assert "provider authentication error" not in msg_a.lower()
    assert "provider authentication error" not in msg_b.lower()


def test_embedded_timeout_wording_not_labeled_provider_timeout():
    error = (
        "RuntimeError: job crashed\n"
        "details: the upstream request timed out after 30s"
    )
    job = {"name": "digest", "no_agent": False}
    msg = _summarize_cron_failure_for_delivery(job, error)
    assert "provider timeout" not in msg.lower()


def test_first_line_provider_auth_still_classifies():
    job = {"name": "daily-digest", "no_agent": False}
    msg = _summarize_cron_failure_for_delivery(job, "HTTP 401 authentication failed")
    assert "provider authentication error" in msg
