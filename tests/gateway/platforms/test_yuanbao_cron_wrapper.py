"""Regression tests for Yuanbao cron delivery wrapper stripping."""

import pytest

from gateway.platforms.yuanbao import MessageSender


@pytest.mark.parametrize(
    "footer",
    [
        'To stop or manage this job, send me a new message (e.g. "stop reminder report").',
        'This was a one-time job and will not run again. To create or manage cron jobs, send me a message like "cron list".',
    ],
)
def test_strip_cron_wrapper_preserves_multi_paragraph_body(footer: str) -> None:
    body = "first paragraph\n\nsecond paragraph\n\nfinal paragraph"
    wrapped = (
        "Cronjob Response: report\n"
        "(job_id: job-123)\n"
        "-------------\n\n"
        f"{body}\n\n"
        f"{footer}"
    )

    assert MessageSender.strip_cron_wrapper(wrapped) == body


def test_strip_cron_wrapper_does_not_drop_unrecognized_final_paragraph() -> None:
    body = "first paragraph\n\nthis is ordinary output, not a scheduler footer"
    wrapped = f"Cronjob Response: report\n(job_id: job-123)\n-------------\n\n{body}"

    assert MessageSender.strip_cron_wrapper(wrapped) == body
