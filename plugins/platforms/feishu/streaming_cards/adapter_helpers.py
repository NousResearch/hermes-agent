"""Cron wrap-envelope parsing for cron result cards.

The scheduler wraps cron delivery payloads in a header/divider/footer envelope
(cron.wrap_response, default on). The card layer unwraps it to render the
job name in the card header; a mismatched shape falls through unchanged.
"""

from __future__ import annotations

import re

CRON_WRAP_HEADER = "Cronjob Response: "
CRON_WRAP_JOBID_LINE = "(job_id: "
CRON_WRAP_DIVIDER = "\n-------------\n\n"
CRON_WRAP_FOOTER_PREFIX = '\n\nTo stop or manage this job, send me a new message (e.g. "stop reminder '



_CRON_FAILURE_RE = re.compile(r"Cron '[^']{0,120}' failed: |\*\*Status:\*\* [^\n]{0,60}(?:failed|error)")


def _looks_like_cron_failure(body: str) -> bool:
    """Whether a cron body is a failure notice (red header; a misjudgment only affects color, not delivery)."""
    return _CRON_FAILURE_RE.search(body) is not None


def _parse_cron_payload(content: str) -> tuple[str, str, str, bool]:
    """Unwrap the cron envelope → ``(task_name, card_body, job_id, is_failure)``.

    Body = the raw output plus the management-hint footer (with job_id, keeping
    the traceability of the native text); on a shape mismatch (wrap_response
    disabled / upstream rewording) the original content is returned whole.
    """
    if not content.startswith(CRON_WRAP_HEADER):
        return "", content, "", _looks_like_cron_failure(content)
    divider_pos = content.find(CRON_WRAP_DIVIDER)
    footer_pos = content.rfind(CRON_WRAP_FOOTER_PREFIX)
    if (divider_pos < 0 or footer_pos < 0 or footer_pos <= divider_pos
            or CRON_WRAP_JOBID_LINE not in content[:divider_pos]):
        return "", content, "", _looks_like_cron_failure(content)
    head = content[len(CRON_WRAP_HEADER):divider_pos].split("\n")
    task_name = head[0].strip()
    job_id = ""
    for line in head[1:]:
        line = line.strip()
        if line.startswith(CRON_WRAP_JOBID_LINE) and line.endswith(")"):
            job_id = line[len(CRON_WRAP_JOBID_LINE):-1]
            break
    payload = content[divider_pos + len(CRON_WRAP_DIVIDER):footer_pos].strip()
    body = payload or content
    hint = content[footer_pos:].lstrip()
    if hint:
        body += "\n\n" + hint
        if job_id:
            body += f" (job_id: {job_id})"
    elif job_id:
        body += f"\n\n(job_id: {job_id})"
    return task_name, body, job_id, _looks_like_cron_failure(payload)
