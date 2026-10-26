"""Execution-mode invariants every cron job definition must satisfy.

``create_job`` and ``update_job`` (``cron/jobs.py``) both run these, so an update cannot
bypass a rule the create door enforces.
"""
from typing import Optional

from hermes_constants import display_hermes_home

NO_AGENT_WITHOUT_SCRIPT_ERROR = (
    "no_agent=True requires a script — with no agent and no script "
    "there is nothing for the job to run."
)

SCRIPT_IS_COMMAND_LINE_ERROR = (
    "script must name a file under {scripts_dir} (e.g. 'watchdog.sh'), not a shell command line: "
    "{script!r} has no such file. Write the command into a script there and pass its filename."
)


def _validate_job_mode_invariants(
    monitor_script: Optional[str],
    monitor_url: Optional[str],
    no_agent: bool,
    script: Optional[str],
) -> None:
    """Execution-mode invariants shared by create_job and update_job (no bypass via the update
    door)."""
    if monitor_script and monitor_url:
        raise ValueError(
            "monitor_script and monitor_url are mutually exclusive — a job "
            "can only have one monitor source.")
    if (monitor_script or monitor_url) and no_agent:
        raise ValueError(
            "monitor_script/monitor_url cannot be combined with no_agent=True — "
            "the whole point of a monitor job is to suppress or wake the AGENT "
            "based on source changes. Use a plain no_agent script job instead.")
    if no_agent and not script:
        raise ValueError(NO_AGENT_WITHOUT_SCRIPT_ERROR)
    if script and _script_is_command_line(script):
        raise ValueError(SCRIPT_IS_COMMAND_LINE_ERROR.format(
            scripts_dir=display_hermes_home() + "/scripts/", script=script))


def _script_is_command_line(script: str) -> bool:
    """A ``script`` with whitespace that resolves to no file is a command line (``echo hi``),
    not a path; run every tick it would deliver "Script not found" as the payload. A plain
    missing filename stays creatable (``hermes cron doctor`` reports it)."""
    if not any(ch.isspace() for ch in script.strip()):
        return False
    from cron.lifecycle_guard import _resolve_script_path
    path = _resolve_script_path(script)
    return path is None or not path.is_file()
