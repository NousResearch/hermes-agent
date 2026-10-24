"""The hosted dashboard retires through its admission fence, not an external SIGTERM."""

from pathlib import Path

from hermes_cli.web_server_skew_exit import (
    DEFAULT_SKEW_POLL_S,
    _run_retirement_watchdog,
)
from pm.generation_lifecycle import GenerationObserver


def start_generation_watchdog(
    server,
    *,
    observer=None,
    fence=None,
    poll_s=DEFAULT_SKEW_POLL_S,
    max_polls=None,
    owned=False,
):
    if not owned:
        return None
    observer = observer or GenerationObserver(Path(__file__).resolve().parent.parent)

    def observe():
        reason = observer.probe()
        if reason == "":
            observer.collect()
        return reason or None

    return _run_retirement_watchdog(
        server,
        observe=observe,
        fence=fence,
        poll_s=poll_s,
        max_polls=max_polls,
        thread_name="dependency-generation-watchdog",
    )
