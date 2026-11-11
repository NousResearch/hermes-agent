"""Process-context checks shared by gateway lifecycle callers."""

import os


def _is_running_inside_gateway_process_tree() -> bool:
    """True for descendants of the active profile's validated gateway."""
    try:
        from gateway.status import get_running_pid
        from hermes_cli.gateway import _is_pid_ancestor_of_current_process

        gateway_pid = get_running_pid(cleanup_stale=False)
        if gateway_pid is None or gateway_pid == os.getpid():
            return False
        return _is_pid_ancestor_of_current_process(gateway_pid)
    except Exception:
        return False
