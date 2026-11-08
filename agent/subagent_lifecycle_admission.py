"""Serialize lifecycle launch identities without holding the host admission lock."""

import functools
import threading

from tools.host_policy import HostPolicyError

_LAUNCH_LOCK = threading.RLock()


def serialized_launch(function):
    @functools.wraps(function)
    def launch(*args, **kwargs):
        # Construction is already process-serialized; this also includes the
        # correlation check and publication so duplicate launches cannot race.
        with _LAUNCH_LOCK:
            try:
                return function(*args, **kwargs)
            except HostPolicyError as exc:
                from agent.subagent_lifecycle import SubagentLifecycleError
                raise SubagentLifecycleError(str(exc)) from exc
    return launch
