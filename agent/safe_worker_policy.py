"""Private bootstrap policy for admission-owned troubleshooting workers.

Bind once in a fresh exec, before config/agent/plugin imports. This is not a
sandbox or a public launch switch: only the managed worker bootstrap consumes
the owner's validated assignment. No environment flag grants this policy and
no reset exists; helper threads see the same immutable snapshot.
"""
from dataclasses import dataclass
import json
import os
import sys
from threading import Lock


@dataclass(frozen=True, slots=True)
class _SafeWorkerPolicy:
    pid: int
    safe_mode: bool
    ignore_user_config: bool
    config_json: str


_policy: _SafeWorkerPolicy | None = None
_bind_lock = Lock()
_RUNTIME_IMPORTS = frozenset({
    "run_agent", "model_tools", "hermes_cli.config", "hermes_cli.env_loader",
    "hermes_cli.plugins", "providers", "agent.agent_init",
})


def _bind_safe_worker_policy(*, safe_mode: bool, ignore_user_config: bool,
                             config: dict) -> _SafeWorkerPolicy:
    """Called by the private worker bootstrap, never by the owner or public CLI."""
    global _policy
    if type(safe_mode) is not bool or type(ignore_user_config) is not bool:
        raise TypeError("worker bypass policy fields must be booleans")
    if not (safe_mode or ignore_user_config):
        raise ValueError("ordinary execution does not bind a bypass policy")
    if type(config) is not dict:
        raise TypeError("worker config must be a JSON object")
    serialized = json.dumps(config, allow_nan=False)
    with _bind_lock:
        if _policy is not None:
            raise RuntimeError("worker policy is already bound")
        if _RUNTIME_IMPORTS.intersection(sys.modules):
            raise RuntimeError("worker policy must bind before runtime imports")
        _policy = _SafeWorkerPolicy(os.getpid(), safe_mode,
                                    safe_mode or ignore_user_config, serialized)
        return _policy


def _current_policy() -> _SafeWorkerPolicy | None:
    if _policy is not None and _policy.pid != os.getpid():
        raise RuntimeError("worker policy cannot cross a fork; use a fresh exec")
    return _policy


def safe_worker_enabled() -> bool:
    policy = _current_policy()
    return policy is not None and policy.safe_mode


def worker_config_snapshot() -> dict | None:
    """Return detached explicit config; None means the ordinary loader owns it."""
    policy = _current_policy()
    return json.loads(policy.config_json) if policy is not None else None
