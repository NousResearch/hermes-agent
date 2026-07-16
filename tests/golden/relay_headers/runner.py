from __future__ import annotations

import os
from types import SimpleNamespace

from agent.chat_completion_helpers import _pool_affinity_headers


def run_case(case: dict):
    old_env = os.environ.get("HERMES_SESSION_SOURCE")
    try:
        os.environ.pop("HERMES_SESSION_SOURCE", None)
        for key, value in (case.get("env") or {}).items():
            os.environ[key] = value
        agent = SimpleNamespace(**case["agent"])
        return {
            "return": _pool_affinity_headers(agent, case.get("aux_task")),
            "messages": [],
            "db": [],
        }
    finally:
        if old_env is None:
            os.environ.pop("HERMES_SESSION_SOURCE", None)
        else:
            os.environ["HERMES_SESSION_SOURCE"] = old_env
