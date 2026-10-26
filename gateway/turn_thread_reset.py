"""Make a reused turn-executor thread look like a new one to its next item.

Turn code keys some state by the running thread instead of by turn: ``tools.interrupt`` bits and
reasons by thread id, per-turn callbacks in module ``threading.local`` objects, and the sudo
password cache's thread scope (the fallback when a turn has no session key). A new thread starts
with none of it; a parked executor worker (gateway/turn_executor.py) must drop all of it before it
takes another session's turn, or turn 1's interrupt, approval callback or cached sudo password is
live in turn 2. Context variables need nothing here: every item runs in a fresh context.

Modules are looked up in ``sys.modules`` only: one that was never imported holds no state.
A new per-turn ``threading.local`` in turn code belongs in ``_PER_TURN_THREAD_LOCALS``.
"""

from __future__ import annotations

import sys
import threading

_PER_TURN_THREAD_LOCALS = (
    ("tools.terminal_tool", "_callback_tls"),
    ("tools.environments.base", "_activity_callback_local"),
    ("tools.feishu_lark", "_local"),
    ("agent.vault_backends.unlock", "_callback_tls"),
    ("agent.vault_backends.unlock", "_current_session_tls"),
    ("agent.auxiliary_client", "_aux_probe_state"),
    ("agent.auxiliary_client", "_aux_interrupt_protection"),
    ("agent.auxiliary_client", "_aux_progress"),
    ("agent.auxiliary_client", "_aux_dispatch"),
    ("agent.auxiliary_client", "_aux_provider_response"),
    ("agent.auxiliary_client", "_aux_stream_deadline"),
    ("agent.local_prompt_progress", "_reading"),
    ("hermes_cli.plugins", "_thread_tool_whitelist"),
    ("hermes_cli.observability.shared_metrics_events", "_compression_attempt"),
)


def reset_reused_thread_state() -> None:
    """Drop the calling thread's turn-keyed state (call from the worker itself, between items)."""
    tid = threading.get_ident()
    interrupt = sys.modules.get("tools.interrupt")
    if interrupt is not None:
        interrupt.set_interrupt(False, tid)  # clears the bit, its reason and a pending yield
    sudo = sys.modules.get("tools.terminal_tool_sudo")
    if sudo is not None:
        with sudo._sudo_password_cache_lock:
            sudo._sudo_password_cache.pop(f"thread:{tid}", None)
    for module_name, attr in _PER_TURN_THREAD_LOCALS:
        local = getattr(sys.modules.get(module_name), attr, None)
        if local is not None:
            local.__dict__.clear()  # a threading.local's __dict__ is the calling thread's namespace
