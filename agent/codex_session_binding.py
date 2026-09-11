"""Codex continuation in the existing Hermes session metadata, before native work."""

import os
from pathlib import Path


def session_binding_options(agent, cwd: str) -> dict:
    codex_home = str(Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").resolve())
    scope = {"session_id": agent.session_id, "cwd": str(Path(cwd).resolve()), "codex_home": codex_home}
    db = getattr(agent, "_session_db", None)
    binding = (db.get_session_model_config_value(agent.session_id, "codex_native_session")
               if db is not None else getattr(agent, "_codex_thread_binding", None))
    if binding is None and db is not None and db.get_session_model_config_value(agent.session_id, "_branched_from"):
        # Hermes' copied projection is not a fork of Codex's authoritative history.
        raise RuntimeError("Codex native history branching is unsupported; use /new or resume the original session")
    if binding is None and db is not None and any(
        message.get("role") == "assistant"
        for message in db.get_messages(agent.session_id, include_inactive=True)
    ):
        # Legacy sessions have no native marker. Their projection cannot prove
        # a fresh thread is safe, including tool-only or archived assistant rows.
        raise RuntimeError("Codex native history has no binding; use /new or resume a bound session")
    if binding is not None:
        if not isinstance(binding, dict) or any(binding.get(k) != v for k, v in scope.items()):
            raise RuntimeError("Codex native session scope changed; explicit new session required")
        if not isinstance(binding.get("thread_id"), str) or not binding["thread_id"]:
            raise RuntimeError("Codex native session has no resumable thread ID")

    def persist(thread_id: str) -> None:
        value = {**scope, "thread_id": thread_id}
        if db is not None:
            db.patch_session_model_config(agent.session_id, {"codex_native_session": value})
            if db.get_session_model_config_value(agent.session_id, "codex_native_session") != value:
                raise RuntimeError("Codex native binding was not persisted; turn not submitted")
        agent._codex_thread_binding = value

    return {"thread_id": binding["thread_id"] if binding else None,
            "codex_home": codex_home, "on_thread_ready": persist}
