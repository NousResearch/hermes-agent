"""Hermes JIT Context OS - Native Context Engine.

Implements the ContextEngine ABC to act as a native context engine in Hermes (context.engine: jit).
Unlike a passive pre_llm_call hook, JitContextEngine actively selects and filters the wire messages
sent to LLM APIs:
    [System Prompt + <ONA_CONTEXT>] + [Last 3-4 User Turns]
ensuring stable, predictable request size (3,000 - 5,000 tokens) regardless of session length.
Full history remains intact in the SQLite session transcript.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.plugins.context_engine.jit")

try:
    from agent.context_engine import ContextEngine
except ImportError:
    class ContextEngine:  # type: ignore[no-redef]
        name: str = "jit"
        last_prompt_tokens: int = 0
        last_completion_tokens: int = 0
        last_total_tokens: int = 0
        threshold_tokens: int = 35000
        context_length: int = 200000
        compression_count: int = 0
        protect_first_n: int = 1
        protect_last_n: int = 8
        emit_automatic_compaction_status: bool = False

        def update_from_response(self, usage: Dict[str, Any]) -> None:
            pass

        def should_compress(self, prompt_tokens: int = None) -> bool:
            return False

        def compress(self, messages: List[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
            return messages

        def select_context(self, request_messages: List[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
            return request_messages


_LATEST_CAPSULE_CACHE: Dict[str, Any] = {
    "session_id": "",
    "capsule": "",
    "timestamp": 0.0,
    "scope": "general",
}

_JIT_ENGINE_ACTIVE: bool = False


def set_jit_engine_active(active: bool = True) -> None:
    global _JIT_ENGINE_ACTIVE
    _JIT_ENGINE_ACTIVE = active


def is_jit_engine_active() -> bool:
    global _JIT_ENGINE_ACTIVE
    return _JIT_ENGINE_ACTIVE or os.environ.get("HERMES_CONTEXT_ENGINE") == "jit"


def update_latest_capsule(session_id: str, capsule: str, scope: str = "general") -> None:
    global _LATEST_CAPSULE_CACHE
    _LATEST_CAPSULE_CACHE = {
        "session_id": session_id,
        "capsule": capsule,
        "timestamp": time.time(),
        "scope": scope,
    }


def get_latest_capsule(session_id: str, max_age_seconds: float = 60.0) -> Optional[str]:
    global _LATEST_CAPSULE_CACHE
    if (
        _LATEST_CAPSULE_CACHE.get("capsule")
        and _LATEST_CAPSULE_CACHE.get("session_id") == session_id
        and (time.time() - _LATEST_CAPSULE_CACHE.get("timestamp", 0.0)) < max_age_seconds
    ):
        return _LATEST_CAPSULE_CACHE["capsule"]
    return None


class JitContextEngine(ContextEngine):
    """Native ContextEngine implementation for JIT Context OS."""

    def __init__(
        self,
        keep_turns: int = 4,
        max_older_tool_chars: int = 1200,
        max_current_tool_chars: int = 3500,
        session_id: Optional[str] = None,
        db_path: Optional[str | Path] = None,
        **kwargs: Any,
    ):
        self._name = "jit"
        self.keep_turns = keep_turns
        self.max_older_tool_chars = max_older_tool_chars
        self.max_current_tool_chars = max_current_tool_chars
        self.session_id = session_id
        self.db_path = db_path
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 35000
        self.context_length = 200000
        self.compression_count = 0
        self.protect_first_n = 1
        self.protect_last_n = 8
        self.emit_automatic_compaction_status = False

    @property
    def name(self) -> str:
        return self._name

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if not usage:
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return False

    def compress(self, messages: List[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
        return messages

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        incoming_message: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Active Context Projection (Deterministic History Selection)."""
        if not request_messages:
            return []

        # 1. Separate system prompt from conversational turns
        sys_msg = None
        other_msgs = []
        for m in request_messages:
            if m.get("role") == "system" and sys_msg is None:
                sys_msg = dict(m)
            else:
                other_msgs.append(m)

        if sys_msg is None:
            sys_msg = {"role": "system", "content": ""}

        # 2. Compile or retrieve <ONA_CONTEXT> capsule
        capsule = self._resolve_capsule(incoming_message, other_msgs)

        # 3. Inject capsule into system prompt
        sys_content = sys_msg.get("content", "") or ""
        if capsule:
            if "<ONA_CONTEXT" in sys_content:
                sys_content = re.sub(r"<ONA_CONTEXT[\s\S]*?</ONA_CONTEXT>", capsule.strip(), sys_content)
            else:
                sys_content = f"{sys_content.rstrip()}\n\n{capsule.strip()}"
        sys_msg["content"] = sys_content

        # 4. Slice to the last N user turns and prune tool outputs
        selected_history = self._slice_turns(
            other_msgs,
            keep_turns=self.keep_turns,
            prune_tools=True,
        )

        return [sys_msg] + selected_history

    def _resolve_capsule(
        self,
        incoming_message: Optional[Dict[str, Any]],
        history_messages: List[Dict[str, Any]],
    ) -> str:
        """Compile capsule using self-contained compiler and L0 overlay."""
        session_id = self.session_id or "default"
        cached = get_latest_capsule(session_id)
        if cached:
            return cached

        try:
            from .compiler import compile_context
            from .l0.db import get_db
            from .l0.overlay import ensure_session, get_session_cwd
            from .l1.scope import resolve_scope

            user_text = ""
            if incoming_message and isinstance(incoming_message, dict):
                user_text = incoming_message.get("content", "")
            if not user_text and history_messages:
                for m in reversed(history_messages):
                    if m.get("role") == "user" and isinstance(m.get("content"), str):
                        user_text = m["content"]
                        break

            conn = get_db(db_path=self.db_path)
            try:
                session_cwd = get_session_cwd(conn, session_id)
                default_scope = "general"
                if session_cwd and Path(session_cwd).exists():
                    cname = Path(session_cwd).name
                    if cname not in ("wojciechwiesner", "Projects", "active"):
                        default_scope = cname

                sess_obj = ensure_session(conn, session_id, default_scope=default_scope)
                curr_scope = sess_obj.get("active_scope") or default_scope
                active_scope, _, _, _ = resolve_scope(user_text, curr_scope)

                res = compile_context(
                    conn=conn,
                    session_id=session_id,
                    user_message=user_text,
                    transcript_messages=history_messages,
                    active_scope=active_scope,
                    default_scope=default_scope,
                    session_cwd=session_cwd,
                )
                capsule = str(res)
                update_latest_capsule(session_id, capsule, active_scope)
                return capsule
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("JitContextEngine failed to compile capsule: %s", exc)
            return ""

    def _slice_turns(
        self,
        messages: List[Dict[str, Any]],
        keep_turns: int = 4,
        prune_tools: bool = True,
    ) -> List[Dict[str, Any]]:
        """Slice history to keep the last `keep_turns` user turns intact."""
        if not messages:
            return []

        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not user_indices:
            sliced = [dict(m) for m in messages[-keep_turns * 2:]]
        elif len(user_indices) <= keep_turns:
            sliced = [dict(m) for m in messages]
        else:
            cut_idx = user_indices[-keep_turns]
            sliced = [dict(m) for m in messages[cut_idx:]]

        if prune_tools:
            self._prune_tool_outputs(sliced)

        # Strip any <ONA_CONTEXT> from user messages to prevent duplication
        for m in sliced:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                if "<ONA_CONTEXT" in m["content"]:
                    m["content"] = re.sub(r"<ONA_CONTEXT[\s\S]*?</ONA_CONTEXT>", "", m["content"]).strip()

        return sliced

    def _prune_tool_outputs(self, messages: List[Dict[str, Any]]) -> None:
        """Truncate massive tool dumps from both older turns and current turn.
        
        Preserves valid tool-call / result pairing: tool_call_id, role, and name are NEVER modified.
        """
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        last_user_idx = user_indices[-1] if user_indices else -1

        for i, m in enumerate(messages):
            if m.get("role") == "tool":
                content = m.get("content")
                if not isinstance(content, str):
                    continue

                is_older_turn = (last_user_idx != -1 and i < last_user_idx)
                max_chars = self.max_older_tool_chars if is_older_turn else self.max_current_tool_chars

                if len(content) > max_chars:
                    turn_desc = "older turn" if is_older_turn else "current turn"
                    m["content"] = (
                        content[:max_chars // 2]
                        + f"\n... [tool output truncated by JIT context engine ({len(content)} chars, {turn_desc})] ...\n"
                        + content[-(max_chars // 4):]
                    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "keep_turns": self.keep_turns,
            "max_older_tool_chars": self.max_older_tool_chars,
            "max_current_tool_chars": self.max_current_tool_chars,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
        }
