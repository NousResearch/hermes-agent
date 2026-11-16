"""Ephemeral popup chat (``/tmp``): a modal overlay that talks to a fresh, context-free agent.

Each message opens its own throwaway ``AIAgent`` (unique session id, ``skip_memory=True``,
``side_agent=True``) and runs exactly one ``run_conversation`` call seeded only with that single
message -- no main-session history, no memory, and nothing carried between popup turns. ESC closes
the popup and discards everything. Mirrors the established modal-overlay convention used by the
clarify / approval / slash-confirm panels (a ``self._temp_chat_state`` dict + a
``_get_temp_chat_display_fragments`` renderer wired into ``_build_tui_layout_children``).
"""
from __future__ import annotations

import threading
import uuid
from contextlib import suppress
from datetime import datetime

from agent.i18n import t


class CLITempChatMixin:
    """`/tmp` ephemeral popup chat: no context, each message independent, ESC closes."""

    def _open_temp_chat(self) -> bool:
        """Open the ephemeral popup. Returns True (kept on the REPL, like other modal openers)."""
        from cli import _cprint
        if not getattr(self, "_app", None):
            _cprint("  " + t("cli.tui.temp_chat_no_tui"))
            return True
        if self._temp_chat_state is not None:
            return True
        self._capture_modal_input_snapshot()
        self._temp_chat_state = {"messages": [], "pending": False, "seq": 0}
        self._invalidate(min_interval=0.0)
        return True

    def _close_temp_chat(self) -> None:
        """Tear down the popup and restore any draft that was in the composer."""
        self._temp_chat_state = None
        self._restore_modal_input_snapshot()
        self._invalidate(min_interval=0.0)

    def _get_temp_chat_display_fragments(self):
        """Bordered panel: title, each popup message (You / reply), and a status/hint footer."""
        from cli import _wrap_panel_text
        state = self._temp_chat_state
        if state is None:
            return []
        title = t("cli.tui.temp_chat_title")
        messages = state.get("messages") or []
        you_label = t("cli.tui.temp_chat_you")
        reply_label = t("cli.tui.temp_chat_reply")
        hint = t("cli.tui.temp_chat_hint")
        thinking = t("cli.tui.temp_chat_thinking")

        # Compute width from the longest raw line so the panel is stable as the chat grows.
        raw_lines: list[str] = []
        for role, text in messages:
            prefix = you_label if role == "user" else reply_label
            for line in (text or "").splitlines() or [""]:
                raw_lines.append(f"{prefix} {line}")
        raw_lines.append(hint)
        from cli import _panel_box_width
        box_width = _panel_box_width(title, raw_lines)
        inner = box_width - 2

        rows: list[tuple[str, str]] = []
        for role, text in messages:
            is_user = role == "user"
            prefix = you_label if is_user else reply_label
            style = "class:temp-chat-you" if is_user else "class:temp-chat-reply"
            body = f"{prefix} {text}" if text else prefix
            for wrapped in _wrap_panel_text(body, inner, subsequent_indent="  "):
                rows.append((style, wrapped))
        if state.get("pending"):
            rows.append(("class:temp-chat-pending", thinking))
        if not rows:
            rows.append(("class:temp-chat-hint", t("cli.tui.temp_chat_empty")))
        rows.append(("class:temp-chat-hint", hint))
        row_styles = [s for s, _ in rows]
        body_lines = [txt for _, txt in rows]
        return self._render_sudo_style_panel(title, body_lines, row_styles=row_styles)

    def _tui_temp_chat_submit(self, event) -> None:
        """Enter inside the popup: send the typed line to a fresh no-context agent."""
        buf = event.app.current_buffer
        text = buf.text.strip()
        buf.reset()
        state = self._temp_chat_state
        if state is None:
            return
        if not text or state.get("pending"):
            event.app.invalidate()
            return
        state["messages"].append(("user", text))
        state["pending"] = True
        state["seq"] = state.get("seq", 0) + 1
        seq = state["seq"]
        event.app.invalidate()
        self._temp_chat_run(text, seq)

    def _temp_chat_run(self, prompt: str, seq: int) -> None:
        """Spawn a daemon thread that answers ``prompt`` with a throwaway, context-free agent."""
        from cli import set_approval_callback, set_secret_capture_callback, set_sudo_password_callback
        from run_agent import AIAgent
        from hermes_cli.cli_commands_mixin import _BG_PROVIDER_KWARGS

        if not self._ensure_runtime_credentials():
            self._temp_chat_deliver(seq, t("cli.tui.temp_chat_no_credentials"))
            return
        turn_route = self._resolve_turn_agent_config(prompt)
        runtime = turn_route["runtime"]
        task_id = f"tmp_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"

        def produce() -> str:
            from agent.vault_backends.unlock import (
                set_code_prompt_callback, set_save_login_prompt_callback, set_unlock_prompt_callback)
            set_sudo_password_callback(self._sudo_password_callback)
            set_approval_callback(self._approval_callback)
            set_unlock_prompt_callback(self._vault_unlock_callback)
            set_save_login_prompt_callback(self._vault_save_login_callback)
            set_code_prompt_callback(self._vault_code_callback)
            with suppress(Exception):
                set_secret_capture_callback(self._secret_capture_callback)
            try:
                agent = AIAgent(
                    model=turn_route["model"], acp_command=runtime.get("command"),
                    acp_args=runtime.get("args"), max_iterations=self.max_turns,
                    **{k: runtime.get(k) for k in (
                        "api_key", "base_url", "provider", "api_mode", "max_tokens")},
                    enabled_toolsets=self.enabled_toolsets, quiet_mode=True, verbose_logging=False,
                    session_id=task_id, platform="cli", side_agent=True, skip_memory=True,
                    skip_background_review=True, session_db=self._session_db,
                    reasoning_config=self.reasoning_config, service_tier=self.service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    **{kw: getattr(self, attr) for kw, attr in _BG_PROVIDER_KWARGS.items()})
                agent._print_fn = lambda *_a, **_kw: None
                try:
                    result = agent.run_conversation(user_message=prompt, task_id=task_id)
                    response = result.get("final_response", "") if result else ""
                    if not response and result and result.get("error"):
                        response = t("cli.tui.temp_chat_error", error=result["error"])
                    return response or t("cli.tui.temp_chat_no_response")
                finally:
                    with suppress(Exception):
                        agent.close()
            finally:
                with suppress(Exception):
                    set_sudo_password_callback(None)
                    set_approval_callback(None)
                    set_secret_capture_callback(None)
                    set_unlock_prompt_callback(None)
                    set_save_login_prompt_callback(None)
                    set_code_prompt_callback(None)

        def run() -> None:
            try:
                body = produce()
            except Exception as exc:
                body = t("cli.tui.temp_chat_error", error=exc)
            self._temp_chat_deliver(seq, body)

        threading.Thread(target=run, daemon=True, name=f"temp-chat-{task_id}").start()

    def _temp_chat_deliver(self, seq: int, body: str) -> None:
        """Append a reply to the popup, but only if the popup is still open on the same turn."""
        state = self._temp_chat_state
        # Dropped if the user closed the popup (state is None) or sent another message meanwhile.
        if state is None or state.get("seq") != seq:
            return
        state["messages"].append(("assistant", body))
        state["pending"] = False
        if getattr(self, "_app", None):
            self._invalidate(min_interval=0.0)
