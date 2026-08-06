"""/kanban slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

from agent.i18n import t
from gateway.platforms.base import MessageEvent

from gateway.slash_commands._shared import logger

class KanbanCommandsMixin:
    """/kanban handlers."""

    async def _handle_kanban_command(self, event: MessageEvent) -> str:
        """Handle /kanban — delegate to the shared kanban CLI.

        Run the potentially-blocking DB work in a thread pool so the
        gateway event loop stays responsive.  Read operations (list,
        show, context, tail) are permitted while an agent is running;
        mutations are allowed too because the board is profile-agnostic
        and does not touch the running agent's state.

        For ``/kanban create`` invocations we also auto-subscribe the
        originating gateway source (platform + chat + thread) to the new
        task's terminal events, so the user hears back when the worker
        completes / blocks / auto-blocks / crashes without having to poll.
        """
        import asyncio
        import re
        import shlex
        from hermes_cli.kanban import run_slash

        text = (event.text or "").strip()
        # Strip the leading "/kanban" (with or without slash), leaving args.
        if text.startswith("/"):
            text = text.lstrip("/")
        if text.startswith("kanban"):
            text = text[len("kanban"):].lstrip()

        tokens = shlex.split(text) if text else []
        requested_board = None
        action = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--board":
                if i + 1 >= len(tokens):
                    break
                requested_board = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--board="):
                requested_board = tok.split("=", 1)[1]
                i += 1
                continue
            action = tok
            break

        is_create = action == "create"

        try:
            output = await asyncio.to_thread(run_slash, text)
        except Exception as exc:  # pragma: no cover - defensive
            return t("gateway.kanban.error_prefix", error=exc)

        # Auto-subscribe on create. Parse the task id from the CLI's standard
        # success line ("Created t_abcd  (ready, assignee=...)"). If the user
        # passed --json we don't subscribe; they're clearly scripting and
        # can call /kanban notify-subscribe explicitly.
        if is_create and output:
            m = re.search(r"Created\s+(t_[0-9a-f]+)\b", output)
            if m:
                task_id = m.group(1)
                try:
                    source = event.source
                    platform = getattr(source, "platform", None)
                    platform_str = (
                        platform.value if hasattr(platform, "value") else str(platform or "")
                    ).lower()
                    chat_id = str(getattr(source, "chat_id", "") or "")
                    chat_type = str(getattr(source, "chat_type", "") or "") or None
                    thread_id = str(getattr(source, "thread_id", "") or "")
                    user_id = str(getattr(source, "user_id", "") or "") or None
                    delivery_metadata = self._thread_metadata_for_source(
                        source, self._reply_anchor_for_event(event)
                    ) or None
                    if isinstance(delivery_metadata, dict):
                        chat_type = str(getattr(source, "chat_type", "") or "")
                        if chat_type:
                            delivery_metadata.setdefault("chat_type", chat_type)
                    if platform_str and chat_id:
                        def _sub():
                            from hermes_cli import kanban_db as _kb
                            conn = _kb.connect(board=requested_board)
                            try:
                                _kb.add_notify_sub(
                                    conn, task_id=task_id,
                                    platform=platform_str, chat_id=chat_id,
                                    chat_type=chat_type,
                                    thread_id=thread_id or None,
                                    user_id=user_id,
                                    notifier_profile=getattr(self, "_kanban_notifier_profile", None) or self._active_profile_name(),
                                    delivery_metadata=delivery_metadata,
                                )
                            finally:
                                conn.close()
                        await asyncio.to_thread(_sub)
                        output = (
                            output.rstrip()
                            + "\n"
                            + t("gateway.kanban.subscribed_suffix", task_id=task_id)
                        )
                except Exception as exc:
                    logger.warning("kanban create auto-subscribe failed: %s", exc)

        # Gateway messages have practical length caps; truncate long
        # listings to keep the UX reasonable.
        if len(output) > 3800:
            output = output[:3800] + "\n" + t("gateway.kanban.truncated_suffix")
        return output or t("gateway.kanban.no_output")
