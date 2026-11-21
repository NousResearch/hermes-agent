"""``/btw`` on the classic CLI: a side question answered from a snapshot of the conversation
(``agent/side_question.py``) without touching the live session.

With a gateway connection the verb runs in the gateway (``prompt.btw``) and this process only
renders (``hermes_cli/gateway_side.py``).
"""

from __future__ import annotations

from hermes_cli.cli_commands_session_tools import _command_arg, _cp, _gt, _lines, _t


def _preview(text: str) -> str:
    from hermes_cli.cli_commands_mixin import _ellipsize
    return _ellipsize(text, 60)


class CLISideThreadMixin:
    """The ``/btw`` handler; ``_side_worker`` comes from CLICommandsMixin."""

    def _handle_btw_command(self, cmd: str):
        """Handle /btw <question> — answer a side question about this conversation from a
        history snapshot via a one-shot auxiliary call. The live session is never touched
        (no history mutation, no role-alternation risk, no cache invalidation)."""
        question = _command_arg(cmd)
        if not question:
            return _cp(*_lines(_t("btw.usage")))
        from hermes_cli import gateway_side
        connection = gateway_side.connection_for(self)
        if connection is not None:
            # The gateway answers from its own copy of the conversation (prompt.btw): the live
            # agent's fork when it has one, else the stored transcript.
            _task_id, wait = gateway_side.start(connection, self.session_id, "btw", question)
            preview = _preview(question)
            _cp(*_lines(_gt("btw.started", preview=preview)), "")
            self._side_worker(wait, name="btw-side-question", fail_label="/btw",
                              header_lines=[f"  {_t('btw.header', preview=preview)}"],
                              title_suffix=_t("btw.title_suffix"),
                              empty_note=f"  {_t('btw.no_answer')}").start()
            return
        if not self._ensure_runtime_credentials():
            return _cp(f"  {_t('btw.no_credentials')}")
        # Snapshot NOW, on the UI thread — the foreground turn keeps appending to
        # conversation_history while the worker runs.
        history_snapshot = list(self.conversation_history or [])
        # Live agent → cache-parity fork (full context, warm cache reads).
        parent_agent = self.agent
        turn_route = self._resolve_turn_agent_config(question)
        runtime = turn_route["runtime"]
        main_runtime = {
            "model": turn_route["model"],
            **{k: runtime.get(k) for k in ("provider", "base_url", "api_key", "api_mode")},
            "session_id": getattr(parent_agent, "session_id", None),
        }
        preview = _preview(question)
        _cp(*_lines(_gt("btw.started", preview=preview)), "")

        def produce():
            from agent.side_question import answer_side_question
            return answer_side_question(
                question, history_snapshot, parent_agent=parent_agent, main_runtime=main_runtime)

        self._side_worker(produce, name="btw-side-question", fail_label="/btw",
                          header_lines=[f"  {_t('btw.header', preview=preview)}"],
                          title_suffix=_t("btw.title_suffix"),
                          empty_note=f"  {_t('btw.no_answer')}").start()
