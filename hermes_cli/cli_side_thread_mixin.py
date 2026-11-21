"""``/btw`` on the classic CLI: side questions and side exploration threads.

``/btw <question>`` answers from a snapshot of the conversation (``agent/side_question.py``).
``/btw --explore <task>`` forks the live agent into a tool-using side thread
(``agent/side_thread.py``) whose steps print as dim ``┊`` lines while the main turn keeps going;
its report lands in a side-result panel. ``--threads`` / ``--stop`` / ``--promote`` manage those
threads; a promotion is queued as the next normal user turn (the ``/plan`` pattern), so the main
transcript only ever grows by real turns.

With a gateway connection the whole verb runs in the gateway (``prompt.btw``) and this process
only renders (``hermes_cli/gateway_side.py``).
"""

from __future__ import annotations

from agent.i18n import t
from hermes_cli.cli_commands_session_tools import _command_arg, _cp, _dim_line, _gt, _lines, _t


def _preview(text: str) -> str:
    from hermes_cli.cli_commands_mixin import _ellipsize
    return _ellipsize(text, 60)


def _report_header(thread_id: str, preview: str) -> str:
    """The first line of ``side_thread.report``: the panel header above the report body."""
    return t("side_thread.report", thread_id=thread_id, preview=preview, report="").strip()


class CLISideThreadMixin:
    """The ``/btw`` handler; ``_side_worker`` / ``_queue_prompt_turn`` come from CLICommandsMixin."""

    def _handle_btw_command(self, cmd: str):
        """Handle /btw — a side question about this conversation, or (with flags) a side
        exploration thread. The live session is never touched (no history mutation, no
        role-alternation risk, no cache invalidation)."""
        from agent.side_thread import parse_btw_args, run_thread_command
        arg = _command_arg(cmd)
        request = parse_btw_args(arg)
        if not request.text and request.action in {"ask", "explore"}:
            return _cp(*_lines(_t("btw.usage")), *_lines(t("side_thread.usage")))
        from hermes_cli import gateway_side
        connection = gateway_side.connection_for(self)
        if connection is not None:
            return self._btw_via_gateway(connection, arg, request)
        if request.action == "ask":
            return self._btw_ask(request.text)
        if request.action == "explore":
            return self._btw_explore(request.text)
        result = run_thread_command(self.session_id, request)
        return self._btw_thread_result(result.text, result.prompt)

    def _btw_thread_result(self, text: str, prompt: str = "") -> None:
        """Show a --threads/--stop/--promote result; a promotion's prompt becomes the next turn."""
        if prompt:
            return self._queue_prompt_turn(prompt, "/btw --promote", "  " + text)
        _cp(*_lines(text))

    def _btw_ask(self, question: str) -> None:
        """The classic /btw: one tool-less answer from a history snapshot."""
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

    def _btw_explore(self, task: str) -> None:
        """/btw --explore: fork the live agent into a tool-using side thread."""
        from agent.side_thread import (
            SideThreadUnavailable, explore_refusal, format_report, register_thread, run_side_exploration,
        )
        if (refusal := explore_refusal(self.agent)) is not None:
            return _cp(f"  {refusal.user_message()}")
        try:
            thread = register_thread(self.session_id, task)
        except SideThreadUnavailable as exc:
            return _cp(f"  {exc.user_message()}")
        parent_agent, history_snapshot = self.agent, list(self.conversation_history or [])
        tid, preview = thread.thread_id, _preview(task)
        _cp(*_lines(t("side_thread.started", thread_id=tid, preview=preview)), "")

        def produce():
            run_side_exploration(thread, parent_agent, history_snapshot,
                                 on_progress=lambda line: _cp(_dim_line(f"┊ 🔭 {tid}  {line}")))
            return format_report(thread)

        self._side_worker(produce, name=f"btw-explore-{tid}", fail_label=f"/btw {tid}",
                          header_lines=[f"  {_report_header(tid, preview)}"], title_suffix=f"(btw {tid})",
                          empty_note=f"  {t('side_thread.no_report', thread_id=tid)}", bell=True).start()

    def _btw_via_gateway(self, connection, arg: str, request) -> None:
        """The gateway answers from its own copy of the conversation (``prompt.btw``): the live
        agent's fork when it has one, else the stored transcript. Thread commands reply at once."""
        from hermes_cli import gateway_side
        explore = request.action == "explore"
        tid_box: list[str] = []

        def on_progress(line: str) -> None:
            _cp(_dim_line(f"┊ 🔭 {tid_box[0] if tid_box else ''}  {line}"))

        task_id, wait, reply = gateway_side.start_btw(
            connection, self.session_id, arg, on_progress=on_progress if explore else None)
        if task_id is None:
            return self._btw_thread_result(str(reply.get("text") or ""), str(reply.get("prompt") or ""))
        tid_box.append(task_id)
        preview = _preview(request.text)
        if explore:
            _cp(*_lines(t("side_thread.started", thread_id=task_id, preview=preview)), "")
            header, suffix = _report_header(task_id, preview), f"(btw {task_id})"
            empty = t("side_thread.no_report", thread_id=task_id)
        else:
            _cp(*_lines(_gt("btw.started", preview=preview)), "")
            header, suffix, empty = _t("btw.header", preview=preview), _t("btw.title_suffix"), _t("btw.no_answer")
        self._side_worker(wait, name=f"btw-{task_id}", fail_label=f"/btw {task_id}" if explore else "/btw",
                          header_lines=[f"  {header}"], title_suffix=suffix, empty_note=f"  {empty}",
                          bell=explore).start()
