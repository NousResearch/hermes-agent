"""The ``/bg`` slash command: a prompt answered by a separate background session (``cli_commands_mixin``
split, file-line ratchet). Locally the side agent is its own ``AIAgent`` on a daemon thread of this
process; on the gateway it runs on the session's route (``prompt.background``). Either way the result
prints into this CLI's scrollback and the task is recorded for the status bar and ``/agents``.
Same lazy-binding contract as the other siblings: ``cli``-level names are imported at call time.
"""

from __future__ import annotations

import time
import uuid
from contextlib import suppress
from datetime import datetime

from hermes_cli.cli_commands_session_tools import _command_arg, _cp, _gt, _lines, _t

# AIAgent keyword -> HermesCLI attribute copied onto the /bg side agent (provider routing + fallback).
_BG_PROVIDER_KWARGS = {
    "providers_allowed": "_providers_only", "providers_ignored": "_providers_ignore",
    "providers_order": "_providers_order", "provider_sort": "_provider_sort",
    "provider_require_parameters": "_provider_require_params",
    "provider_data_collection": "_provider_data_collection",
    "openrouter_min_coding_score": "_openrouter_min_coding_score", "fallback_model": "_fallback_model"}


class CLICommandsBackgroundMixin:
    """``/bg`` for :class:`cli.HermesCLI` (mixed into ``CLICommandsMixin``)."""

    def _handle_background_command(self, cmd: str):
        """Handle /bg <prompt> — run a prompt in a separate background session (its own AIAgent
        on a thread); the result prints here without touching the active history."""
        prompt = _command_arg(cmd)
        if not prompt:
            return _cp(*_lines(_t("background.usage")))
        from hermes_cli.cli_commands_mixin import _ellipsize
        self._background_task_counter += 1
        task_num = self._background_task_counter
        preview = _ellipsize(prompt, 60)
        from hermes_cli import gateway_side
        connection = gateway_side.connection_for(self)
        if connection is not None:
            # The gateway runs the side agent on this session's route (prompt.background); this
            # process never builds one.
            task_id, wait = gateway_side.start(connection, self.session_id, "background", prompt)
            _cp(f"  {_t('background.started', number=task_num, preview=preview)}",
                f"  {_t('background.task_id', task_id=task_id)}", f"  {_t('background.keep_chatting')}\n")
            return self._start_background_worker(task_id, task_num, preview, wait, prompt)
        from cli import set_approval_callback, set_secret_capture_callback, set_sudo_password_callback
        from run_agent import AIAgent
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
        if not self._ensure_runtime_credentials():
            return _cp(f"  {_t('background.no_credentials')}")
        _cp(f"  {_t('background.started', number=task_num, preview=preview)}",
            f"  {_t('background.task_id', task_id=task_id)}", f"  {_t('background.keep_chatting')}\n")
        turn_route = self._resolve_turn_agent_config(prompt)
        runtime = turn_route["runtime"]

        def produce():
            from agent.vault_backends.unlock import set_code_prompt_callback, set_save_login_prompt_callback, set_unlock_prompt_callback
            set_sudo_password_callback(self._sudo_password_callback)
            set_approval_callback(self._approval_callback)
            set_unlock_prompt_callback(self._vault_unlock_callback)
            set_save_login_prompt_callback(self._vault_save_login_callback)
            set_code_prompt_callback(self._vault_code_callback)
            with suppress(Exception):
                set_secret_capture_callback(self._secret_capture_callback)
            try:
                bg_agent = AIAgent(
                    model=turn_route["model"], acp_command=runtime.get("command"),
                    acp_args=runtime.get("args"), max_iterations=self.max_turns,
                    **{k: runtime.get(k) for k in ("api_key", "base_url", "provider", "api_mode",
                                                   "max_tokens")}, enabled_toolsets=self.enabled_toolsets,
                    quiet_mode=True, verbose_logging=False, session_id=task_id, platform="cli",
                    side_agent=True,
                    session_db=self._session_db, reasoning_config=self.reasoning_config,
                    service_tier=self.service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    **{kw: getattr(self, attr) for kw, attr in _BG_PROVIDER_KWARGS.items()})
                self._background_task_meta[task_id]["agent"] = bg_agent  # live activity for /agents
                # Silence raw spinner; route thinking through TUI widget when no foreground agent is active.
                bg_agent._print_fn = lambda *_a, **_kw: None

                def _bg_thinking(text: str) -> None:
                    # Concurrent bg tasks may race on _spinner_text; acceptable for best-effort UI.
                    if not self._agent_running:
                        self._spinner_text = text
                        if self._app:
                            # Display result in the CLI (thread-safe via patch_stdout). Force a TUI refresh
                            # first so spinner/status bar don't overlap with the output (fixes #2718).
                            # Same TUI refresh pattern as success path (#2718)
                            self._app.invalidate()

                bg_agent.thinking_callback = _bg_thinking
                try:
                    result = bg_agent.run_conversation(user_message=prompt, task_id=task_id)
                    response = result.get("final_response", "") if result else ""
                    if not response and result and result.get("error"):
                        response = _gt("model.error_prefix", error=result["error"])
                    return response
                finally:
                    # One agent per /bg task in a long-lived CLI process: close()
                    # is the owner boundary (memory shutdown, tool subprocesses,
                    # httpx clients); an unclosed side agent leaks all of them
                    # until the CLI exits (#50197).
                    with suppress(Exception):
                        bg_agent.close()
            finally:
                with suppress(Exception):
                    set_sudo_password_callback(None)
                    set_approval_callback(None)
                    set_secret_capture_callback(None)
                    set_unlock_prompt_callback(None)
                    set_save_login_prompt_callback(None)
                    set_code_prompt_callback(None)

        self._start_background_worker(task_id, task_num, preview, produce, prompt)

    def _start_background_worker(self, task_id, task_num, preview, produce, prompt):
        """The /bg result panel's daemon thread (``produce()`` returns the answer text). The task's
        record for /agents (number, prompt, start time; the local side agent once it exists) lives
        in ``_background_task_meta`` until ``done()``, so /agents can list what /bg started."""
        self._background_task_meta[task_id] = {"number": task_num, "prompt": prompt, "started_at": time.time(), "agent": None}

        def done():
            self._background_tasks.pop(task_id, None)
            self._background_task_meta.pop(task_id, None)
            if not self._agent_running:  # clear spinner only if no foreground agent owns it
                self._spinner_text = ""

        thread = self._side_worker(
            produce, name=f"bg-task-{task_id}", fail_label=_t("background.label", number=task_num),
            header_lines=[f"  {_t('background.complete', number=task_num)}",
                          f"  {_t('background.prompt', preview=preview)}"],
            title_suffix=_t("background.title_suffix", number=task_num),
            empty_note=f"  {_t('background.no_response')}", bell=True, on_done=done)
        self._background_tasks[task_id] = thread
        thread.start()
