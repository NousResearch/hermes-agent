"""Caller context and terminal-resource ownership for one cron run."""

from __future__ import annotations

import uuid
from typing import Optional

from agent.delegation_context import (
    enter_non_dispatcher_owned_context, exit_non_dispatcher_owned_context)


class _CronRunScope:
    """Per-run ContextVar / tool-cwd scope for ``run_job`` (ContextVars, not os.environ, so
    parallel jobs don't clobber each other). ``exit()`` restores the caller context;
    ``release()`` drops tool ownership only after the worker and actual cleanup complete.

    HERMES_SESSION_* are deliberately NOT seeded from job["origin"]: it is delivery metadata, not
    a sender, and terminal/tts/skills/send_message tools would act as if the origin user were
    driving the agent. Delivery reads job["origin"] / HERMES_CRON_AUTO_DELIVER_* directly.
    """

    def __init__(self, job: dict, job_id: str, execution_id: Optional[str]):
        from gateway.session_context import set_session_vars, _VAR_MAP
        from tools.terminal_tool import record_session_cwd, register_run_scoped_task
        from cron.scheduler import _CRON_DELIVERY_VARS, _resolve_job_workdir

        self._delivery_vars = _CRON_DELIVERY_VARS
        self._var_map = _VAR_MAP
        # Resolve workdir BEFORE set_session_vars so it owns the _SESSION_CWD set/clear.
        self.workdir = _resolve_job_workdir(job, job_id)
        self._ctx_tokens = set_session_vars(
            platform="",
            chat_id="",
            chat_name="",
            # Cron can't receive completions after its turn; async delegation output could
            # otherwise route to an unrelated chat via the ambient session key => inline delegation.
            # We clear the HERMES_SESSION_* routing keys just below, so an async delegation's completion
            # event carries session_key="" — _enrich_async_delegation_routing cannot resolve it and
            # _inject_watch_notification drops it ("no routing metadata"). And by the time a child finishes,
            # run_job has already shipped the job's final response via _deliver_result; there is no turn
            # left to re-enter. (Worse, get_current_session_key() can fall back to the ambient os.environ
            # HERMES_SESSION_KEY, which risks routing a cron subagent's output into an unrelated user chat.)
            # Declaring the channel stateless routes delegate_task to its existing inline/synchronous path,
            # so results return within the job's own turn. See declare_stateless_channel(). Upstream:
            # #53027, #63142.
            async_delivery=False,
            cwd=self.workdir or "",
        )
        for name in self._delivery_vars:
            _VAR_MAP[name].set("")
        # Workdir binds to the per-run task id (tool-layer cwd authority) instead of mutating
        # global TERMINAL_CWD; _SESSION_CWD above remains the prompt/context-file authority.
        self.task_id = f"cron:{job_id}:{execution_id or job.get('execution_id') or uuid.uuid4().hex}"
        if self.workdir:
            record_session_cwd(self.task_id, self.workdir)
        self._cron_session_var = _VAR_MAP["HERMES_CRON_SESSION"]
        self._cron_session_token = None
        self._non_dispatcher_token = None
        register_run_scoped_task(self.task_id)

    def enter(self) -> None:
        # Scope cron approval policy; exit() RESETS via token (pinning "" would suppress the legacy
        # os.environ fallback used by standalone entrypoints/tests).
        self._cron_session_token = self._cron_session_var.set("1")
        # Mark NOT the kanban worker: a worker's cronjob(action="run") lands here with
        # HERMES_KANBAN_TASK in env, and an unrelated job could close the worker's task. Must be a
        # ContextVar, NOT an os.environ clear (env is shared with the worker heartbeat and
        # concurrent jobs); copy_context() carries it into the agent thread.
        self._non_dispatcher_token = enter_non_dispatcher_owned_context()

    def exit(self) -> None:
        from gateway.session_context import clear_session_vars

        clear_session_vars(self._ctx_tokens)  # also clears _SESSION_CWD
        if self._cron_session_token is not None:
            self._cron_session_var.reset(self._cron_session_token)
        if self._non_dispatcher_token is not None:
            exit_non_dispatcher_owned_context(self._non_dispatcher_token)
        for name in self._delivery_vars:
            self._var_map[name].set("")

    def release(self) -> None:
        """Release tool state in the owning profile, independently of caller ContextVar tokens."""
        from tools.terminal_tool import clear_run_scoped_task, clear_session_cwd

        clear_session_cwd(self.task_id)
        clear_run_scoped_task(self.task_id)
