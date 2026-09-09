"""Gateway-owned fresh plugin conversations. Never enters the parent turn/cache path."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
import json
import threading
import uuid

from gateway.session import SessionContext
from hermes_cli.plugin_side_runs import SideRunConfig, bounded_number


def owner_identity(source):
    """A shared room is not shared authority for a privately initiated child."""
    fields = ("chat_id", "chat_type", "thread_id", "user_id", "user_id_alt", "scope_id", "profile")
    return {"platform": source.platform.value, **{k: str(getattr(source, k, None) or "") for k in fields}}


def owns_side_run(source, model_config):
    if isinstance(model_config, str):
        model_config = json.loads(model_config)
    side = (model_config or {}).get("side_run")
    if side is None:
        return None
    return bool(source.user_id and isinstance(side, dict) and side.get("owner") == owner_identity(source))


@dataclass
class SideRun:
    session_id: str
    key: str
    event: object
    plugin_id: str
    config: SideRunConfig
    parent_id: str | None
    adapter: object
    db: object
    listing_key: str
    cancelled: threading.Event = field(default_factory=threading.Event)
    agent: object = None
    task: asyncio.Task | None = None
    registration: object = None

    def interrupt(self, reason="Side run cancelled", *, hard_cancel=True):
        from tools.approval import unregister_gateway_notify
        self.cancelled.set()
        unregister_gateway_notify(self.key)
        if self.agent is not None:
            self.agent.interrupt(reason, hard_cancel=hard_cancel)


class SideRunService:
    def __init__(self, runner, *, max_concurrent=None):
        from hermes_cli.config import load_config
        if max_concurrent is None:
            max_concurrent = (load_config().get("gateway", {}).get("side_runs", {}) or {}).get("max_concurrent", 3)
        self.max_concurrent = bounded_number(max_concurrent, "side run concurrency", 1, 10, integer=True)
        self.runner = runner
        self.loop = asyncio.get_running_loop()
        self.runs: dict[str, SideRun] = {}
        self.closing = False

    def start(self, event, plugin_id, prompt, config, *, registration_context=None):
        if self.closing or self.runner._draining:
            raise ValueError("Side runs are unavailable while the gateway is draining")
        if len(self.runs) >= self.max_concurrent:
            raise ValueError("Side-run capacity reached; wait or cancel an existing run")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 100000:
            raise ValueError("Side-run prompt must contain 1 to 100000 characters")
        # Revalidate even programmatically constructed dataclasses at the host boundary.
        config = SideRunConfig.from_mapping(config.to_dict())
        source = event.source
        if not event.allow_gateway_control or not source.user_id:
            raise ValueError("Side runs require an authenticated owner")
        adapter = self.runner._adapter_for_source(source)
        if adapter is None or not callable(getattr(adapter, "send", None)):
            raise ValueError("Side runs require a reply transport")
        with self.runner._profile_scope_for_source(source):
            session_db = getattr(self.runner, "_session_db", None)
            if session_db is None:
                raise ValueError("Side runs require a durable session database")
            db = session_db._db
            parent_key = self.runner._session_key_for_source(source)
            parent = self.runner.session_store._entries.get(parent_key)
        sid = "side-" + uuid.uuid4().hex
        run = SideRun(sid, "plugin-side:" + sid, deepcopy(event), plugin_id, config,
                      getattr(parent, "session_id", None), adapter, db, parent_key)
        self.runs[sid] = run
        if registration_context is not None:
            def cancel_on_unload():
                run.cancelled.set()
                self.loop.call_soon_threadsafe(self._cancel_run, run)
            run.registration = registration_context.on_unload(cancel_on_unload)
        run.task = self.loop.create_task(self._execute(run, prompt.strip()), name=f"plugin-side:{sid}")
        # Reuse the host's detached-work drain/interrupt ledger, including admission
        # before construction and final delivery after the synchronous worker exits.
        self.runner._track_deferred_agent_worker(run.task, run)
        return sid

    def cancel(self, source, session_id, *, plugin_id=None):
        run = self.runs.get(session_id)
        if run is None or owner_identity(source) != owner_identity(run.event.source):
            return False
        if plugin_id is not None and plugin_id != run.plugin_id:
            return False
        self._cancel_run(run)
        return True

    def _cancel_run(self, run):
        if self.runs.get(run.session_id) is not run:
            return
        run.interrupt()

    def shutdown(self):
        self.closing = True
        for run in list(self.runs.values()):
            self._cancel_run(run)

    async def wait(self):
        tasks = [run.task for run in self.runs.values() if run.task is not None]
        if tasks:
            await asyncio.gather(*tasks)

    async def _send(self, run, text):
        from gateway.run import _redact_gateway_user_facing_secrets
        source = run.event.source
        metadata = dict(self.runner._thread_metadata_for_source(
            source, self.runner._reply_anchor_for_event(run.event)) or {})
        # This is an independent message beside the parent's native stream; never seal that stream.
        metadata["_interim_send"] = True
        result = await run.adapter.send(
            source.chat_id, _redact_gateway_user_facing_secrets(text),
            reply_to=self.runner._reply_anchor_for_event(run.event),
            metadata=metadata,
        )
        if result is not None and getattr(result, "success", True) is False:
            raise RuntimeError("Side-run delivery failed")

    async def _execute(self, run, prompt):
        # The executor Future is shielded: cancelling an asyncio waiter cannot kill a sync worker.
        # Capacity and callbacks stay owned until that worker's finally/close has actually completed.
        timer = self.loop.call_later(run.config.run_budget_seconds, self._cancel_run, run)
        worker = None
        try:
            source = run.event.source
            with self.runner._profile_scope_for_source(source):
                tokens = self.runner._set_session_env(SessionContext(source, [], {}, session_key=run.key, session_id=run.session_id))
                try:
                    worker = asyncio.create_task(self.runner._run_in_executor_with_context(self._work, run, prompt))
                finally:
                    self.runner._clear_session_env(tokens)
            while True:
                try:
                    answer = await asyncio.shield(worker)
                    break
                except asyncio.CancelledError:
                    self._cancel_run(run)
                    if self.closing:
                        # The host has already budgeted drain/settle. Its executor
                        # quiesce policy retains DB handles for live sync workers.
                        raise
                    if worker.cancelled():
                        raise RuntimeError("Side-run worker was cancelled before execution") from None
            timer.cancel()
            text = f"Side run {run.session_id} cancelled." if run.cancelled.is_set() else answer
            await self._send(run, text)
        except Exception:
            # Never return/log raw constructor, provider or tool exceptions (often contain secrets).
            try:
                await self._send(run, f"Side run {run.session_id} failed. Check the route configuration or provider availability.")
            except Exception:
                # Session transcript/outcome remains available when delivery is unavailable.
                pass
        finally:
            timer.cancel()
            if worker is not None and not worker.done() and not self.closing:
                self._cancel_run(run)
                await asyncio.shield(worker)
            self.runs.pop(run.session_id, None)
            if run.registration is not None:
                run.registration.dispose()

    def _work(self, run, prompt):
        from run_agent import AIAgent
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from tools.approval import register_gateway_notify, unregister_gateway_notify
        from tools.approval_context import set_current_session_key, reset_current_session_key
        from gateway.session_context import isolated_session_context
        source, config = run.event.source, run.config
        token = set_current_session_key(run.key)
        agent = None
        status = "failed"
        try:
            with isolated_session_context(run.session_id):
                db = run.db
                # Independent, user-visible branch with an empty transcript: reuse the durable
                # branch marker so parent compression/resume never treats this as a continuation.
                saved = {"_branched_from": run.parent_id,
                         "side_run": {"owner": owner_identity(source), "plugin": run.plugin_id,
                                     "effective_config": config.to_dict()},
                         "gateway_runtime": {"provider": config.provider, "model": config.model},
                         "max_iterations": config.max_iterations, "max_tokens": config.max_tokens,
                         "reasoning_config": config.reasoning,
                         "codex_app_server_auto_compaction": "off"}
                db.create_session(run.session_id, source.platform.value, model=config.model,
                                  model_config=saved, user_id=source.user_id, chat_id=source.chat_id,
                                  chat_type=source.chat_type, thread_id=source.thread_id, session_key=run.listing_key,
                                  parent_session_id=run.parent_id, origin_json=json.dumps(source.to_dict()))
                if run.cancelled.is_set():
                    return "Cancelled"
                runtime = resolve_runtime_provider(requested=config.provider, target_model=config.model, strict=True)
                from hermes_cli.runtime_provider_strict import validate_explicit_runtime
                validate_explicit_runtime(config.provider, config.model, runtime)
                if (runtime.get("command") or runtime.get("acp_command")
                        or str(runtime.get("base_url", "")).startswith(("acp:", "moa:"))
                        or runtime.get("api_mode") == "codex_app_server"):
                    raise ValueError("External agent/virtual transports do not support isolated host tool approvals")
                saved["side_run"]["runtime"] = {key: runtime[key] for key in ("provider", "api_mode") if key in runtime}
                db.update_session_meta(run.session_id, json.dumps(saved), model=config.model)
                if config.tools is not None:
                    from toolsets import get_all_toolsets
                    if set(config.tools) - set(get_all_toolsets()):
                        raise ValueError("Unknown toolset in side-run configuration")
                kwargs = {key: runtime[key] for key in (
                    "provider", "api_key", "base_url", "api_mode", "credential_pool", "request_overrides", "capabilities",
                ) if key in runtime}
                overrides = dict(kwargs.get("request_overrides") or {})
                # SDK transports merge provider overrides after request assembly. A preset's
                # empty history, selected tools and output/reasoning limits must remain authoritative.
                reserved = {"messages", "input", "system", "instructions", "tools", "tool_choice",
                            "max_tokens", "max_completion_tokens", "max_output_tokens",
                            "reasoning", "reasoning_effort", "thinking"}
                for body in (overrides, overrides.get("extra_body") or {}):
                    if not isinstance(body, dict) or reserved.intersection(body):
                        raise ValueError("Provider overrides conflict with isolated side-run configuration")
                if runtime.get("extra_headers"):
                    overrides["extra_headers"] = runtime["extra_headers"]
                kwargs["request_overrides"] = overrides
                agent = AIAgent(
                    **kwargs, model=config.model, requested_provider=config.provider,
                    session_id=run.session_id, session_db=db, parent_session_id=run.parent_id,
                    platform=source.platform.value, user_id=source.user_id, user_id_alt=source.user_id_alt,
                    user_name=source.user_name, chat_id=source.chat_id, chat_name=source.chat_name,
                    chat_type=source.chat_type, thread_id=source.thread_id, gateway_session_key=run.key,
                    enabled_toolsets=None if config.tools is None else list(config.tools),
                    reasoning_config=deepcopy(config.reasoning), max_iterations=config.max_iterations,
                    max_tokens=config.max_tokens, run_budget_seconds=config.run_budget_seconds,
                    fallback_model=[], skip_memory=True, skip_context_files=True, skip_background_review=True,
                    quiet_mode=True,
                )
                run.agent = agent
                agent._max_output_tokens_ceiling = config.max_tokens
                # Native app-server promotion would replace Hermes' tool/approval loop mid-run.
                agent.codex_app_server_auto_compaction = "off"
                if getattr(agent, "api_mode", None) == "codex_app_server":
                    raise ValueError("App-server transport cannot enforce host side-run boundaries")
                if agent.model != config.model or agent.provider != runtime["provider"]:
                    raise ValueError("Agent changed explicit primary route")
                if run.cancelled.is_set():
                    agent.interrupt("Side run cancelled", hard_cancel=True)
                    return "Cancelled"
                register_gateway_notify(run.key, lambda data: self._notify(run, data))
                if run.cancelled.is_set():
                    unregister_gateway_notify(run.key)
                    return "Cancelled"
                result = agent.run_conversation(user_message=prompt, conversation_history=None, task_id=run.session_id)
                if result.get("failed") or result.get("error"):
                    raise RuntimeError("Side run failed")
                answer = result.get("final_response")
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("Side run returned no answer")
                status = "completed"
                return answer
        finally:
            unregister_gateway_notify(run.key)
            from tools.approval import clear_session
            try:
                clear_session(run.key)
                if run.db.get_session(run.session_id):
                    # SessionDB preserves the first end reason. Stamp the host outcome before
                    # AIAgent.close() writes its generic agent_close marker.
                    run.db.end_session(run.session_id, "side_run_" + ("cancelled" if run.cancelled.is_set() else status))
            finally:
                try:
                    if agent is not None:
                        agent.close()
                finally:
                    reset_current_session_key(token)

    def _notify(self, run, data):
        from gateway.run import _redact_approval_command
        request_id = data["request_id"]
        if run.cancelled.is_set() or self.closing:
            raise RuntimeError("Side run cancelled")
        text = (f"Side run {run.session_id} requests approval:\n"
                f"{_redact_approval_command(data.get('command', ''))}\n"
                f"/approve side:{request_id}\n/deny side:{request_id}")
        future = asyncio.run_coroutine_threadsafe(self._send(run, text), self.loop)
        try:
            future.result(timeout=30)
        except BaseException:
            future.cancel()
            raise RuntimeError("Approval transport unavailable") from None

    def approval_reply(self, event):
        command = event.get_command()
        if command not in {"approve", "deny"}:
            return None
        argument = event.get_command_args().strip()
        if not argument.startswith("side:"):
            return None
        request_id = argument[5:]
        from tools.approval import list_gateway_approvals, resolve_gateway_approval
        for run in list(self.runs.values()):
            if any(p.get("request_id") == request_id for p in list_gateway_approvals(run.key)):
                denied = self.runner._check_slash_access(event.source, command)
                if denied is not None:
                    return denied
                if not event.allow_gateway_control or owner_identity(event.source) != owner_identity(run.event.source):
                    return "Approval request is unavailable for this sender."
                if run.cancelled.is_set():
                    return "Approval request has expired."
                count = resolve_gateway_approval(run.key, "once" if command == "approve" else "deny", request_id=request_id)
                return "Side-run approval recorded." if count else "Approval request has expired."
        return "Approval request has expired or is unavailable."
