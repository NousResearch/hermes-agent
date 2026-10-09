"""Plugin and user extension points on the inbound pipeline (moved out of ``gateway/run_inbound.py``).

``pre_gateway_dispatch`` (before auth), ``post_gateway_admission`` (once per admitted inbound
message, #129958), ``pre_command`` / ``command:<name>`` hooks, user ``quick_commands`` and
plugin-registered slash commands. Bound onto ``GatewayRunner`` through ``GatewayInboundMixin``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Optional, Tuple

from agent.i18n import t
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_context

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayInboundHooksMixin:
    """Hook, quick-command and plugin-command dispatch for ``GatewayInboundMixin``."""

    async def _hm_post_admission_consume(
        self, event: "MessageEvent", source: SessionSource, session_key: str
    ) -> Tuple[bool, Optional[str]]:
        """``post_gateway_admission`` exactly once per inbound message: ``(consumed, reply)``.

        Never while an admitted row EXECUTES (first run or restart replay): the authority path
        already offered the message at ingress, before its commit, so a consumed message is never
        admitted and an admitted one is never offered twice."""
        from gateway.session_ingress import executing_admission
        if executing_admission.get():
            return False, None
        from gateway.run_inbound_consumer import run_post_admission_hook
        return await run_post_admission_hook(self, event, source, session_key)

    async def _hm_pre_gateway_dispatch_hook(
        self, event: "MessageEvent", source: SessionSource
    ) -> Optional["MessageEvent"]:
        """Run the ``pre_gateway_dispatch`` plugin hook; None = drop, else the (maybe rewritten) event.
        Results: ``{"action": "skip"}`` → drop; ``{"action": "rewrite", "text"}`` → replace ``event.text``;
        ``allow``/None → normal dispatch. Runs BEFORE auth so plugins can handle unauthorized senders."""
        try:
            from hermes_cli.lifecycle import ainvoke_hook as _ainvoke_hook
            _hook_results = await _ainvoke_hook(
                "pre_gateway_dispatch", event=event, gateway=self,
                # getattr: bare-runner tests build GatewayRunner via object.__new__ without __init__.
                session_store=getattr(self, "session_store", None),
            )
        except Exception as _hook_exc:
            logger.warning("pre_gateway_dispatch invocation failed: %s", _hook_exc)
            _hook_results = []

        for _result in _hook_results:
            if not isinstance(_result, dict):
                continue
            _action = _result.get("action")
            if _action == "skip":
                logger.info(
                    "pre_gateway_dispatch skip: reason=%s platform=%s chat=%s",
                    _result.get("reason"), source.platform.value if source.platform else "unknown",
                    source.chat_id or "unknown",
                )
                return None
            if _action == "rewrite":
                _new_text = _result.get("text")
                if isinstance(_new_text, str):
                    event = dataclasses.replace(event, text=_new_text)
                break
            if _action == "allow":
                break
        return event

    def _hm_quick_commands(self) -> dict:
        """User-defined ``quick_commands`` mapping from config (empty dict when unset/malformed)."""
        cfg = self.config
        qc = (cfg.get("quick_commands") if isinstance(cfg, dict) else getattr(cfg, "quick_commands", None)) or {}
        return qc if isinstance(qc, dict) else {}

    @staticmethod
    def _hm_expand_alias_quick_command(event: "MessageEvent", qcmd: dict) -> Optional[str]:
        """Rewrite ``event.text`` to an alias quick command's target; returns the new command name."""
        target = (qcmd.get("target") or "").strip()
        if not target:
            return None
        target = target if target.startswith("/") else f"/{target}"
        event.text = f"{target} {event.get_command_args().strip()}".strip()
        target_command = target.lstrip("/")
        return target_command.split()[0] if target_command else target_command

    async def _hm_command_hooks(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str, command: str, canonical: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Fire ``pre_command`` (observer) and ``command:<canonical>`` (interceptor) hooks →
        ``(handled, result, new_command)`` (``new_command`` set when a handler rewrote the command).
        The running-agent path deliberately does NOT fire these — a slow or hostile plugin must not
        interfere with the operator's escape hatches for a live agent."""
        raw_args = event.get_command_args().strip()
        platform = source.platform.value if source.platform else ""
        try:
            from hermes_cli.plugins import fire_pre_command_hook
            fire_pre_command_hook(
                surface="gateway", command=str(canonical), alias_used=str(command),
                args_raw=raw_args, session_key=_quick_key, platform=platform,
            )
        except Exception as _pre_cmd_err:
            logger.debug("pre_command hook dispatch failed (non-fatal): %s", _pre_cmd_err)

        # Handlers may return ``{"decision": "deny" | "handled" | "rewrite", ...}`` to intercept
        # dispatch; handlers returning nothing behave as plain observers.
        hook_ctx = {
            "platform": platform, "user_id": source.user_id, "command": canonical,
            "raw_command": command, "args": raw_args, "raw_args": raw_args,
        }
        try:
            hook_results = await self.hooks.emit_collect(f"command:{canonical}", hook_ctx)
        except Exception as _hook_err:
            logger.debug("command:%s hook dispatch failed (non-fatal): %s", canonical, _hook_err)
            hook_results = []

        for hook_result in hook_results:
            if not isinstance(hook_result, dict):
                continue
            decision = str(hook_result.get("decision", "")).strip().lower()
            message = hook_result.get("message")
            message = message if isinstance(message, str) and message else None
            if decision == "deny":
                return True, message or t("gateway.hooks.command_blocked", command=command), None
            if decision == "handled":
                return True, message, None
            if decision == "rewrite":
                new_command = str(hook_result.get("command_name", "")).strip().lstrip("/")
                if new_command:
                    event.text = f"/{new_command} {str(hook_result.get('raw_args', '')).strip()}".strip()
                    return False, None, event.get_command()
        return False, None, None

    async def _hm_run_exec_quick_command(self, command: str, exec_cmd: str) -> str:
        """Run a ``type: exec`` quick command in the gateway process (30 s cap, sanitized env — the
        gateway process has every API key in os.environ; output is redacted too)."""
        try:
            from tools.environments.local import build_subprocess_env
            proc = await asyncio.create_subprocess_shell(
                exec_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=build_subprocess_env(),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = (stdout or stderr).decode().strip()
            if output:
                from agent.redact import redact_sensitive_text
                output = redact_sensitive_text(output)
            return output or t("gateway.quick_command.no_output")
        except asyncio.TimeoutError:
            return t("gateway.quick_command.timed_out")
        except Exception as e:
            return t("gateway.quick_command.error", error=e)

    async def _hm_dispatch_quick_and_plugin_commands(
        self, event: "MessageEvent", source: SessionSource, command: Optional[str]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Drain gate, user-defined quick commands (exec/alias) and plugin slash commands →
        ``(handled, result, command)``; an alias quick command rewrites ``command``."""
        if self._draining:
            return True, t("gateway.busy.drain_rejected_new_work", action=self._status_action_gerund()), command

        # User-defined quick commands (bypass agent loop, no LLM call)
        qcmd = self._hm_quick_commands().get(command) if command else None
        if qcmd is not None:
            # Quick commands are slash capabilities too — and type:exec ones run a shell command in
            # the gateway process. They are never in the registry, so the early gate never fires for
            # them; apply the same admin/user policy to the raw typed name here.
            # The early gate above only fires for registry-known commands, so quick commands (never in the
            # registry) would otherwise reach this dispatch sink unchecked. (#44727)
            _denied = self._check_slash_access(source, command)
            if _denied is not None:
                return True, _denied, command
            qtype = qcmd.get("type")
            if qtype == "exec":
                exec_cmd = qcmd.get("command", "")
                if not exec_cmd:
                    return True, t("gateway.quick_command.no_command", command=command), command
                return True, await self._hm_run_exec_quick_command(command, exec_cmd), command
            if qtype != "alias":
                return True, t("gateway.quick_command.unsupported_type", command=command), command
            new_command = self._hm_expand_alias_quick_command(event, qcmd)
            if new_command is None:
                return True, t("gateway.quick_command.no_target", command=command), command
            command = new_command  # Fall through to normal command dispatch below

        # Plugin-registered slash commands. Underscores normalize to hyphens so Telegram's
        # underscored autocomplete form matches plugin commands registered with hyphens.
        if command:
            try:
                from hermes_cli.plugins import get_plugin_command_handler
                plugin_handler = get_plugin_command_handler(command.replace("_", "-"))
                if plugin_handler:
                    # The agent-turn path binds HERMES_SESSION_* via _set_session_env; this dispatch
                    # sits before it, so a handler reading get_session_env() would see an empty or a
                    # foreign (cron agent's os.environ) session (#108698). No session_entry exists yet,
                    # so session_key is derived from source. Sync handlers run on the gateway pool
                    # (contextvars carried), never the loop thread: blocking I/O there starves the
                    # liveness watchdog and the process exits 75 mid-handler (#105279).
                    _plugin_context = build_session_context(source, self.config)
                    _plugin_context.session_key = self._session_key_for_source(source)
                    user_args = event.get_command_args().strip()
                    with self._session_env_scope(_plugin_context):
                        if asyncio.iscoroutinefunction(plugin_handler):
                            result = await plugin_handler(user_args)
                        else:
                            result = await self._run_in_executor_with_context(plugin_handler, user_args)
                            if asyncio.iscoroutine(result):
                                result = await result
                    return True, str(result) if result else None, command
            except Exception as e:
                logger.warning("Plugin command dispatch failed: %s", e)
        return False, None, command
