"""User-defined ``quick_commands`` on the messaging gateway: ``exec`` (shell, no LLM), ``alias``
(rewrite to another slash command) and ``prompt`` (the saved prompt becomes the agent turn)."""
from __future__ import annotations

import asyncio
from typing import Optional

from agent.i18n import t
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class GatewayQuickCommandsMixin:
    """Quick-command resolution and dispatch for GatewayRunner (mixed into GatewayInboundMixin)."""

    def _hm_quick_commands(self) -> dict:
        """User-defined ``quick_commands`` mapping from config (empty dict when unset/malformed)."""
        cfg = self.config
        qc = (cfg.get("quick_commands") if isinstance(cfg, dict) else getattr(cfg, "quick_commands", None)) or {}
        return qc if isinstance(qc, dict) else {}

    @staticmethod
    def _hm_expand_alias_quick_command(event: MessageEvent, qcmd: dict) -> Optional[str]:
        """Rewrite ``event.text`` to an alias quick command's target; returns the new command name."""
        target = (qcmd.get("target") or "").strip()
        if not target:
            return None
        target = target if target.startswith("/") else f"/{target}"
        event.text = f"{target} {event.get_command_args().strip()}".strip()
        target_command = target.lstrip("/")
        return target_command.split()[0] if target_command else target_command

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
        except TimeoutError:
            return t("gateway.quick_command.timed_out")
        except Exception as e:
            return t("gateway.quick_command.error", error=e)

    async def _hm_dispatch_quick_command(
        self, event: MessageEvent, source: SessionSource, command: Optional[str]
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """``(handled, result, command)``: an alias rewrites ``command``; a prompt rewrites
        ``event.text`` and answers ``command=None`` so the turn reaches the agent unrouted."""
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
            if qtype == "prompt":
                # The configured prompt becomes this turn's user message; fall through to the agent.
                from hermes_cli.quick_commands import render_prompt_quick_command
                msg = render_prompt_quick_command(qcmd, event.get_command_args())
                if not msg:
                    return True, t("gateway.quick_command.no_prompt", command=command), command
                event.text = msg
                return False, None, None
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
            command = new_command  # Fall through to normal command dispatch
        return False, None, command
