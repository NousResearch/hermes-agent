"""/verbose, /footer, /yolo, /debug, /personality, /busy slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

from typing import Union

from agent.i18n import t
from gateway.platforms.base import EphemeralReply, MessageEvent
from hermes_cli.config import atomic_config_write, cfg_get
from utils import is_truthy_value

from gateway.slash_commands._shared import logger

class RuntimeFlagsCommandsMixin:
    """/verbose, /footer, /yolo, /debug, /personality, /busy handlers."""

    async def _handle_personality_command(self, event: MessageEvent) -> str:
        """Handle /personality command - list or set a personality.

        All resolution/persistence goes through hermes_cli.personality —
        the single owner of personality state on every surface.
        """
        from gateway.run import _load_gateway_config
        from hermes_cli.personality import (
            active_personality_name,
            available_personalities,
            describe_personality,
            persist_personality,
            resolve_personality,
        )

        args = event.get_command_args().strip()

        try:
            config = _load_gateway_config()
        except Exception:
            config = {}
        personalities = available_personalities(config)

        if not args:
            current = active_personality_name(config)
            lines = [t("gateway.personality.header")]
            lines.append(t("gateway.personality.none_option"))
            for name, prompt in personalities.items():
                marker = " ✓" if name == current else ""
                lines.append(
                    t(
                        "gateway.personality.item",
                        name=f"{name}{marker}",
                        preview=describe_personality(prompt),
                    )
                )
            lines.append(t("gateway.personality.usage"))
            return "\n".join(lines)

        try:
            name, _new_prompt = resolve_personality(args, config)
        except ValueError:
            available = "`none`, " + ", ".join(f"`{n}`" for n in personalities)
            return t("gateway.personality.unknown", name=args.lower(), available=available)

        # Persist the selection only — hermes_cli.personality never writes
        # agent.system_prompt (user-owned manual overlay). persist_personality
        # writes get_hermes_home()/config.yaml, i.e. the routed profile under
        # multiplex; the next turn re-resolves the prompt from that file
        # (_get_system_prompt_for_channel), so no process-global state to update.
        if not persist_personality(name):
            return t("gateway.personality.save_failed", error="config write failed")

        if not name:
            return t("gateway.personality.cleared")
        return t("gateway.personality.set_to", name=name)

    async def _handle_yolo_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /yolo — toggle dangerous command approval bypass for this session only."""
        from tools.approval import (
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
        )

        session_key = self._session_key_for_source(event.source)
        current = is_session_yolo_enabled(session_key)
        if current:
            disable_session_yolo(session_key)
            return EphemeralReply(t("gateway.yolo.disabled"))
        else:
            enable_session_yolo(session_key)
            return EphemeralReply(t("gateway.yolo.enabled"))

    async def _handle_verbose_command(self, event: MessageEvent) -> str:
        """Handle /verbose command — cycle tool progress display mode.

        Gated by ``display.tool_progress_command`` in config.yaml (default off).
        When enabled, cycles the tool progress mode through off → new → all →
        verbose → off for the *current platform*.  The setting is saved to
        ``display.platforms.<platform>.tool_progress`` so each channel can
        have its own verbosity level independently.
        """
        from gateway.run import _gateway_config_home, _load_gateway_config, _platform_config_key

        config_path = _gateway_config_home() / "config.yaml"
        platform_key = _platform_config_key(event.source.platform)

        # --- check config gate ------------------------------------------------
        try:
            user_config = _load_gateway_config()
            gate_enabled = is_truthy_value(
                cfg_get(user_config, "display", "tool_progress_command"),
                default=False,
            )
        except Exception:
            gate_enabled = False

        if not gate_enabled:
            return t("gateway.verbose.not_enabled")

        # --- cycle mode (per-platform) ----------------------------------------
        cycle = ["off", "new", "all", "verbose", "log"]
        descriptions = {
            "off": t("gateway.verbose.mode_off"),
            "new": t("gateway.verbose.mode_new"),
            "all": t("gateway.verbose.mode_all"),
            "verbose": t("gateway.verbose.mode_verbose"),
            "log": t("gateway.verbose.mode_log"),
        }

        # Read current effective mode for this platform via the resolver
        from gateway.display_config import resolve_display_setting
        current = resolve_display_setting(user_config, platform_key, "tool_progress", "all")
        if current not in cycle:
            current = "all"
        idx = (cycle.index(current) + 1) % len(cycle)
        new_mode = cycle[idx]

        # Save to display.platforms.<platform>.tool_progress
        try:
            if "display" not in user_config or not isinstance(user_config.get("display"), dict):
                user_config["display"] = {}
            display = user_config["display"]
            if "platforms" not in display or not isinstance(display.get("platforms"), dict):
                display["platforms"] = {}
            if platform_key not in display["platforms"] or not isinstance(display["platforms"].get(platform_key), dict):
                display["platforms"][platform_key] = {}
            display["platforms"][platform_key]["tool_progress"] = new_mode
            atomic_config_write(config_path, user_config)
            return (
                f"{descriptions[new_mode]}\n"
                + t("gateway.verbose.saved_suffix", platform=platform_key)
            )
        except Exception as e:
            logger.warning("Failed to save tool_progress mode: %s", e)
            return f"{descriptions[new_mode]}\n" + t("gateway.verbose.save_failed", error=e)

    async def _handle_busy_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /busy — control what happens when messaging while Hermes is working.

        Usage:
            /busy               Show current busy input mode
            /busy status        Show current busy input mode
            /busy queue         Queue messages for the next turn
            /busy steer         Inject messages mid-run without interrupting
            /busy interrupt     Interrupt the current run (default)
        """
        arg = event.get_command_args().strip().lower()
        if not arg or arg == "status":
            mode = self._effective_busy_input_mode(event.source)
            if mode == "queue":
                behavior = "queues for next turn"
            elif mode == "steer":
                behavior = "steers into current run (after next tool call)"
            else:
                behavior = "interrupts current run"
            return EphemeralReply(
                f"**Busy input mode: `{mode}`" + "\n"
                f"Messages while busy: _{behavior}_" + "\n"
                f"Change with `/busy queue`, `/busy steer`, or `/busy interrupt`."
            )

        if arg not in {"queue", "interrupt", "steer"}:
            return EphemeralReply(
                f"Unknown mode `{arg}`. Use `/busy queue`, `/busy steer`, or `/busy interrupt`."
            )

        # Persist before mutate
        from cli import save_config_value
        if save_config_value("display.busy_input_mode", arg):
            profile_name = self._busy_profile_name_for_source(event.source)
            if profile_name:
                from gateway.run import _load_gateway_runtime_config

                self._snapshot_profile_busy_modes(
                    profile_name,
                    _load_gateway_runtime_config(),
                )
            else:
                self._busy_input_mode = arg
                # busy_input_mode is the source of truth for the text mode
                # too (run.py:_load_busy_text_mode) — re-derive it so the
                # adapter refresh below doesn't read a stale value and keep
                # interrupting after e.g. /busy queue (config IS saved; only
                # the live session lagged until restart).
                self._busy_text_mode = self._load_busy_text_mode()

            adapter = self._adapter_for_source(event.source)
            if adapter is not None:
                adapter._busy_text_mode = self._effective_busy_text_mode(event.source)

            if arg == "queue":
                behavior = "Messages will be queued for the next turn while Hermes is busy."
            elif arg == "steer":
                behavior = "Messages will be steered into the current run (after the next tool call)."
            else:
                behavior = "Messages will interrupt the current run while Hermes is busy."
            return EphemeralReply(
                f"Busy input mode set to **`{arg}`** (saved)." + "\n"
                f"_{behavior}_"
            )
        else:
            return EphemeralReply(
                f"Busy input mode could not be saved to config. Mode unchanged."
            )

    async def _handle_footer_command(self, event: MessageEvent) -> str:
        """Handle /footer command — toggle the runtime-metadata footer.

        Usage:
            /footer           → toggle on/off
            /footer on        → enable globally
            /footer off       → disable globally
            /footer status    → show current state + fields

        The footer is saved to ``display.runtime_footer.enabled`` (global).
        Per-platform overrides under ``display.platforms.<platform>.runtime_footer``
        are respected but not modified here — edit config.yaml directly for
        per-platform control.
        """
        from gateway.run import _gateway_config_home, _load_gateway_config, _platform_config_key, _resolve_gateway_model
        from gateway.runtime_footer import resolve_footer_config

        config_path = _gateway_config_home() / "config.yaml"
        platform_key = _platform_config_key(event.source.platform)

        # --- parse argument -------------------------------------------------
        arg = ""
        try:
            text = (getattr(event, "message", None) or "").strip()
            if text.startswith("/"):
                parts = text.split(None, 1)
                if len(parts) > 1:
                    arg = parts[1].strip().lower()
        except Exception:
            arg = ""

        # --- load config ----------------------------------------------------
        try:
            user_config: dict = _load_gateway_config()
        except Exception as e:
            return t("gateway.config_read_failed", error=e)

        effective = resolve_footer_config(user_config, platform_key)

        if arg in {"status", "?"}:
            state = t("gateway.footer.state_on") if effective["enabled"] else t("gateway.footer.state_off")
            fields = ", ".join(effective.get("fields") or [])
            return t(
                "gateway.footer.status",
                state=state,
                fields=fields,
                platform=platform_key,
            )

        if arg in {"on", "enable", "true", "1"}:
            new_state = True
        elif arg in {"off", "disable", "false", "0"}:
            new_state = False
        elif arg == "":
            new_state = not effective["enabled"]
        else:
            return t("gateway.footer.usage")

        # --- write global flag ---------------------------------------------
        try:
            if not isinstance(user_config.get("display"), dict):
                user_config["display"] = {}
            display = user_config["display"]
            if not isinstance(display.get("runtime_footer"), dict):
                display["runtime_footer"] = {}
            display["runtime_footer"]["enabled"] = new_state
            atomic_config_write(config_path, user_config)
        except Exception as e:
            logger.warning("Failed to save runtime_footer.enabled: %s", e)
            return t("gateway.config_save_failed", error=e)

        state = t("gateway.footer.state_on") if new_state else t("gateway.footer.state_off")
        example = ""
        if new_state:
            # Show a preview using current agent state if available.
            from gateway.runtime_footer import format_runtime_footer
            preview = format_runtime_footer(
                model=_resolve_gateway_model(user_config) or None,
                context_tokens=0,
                context_length=None,
                fields=effective.get("fields") or ["model", "context_pct", "cwd"],
            )
            if preview:
                example = t("gateway.footer.example_line", preview=preview)
        return t("gateway.footer.saved", state=state, example=example)

    async def _handle_debug_command(self, event: MessageEvent) -> str:
        """Handle /debug — upload debug report (summary only) and return paste URLs.

        Gateway uploads ONLY the summary report (system info + log tails),
        NOT full log files, to protect conversation privacy.  Users who need
        full log uploads should use ``hermes debug share`` from the CLI.
        """
        from hermes_cli.debug import (
            _capture_dump, collect_debug_report,
            upload_to_pastebin, _schedule_auto_delete,
            _GATEWAY_PRIVACY_NOTICE, _best_effort_sweep_expired_pastes,
        )

        # Run blocking I/O (dump capture, log reads, uploads) in a thread.
        def _collect_and_upload():
            _best_effort_sweep_expired_pastes()
            dump_text = _capture_dump()
            report = collect_debug_report(log_lines=200, dump_text=dump_text)

            urls = {}
            try:
                urls["Report"] = upload_to_pastebin(report)
            except Exception as exc:
                return t("gateway.debug.upload_failed", error=exc)

            # Schedule auto-deletion after 6 hours
            _schedule_auto_delete(list(urls.values()))

            lines = [_GATEWAY_PRIVACY_NOTICE, "", t("gateway.debug.header"), ""]
            label_width = max(len(k) for k in urls)
            for label, url in urls.items():
                lines.append(f"`{label:<{label_width}}`  {url}")

            lines.append("")
            lines.append(t("gateway.debug.auto_delete"))
            lines.append(t("gateway.debug.full_logs_hint"))
            lines.append(t("gateway.debug.share_hint"))
            return "\n".join(lines)

        # _run_in_executor_with_context, not a bare hop: this collects the
        # profile's logs/config off ``get_hermes_home()`` and uploads them to a
        # public paste. Losing the contextvar override would publish the DEFAULT
        # profile's diagnostics from another profile's chat.
        return await self._run_in_executor_with_context(_collect_and_upload)
