"""/reasoning, /fast slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

from typing import Optional
import asyncio

from agent.i18n import t
from gateway.platforms.base import MessageEvent
from hermes_cli.config import atomic_config_write

from gateway.slash_commands._shared import logger

class ReasoningCommandsMixin:
    """/reasoning, /fast handlers."""

    def _save_gateway_config_key(self, key_path: str, value) -> bool:
        """Save a dot-separated key to config.yaml (shared by /reasoning, /fast
        and their interactive pickers)."""
        from gateway.run import _hermes_home
        from hermes_cli.config import read_user_config_raw
        config_path = _hermes_home / "config.yaml"
        try:
            # Write-back round-trip: raw read is correct (merged defaults must
            # not be persisted back to the user's file).
            user_config = read_user_config_raw(config_path)
            keys = key_path.split(".")
            current = user_config
            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            atomic_config_write(config_path, user_config)
            return True
        except Exception as e:
            logger.error("Failed to save config key %s: %s", key_path, e)
            return False

    def _apply_reasoning_selection(
        self,
        session_key: str,
        platform_key: str,
        value: str,
        persist_global: bool = False,
    ) -> str:
        """Apply a /reasoning argument (typed or picked) and return the reply.

        Single application path shared by the typed `/reasoning <arg>` branch
        and the interactive choice picker, so both surfaces stay in lockstep
        with the canonical parser.
        """
        from hermes_constants import parse_reasoning_effort

        value = (value or "").strip().lower()

        # Display toggle (per-platform)
        if value in {"show", "on"}:
            self._show_reasoning = True
            self._save_gateway_config_key(
                f"display.platforms.{platform_key}.show_reasoning", True
            )
            return t("gateway.reasoning.display_set_on", platform=platform_key)
        if value in {"hide", "off"}:
            self._show_reasoning = False
            self._save_gateway_config_key(
                f"display.platforms.{platform_key}.show_reasoning", False
            )
            return t("gateway.reasoning.display_set_off", platform=platform_key)

        if value == "reset":
            if persist_global:
                return t("gateway.reasoning.reset_global_unsupported")
            self._set_session_reasoning_override(session_key, None)
            self._reasoning_config = self._load_reasoning_config()
            self._evict_cached_agent(session_key)
            return t("gateway.reasoning.reset_done")

        parsed = parse_reasoning_effort(value)
        if parsed is None:
            return t("gateway.reasoning.unknown_arg", arg=value)

        self._reasoning_config = parsed
        if persist_global:
            if self._save_gateway_config_key("agent.reasoning_effort", value):
                self._set_session_reasoning_override(session_key, None)
                self._evict_cached_agent(session_key)
                return t("gateway.reasoning.set_global", effort=value)
            self._set_session_reasoning_override(session_key, parsed)
            self._evict_cached_agent(session_key)
            return t("gateway.reasoning.set_global_save_failed", effort=value)

        self._set_session_reasoning_override(session_key, parsed)
        self._evict_cached_agent(session_key)
        return t("gateway.reasoning.set_session", effort=value)

    def _reasoning_picker_choices(self, current_effort: str) -> list:
        """Build the choice list for the interactive /reasoning picker."""
        from hermes_constants import VALID_REASONING_EFFORTS

        choices = [
            {
                "value": "none",
                "label": t("gateway.reasoning.choice_none"),
                "is_current": current_effort == "none",
            }
        ]
        for level in VALID_REASONING_EFFORTS:
            choices.append(
                {
                    "value": level,
                    "label": level,
                    "is_current": level == current_effort,
                }
            )
        choices.extend(
            [
                {"value": "reset", "label": t("gateway.reasoning.choice_reset"), "is_current": False},
                {"value": "show", "label": t("gateway.reasoning.choice_show"), "is_current": False},
                {"value": "hide", "label": t("gateway.reasoning.choice_hide"), "is_current": False},
            ]
        )
        return choices

    async def _try_send_choice_picker(
        self,
        event: MessageEvent,
        session_key: str,
        title: str,
        choices: list,
        on_choice_selected,
    ) -> bool:
        """Send an interactive choice picker when the platform supports it.

        Mirrors the `/model` picker gate: the capability is detected on the
        adapter *type* (``send_choice_picker``), and a failed send falls back
        to the text path (returns False) instead of erroring the command.
        """
        adapter = getattr(self, "_adapter_for_source")(event.source)
        has_picker = (
            adapter is not None
            and getattr(type(adapter), "send_choice_picker", None) is not None
        )
        if not has_picker:
            return False
        try:
            metadata = self._thread_metadata_for_source(
                event.source, self._reply_anchor_for_event(event)
            )
            result = await adapter.send_choice_picker(
                chat_id=event.source.chat_id,
                title=title,
                choices=choices,
                session_key=session_key,
                on_choice_selected=on_choice_selected,
                metadata=metadata,
            )
            return bool(getattr(result, "success", False))
        except Exception as e:
            logger.warning("send_choice_picker failed, falling back to text: %s", e)
            return False

    async def _handle_reasoning_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /reasoning command — manage reasoning effort and display toggle.

        Usage:
            /reasoning                       Show current effort level and display state
            /reasoning <level>               Set reasoning effort for this session only
            /reasoning <level> --global      Persist reasoning effort to config.yaml
            /reasoning reset                 Clear this session's reasoning override
            /reasoning show|on               Show model reasoning in responses
            /reasoning hide|off              Hide model reasoning from responses
        """
        from gateway.run import _platform_config_key

        raw_args = event.get_command_args().strip()
        args, persist_global = self._parse_reasoning_command_args(raw_args)
        # Normalize the source (Telegram DM topic recovery) before deriving
        # the override key so storage matches the key the next message turn
        # reads — same fix as /model (#30479).
        _reasoning_source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(_reasoning_source)
        self._show_reasoning = self._load_show_reasoning()
        # Use the session's effective model (session /model override wins over
        # config default) so per-model reasoning_overrides display correctly.
        _session_model = str(
            ((getattr(self, "_session_model_overrides", {}) or {}).get(session_key) or {}).get("model") or ""
        )
        self._reasoning_config = self._resolve_session_reasoning_config(
            source=event.source,
            session_key=session_key,
            model=_session_model,
        )

        if not raw_args:
            # Show current state
            rc = self._reasoning_config
            if rc is None:
                level = t("gateway.reasoning.level_default")
                current_effort = "medium"
            elif rc.get("enabled") is False:
                level = t("gateway.reasoning.level_disabled")
                current_effort = "none"
            else:
                level = rc.get("effort", "medium")
                current_effort = level
            display_state = (
                t("gateway.reasoning.display_on")
                if self._show_reasoning
                else t("gateway.reasoning.display_off")
            )
            has_session_override = session_key in (getattr(self, "_session_reasoning_overrides", {}) or {})
            scope = (
                t("gateway.reasoning.scope_session")
                if has_session_override
                else t("gateway.reasoning.scope_global")
            )

            # Interactive picker on platforms that support it (parity with the
            # /model picker). Falls through to the text status card otherwise.
            _picker_platform_key = _platform_config_key(event.source.platform)

            async def _on_reasoning_choice(_chat_id: str, value: str) -> str:
                return self._apply_reasoning_selection(
                    session_key, _picker_platform_key, value
                )

            picker_sent = await self._try_send_choice_picker(
                event,
                session_key,
                title=t(
                    "gateway.reasoning.picker_title",
                    level=level,
                    scope=scope,
                    display=display_state,
                ),
                choices=self._reasoning_picker_choices(current_effort),
                on_choice_selected=_on_reasoning_choice,
            )
            if picker_sent:
                return None  # Picker sent — adapter handles the response

            return t(
                "gateway.reasoning.status",
                level=level,
                scope=scope,
                display=display_state,
            )

        # Typed argument path — same applier the picker uses.
        platform_key = _platform_config_key(event.source.platform)
        return self._apply_reasoning_selection(
            session_key, platform_key, args, persist_global=persist_global
        )

    async def _handle_fast_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /fast — mirror the CLI Priority Processing toggle in gateway chats.

        Session-scoped by default; ``--global`` persists agent.service_tier
        to config.yaml (parity with /model and /reasoning).
        """
        from gateway.run import _load_gateway_config, _resolve_gateway_model
        from hermes_cli.models import model_supports_fast_mode

        raw_args = event.get_command_args().strip().lower()
        # Reuse the /reasoning arg parser: strips --global (any position),
        # normalizes unicode dashes.
        args, persist_global = self._parse_reasoning_command_args(raw_args)
        session_key = self._session_key_for_source(event.source)
        self._service_tier = self._resolve_session_service_tier(
            session_key=session_key
        )

        user_config = _load_gateway_config()
        model = _resolve_gateway_model(user_config)
        if not model_supports_fast_mode(model):
            return t("gateway.fast.not_supported")

        def _apply_fast_selection(value: str, persist: bool = False) -> str:
            """Apply a /fast argument (typed or picked) and return the reply."""
            if value in {"fast", "on"}:
                tier = "priority"
                saved_value = "fast"
                label = t("gateway.fast.label_fast")
            elif value in {"normal", "off"}:
                tier = None
                saved_value = "normal"
                label = t("gateway.fast.label_normal")
            else:
                return t("gateway.fast.unknown_arg", arg=value)
            self._service_tier = tier
            if persist:
                if self._save_gateway_config_key("agent.service_tier", saved_value):
                    # Global write supersedes any session override.
                    self._set_session_service_tier_override(
                        session_key, None, clear=True
                    )
                    self._evict_cached_agent(session_key)
                    return t("gateway.fast.saved", label=label)
                # Config write failed — fall back to a session override so the
                # user's choice still applies (mirrors /reasoning --global).
                self._set_session_service_tier_override(session_key, tier)
                self._evict_cached_agent(session_key)
                return t("gateway.fast.session_only", label=label)
            self._set_session_service_tier_override(session_key, tier)
            self._evict_cached_agent(session_key)
            return t("gateway.fast.session_only", label=label)

        if not args or args == "status":
            is_fast = self._service_tier == "priority"
            status = t("gateway.fast.status_fast") if is_fast else t("gateway.fast.status_normal")

            async def _on_fast_choice(_chat_id: str, value: str) -> str:
                return _apply_fast_selection(value, persist=persist_global)

            picker_sent = await self._try_send_choice_picker(
                event,
                session_key,
                title=t("gateway.fast.picker_title", mode=status),
                choices=[
                    {
                        "value": "fast",
                        "label": t("gateway.fast.choice_fast"),
                        "is_current": is_fast,
                    },
                    {
                        "value": "normal",
                        "label": t("gateway.fast.choice_normal"),
                        "is_current": not is_fast,
                    },
                ],
                on_choice_selected=_on_fast_choice,
            )
            if picker_sent:
                return None  # Picker sent — adapter handles the response

            return t("gateway.fast.status", mode=status)

        return _apply_fast_selection(args, persist=persist_global)
