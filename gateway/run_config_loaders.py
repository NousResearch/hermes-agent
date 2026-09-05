"""Config/env loaders (busy modes, reasoning, service tier, timeouts, fallback) for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from gateway.config import Platform
from gateway.restart import (
    DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT, DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT,
    DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT, DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT, parse_cron_drain_timeout,
    parse_restart_after_turn_timeout, parse_restart_drain_timeout,
    parse_signal_interrupt_grace_timeout,
)
from gateway.session import SessionSource
from gateway.session_state import SERVICE_TIER_UNSET as _SERVICE_TIER_UNSET
from hermes_cli.config import cfg_get, resolve_ephemeral_system_prompt_from_config
from hermes_cli.fallback_config import get_fallback_chain
from utils import is_truthy_value

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401
    from gateway.run_turn_runner import TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

_BUSY_INPUT_MODES = {"interrupt", "queue", "steer"}


class GatewayConfigLoadersMixin:
    """Config/env loaders (busy modes, reasoning, service tier, timeouts, fallback) for GatewayRunner."""

    @staticmethod
    def _cfg_str(section: str, key: str) -> str:
        """``<section>.<key>`` from the gateway runtime config as a stripped string ("" when unset)."""
        from gateway.run import _load_gateway_runtime_config
        return str(cfg_get(_load_gateway_runtime_config(), section, key, default="") or "").strip()

    @classmethod
    def _env_or_cfg_str(cls, env_var: str, section: str, key: str) -> str:
        """Non-empty stripped env var, else :meth:`_cfg_str`."""
        return os.getenv(env_var, "").strip() or cls._cfg_str(section, key)

    @staticmethod
    def _load_prefill_messages() -> List[Dict[str, Any]]:
        """Load ephemeral prefill messages from config or env var.

        HERMES_PREFILL_MESSAGES_FILE env wins, then top-level prefill_messages_file in config.yaml,
        then legacy agent.prefill_messages_file. Relative paths resolve from ~/.hermes/.
        """
        from gateway.run import _hermes_home, _load_gateway_runtime_config
        file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
        if not file_path:
            cfg = _load_gateway_runtime_config()
            file_path = str(
                cfg.get("prefill_messages_file", "") or cfg_get(cfg, "agent", "prefill_messages_file", default="") or ""
            )
        if not file_path:
            return []
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = _hermes_home / path
        if not path.exists():
            logger.warning("Prefill messages file not found: %s", path)
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning("Prefill messages file must contain a JSON array: %s", path)
                return []
            return data
        except Exception as e:
            logger.warning("Failed to load prefill messages from %s: %s", path, e)
            return []

    @staticmethod
    def _load_ephemeral_system_prompt() -> str:
        """HERMES_EPHEMERAL_SYSTEM_PROMPT env first, then ``display.personality`` / ``agent.system_prompt``."""
        from gateway.run import _load_gateway_runtime_config
        prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
        if prompt:
            return prompt
        return resolve_ephemeral_system_prompt_from_config(_load_gateway_runtime_config())

    def _channel_override(self, platform: Platform, chat_id: str, thread_id, parent_id):
        """``channel_overrides`` entry for this channel/thread, or None (also when no config is bound)."""
        from gateway.run import _get_channel_override
        config = getattr(self, "config", None)
        if not config:
            return None
        return _get_channel_override(config, platform, chat_id, thread_id=thread_id, parent_id=parent_id)

    def _resolve_model_for_channel(
        self, platform: Platform, chat_id: str, *, user_config: Optional[dict] = None,
        thread_id: Optional[str] = None, parent_id: Optional[str] = None,
    ) -> str:
        """Resolve model for this channel: channel_overrides else global default.

        Precedence lives in :func:`hermes_cli.model_switch.resolve_effective_model` (shared with the
        API server so the surfaces cannot diverge). No session tier here: session /model overrides
        are applied later by ``_apply_session_model_override``.
        """
        from gateway.run import _resolve_gateway_model
        from hermes_cli.model_switch import resolve_effective_model
        return resolve_effective_model(
            None,  # session tier applied downstream (_apply_session_model_override)
            self._channel_override(platform, chat_id, thread_id, parent_id),
            _resolve_gateway_model(user_config),
        )

    def _get_system_prompt_for_channel(
        self, platform: Platform, chat_id: str, *, thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Ephemeral system prompt for this channel/thread.

        ``channel_overrides`` when set, else the gateway prompt resolved from the CURRENT profile's
        config on every call (callers run inside ``_profile_runtime_scope``, so routed multiplex
        profiles get their own personality/system_prompt and ``/personality`` edits apply next turn).
        Legacy ``channel_prompts`` are applied separately via ``event.channel_prompt`` in ``run_sync``.

        Callers run inside ``_profile_runtime_scope`` (``run_sync`` under ``_run_agent``), so a routed
        multiplex profile gets its own ``display.personality`` / ``agent.system_prompt`` instead of a
        boot-time snapshot of the launch profile's (#89161); ``/personality`` edits take effect on the next
        turn for the same reason.
        """
        override = self._channel_override(platform, chat_id, thread_id, parent_id)
        if override and override.system_prompt:
            return (override.system_prompt or "").strip()
        return self._load_ephemeral_system_prompt()

    @staticmethod
    def _load_reasoning_config(model: str = "") -> dict | None:
        """Reasoning effort from config.yaml via :func:`hermes_constants.resolve_reasoning_config`.

        Per-model override > global ``agent.reasoning_effort``; YAML False = disabled. Empty
        ``model`` uses ``model.default``.

        Closes #21256.
        """
        from gateway.run import _load_gateway_runtime_config
        from hermes_constants import resolve_reasoning_config
        return resolve_reasoning_config(_load_gateway_runtime_config(), model)

    @staticmethod
    def _parse_reasoning_command_args(raw_args: str) -> tuple[str, bool]:
        """Parse `/reasoning` args into `(value, persist_global)`; `--global` anywhere persists to config."""
        import shlex
        text = str(raw_args or "").strip().replace("—", "--")
        if not text:
            return "", False
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()
        value = " ".join(token for token in tokens if token != "--global")
        return value.strip().lower(), "--global" in tokens

    def _resolve_session_reasoning_config(
        self, *, source: Optional[SessionSource] = None, session_key: Optional[str] = None,
        model: str = "",
    ) -> dict | None:
        """Session ``/reasoning --session`` > channel override > per-model
        ``agent.reasoning_overrides`` > global.

        ``model`` must be the session's *effective* model (session ``/model`` override included);
        empty uses ``model.default``.
        """
        resolved_session_key = self._resolve_session_key_or_none(source, session_key)
        if resolved_session_key:
            _r_state = self._peek_session_state(resolved_session_key)
            if _r_state is not None and _r_state.conversation.reasoning_override is not None:
                return _r_state.conversation.reasoning_override
        if source is not None:
            try:
                override = self._channel_override(
                    source.platform,
                    str(source.chat_id) if source.chat_id else "",
                    str(source.thread_id) if getattr(source, "thread_id", None) else None,
                    str(source.parent_chat_id) if getattr(source, "parent_chat_id", None) else None,
                )
                channel_effort = getattr(override, "reasoning_effort", None)
                if channel_effort is not None:
                    from hermes_constants import parse_reasoning_effort

                    parsed = parse_reasoning_effort(channel_effort)
                    if parsed is not None:
                        return parsed
                    if str(channel_effort).strip():
                        logger.warning(
                            "Unknown channel reasoning_effort '%s', using global default",
                            channel_effort,
                        )
            except Exception:
                logger.debug("Failed to resolve channel reasoning override", exc_info=True)
        return self._load_reasoning_config(model)

    def _set_session_reasoning_override(self, session_key: str, reasoning_config: Optional[dict]) -> None:
        """Set or clear the session-scoped reasoning override."""
        if not session_key:
            return
        # Per-session field write: a lazy ``_session_reasoning_overrides = {}`` init replaced the
        # WHOLE dict, racing concurrent sessions; a SessionState field reset cannot cross sessions.
        self._session_state(session_key).conversation.reasoning_override = (
            None if reasoning_config is None else dict(reasoning_config)
        )

    def _resolve_session_service_tier(self, source=None, session_key: Optional[str] = None) -> Optional[str]:
        """Effective service tier: a session-scoped /fast override beats the config default.

        The override stores "priority" or None (explicit normal), so presence — not truthiness — decides.
        """
        resolved_session_key = self._resolve_session_key_or_none(source, session_key)
        if resolved_session_key:
            _t_state = self._peek_session_state(resolved_session_key)
            if _t_state is not None and _t_state.conversation.service_tier_override is not _SERVICE_TIER_UNSET:
                return _t_state.conversation.service_tier_override
        return self._load_service_tier()

    def _set_session_service_tier_override(self, session_key: str, service_tier, clear: bool = False) -> None:
        """Set ("priority" / None = explicit normal) or ``clear`` the session-scoped /fast override."""
        if not session_key:
            return
        # Presence-sensitive: "priority" or None (explicit normal) both count as an override; the
        # sentinel means "no override". Per-session field write: a lazy dict replace races sessions.
        self._session_state(session_key).conversation.service_tier_override = (
            _SERVICE_TIER_UNSET if clear else service_tier
        )

    @classmethod
    def _load_service_tier(cls) -> str | None:
        """``agent.service_tier``: fast/priority/on => "priority"; normal/off => None; None when unset/unknown."""
        raw = cls._cfg_str("agent", "service_tier")
        value = raw.lower()
        if not value or value in {"normal", "default", "standard", "off", "none"}:
            return None
        if value in {"fast", "priority", "on"}:
            return "priority"
        if value in {"auto", "cold"}:
            return value
        logger.warning("Unknown service_tier '%s', ignoring", raw)
        return None

    @staticmethod
    def _load_show_reasoning() -> bool:
        """``display.show_reasoning`` toggle."""
        from gateway.run import _load_gateway_runtime_config
        return is_truthy_value(cfg_get(_load_gateway_runtime_config(), "display", "show_reasoning"), default=False)

    @classmethod
    def _load_busy_input_mode(cls) -> str:
        """Gateway drain-time busy-input behavior from env/config (default ``interrupt``)."""
        mode = cls._env_or_cfg_str("HERMES_GATEWAY_BUSY_INPUT_MODE", "display", "busy_input_mode").lower()
        return mode if mode in {"queue", "steer"} else "interrupt"

    @classmethod
    def _load_busy_text_mode(cls) -> str:
        """Normal busy TEXT follow-up behavior.

        ``busy_input_mode`` is the source of truth (default ``interrupt``); legacy ``busy_text_mode``
        is honored only when explicitly set so existing queue setups keep working.
        """
        from gateway.run import GatewayRunner
        legacy = cls._env_or_cfg_str("HERMES_GATEWAY_BUSY_TEXT_MODE", "display", "busy_text_mode").lower()
        if legacy in {"interrupt", "queue"}:
            return legacy
        return "queue" if GatewayRunner._load_busy_input_mode() == "queue" else "interrupt"

    @staticmethod
    def _busy_modes_from_config(config: dict, *, fallback_input: str, fallback_text: str) -> tuple[str, str]:
        """Resolve one profile's busy modes without consulting process env."""
        raw_input = str(cfg_get(config, "display", "busy_input_mode", default="") or "").strip().lower()
        input_mode = raw_input if raw_input in _BUSY_INPUT_MODES else fallback_input
        raw_text = str(cfg_get(config, "display", "busy_text_mode", default="") or "").strip().lower()
        if raw_text in {"interrupt", "queue"}:
            text_mode = raw_text
        elif raw_input in _BUSY_INPUT_MODES:
            text_mode = "queue" if input_mode == "queue" else "interrupt"
        else:
            text_mode = fallback_text
        return input_mode, text_mode

    def _snapshot_profile_busy_modes(self, profile_name: str, config: dict) -> None:
        """Cache a routed profile's busy policy for this gateway lifetime."""
        input_mode, text_mode = self._busy_modes_from_config(
            config, fallback_input=getattr(self, "_busy_input_mode", "interrupt"),
            fallback_text=getattr(self, "_busy_text_mode", "interrupt"),
        )
        self.__dict__.setdefault("_busy_input_modes_by_profile", {})[profile_name] = input_mode
        self.__dict__.setdefault("_busy_text_modes_by_profile", {})[profile_name] = text_mode

    def _busy_profile_name_for_source(self, source: SessionSource) -> Optional[str]:
        """Return the routed profile whose busy policy applies, if any."""
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return None
        name = str(getattr(source, "profile", "") or "").strip()
        if not name:
            try:
                name = str(self._profile_name_for_source(source) or "").strip()
            except Exception:
                name = ""
        return name or None

    def _effective_busy_mode(self, source: SessionSource, attr: str) -> str:
        """Busy mode from the routed profile snapshot (``attr``: ``_busy_input_mode`` / ``_busy_text_mode``)."""
        fallback = getattr(self, attr, "interrupt")
        profile_name = self._busy_profile_name_for_source(source)
        if not profile_name:
            return fallback
        modes = getattr(self, attr + "s_by_profile", None)
        return modes.get(profile_name, fallback) if isinstance(modes, dict) else fallback

    def _effective_busy_input_mode(self, source: SessionSource) -> str:
        """Resolve busy input mode from the routed profile startup snapshot."""
        return self._effective_busy_mode(source, "_busy_input_mode")

    def _effective_busy_text_mode(self, source: SessionSource) -> str:
        """Resolve legacy busy text mode from the routed profile snapshot."""
        return self._effective_busy_mode(source, "_busy_text_mode")

    @staticmethod
    def _warn_unparsable_timeout(cfg_key: str, raw: object, default: float) -> None:
        """Warn when a supplied timeout value is not a number (the parser already fell back to ``default``)."""
        try:
            float(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid %s '%s', using default %.0fs", cfg_key, raw, default)

    @classmethod
    def _load_restart_drain_timeout(cls) -> float:
        """Graceful gateway restart/stop drain timeout in seconds."""
        raw = cls._env_or_cfg_str("HERMES_RESTART_DRAIN_TIMEOUT", "agent", "restart_drain_timeout")
        value = parse_restart_drain_timeout(raw)
        if raw and value == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT:
            cls._warn_unparsable_timeout("restart_drain_timeout", raw, DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT)
        return value

    @classmethod
    def _load_env_or_agent_cfg_timeout(cls, env_var: str, cfg_key: str, parse, default: float) -> float:
        """Env var (non-empty) else ``agent.<cfg_key>``; warn once when a supplied value fails to parse.

        ``0`` is a valid value; the parser falls back to ``default`` on garbage."""
        from gateway.run import _load_gateway_runtime_config
        env_raw = os.getenv(env_var)
        if env_raw is not None and str(env_raw).strip() != "":
            raw: object = env_raw
        else:
            raw = cfg_get(_load_gateway_runtime_config(), "agent", cfg_key, default=None)
        value = parse(raw)
        if raw is not None and str(raw).strip() != "":
            cls._warn_unparsable_timeout(cfg_key, raw, default)
        return value

    @classmethod
    def _load_restart_after_turn_timeout(cls) -> float:
        """In-band restart wait-for-idle timeout in seconds."""
        return cls._load_env_or_agent_cfg_timeout(
            "HERMES_RESTART_AFTER_TURN_TIMEOUT", "restart_after_turn_timeout",
            parse_restart_after_turn_timeout, DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
        )

    @classmethod
    def _load_cron_drain_timeout(cls) -> float:
        """The cron-only floor under the stop()/drain wait.

        See #82161.
        """
        return cls._load_env_or_agent_cfg_timeout(
            "HERMES_CRON_DRAIN_TIMEOUT", "cron_drain_timeout",
            parse_cron_drain_timeout, DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
        )

    @classmethod
    def _load_signal_interrupt_grace_timeout(cls) -> float:
        """``gateway.signal_interrupt_grace_timeout``: unexpected-signal post-interrupt grace in seconds."""
        from gateway.run import _load_gateway_runtime_config
        raw = cfg_get(_load_gateway_runtime_config(), "gateway", "signal_interrupt_grace_timeout", default=None)
        value = parse_signal_interrupt_grace_timeout(raw)
        if raw is not None and raw != "":
            cls._warn_unparsable_timeout(
                "signal_interrupt_grace_timeout", raw, DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
            )
        return value

    def _post_interrupt_grace_timeout(self) -> float:
        """Grace before teardown after forcibly interrupting agents (longer on an unexpected signal)."""
        if getattr(self, "_signal_initiated_shutdown", False) and not getattr(self, "_restart_requested", False):
            grace = getattr(self, "_signal_interrupt_grace_timeout", DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT)
            return max(0.0, float(grace))
        return DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT

    @staticmethod
    def _load_background_notifications_mode() -> str:
        """Background process notification mode from env/config (default ``concise``)."""
        from gateway.run import _load_gateway_runtime_config
        mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
        if not mode:
            raw = cfg_get(_load_gateway_runtime_config(), "display", "background_process_notifications")
            if raw is False:
                mode = "off"
            elif raw not in {None, ""}:
                mode = str(raw)
        mode = (mode or "concise").strip().lower()
        if mode not in {"concise", "all", "result", "error", "off"}:
            logger.warning("Unknown background_process_notifications '%s', defaulting to 'concise'", mode)
            return "concise"
        return mode

    @staticmethod
    def _load_provider_routing() -> dict:
        """OpenRouter provider routing preferences (canonical fail-open loader: managed overlay + ${VAR})."""
        from gateway.run import _load_gateway_runtime_config
        try:
            return _load_gateway_runtime_config().get("provider_routing", {}) or {}
        except Exception:
            return {}

    @staticmethod
    def _load_fallback_model() -> list | None:
        """Fallback chain: ``fallback_providers`` (kept first) merged with legacy ``fallback_model``."""
        from gateway.run import _load_gateway_runtime_config
        try:
            # Canonical gateway loader (fail-open): managed overlay + ${VAR} expansion apply here too.
            return get_fallback_chain(_load_gateway_runtime_config()) or None
        except Exception:
            return None

    def _refresh_fallback_model(self) -> list | None:
        """Re-read fallback_providers from disk for the next agent create/reuse.

        Lets a chain edited after startup reach messaging sessions (cron already re-reads per job).
        A TRANSIENT read/parse failure (user mid-edit, non-atomic write) keeps the last known-good
        chain; only a successful read that genuinely lacks the key clears it.

        Cron already does this per job via ``get_fallback_chain``; the gateway previously froze
        ``self._fallback_model`` at process start, so a chain configured (or changed) after ``hermes
        gateway`` was running never reached messaging sessions even though the same process's cron jobs fell
        back correctly. Fixes #60955.
        """
        from gateway.run import _hermes_home
        try:
            from hermes_cli.config import read_user_config_raw
            cfg_path = _hermes_home / "config.yaml"
            if not cfg_path.exists():
                self._fallback_model = None
                return self._fallback_model
            # Raw primitive (raises on parse failure) is required here: the canonical fail-open
            # loader would return {} on a torn mid-edit write and WIPE the last known-good chain.
            # The overlay/expansion below fixes the managed-scope/${VAR} drift without losing that.
            cfg = read_user_config_raw(cfg_path)
            with suppress(Exception):
                from hermes_cli import managed_scope
                cfg = managed_scope.apply_managed_overlay(cfg)
            with suppress(Exception):
                from hermes_cli.config import _expand_env_vars
                expanded = _expand_env_vars(cfg)
                if isinstance(expanded, dict):
                    cfg = expanded
        except Exception:
            logger.debug("fallback_providers refresh: config.yaml read failed; keeping last known-good chain", exc_info=True)
            return self._fallback_model
        self._fallback_model = get_fallback_chain(cfg) or None
        return self._fallback_model

    @staticmethod
    def _apply_fallback_chain_to_agent(agent: Any, chain: list | None) -> None:
        """Keep a cached agent's fallback chain aligned with current config.

        Skips the rewrite while a cooldown holds the agent on an activated fallback provider
        (``restore_primary_runtime`` owns that lifecycle); otherwise replaces the chain so
        mid-uptime ``fallback_providers`` edits apply without a restart.

        When primary is active (or cooldown expired), replace the chain so mid-uptime ``fallback_providers``
        edits take effect without requiring a gateway restart (#60955).
        """
        if agent is None:
            return
        new_chain = list(chain or [])
        rate_limited_until = getattr(agent, "_rate_limited_until", 0) or 0
        if getattr(agent, "_fallback_activated", False) and rate_limited_until > time.monotonic():
            return
        old_chain = list(getattr(agent, "_fallback_chain", []) or [])
        agent._fallback_chain = new_chain
        agent._fallback_model = new_chain[0] if new_chain else None
        if not getattr(agent, "_fallback_activated", False):
            agent._fallback_index = 0
        # A config edit means the user changed something — drop the session-scoped unavailability
        # memo so re-configured entries (e.g. credentials added mid-uptime) get retried. Only on real
        # content change, so the per-message no-op refresh keeps the memo's rate-limiting benefit.
        # See #60955.
        if new_chain != old_chain:
            unavailable = getattr(agent, "_unavailable_fallback_keys", None)
            if unavailable:
                unavailable.clear()
