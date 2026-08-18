"""Actor and serving-profile policy for typed commands and native pickers."""
import logging
from pathlib import Path
from typing import Optional
from gateway.config import GatewayConfig
from gateway.session import SessionSource
from gateway.message_actor import copy_session_source_with
from hermes_constants import get_hermes_home

logger = logging.getLogger("gateway.run")

class GatewaySlashPolicyMixin:
    def _register_profile_gateway_config(
        self, profile_name: str, gateway_config: GatewayConfig
    ) -> None:
        """Remember one served profile's effective slash-policy config."""
        registry = getattr(self, "_profile_gateway_configs", None)
        if isinstance(registry, dict):
            registry[profile_name] = gateway_config
        else:
            self._profile_gateway_configs = {profile_name: gateway_config}


    def _make_profile_slash_access_check(
        self,
        profile_name: str,
        *,
        gateway_config: Optional[GatewayConfig] = None,
    ):
        """Return a slash checker scoped to one multiplexed profile runtime."""
        from gateway.run import _profile_runtime_scope
        from hermes_cli.profiles import get_profile_dir

        try:
            profile_home = get_profile_dir(profile_name)
        except Exception:
            profile_home = None

        def _check(source: SessionSource, canonical_cmd: str) -> Optional[str]:
            if profile_home is None:
                return f"⛔ /{canonical_cmd} is unavailable without profile policy context."
            scoped_source = source
            if not getattr(source, "profile", None):
                scoped_source = copy_session_source_with(source, profile=profile_name)
            with _profile_runtime_scope(profile_home):
                return self._check_slash_access(
                    scoped_source,
                    canonical_cmd,
                    gateway_config=gateway_config,
                )

        return _check


    def _make_default_profile_slash_access_check(self):
        """Scope primary-adapter callback policy to its routed serving profile."""
        from gateway.run import _profile_runtime_scope
        default_home = Path(get_hermes_home())

        def _check(source: SessionSource, canonical_cmd: str) -> Optional[str]:
            from gateway.profile_routing import ProfileRouteRejected

            scoped_source = source
            try:
                profile_name = str(getattr(source, "profile", "") or "").strip()
                if not profile_name:
                    profile_name = self._profile_name_for_source(source) or ""
                    if profile_name:
                        scoped_source = copy_session_source_with(
                            source, profile=profile_name
                        )
                profile_home = (
                    self._resolve_profile_home_for_source(scoped_source)
                    if profile_name
                    else default_home
                )
            except ProfileRouteRejected:
                return (
                    f"⛔ /{canonical_cmd} is unavailable without profile "
                    "policy context."
                )
            except Exception:
                logger.warning(
                    "Could not resolve serving profile for /%s callback; "
                    "failing closed",
                    canonical_cmd,
                    exc_info=True,
                )
                return (
                    f"⛔ /{canonical_cmd} is unavailable without profile "
                    "policy context."
                )
            with _profile_runtime_scope(profile_home):
                return self._check_slash_access(scoped_source, canonical_cmd)

        return _check


    def _primary_slash_access_check(self):
        """Return the slash-policy callback for a primary adapter instance."""
        if getattr(self.config, "multiplex_profiles", False):
            return self._make_default_profile_slash_access_check()
        return self._check_slash_access


    def _effective_gateway_config_for_source(
        self, source: Optional[SessionSource]
    ) -> tuple:
        """Return ``(gateway_config, resolved)`` for authorization decisions.

        The GatewayConfig whose policy must govern *source*: under
        ``gateway.multiplex_profiles`` that is the named profile actually
        serving the source (``source.profile``, stamped by per-credential
        adapter handlers or profile-route matching) — never blindly
        ``self.config``, which is the multiplexer's OWN profile config.
        Native picker callbacks already capture the effective profile
        config; this resolver gives typed command paths the same property.

        Returns ``resolved=False`` when a named profile is requested but no
        trustworthy policy context exists for it. Callers MUST fail closed
        in that case — falling back to ``self.config`` would apply another
        profile's allow_admin_from / user_allowed_commands lists (the exact
        defect this closes).

        When multiplexing is off, returns ``(self.config, True)`` so every
        caller keeps byte-identical single-profile behavior.
        """
        from gateway.run import _profile_runtime_scope
        base_cfg = getattr(self, "config", None)
        if not getattr(base_cfg, "multiplex_profiles", False):
            return base_cfg, True
        profile_name = ""
        if source is not None:
            profile_name = str(getattr(source, "profile", "") or "").strip()
        if not profile_name:
            # Unstamped ingress is the multiplexer's own lane: self.config IS
            # this profile's config. (Do NOT branch on
            # ``_active_profile_name()`` here — it is derived from
            # ``get_hermes_home()``, which the per-message runtime scope
            # redirects to the SECONDARY profile's home while this resolver
            # runs inside it.)
            return base_cfg, True
        registry = getattr(self, "_profile_gateway_configs", None)
        cfg = registry.get(profile_name) if isinstance(registry, dict) else None
        if cfg is None and isinstance(registry, dict):
            try:
                from hermes_cli.profiles import normalize_profile_name

                cfg = registry.get(normalize_profile_name(profile_name))
            except Exception:
                cfg = None
        if cfg is not None:
            return cfg, True
        # Routed-but-never-connected profile: resolve its own config once,
        # mirroring how secondary adapters are built at startup (scoped load).
        try:
            from gateway.config import load_gateway_config as _load_profile_cfg
            from hermes_cli.profiles import (
                get_profile_dir,
                normalize_profile_name,
                profile_exists,
            )

            norm_name = normalize_profile_name(profile_name)
            if profile_exists(norm_name):
                profile_home = get_profile_dir(norm_name)
                with _profile_runtime_scope(profile_home):
                    loaded = _load_profile_cfg()
                if isinstance(registry, dict):
                    registry[norm_name] = loaded
                else:
                    self._profile_gateway_configs = {norm_name: loaded}
                return loaded, True
        except Exception as exc:
            logger.warning(
                "Could not resolve GatewayConfig for multiplexed profile %r; "
                "failing closed for authorization-sensitive paths: %s",
                profile_name,
                exc,
            )
        return None, False


    def _quick_commands_for_source(self, source: Optional[SessionSource]) -> dict:
        """User-defined quick commands of the profile serving *source*.

        Quick commands are slash capabilities — type:exec entries run shell
        commands in the gateway process — so the lookup must use the
        effective named-profile config, not the multiplexer's dict.
        """
        cfg, resolved = self._effective_gateway_config_for_source(source)
        if not resolved or cfg is None:
            return {}
        if isinstance(cfg, dict):
            quick_commands = cfg.get("quick_commands", {}) or {}
        else:
            quick_commands = getattr(cfg, "quick_commands", {}) or {}
        return quick_commands if isinstance(quick_commands, dict) else {}


    def _check_slash_access(
        self,
        source: SessionSource,
        canonical_cmd: str,
        *,
        gateway_config: Optional[GatewayConfig] = None,
    ) -> Optional[str]:
        """Return a denial message if ``source`` cannot run ``canonical_cmd``,
        else None. Used by both the cold and running-agent dispatch paths
        in ``_handle_message`` so admin/user gating can't be bypassed by
        an in-flight agent.

        Backward-compat semantics live in
        :func:`gateway.slash_access.policy_for_source` — when the operator
        hasn't set ``allow_admin_from`` for the scope, the policy returns
        ``enabled=False`` and this method always returns None.
        """
        from gateway.slash_access import (
            IDENTITY_FREE_FLOOR_COMMANDS,
            policy_for_source as _policy_for_source,
        )

        if not canonical_cmd:
            return None
        # Shared routing sources intentionally omit participant identity. The
        # dispatch path must rehydrate the actual actor from MessageEvent; if it
        # cannot, fail closed before any stateful command runs even when slash
        # policy is unconfigured. Keep only read-only diagnostics available.
        if (
            not getattr(source, "user_id", None)
            and canonical_cmd not in IDENTITY_FREE_FLOOR_COMMANDS
        ):
            logger.info(
                "Slash command /%s denied for %s (missing actor identity)",
                canonical_cmd,
                source.platform.value if source and source.platform else "?",
            )
            return f"⛔ /{canonical_cmd} requires an identifiable user."
        if gateway_config is not None:
            policy_cfg = gateway_config
        else:
            # Typed dispatch paths (busy / cold / quick-command gates) don't
            # know which multiplexed profile serves this source. Resolve it —
            # and fail closed rather than silently gating by the multiplexer's
            # own policy (same contract as the native picker callbacks). A
            # resolved-but-absent config (config=None bare runners) keeps the
            # legacy disabled-policy path below.
            policy_cfg, _policy_resolved = (
                self._effective_gateway_config_for_source(source)
            )
            if not _policy_resolved:
                logger.info(
                    "Slash command /%s denied for %s (no policy context for "
                    "profile %r)",
                    canonical_cmd,
                    source.platform.value if source.platform else "?",
                    getattr(source, "profile", "") or "?",
                )
                return (
                    f"⛔ /{canonical_cmd} is unavailable without profile "
                    "policy context."
                )
        policy = _policy_for_source(policy_cfg, source)
        if not policy.enabled or policy.can_run(source.user_id, canonical_cmd):
            return None
        logger.info(
            "Slash command /%s denied for %s:%s (not admin, not in user_allowed_commands)",
            canonical_cmd,
            source.platform.value if source.platform else "?",
            source.user_id,
        )
        allowed_preview = sorted(policy.user_allowed_commands)
        if allowed_preview:
            suffix = (
                "You can run: "
                + ", ".join(f"/{c}" for c in allowed_preview[:12])
                + ("…" if len(allowed_preview) > 12 else "")
                + ". Use /whoami for the full list."
            )
        else:
            suffix = (
                "No slash commands are enabled for non-admins on this "
                "platform. Ask an admin to add you to allow_admin_from "
                "or to set user_allowed_commands."
            )
        return f"⛔ /{canonical_cmd} is admin-only here. {suffix}"
