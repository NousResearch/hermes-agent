"""Capability-gated platform action facade for plugins (#64176, action half).

Every verb returns a structured result dict — ``{"ok": True, ...}`` on success, ``{"ok": False,
"error": <code>, "detail": <str>}`` on failure — and never raises into hook dispatch.
"""

from __future__ import annotations

import logging
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

ACTIONS_CONTRACT_VERSION = 1

CAPABILITY_ID = "gateway.platform_actions"


def _err(code: str, detail: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "error": code}
    if detail:
        result["detail"] = detail
    return result


def _ok(**fields: Any) -> Dict[str, Any]:
    return {"ok": True, **fields}


# -- per-platform verb implementations (adapter, *args) -> result ------------


async def _telegram_add_reaction(adapter, chat_id, message_id, emoji):
    if await adapter._set_reaction(chat_id, message_id, emoji):
        return _ok(action="add_reaction")
    return _err("action_failed", "telegram set_message_reaction failed")


async def _discord_add_reaction(adapter: Any, chat_id: str, message_id: str, emoji: str) -> Dict[str, Any]:
    client = getattr(adapter, "_client", None)
    if client is None:
        return _err("adapter_disconnected", "discord client unavailable")
    try:
        channel_id = int(str(chat_id))
        msg_id = int(str(message_id))
    except (TypeError, ValueError):
        return _err("invalid_argument", "discord ids must be numeric")
    channel = client.get_channel(channel_id)
    if channel is None:
        channel = await client.fetch_channel(channel_id)
    message = await channel.fetch_message(msg_id)
    await message.add_reaction(emoji)
    return _ok(action="add_reaction")


async def _telegram_set_thread_title(adapter, chat_id, thread_id, title):
    await adapter.rename_dm_topic(chat_id, int(thread_id), title)
    return _ok(action="set_thread_title")


async def _discord_set_thread_title(adapter, chat_id, thread_id, title):
    if await adapter.rename_thread(thread_id, title):
        return _ok(action="set_thread_title")
    return _err("action_failed", "discord thread rename failed")


_VERBS = {
    "add_reaction": {"telegram": _telegram_add_reaction, "discord": _discord_add_reaction},
    "set_thread_title": {"telegram": _telegram_set_thread_title, "discord": _discord_set_thread_title},
}


class PlatformActions:
    """Per-plugin facade over the live gateway adapter registry.

    Instances are cheap and hold only the owning plugin id; the gateway runner and adapters are
    resolved at call time so a facade created before the gateway starts (plugin ``register()`` runs
    first) still works once adapters connect.
    """

    def __init__(self, plugin_id: str):
        self._plugin_id = plugin_id

    def for_source(
        self,
        *,
        platform: str,
        channel_id: str,
        thread_id: Optional[str],
        chat_type: str,
        routed_profile: str,
        source_identity_candidates: Tuple[str, ...],
        transport_profile: Optional[str] = None,
    ) -> "SourceBoundPlatformActions":
        """Bind the mutation-only facade to one immutable gateway source."""
        return SourceBoundPlatformActions(
            owner=self,
            platform=str(platform or ""),
            channel_id=str(channel_id or ""),
            thread_id=str(thread_id) if thread_id is not None else None,
            chat_type=str(chat_type or ""),
            routed_profile=str(routed_profile or ""),
            source_identity_candidates=tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in source_identity_candidates
                    if str(value).strip()
                )
            ),
            transport_profile=(
                str(transport_profile).strip()
                if transport_profile is not None
                else None
            ),
        )

    # -- shared plumbing ----------------------------------------------------

    def _capability_granted(self) -> bool:
        try:
            from hermes_cli.plugin_capabilities import plugin_capability_granted

            return plugin_capability_granted(self._plugin_id, CAPABILITY_ID)
        except Exception:
            # Ground rule: failure to read consent state = not granted.
            logger.debug("platform_actions capability check failed for %s", self._plugin_id, exc_info=True)
            return False

    def _resolve_adapter(self, platform: str):
        """Return ``(adapter, error_dict)``; exactly one is non-None."""
        try:
            from gateway.run import _gateway_runner_ref

            runner = _gateway_runner_ref()
        except Exception:
            runner = None
        if runner is None:
            return None, _err("gateway_unavailable", "no gateway runner is active in this process")
        try:
            from gateway.config import Platform

            platform_enum = Platform(str(platform).strip().lower())
        except Exception:
            return None, _err("unknown_platform", f"unknown platform {platform!r}")
        # Multiplex/Team-Gateway: a secondary profile's adapters live in runner._profile_adapters,
        # not runner.adapters. Every adapter-resolution path goes through the same profile-aware,
        # fail-closed lookup so a plugin scoped to one profile can never act through another
        # profile's bot identity. The bare default-profile lookup is only for a runner predating
        # _authorization_adapter (defensive, not expected).
        resolve_fn = getattr(runner, "_authorization_adapter", None)
        if callable(resolve_fn):
            try:
                from hermes_cli.profiles import get_active_profile_name

                profile_name = get_active_profile_name()
            except Exception:
                # Fail closed: an unresolvable profile must not degrade to the default profile's bot.
                logger.debug(
                    "platform_actions: profile resolution failed for %s",
                    self._plugin_id, exc_info=True,
                )
                return None, _err(
                    "adapter_not_registered",
                    f"no {platform_enum.value} adapter is registered "
                    "(active profile could not be resolved)",
                )
            adapter = resolve_fn(platform_enum, profile_name)
        else:
            adapter = getattr(runner, "adapters", {}).get(platform_enum)
        if adapter is None:
            return None, _err("adapter_not_registered", f"no {platform_enum.value} adapter is registered")
        try:
            connected = bool(adapter.is_connected)
        except Exception:
            connected = False
        if not connected:
            return None, _err("adapter_disconnected", f"the {platform_enum.value} adapter is not connected")
        return adapter, None

    def _gate(self, platform: str, **required: Any):
        """Run the shared gate chain. Returns ``(adapter, error_dict)``."""
        if not self._capability_granted():
            return None, _err(
                "capability_not_granted",
                f"plugin {self._plugin_id!r} lacks the {CAPABILITY_ID!r} "
                "capability (grant via consent flow or "
                f"plugins.entries.{self._plugin_id}.allow_platform_actions)",
            )
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                return None, _err("invalid_argument", f"{name} must be a non-empty string")
        return self._resolve_adapter(platform)

    async def _run(self, verb: str, platform: str, *args: str, **required: Any) -> Dict[str, Any]:
        """Gate, dispatch *verb* to the adapter's platform implementation, audit, return."""
        adapter, error = self._gate(platform, **required)
        if error is None and adapter is not None:
            try:
                impl = _VERBS[verb].get(getattr(adapter.platform, "value", None))
                if impl is None:
                    result = _err("unsupported_platform_action", f"{verb} is not implemented for {platform}")
                else:
                    result = await impl(adapter, *args)
            except Exception as exc:
                result = _err("action_failed", str(exc)[:512])
        else:
            result = error or _err("gateway_unavailable")
        self._audit(verb, platform, result)
        return result

    # -- v1 verbs -----------------------------------------------------------

    async def add_reaction(self, platform: str, chat_id: str, message_id: str, emoji: str) -> Dict[str, Any]:
        """Add/set an emoji reaction on a platform message."""
        return await self._run(
            "add_reaction", platform, chat_id, message_id, emoji,
            chat_id=chat_id, message_id=message_id, emoji=emoji,
        )

    async def set_thread_title(self, platform: str, chat_id: str, thread_id: str, title: str) -> Dict[str, Any]:
        """Rename a thread / forum topic."""
        return await self._run(
            "set_thread_title", platform, chat_id, thread_id, title,
            chat_id=chat_id, thread_id=thread_id, title=title,
        )

    def _audit(self, verb: str, platform: str, result: Dict[str, Any]) -> None:
        """Every platform action is logged (the #64176 'all actions logged' rule)."""
        logger.info(
            "platform_action plugin=%s verb=%s platform=%s ok=%s%s",
            self._plugin_id,
            verb,
            platform,
            result.get("ok"),
            "" if result.get("ok") else f" error={result.get('error')}",
        )


@dataclass(frozen=True)
class SourceBoundPlatformActions:
    """Mutation facade bound to one source; no adapter or profile path leaks."""

    owner: PlatformActions
    platform: str
    channel_id: str
    thread_id: Optional[str]
    chat_type: str
    routed_profile: str
    source_identity_candidates: Tuple[str, ...]
    # Opaque host-owned adapter-registry identity. It distinguishes a routed
    # runtime profile from the profile whose live transport received the
    # source without exposing an adapter, SDK client, credential, or path.
    transport_profile: Optional[str] = None

    def _validated_host(
        self, *, mutation: bool
    ) -> Tuple[
        Optional[Any],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[Dict[str, Any]],
    ]:
        if not self.owner._capability_granted():
            return None, None, None, None, _err(
                "capability_not_granted",
                f"plugin {self.owner._plugin_id!r} lacks {CAPABILITY_ID!r}",
            )
        required = {
            "platform": self.platform,
            "channel_id": self.channel_id,
            "routed_profile": self.routed_profile,
        }
        for name, raw in required.items():
            if not isinstance(raw, str) or not raw.strip():
                return None, None, None, None, _err(
                    "invalid_argument", f"{name} must be a non-empty string"
                )
        try:
            from gateway.config import Platform

            platform = Platform(self.platform.strip().lower()).value
        except Exception:
            return None, None, None, None, _err(
                "unknown_platform", f"unknown platform {self.platform!r}"
            )
        try:
            from hermes_cli.profiles import normalize_profile_name, validate_profile_name

            profile = normalize_profile_name(self.routed_profile)
            validate_profile_name(profile)
            transport_profile = normalize_profile_name(
                self.transport_profile or self.routed_profile
            )
            validate_profile_name(transport_profile)
        except Exception:
            return None, None, None, None, _err(
                "invalid_argument", "routed or transport profile is invalid"
            )
        if mutation and self.chat_type.strip().lower() in {"", "dm", "direct", "private"}:
            return None, None, None, None, _err(
                "unsupported_context", "channel policies require a group or channel source"
            )
        if mutation and not self.source_identity_candidates:
            return None, None, None, None, _err(
                "explicit_admin_required", "no normalized sender identity is available"
            )
        try:
            from gateway.run import _gateway_runner_ref

            runner = _gateway_runner_ref()
        except Exception:
            runner = None
        if runner is None:
            return None, None, None, None, _err(
                "gateway_unavailable", "no gateway runner is active in this process"
            )
        return runner, platform, profile, transport_profile, None

    async def get_channel_policy_status(self) -> Dict[str, Any]:
        """Read current source-bound policy state without granting mutation access."""
        runner, platform, profile, transport_profile, error = self._validated_host(
            mutation=False
        )
        if error is not None:
            result = error
            self.owner._audit("get_channel_policy_status", self.platform, result)
            return result
        assert (
            runner is not None
            and platform is not None
            and profile is not None
            and transport_profile is not None
        )
        service = getattr(runner, "_get_plugin_channel_policy_status_action", None)
        if not callable(service):
            result = _err(
                "unsupported_platform_action", "the gateway has no channel policy status service"
            )
            self.owner._audit("get_channel_policy_status", self.platform, result)
            return result
        try:
            result = service(
                plugin_id=self.owner._plugin_id,
                platform=platform,
                routed_profile=profile,
                transport_profile=transport_profile,
                channel_id=self.channel_id,
                thread_id=self.thread_id,
                chat_type=self.chat_type,
                source_identity_candidates=self.source_identity_candidates,
            )
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
                result = _err("action_failed", "channel policy status service returned an invalid result")
        except Exception as exc:
            result = _err("action_failed", str(exc)[:512])
        self.owner._audit("get_channel_policy_status", self.platform, result)
        return result

    async def set_channel_policy(self, policy: str, value: str) -> Dict[str, Any]:
        """Request a host-owned channel policy mutation after independent gates."""
        runner, platform, profile, transport_profile, error = self._validated_host(
            mutation=True
        )
        if error is not None:
            result = error
            self.owner._audit("set_channel_policy", self.platform, result)
            return result
        for name, raw in {"policy": policy, "value": value}.items():
            if not isinstance(raw, str) or not raw.strip():
                result = _err("invalid_argument", f"{name} must be a non-empty string")
                self.owner._audit("set_channel_policy", self.platform, result)
                return result
        assert (
            runner is not None
            and platform is not None
            and profile is not None
            and transport_profile is not None
        )
        service = getattr(runner, "_apply_plugin_channel_policy_action", None)
        if not callable(service):
            result = _err("unsupported_platform_action", "the gateway has no channel policy service")
            self.owner._audit("set_channel_policy", self.platform, result)
            return result
        try:
            result = service(
                plugin_id=self.owner._plugin_id,
                platform=platform,
                routed_profile=profile,
                transport_profile=transport_profile,
                channel_id=self.channel_id,
                thread_id=self.thread_id,
                chat_type=self.chat_type,
                source_identity_candidates=self.source_identity_candidates,
                policy=policy.strip(),
                value=value.strip(),
            )
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
                result = _err("action_failed", "channel policy service returned an invalid result")
        except Exception as exc:
            result = _err("action_failed", str(exc)[:512])
        self.owner._audit("set_channel_policy", self.platform, result)
        return result


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
# ---- END PLUGIN-COMPAT ----
