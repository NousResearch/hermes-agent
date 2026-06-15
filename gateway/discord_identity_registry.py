"""Safe identity registry for Discord Native Multi-Bot Protocol v2.

Slice 1.1 is intentionally limited to loading configured identity metadata,
validating registry-level uniqueness, persisting non-secret metadata to the v2
store, and resolving token secret refs on demand.  It does not instantiate
Discord clients, verify credentials, wire gateway runtime routing, or send
messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gateway.config import (
    DiscordNativeMultibotConfig,
    DiscordNativeMultibotIdentityConfig,
)
from gateway.secret_refs import (
    SecretRefError,
    SecretResolutionError,
    SecretResolver,
    SensitiveToken,
    redact_secret_ref,
    validate_secret_ref,
)


class DiscordIdentityRegistryError(ValueError):
    """Base validation error for the Discord v2 identity registry."""


def _redact_secret_ref(ref: str | None) -> str | None:
    return redact_secret_ref(ref)


def _validate_secret_ref(ref: Any) -> str:
    try:
        return validate_secret_ref(
            ref,
            allow_env=False,
            field_name="discord_native_multibot token_secret_ref",
        )
    except SecretRefError as exc:
        raise DiscordIdentityRegistryError(
            str(exc)
        ) from None


def _required_identity_field(identity: DiscordNativeMultibotIdentityConfig, field: str) -> str:
    value = getattr(identity, field)
    if not isinstance(value, str) or not value.strip():
        raise DiscordIdentityRegistryError(
            f"discord_native_multibot identities require {field}"
        )
    return value.strip()


def _allowed_scopes_dict(identity: DiscordNativeMultibotIdentityConfig) -> dict[str, Any]:
    scopes = identity.allowed_scopes
    if hasattr(scopes, "to_dict"):
        result = scopes.to_dict()
    elif isinstance(scopes, dict):
        result = dict(scopes)
    else:
        result = {}
    return dict(result)


@dataclass(frozen=True, repr=False)
class DiscordIdentityMetadata:
    """Safe Discord v2 identity metadata.

    The resolved Discord bot token is intentionally not a field on this
    dataclass.  Only the configured secret reference is retained, and diagnostic
    views redact that reference.
    """

    agent_id: str
    hermes_profile: str
    discord_application_id: str
    discord_bot_user_id: str
    token_secret_ref: str
    capabilities: tuple[str, ...] = ()
    allowed_scopes: dict[str, Any] | None = None
    enabled: bool = True

    def __repr__(self) -> str:
        return (
            "DiscordIdentityMetadata("
            f"agent_id={self.agent_id!r}, "
            f"hermes_profile={self.hermes_profile!r}, "
            f"discord_application_id={self.discord_application_id!r}, "
            f"discord_bot_user_id={self.discord_bot_user_id!r}, "
            f"token_secret_ref={_redact_secret_ref(self.token_secret_ref)!r}, "
            f"capabilities={self.capabilities!r}, "
            f"allowed_scopes={self.allowed_scopes!r}, "
            f"enabled={self.enabled!r})"
        )

    def redacted_snapshot(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "hermes_profile": self.hermes_profile,
            "discord_application_id": self.discord_application_id,
            "discord_bot_user_id": self.discord_bot_user_id,
            "token_secret_ref": _redact_secret_ref(self.token_secret_ref),
            "capabilities": list(self.capabilities),
            "allowed_scopes": dict(self.allowed_scopes or {}),
            "enabled": self.enabled,
        }


class DiscordIdentityRegistry:
    """In-memory safe metadata registry for Discord protocol v2 identities."""

    def __init__(
        self,
        *,
        enabled: bool,
        mode: str,
        identities: dict[str, DiscordIdentityMetadata],
        active_agent_ids: set[str],
        secret_resolver: SecretResolver | None,
    ) -> None:
        self.enabled = enabled
        self.mode = mode
        self._identities = dict(identities)
        self._active_agent_ids = set(active_agent_ids)
        self._secret_resolver = secret_resolver

    def __repr__(self) -> str:
        return (
            "DiscordIdentityRegistry("
            f"enabled={self.enabled!r}, mode={self.mode!r}, "
            f"identity_count={len(self._identities)!r}, "
            f"active_agent_ids={sorted(self._active_agent_ids)!r})"
        )

    @classmethod
    def load(
        cls,
        config: DiscordNativeMultibotConfig,
        store: Any | None,
        secret_resolver: SecretResolver | None,
        *,
        allow_duplicate_token_secret_refs: bool = False,
    ) -> "DiscordIdentityRegistry":
        """Load and validate configured Discord v2 identities.

        ``store`` is optional and is only used for persisting safe metadata.  The
        resolver is retained for later ``resolve_token()`` calls; it is never
        invoked during load, which keeps plaintext tokens out of dataclasses,
        snapshots, logs, and the SQLite store.
        """

        registry_enabled = bool(config.enabled and config.mode != "off")
        identities: dict[str, DiscordIdentityMetadata] = {}
        active_agent_ids: set[str] = set()
        seen_application_ids: set[str] = set()
        seen_bot_user_ids: set[str] = set()
        seen_token_refs: set[str] = set()

        for identity in config.identities:
            agent_id = _required_identity_field(identity, "agent_id")
            hermes_profile = (
                _required_identity_field(identity, "hermes_profile")
                if getattr(identity, "hermes_profile", None)
                else "default"
            )
            discord_application_id = _required_identity_field(
                identity, "discord_application_id"
            )
            discord_bot_user_id = _required_identity_field(
                identity, "discord_bot_user_id"
            )
            token_secret_ref = _validate_secret_ref(identity.token_secret_ref)

            if agent_id in identities:
                raise DiscordIdentityRegistryError(
                    "discord_native_multibot identities require unique agent_id"
                )
            if discord_application_id in seen_application_ids:
                raise DiscordIdentityRegistryError(
                    "discord_native_multibot identities require unique discord_application_id"
                )
            if discord_bot_user_id in seen_bot_user_ids:
                raise DiscordIdentityRegistryError(
                    "discord_native_multibot identities require unique discord_bot_user_id"
                )
            if (
                token_secret_ref in seen_token_refs
                and not allow_duplicate_token_secret_refs
            ):
                raise DiscordIdentityRegistryError(
                    "discord_native_multibot identities require unique token_secret_ref"
                )

            seen_application_ids.add(discord_application_id)
            seen_bot_user_ids.add(discord_bot_user_id)
            seen_token_refs.add(token_secret_ref)

            metadata = DiscordIdentityMetadata(
                agent_id=agent_id,
                hermes_profile=hermes_profile,
                discord_application_id=discord_application_id,
                discord_bot_user_id=discord_bot_user_id,
                token_secret_ref=token_secret_ref,
                capabilities=tuple(str(capability) for capability in identity.capabilities),
                allowed_scopes=_allowed_scopes_dict(identity),
                enabled=bool(identity.enabled),
            )
            identities[agent_id] = metadata
            is_active = registry_enabled and metadata.enabled
            if is_active:
                active_agent_ids.add(agent_id)

        if store is not None:
            for metadata in identities.values():
                store.upsert_identity(
                    agent_id=metadata.agent_id,
                    hermes_profile=metadata.hermes_profile,
                    discord_application_id=metadata.discord_application_id,
                    discord_bot_user_id=metadata.discord_bot_user_id,
                    token_secret_ref=metadata.token_secret_ref,
                    capabilities=list(metadata.capabilities),
                    scopes=dict(metadata.allowed_scopes or {}),
                    enabled=metadata.agent_id in active_agent_ids,
                )

        return cls(
            enabled=registry_enabled,
            mode=config.mode,
            identities=identities,
            active_agent_ids=active_agent_ids,
            secret_resolver=secret_resolver,
        )

    @property
    def identities(self) -> dict[str, DiscordIdentityMetadata]:
        """Return a copy of all loaded identities, including disabled ones."""

        return dict(self._identities)

    @property
    def active_agent_ids(self) -> tuple[str, ...]:
        """Enabled identities available for runtime token lookup."""

        return tuple(sorted(self._active_agent_ids))

    def get_identity(
        self, agent_id: str, *, include_disabled: bool = True
    ) -> DiscordIdentityMetadata | None:
        identity = self._identities.get(agent_id)
        if identity is None:
            return None
        if not include_disabled and agent_id not in self._active_agent_ids:
            return None
        return identity

    def resolve_token(self, agent_id: str) -> str:
        """Resolve an active identity token in memory without persisting it."""

        identity = self.get_identity(agent_id, include_disabled=False)
        if identity is None:
            raise KeyError(
                f"unknown or inactive Discord v2 identity agent_id={agent_id!r}"
            )
        if self._secret_resolver is None:
            raise SecretResolutionError(
                f"no secret resolver configured for agent_id={agent_id!r}"
            )

        resolver_error_type: str | None = None
        token: Any = None
        try:
            token = self._secret_resolver.resolve(identity.token_secret_ref)
        except Exception as exc:  # pragma: no cover - message is tested indirectly
            resolver_error_type = type(exc).__name__

        if resolver_error_type is not None:
            raise SecretResolutionError(
                "failed to resolve Discord v2 token for "
                f"agent_id={agent_id!r} token_secret_ref="
                f"{_redact_secret_ref(identity.token_secret_ref)!r}: "
                f"{resolver_error_type}"
            )

        if isinstance(token, SensitiveToken):
            token_value = token.reveal()
        elif isinstance(token, str):
            token_value = token
        else:
            token_value = None

        if not isinstance(token_value, str) or not token_value:
            raise SecretResolutionError(
                f"secret resolver returned no token for agent_id={agent_id!r}"
            )
        return token_value

    def redacted_snapshot(self) -> dict[str, Any]:
        """Safe diagnostic snapshot that never includes resolved tokens."""

        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "identity_count": len(self._identities),
            "active_agent_ids": list(self.active_agent_ids),
            "identities": [
                identity.redacted_snapshot()
                for identity in sorted(
                    self._identities.values(), key=lambda item: item.agent_id
                )
            ],
        }
