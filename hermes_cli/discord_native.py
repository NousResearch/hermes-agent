"""Operator CLI skeleton for Discord Native Multi-Bot Protocol v2.

Slice 1.2 is intentionally default-off and offline-safe: the commands below load
configuration, persist safe identity metadata when explicitly syncing, and render
deterministic diagnostics.  They do not connect the active gateway, resolve bot
tokens, call Discord, invoke Hermes, or use the real secret-handoff MCP server.
Network/secret integrations are represented by injectable protocols so tests and
future slices can provide fakes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence
from urllib.parse import urlencode

from gateway.config import (
    DiscordNativeMultibotConfig,
    DiscordNativeMultibotIdentityConfig,
    load_gateway_config,
)
from gateway.discord_identity_registry import DiscordIdentityRegistry
from gateway.discord_protocol_v2_diagnostics import attach_health_snapshot
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store, default_db_path
from gateway.secret_refs import redact_secret_ref, redact_sensitive_data


DEFAULT_OAUTH_SCOPES = ("bot", "applications.commands")
CONFIG_HELP = (
    "Path to a YAML/JSON config containing discord_native_multibot. "
    "Defaults to the active Hermes gateway config."
)


class SecretPresenceProvider(Protocol):
    """Optional test/future hook for deciding whether a secret ref exists.

    Implementations must never return or print the secret value itself.
    """

    def has_secret(self, ref: str) -> bool:
        raise NotImplementedError


class SecretHandoffProvider(Protocol):
    """Injectable secret handoff creator.

    Slice 1.2 only returns URLs/references.  The provider is deliberately passed
    identity metadata and the secret ref, never a resolved Discord token.
    """

    def create_handoff(
        self,
        *,
        identity: DiscordNativeMultibotIdentityConfig,
        redacted_secret_ref: str,
    ) -> dict[str, Any]:
        raise NotImplementedError


class DiscordRestClient(Protocol):
    """Fakeable Discord REST surface for CLI diagnostics.

    The production CLI does not instantiate a client in Slice 1.2.  Tests and
    later slices can inject an object with these methods.
    """

    def verify_identity(self, identity: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def list_guild_bots(self, guild_id: str) -> list[Any]:
        raise NotImplementedError


REQUIRED_VERIFY_CHECKS = (
    "bot_user_id",
    "guild_membership",
    "message_content_intent",
    "scopes",
    "permissions",
    "secret_ref",
)


@dataclass(frozen=True)
class PlaceholderSecretHandoffProvider:
    """Offline placeholder handoff provider used until real MCP integration.

    It produces deterministic local references and HTTPS-looking placeholder
    URLs.  No external calls are made and no plaintext token is accepted.
    """

    base_url: str = "https://secret-handoff.invalid/hermes/discord-native"

    def create_handoff(
        self,
        *,
        identity: DiscordNativeMultibotIdentityConfig,
        redacted_secret_ref: str,
    ) -> dict[str, Any]:
        handoff_ref = f"secret-handoff://discord-native/{identity.agent_id}/bot-token"
        query = urlencode(
            {
                "agent_id": identity.agent_id,
                "secret_ref": redacted_secret_ref,
            }
        )
        return {
            "agent_id": identity.agent_id,
            "token_secret_ref": redacted_secret_ref,
            "handoff_ref": handoff_ref,
            "handoff_url": f"{self.base_url}?{query}",
            "status": "placeholder_created_no_network",
        }


def _redact_secret_ref(ref: str | None) -> str | None:
    return redact_secret_ref(ref)


def _identity_snapshot(identity: DiscordNativeMultibotIdentityConfig) -> dict[str, Any]:
    snapshot = identity.redacted_snapshot()
    snapshot["allowed_scopes"] = dict(snapshot.get("allowed_scopes") or {})
    return snapshot


def _secret_present(
    ref: str,
    presence_provider: SecretPresenceProvider | None,
) -> bool:
    if presence_provider is None:
        return False
    return bool(presence_provider.has_secret(ref))


def _missing_secret_entries(
    config: DiscordNativeMultibotConfig,
    presence_provider: SecretPresenceProvider | None = None,
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for identity in sorted(config.identities, key=lambda item: item.agent_id):
        if not _secret_present(identity.token_secret_ref, presence_provider):
            missing.append(
                {
                    "agent_id": identity.agent_id,
                    "token_secret_ref": _redact_secret_ref(identity.token_secret_ref),
                    "status": "missing_or_unchecked",
                }
            )
    return missing


def build_plan(
    config: DiscordNativeMultibotConfig,
    *,
    presence_provider: SecretPresenceProvider | None = None,
) -> dict[str, Any]:
    """Return the offline onboarding plan for configured v2 identities."""

    desired = []
    for identity in sorted(config.identities, key=lambda item: item.agent_id):
        item = _identity_snapshot(identity)
        item["will_activate"] = bool(
            config.enabled and config.mode != "off" and identity.enabled
        )
        desired.append(item)

    return {
        "schema_version": 1,
        "command": "plan",
        "enabled": config.enabled,
        "mode": config.mode,
        "guild_allowlist": list(config.guild_allowlist),
        "default_intake_agent_id": config.default_intake_agent_id,
        "desired_identities": desired,
        "missing_secrets": _missing_secret_entries(config, presence_provider),
        "network": "not_used",
        "active_gateway": "not_connected",
    }


def sync_metadata(
    config: DiscordNativeMultibotConfig,
    *,
    store: Any | None = None,
    store_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Load the registry and optionally persist only safe identity metadata."""

    created_store = None
    effective_store = None
    if not dry_run:
        if store is not None:
            effective_store = store
        else:
            created_store = DiscordProtocolV2Store(store_path)
            effective_store = created_store

    try:
        registry = DiscordIdentityRegistry.load(
            config,
            effective_store,
            secret_resolver=None,
        )
        return {
            "schema_version": 1,
            "command": "sync",
            "dry_run": dry_run,
            "synced_identity_count": 0 if dry_run else len(registry.identities),
            "identity_count": len(registry.identities),
            "active_agent_ids": list(registry.active_agent_ids),
            "store_path": str(
                store_path
                if store_path is not None
                else getattr(effective_store, "db_path", default_db_path())
            ),
            "network": "not_used",
            "token_resolution": "not_used",
        }
    finally:
        if created_store is not None:
            created_store.close()


def create_missing_secret_handoffs(
    config: DiscordNativeMultibotConfig,
    *,
    provider: SecretHandoffProvider | None = None,
    presence_provider: SecretPresenceProvider | None = None,
) -> dict[str, Any]:
    """Create placeholder/fake handoffs for missing refs without exposing tokens."""

    provider = provider or PlaceholderSecretHandoffProvider()
    handoffs = []
    for identity in sorted(config.identities, key=lambda item: item.agent_id):
        if _secret_present(identity.token_secret_ref, presence_provider):
            continue
        handoffs.append(
            _sanitize_for_output(
                provider.create_handoff(
                    identity=identity,
                    redacted_secret_ref=_redact_secret_ref(identity.token_secret_ref)
                    or "<redacted>",
                )
            )
        )
    return {
        "schema_version": 1,
        "command": "handoff-missing-secrets",
        "handoff_count": len(handoffs),
        "handoffs": handoffs,
        "secret_values": "never_printed",
        "network": "not_used",
    }


def _sanitize_for_output(value: Any) -> Any:
    """Conservatively redact secret/token-shaped keys in fake client output."""
    return redact_sensitive_data(value)


def verify_identities(
    config: DiscordNativeMultibotConfig,
    *,
    client: DiscordRestClient | None = None,
    operator_network: bool = False,
    store: DiscordProtocolV2Store | None = None,
    store_path: str | Path | None = None,
    presence_provider: SecretPresenceProvider | None = None,
) -> dict[str, Any]:
    """Verify configured identities via an injected fake/client only."""

    registry = DiscordIdentityRegistry.load(config, store=None, secret_resolver=None)
    results = []
    for identity in sorted(
        registry.redacted_snapshot()["identities"], key=lambda item: item["agent_id"]
    ):
        secret_missing = _is_secret_missing(
            str(identity.get("agent_id") or ""), config, presence_provider
        )
        if client is None:
            results.append(
                {
                    "agent_id": identity["agent_id"],
                    "status": (
                        "operator_network_not_implemented"
                        if operator_network
                        else "not_implemented_without_client"
                    ),
                    "network_gate": (
                        "explicit_operator_flag_present_but_no_live_provider"
                        if operator_network
                        else "blocked_without_injected_client_or_operator_flag"
                    ),
                    "checks": list(REQUIRED_VERIFY_CHECKS),
                    "problems": ["missing_secret"] if secret_missing else [],
                }
            )
        else:
            result = client.verify_identity(dict(identity))
            results.append(
                _sanitize_for_output(
                    _normalize_verify_result(
                        identity=dict(identity),
                        config=config,
                        raw_result=result,
                        secret_missing=secret_missing,
                    )
                )
            )

    payload = {
        "schema_version": 1,
        "command": "verify",
        "identity_count": len(registry.identities),
        "duplicate_bot_id_check": "passed_by_registry_load",
        "results": results,
        "network": "fake_client" if client is not None else "not_used",
        "operator_network": "requested_not_implemented" if operator_network else "not_requested",
    }
    return attach_health_snapshot(
        payload,
        config,
        store=store,
        store_path=store_path,
        presence_provider=presence_provider,
    )


def _is_secret_missing(
    agent_id: str,
    config: DiscordNativeMultibotConfig,
    presence_provider: SecretPresenceProvider | None,
) -> bool:
    identity = next((item for item in config.identities if item.agent_id == agent_id), None)
    if identity is None or presence_provider is None:
        return False
    return not _secret_present(identity.token_secret_ref, presence_provider)


def _as_str_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return {str(key) for key, present in value.items() if present}
    if isinstance(value, (set, frozenset)):
        return {str(item) for item in value}
    if isinstance(value, Sequence):
        return {str(item) for item in value}
    return {str(value)}


def _normalize_verify_result(
    *,
    identity: dict[str, Any],
    config: DiscordNativeMultibotConfig,
    raw_result: dict[str, Any],
    secret_missing: bool,
) -> dict[str, Any]:
    """Derive common fake-client checks while preserving fake-specific fields."""

    result = dict(raw_result or {})
    desired_bot_user_id = str(identity.get("discord_bot_user_id") or "")
    actual_bot_user_id = str(
        result.get("bot_user_id")
        or result.get("discord_bot_user_id")
        or result.get("actual_bot_user_id")
        or desired_bot_user_id
    )
    desired_guild_ids = set(
        _as_str_set(identity.get("allowed_scopes", {}).get("guild_ids"))
        or set(config.guild_allowlist)
    )
    actual_guild_source = next(
        (
            result[key]
            for key in ("guild_ids", "member_guild_ids", "guild_memberships")
            if key in result
        ),
        desired_guild_ids,
    )
    actual_guild_ids = _as_str_set(actual_guild_source)
    desired_scopes = set(DEFAULT_OAUTH_SCOPES)
    actual_scope_source = next(
        (result[key] for key in ("scopes", "oauth_scopes") if key in result),
        desired_scopes,
    )
    actual_scopes = _as_str_set(actual_scope_source)
    missing_scopes = sorted(desired_scopes - actual_scopes)

    checks: dict[str, Any] = {
        "bot_user_id": {
            "desired": desired_bot_user_id,
            "actual": actual_bot_user_id,
            "ok": actual_bot_user_id == desired_bot_user_id,
        },
        "guild_membership": {
            "desired_guild_ids": sorted(desired_guild_ids),
            "present_guild_ids": sorted(actual_guild_ids & desired_guild_ids),
            "missing_guild_ids": sorted(desired_guild_ids - actual_guild_ids),
            "ok": not (desired_guild_ids - actual_guild_ids),
        },
        "message_content_intent": {
            "ok": bool(result.get("message_content_intent", result.get("message_content_intent_enabled", True)))
        },
        "scopes": {"required": sorted(desired_scopes), "missing": missing_scopes, "ok": not missing_scopes},
        "permissions": {"ok": bool(result.get("permissions_ok", result.get("has_required_permissions", True)))},
        "secret_ref": {"ok": not secret_missing},
    }
    problems: list[str] = []
    if not checks["bot_user_id"]["ok"]:
        problems.append("bot_user_id_mismatch")
    if not checks["guild_membership"]["ok"]:
        problems.append("missing_guild_membership")
    if not checks["message_content_intent"]["ok"]:
        problems.append("missing_message_content_intent")
    if not checks["scopes"]["ok"]:
        problems.append("missing_scopes")
    if not checks["permissions"]["ok"]:
        problems.append("missing_permissions")
    if secret_missing:
        problems.append("missing_secret")

    result.setdefault("agent_id", identity.get("agent_id"))
    result["checks"] = checks
    result["problems"] = sorted(set(problems))
    result["status"] = "ok" if not problems else "failed"
    return result


def _oauth_invite_url(
    *,
    application_id: str,
    scopes: Sequence[str] = DEFAULT_OAUTH_SCOPES,
    permissions: str = "0",
    guild_id: str | None = None,
) -> str:
    query: dict[str, str] = {
        "client_id": application_id,
        "permissions": permissions,
        "scope": " ".join(scopes),
    }
    if guild_id:
        query["guild_id"] = guild_id
    return "https://discord.com/oauth2/authorize?" + urlencode(query)


def build_invite_links(
    config: DiscordNativeMultibotConfig,
    *,
    scopes: Sequence[str] = DEFAULT_OAUTH_SCOPES,
    permissions: str = "0",
) -> dict[str, Any]:
    links = []
    for identity in sorted(config.identities, key=lambda item: item.agent_id):
        guild_ids = identity.allowed_scopes.guild_ids or config.guild_allowlist or [None]
        for guild_id in guild_ids:
            links.append(
                {
                    "agent_id": identity.agent_id,
                    "discord_application_id": identity.discord_application_id,
                    "guild_id": guild_id,
                    "scopes": list(scopes),
                    "permissions": permissions,
                    "url": _oauth_invite_url(
                        application_id=str(identity.discord_application_id),
                        scopes=scopes,
                        permissions=permissions,
                        guild_id=guild_id,
                    ),
                }
            )
    return {
        "schema_version": 1,
        "command": "invite-links",
        "links": links,
        "network": "not_used",
    }


def _bot_id(bot: Any) -> str | None:
    if isinstance(bot, str):
        return bot
    if isinstance(bot, dict):
        value = bot.get("id") or bot.get("bot_user_id") or bot.get("user_id")
        return str(value) if value is not None else None
    value = getattr(bot, "id", None) or getattr(bot, "bot_user_id", None)
    return str(value) if value is not None else None


def reconcile_guild(
    config: DiscordNativeMultibotConfig,
    *,
    client: DiscordRestClient | None = None,
    operator_network: bool = False,
    guild_ids: Sequence[str] | None = None,
    store: DiscordProtocolV2Store | None = None,
    store_path: str | Path | None = None,
    presence_provider: SecretPresenceProvider | None = None,
) -> dict[str, Any]:
    """Compare registry metadata with fake guild bot membership."""

    registry = DiscordIdentityRegistry.load(config, store=None, secret_resolver=None)
    desired_by_bot_id = {
        identity.discord_bot_user_id: identity.agent_id
        for identity in registry.identities.values()
    }
    target_guilds = list(guild_ids or config.guild_allowlist)
    if not target_guilds:
        payload = {
            "schema_version": 1,
            "command": "reconcile-guild",
            "status": "no_guilds_configured",
            "desired_bot_user_ids": sorted(desired_by_bot_id),
            "network": "not_used",
        }
        return attach_health_snapshot(
            payload,
            config,
            store=store,
            store_path=store_path,
            presence_provider=presence_provider,
        )

    guild_results = []
    for guild_id in target_guilds:
        if client is None:
            guild_results.append(
                {
                    "guild_id": guild_id,
                    "status": (
                        "operator_network_not_implemented"
                        if operator_network
                        else "not_implemented_without_client"
                    ),
                    "network_gate": (
                        "explicit_operator_flag_present_but_no_live_provider"
                        if operator_network
                        else "blocked_without_injected_client_or_operator_flag"
                    ),
                    "desired_agent_ids": sorted(desired_by_bot_id.values()),
                }
            )
            continue

        actual_bots = client.list_guild_bots(guild_id)
        actual_ids = sorted(
            bot_id for bot_id in (_bot_id(bot) for bot in actual_bots) if bot_id
        )
        present = sorted(
            desired_by_bot_id[bot_id]
            for bot_id in actual_ids
            if bot_id in desired_by_bot_id
        )
        missing = sorted(
            agent_id
            for bot_id, agent_id in desired_by_bot_id.items()
            if bot_id not in actual_ids
        )
        extra = sorted(bot_id for bot_id in actual_ids if bot_id not in desired_by_bot_id)
        guild_results.append(
            {
                "guild_id": guild_id,
                "status": "compared_with_fake_client",
                "present_agent_ids": present,
                "missing_agent_ids": missing,
                "extra_bot_user_ids": extra,
                "guild_bot_user_ids": actual_ids,
            }
        )

    payload = {
        "schema_version": 1,
        "command": "reconcile-guild",
        "guilds": guild_results,
        "network": "fake_client" if client is not None else "not_used",
        "operator_network": "requested_not_implemented" if operator_network else "not_requested",
    }
    return attach_health_snapshot(
        payload,
        config,
        store=store,
        store_path=store_path,
        presence_provider=presence_provider,
    )


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(_sanitize_for_output(data), indent=2, sort_keys=True))


def _load_config_from_path(path: str | Path) -> DiscordNativeMultibotConfig:
    target = Path(path).expanduser()
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        import yaml

        raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise SystemExit("discord-native config file must contain a mapping")
    data = raw.get("discord_native_multibot")
    if data is None and isinstance(raw.get("gateway"), dict):
        data = raw["gateway"].get("discord_native_multibot")
    if data is None:
        data = raw
    return DiscordNativeMultibotConfig.from_dict(data)


def load_discord_native_config(args: argparse.Namespace) -> DiscordNativeMultibotConfig:
    config_path = getattr(args, "config", None)
    if config_path:
        return _load_config_from_path(config_path)
    return load_gateway_config().discord_native_multibot


def _cmd_plan(args: argparse.Namespace) -> int:
    _print_json(build_plan(load_discord_native_config(args)))
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    _print_json(
        sync_metadata(
            load_discord_native_config(args),
            store_path=getattr(args, "store_path", None),
            dry_run=getattr(args, "dry_run", False),
        )
    )
    return 0


def _cmd_handoff_missing_secrets(args: argparse.Namespace) -> int:
    _print_json(create_missing_secret_handoffs(load_discord_native_config(args)))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    _print_json(
        verify_identities(
            load_discord_native_config(args),
            operator_network=getattr(args, "operator_network", False),
            store_path=getattr(args, "store_path", None),
        )
    )
    return 0


def _cmd_invite_links(args: argparse.Namespace) -> int:
    scopes = tuple(getattr(args, "scope", None) or DEFAULT_OAUTH_SCOPES)
    _print_json(
        build_invite_links(
            load_discord_native_config(args),
            scopes=scopes,
            permissions=str(getattr(args, "permissions", "0")),
        )
    )
    return 0


def _cmd_reconcile_guild(args: argparse.Namespace) -> int:
    _print_json(
        reconcile_guild(
            load_discord_native_config(args),
            operator_network=getattr(args, "operator_network", False),
            guild_ids=getattr(args, "guild_id", None),
            store_path=getattr(args, "store_path", None),
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes discord-native",
        description="Offline operator tools for Discord native multi-bot onboarding",
    )
    register_parser(parser)
    return parser


def register_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    def add_config_arg(
        target: argparse.ArgumentParser, *, suppress_default: bool = False
    ) -> None:
        kwargs: dict[str, Any] = {"help": CONFIG_HELP}
        if suppress_default:
            kwargs["default"] = argparse.SUPPRESS
        target.add_argument("--config", **kwargs)

    add_config_arg(parser)
    sub = parser.add_subparsers(dest="discord_native_command")

    plan = sub.add_parser("plan", help="Show desired identities/scopes and missing refs")
    add_config_arg(plan, suppress_default=True)
    plan.set_defaults(func=_cmd_plan)

    sync = sub.add_parser("sync", help="Persist safe identity metadata to the v2 store")
    add_config_arg(sync, suppress_default=True)
    sync.add_argument("--store-path", help="SQLite store path (default: profile gateway store)")
    sync.add_argument("--dry-run", action="store_true", help="Validate and summarize only")
    sync.set_defaults(func=_cmd_sync)

    handoff = sub.add_parser(
        "handoff-missing-secrets",
        help="Create placeholder secure-handoff URLs for missing token refs",
    )
    add_config_arg(handoff, suppress_default=True)
    handoff.set_defaults(func=_cmd_handoff_missing_secrets)

    verify = sub.add_parser("verify", help="Offline/fake-client identity verification skeleton")
    add_config_arg(verify, suppress_default=True)
    verify.add_argument("--store-path", help="SQLite store path (default: profile gateway store)")
    verify.add_argument(
        "--operator-network",
        action="store_true",
        help="Future live Discord verification gate; currently returns not_implemented and performs no network",
    )
    verify.set_defaults(func=_cmd_verify)

    invite = sub.add_parser("invite-links", help="Build Discord OAuth invite URLs")
    add_config_arg(invite, suppress_default=True)
    invite.add_argument(
        "--scope",
        action="append",
        help="OAuth scope to include (repeatable; default: bot + applications.commands)",
    )
    invite.add_argument("--permissions", default="0", help="Discord permissions integer")
    invite.set_defaults(func=_cmd_invite_links)

    reconcile = sub.add_parser(
        "reconcile-guild",
        help="Compare desired registry to fake/injected guild state",
    )
    add_config_arg(reconcile, suppress_default=True)
    reconcile.add_argument(
        "--guild-id",
        action="append",
        help="Guild ID to reconcile (repeatable; default: config guild_allowlist)",
    )
    reconcile.add_argument("--store-path", help="SQLite store path (default: profile gateway store)")
    reconcile.add_argument(
        "--operator-network",
        action="store_true",
        help="Future live Discord reconciliation gate; currently returns not_implemented and performs no network",
    )
    reconcile.set_defaults(func=_cmd_reconcile_guild)
    return parser


def cmd_discord_native(args: argparse.Namespace) -> int:
    if getattr(args, "discord_native_command", None) is None:
        print(
            "usage: hermes discord-native "
            "{plan,sync,handoff-missing-secrets,verify,invite-links,reconcile-guild} ..."
        )
        return 2
    return args.func(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "discord_native_command", None) is None:
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
