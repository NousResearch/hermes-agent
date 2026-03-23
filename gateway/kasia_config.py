from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_KASIA_ADDRESS_PREFIXES = ("kaspa:", "kaspatest:", "kaspasim:")

DEFAULT_KASIA_NETWORK = "mainnet"
DEFAULT_KASIA_FEE_POLICY = "auto"
DEFAULT_KASIA_BRIDGE_PORT = 3010
DEFAULT_KASIA_SEND_WAIT_MS = 5000


def _text(value: Any) -> str:
    return str(value or "").strip()


def _is_truthy(value: Any) -> bool:
    return _text(value).lower() in _TRUE_VALUES


def _has_configured_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_text(item) for item in value)
    return bool(_text(value))


def _split_values(value: Any, *, delimiter: str = ",") -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(delimiter)
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_items = value
    else:
        raw_items = [value]

    normalized_items: list[str] = []
    for item in raw_items:
        normalized_item = _text(item)
        if normalized_item and normalized_item not in normalized_items:
            normalized_items.append(normalized_item)
    return normalized_items


def _first_populated(*values: Any) -> str:
    for value in values:
        normalized_value = _text(value)
        if normalized_value:
            return normalized_value
    return ""


def _coerce_optional_int(
    value: Any,
    *,
    default: Optional[int] = None,
    logger=None,
    field_name: Optional[str] = None,
) -> Optional[int]:
    normalized_value = _text(value)
    if not normalized_value:
        return default
    try:
        return int(normalized_value)
    except (TypeError, ValueError):
        if logger and field_name:
            logger.warning("Invalid %s=%r (expected integer)", field_name, value)
        return default


def _configured_urls(
    singular_value: Any,
    plural_value: Any,
) -> tuple[str, tuple[str, ...]]:
    url_list = tuple(_split_values(plural_value))
    primary_url = _first_populated(singular_value, url_list[0] if url_list else "")
    if not url_list and primary_url:
        url_list = (primary_url,)
    return primary_url, url_list


def _configured_urls_with_precedence(
    extra_singular_value: Any,
    extra_plural_value: Any,
    env_singular_value: Any,
    env_plural_value: Any,
) -> tuple[str, tuple[str, ...]]:
    if _has_configured_value(extra_singular_value) or _has_configured_value(extra_plural_value):
        return _configured_urls(extra_singular_value, extra_plural_value)
    return _configured_urls(env_singular_value, env_plural_value)


def _configured_broadcast_channels(
    raw_subscriptions: str,
    allowed_channels: tuple[str, ...],
) -> list[str]:
    configured_channels: list[str] = []
    for segment in raw_subscriptions.split(";"):
        normalized_segment = _text(segment)
        if not normalized_segment or "=" not in normalized_segment:
            continue
        channel_name = _text(normalized_segment.split("=", 1)[0]).lower()
        if channel_name and channel_name not in configured_channels:
            configured_channels.append(channel_name)

    for channel_name in allowed_channels:
        normalized_channel = channel_name.lower()
        if normalized_channel not in configured_channels:
            configured_channels.append(normalized_channel)
    return configured_channels


def normalized_kasia_address_variants(address: Any) -> set[str]:
    normalized_address = _text(address).lower()
    if not normalized_address:
        return set()

    address_variants = {normalized_address}
    if normalized_address.startswith(_KASIA_ADDRESS_PREFIXES):
        address_variants.add(normalized_address.split(":", 1)[1])
    return address_variants


@dataclass(frozen=True, slots=True)
class KasiaSettings:
    enabled: bool
    seed_phrase: str
    indexer_url: str
    indexer_urls: tuple[str, ...]
    node_wborsh_url: str
    node_wborsh_urls: tuple[str, ...]
    network: str
    kns_url: str
    fee_policy: str
    bridge_port: int
    send_wait_ms: int
    max_multipart_parts: Optional[int]
    broadcast_subscriptions: str
    allowed_broadcast_channels: tuple[str, ...]
    allow_all_broadcast_channels: bool
    home_channel: str
    home_channel_name: str
    allowed_users: str
    allow_all_users: bool

    @property
    def has_required_connection_fields(self) -> bool:
        return bool(self.seed_phrase and self.indexer_url and self.node_wborsh_url)

    def platform_extra(self) -> dict[str, Any]:
        extra = {
            "seed_phrase": self.seed_phrase,
            "indexer_url": self.indexer_url,
            "indexer_urls": list(self.indexer_urls),
            "node_wborsh_url": self.node_wborsh_url,
            "node_wborsh_urls": list(self.node_wborsh_urls),
            "network": self.network,
        }
        if self.kns_url:
            extra["kns_url"] = self.kns_url
        if self.fee_policy:
            extra["fee_policy"] = self.fee_policy
        if self.max_multipart_parts is not None:
            extra["max_multipart_parts"] = self.max_multipart_parts
        if self.broadcast_subscriptions:
            extra["broadcast_subscriptions"] = self.broadcast_subscriptions
        if self.allowed_broadcast_channels:
            extra["allowed_broadcast_channels"] = list(self.allowed_broadcast_channels)
        if self.allow_all_broadcast_channels:
            extra["allow_all_broadcast_channels"] = True
        if self.bridge_port != DEFAULT_KASIA_BRIDGE_PORT:
            extra["bridge_port"] = self.bridge_port
        if self.send_wait_ms != DEFAULT_KASIA_SEND_WAIT_MS:
            extra["send_wait_ms"] = self.send_wait_ms
        return extra

    def bridge_env(self) -> dict[str, str]:
        bridge_env = {
            "KASIA_SEED_PHRASE": self.seed_phrase,
            "KASIA_INDEXER_URL": self.indexer_url,
            "KASIA_NODE_WBORSH_URL": self.node_wborsh_url,
            "KASIA_NETWORK": self.network,
            "KASIA_FEE_POLICY": self.fee_policy,
        }
        if self.indexer_urls:
            bridge_env["KASIA_INDEXER_URLS"] = ",".join(self.indexer_urls)
        if self.node_wborsh_urls:
            bridge_env["KASIA_NODE_WBORSH_URLS"] = ",".join(self.node_wborsh_urls)
        if self.kns_url:
            bridge_env["KASIA_KNS_URL"] = self.kns_url
        if self.broadcast_subscriptions:
            bridge_env["KASIA_BROADCAST_SUBSCRIPTIONS"] = self.broadcast_subscriptions
        if self.allowed_broadcast_channels:
            bridge_env["KASIA_ALLOWED_BROADCAST_CHANNELS"] = ",".join(
                self.allowed_broadcast_channels
            )
        if self.allow_all_broadcast_channels:
            bridge_env["KASIA_ALLOW_ALL_BROADCAST_CHANNELS"] = "true"
        if self.max_multipart_parts is not None:
            bridge_env["KASIA_MAX_MULTIPARTS"] = str(self.max_multipart_parts)
        return bridge_env

    def configured_broadcast_channels(self) -> list[str]:
        return _configured_broadcast_channels(
            self.broadcast_subscriptions,
            self.allowed_broadcast_channels,
        )


def load_kasia_settings(
    *,
    extra: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    logger=None,
) -> KasiaSettings:
    config_extra = dict(extra or {})
    config_env = env or os.environ

    indexer_url, indexer_urls = _configured_urls_with_precedence(
        config_extra.get("indexer_url"),
        config_extra.get("indexer_urls"),
        config_env.get("KASIA_INDEXER_URL", ""),
        config_env.get("KASIA_INDEXER_URLS", ""),
    )

    node_wborsh_url, node_wborsh_urls = _configured_urls_with_precedence(
        config_extra.get("node_wborsh_url"),
        config_extra.get("node_wborsh_urls"),
        config_env.get("KASIA_NODE_WBORSH_URL", ""),
        config_env.get("KASIA_NODE_WBORSH_URLS", ""),
    )

    configured_network = _first_populated(
        config_extra.get("network"),
        config_env.get("KASIA_NETWORK", ""),
        DEFAULT_KASIA_NETWORK,
    )
    configured_fee_policy = _first_populated(
        config_extra.get("fee_policy"),
        config_env.get("KASIA_FEE_POLICY", ""),
        DEFAULT_KASIA_FEE_POLICY,
    )
    allowed_broadcast_channels = tuple(
        _split_values(
            config_extra.get("allowed_broadcast_channels")
            if config_extra.get("allowed_broadcast_channels") is not None
            else config_env.get("KASIA_ALLOWED_BROADCAST_CHANNELS", "")
        )
    )

    return KasiaSettings(
        enabled=_is_truthy(config_extra.get("enabled")) or _is_truthy(config_env.get("KASIA_ENABLED", "")),
        seed_phrase=_first_populated(
            config_extra.get("seed_phrase"),
            config_env.get("KASIA_SEED_PHRASE", ""),
        ),
        indexer_url=indexer_url,
        indexer_urls=indexer_urls,
        node_wborsh_url=node_wborsh_url,
        node_wborsh_urls=node_wborsh_urls,
        network=configured_network,
        kns_url=_first_populated(
            config_extra.get("kns_url"),
            config_env.get("KASIA_KNS_URL", ""),
        ),
        fee_policy=configured_fee_policy,
        bridge_port=_coerce_optional_int(
            config_extra.get("bridge_port")
            if config_extra.get("bridge_port") is not None
            else config_env.get("KASIA_BRIDGE_PORT", ""),
            default=DEFAULT_KASIA_BRIDGE_PORT,
            logger=logger,
            field_name="KASIA_BRIDGE_PORT",
        )
        or DEFAULT_KASIA_BRIDGE_PORT,
        send_wait_ms=_coerce_optional_int(
            config_extra.get("send_wait_ms")
            if config_extra.get("send_wait_ms") is not None
            else config_env.get("KASIA_SEND_WAIT_MS", ""),
            default=DEFAULT_KASIA_SEND_WAIT_MS,
            logger=logger,
            field_name="KASIA_SEND_WAIT_MS",
        )
        or DEFAULT_KASIA_SEND_WAIT_MS,
        max_multipart_parts=_coerce_optional_int(
            config_extra.get("max_multipart_parts")
            if config_extra.get("max_multipart_parts") is not None
            else config_env.get("KASIA_MAX_MULTIPARTS", ""),
            default=None,
            logger=logger,
            field_name="KASIA_MAX_MULTIPARTS",
        ),
        broadcast_subscriptions=_first_populated(
            config_extra.get("broadcast_subscriptions"),
            config_env.get("KASIA_BROADCAST_SUBSCRIPTIONS", ""),
        ),
        allowed_broadcast_channels=allowed_broadcast_channels,
        allow_all_broadcast_channels=bool(
            config_extra.get("allow_all_broadcast_channels")
        )
        or _is_truthy(config_env.get("KASIA_ALLOW_ALL_BROADCAST_CHANNELS", "")),
        home_channel=_first_populated(
            config_extra.get("home_channel"),
            config_env.get("KASIA_HOME_CHANNEL", ""),
        ),
        home_channel_name=_first_populated(
            config_extra.get("home_channel_name"),
            config_env.get("KASIA_HOME_CHANNEL_NAME", ""),
            "Home",
        ),
        allowed_users=_first_populated(
            config_extra.get("allowed_users"),
            config_env.get("KASIA_ALLOWED_USERS", ""),
        ),
        allow_all_users=bool(config_extra.get("allow_all_users"))
        or _is_truthy(config_env.get("KASIA_ALLOW_ALL_USERS", "")),
    )


def is_kasia_address_authorized(
    address: Any,
    *,
    env: Optional[Mapping[str, str]] = None,
    display_name: Optional[str] = None,
) -> bool:
    from gateway.kasia_identity import kasia_target_matches

    config_env = env or os.environ
    address_variants = normalized_kasia_address_variants(address)
    if not address_variants:
        return False

    settings = load_kasia_settings(env=config_env)
    if settings.allow_all_users:
        return True

    global_allowlist = _text(config_env.get("GATEWAY_ALLOWED_USERS", ""))
    if not settings.allowed_users and not global_allowlist:
        return _is_truthy(config_env.get("GATEWAY_ALLOW_ALL_USERS", ""))

    allowed_targets = [
        item.strip()
        for allowlist in (settings.allowed_users, global_allowlist)
        for item in allowlist.split(",")
        if item.strip()
    ]
    return any(
        kasia_target_matches(
            address,
            target,
            env=config_env,
            display_name=display_name,
        )
        for target in allowed_targets
    )
