"""Fail-closed routing contract for the silent WhatsApp Borderô reader.

This module is deliberately transport-agnostic. It does not inspect group names,
phone numbers, or message text to decide a route: only exact configured group
JIDs are accepted. The live adapter can therefore reuse the same contract for
inbound filtering, prompt context, and outbound egress policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


_GROUP_JID_RE = re.compile(r"\d+@g\.us\Z")
_TELEGRAM_CHAT_ID_RE = re.compile(r"-?\d+\Z")
_TELEGRAM_THREAD_ID_RE = re.compile(r"[1-9]\d*\Z")


def _platform_name(platform: Any) -> str:
    """Normalize a Platform enum or string without importing gateway code."""
    return str(getattr(platform, "value", platform) or "").lower()


class BorderoReaderConfigError(ValueError):
    """Raised when an enabled reader configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class BorderoRoute:
    """One immutable WhatsApp group → Braza store → Telegram topic route."""

    group_jid: str
    store: str
    location: str
    telegram_chat_id: str
    telegram_thread_id: str

    @property
    def telegram_target(self) -> str:
        return f"telegram:{self.telegram_chat_id}:{self.telegram_thread_id}"


@dataclass(frozen=True)
class BorderoReaderConfig:
    """Validated reader configuration indexed by exact WhatsApp group JID."""

    enabled: bool
    routes: dict[str, BorderoRoute]

    @property
    def group_jids(self) -> frozenset[str]:
        return frozenset(self.routes)


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise BorderoReaderConfigError(f"borderô route requires non-empty {key}")
    return value


def _validate_route(raw: Any) -> BorderoRoute:
    if not isinstance(raw, Mapping):
        raise BorderoReaderConfigError("each borderô route must be a mapping")

    group_jid = _required_string(raw, "group_jid")
    if not _GROUP_JID_RE.fullmatch(group_jid):
        raise BorderoReaderConfigError(
            "group_jid must be the exact canonical WhatsApp group JID (<digits>@g.us)"
        )

    store = _required_string(raw, "store")
    location = _required_string(raw, "location")
    expected = {"PTT": "UBBO", "ODI": "Saldanha"}
    if store not in expected or location != expected[store]:
        raise BorderoReaderConfigError(
            "borderô routes must map exactly PTT→UBBO and ODI→Saldanha"
        )

    telegram_chat_id = _required_string(raw, "telegram_chat_id")
    if not _TELEGRAM_CHAT_ID_RE.fullmatch(telegram_chat_id):
        raise BorderoReaderConfigError("telegram_chat_id must be a numeric Telegram chat id")

    telegram_thread_id = _required_string(raw, "telegram_thread_id")
    if not _TELEGRAM_THREAD_ID_RE.fullmatch(telegram_thread_id):
        raise BorderoReaderConfigError(
            "telegram_thread_id must be a positive numeric forum-topic id"
        )

    return BorderoRoute(
        group_jid=group_jid,
        store=store,
        location=location,
        telegram_chat_id=telegram_chat_id,
        telegram_thread_id=telegram_thread_id,
    )


def load_bordero_reader_config(extra: Mapping[str, Any] | None) -> BorderoReaderConfig:
    """Parse and validate the opt-in reader settings from ``PlatformConfig.extra``.

    Disabled is the safe default. When enabled, exactly two routes are required,
    one for each canonical Braza store, and every Telegram destination must be
    explicit. No name-based or partial configuration is accepted.
    """

    extra = extra or {}
    enabled = extra.get("bordero_read_only", False) is True
    if not enabled:
        return BorderoReaderConfig(enabled=False, routes={})

    raw_routes = extra.get("bordero_routes")
    if not isinstance(raw_routes, (list, tuple)) or len(raw_routes) != 2:
        raise BorderoReaderConfigError(
            "enabled borderô reader requires exactly two bordero_routes"
        )

    routes: dict[str, BorderoRoute] = {}
    for raw_route in raw_routes:
        route = _validate_route(raw_route)
        if route.group_jid in routes:
            raise BorderoReaderConfigError("borderô group_jid values must be unique")
        routes[route.group_jid] = route

    stores = {(route.store, route.location) for route in routes.values()}
    if stores != {("PTT", "UBBO"), ("ODI", "Saldanha")}:
        raise BorderoReaderConfigError(
            "enabled borderô reader must contain one PTT/UBBO and one ODI/Saldanha route"
        )

    return BorderoReaderConfig(enabled=True, routes=routes)


def route_for_message(
    message: Mapping[str, Any], config: BorderoReaderConfig
) -> BorderoRoute | None:
    """Return a route only for an exact allowlisted group JID.

    ``isGroup`` is checked independently so a malformed or forged DM payload
    can never inherit a group route merely by reusing its JID.
    """

    if not config.enabled or message.get("isGroup") is not True:
        return None
    group_jid = message.get("chatId")
    if not isinstance(group_jid, str):
        return None
    return config.routes.get(group_jid)


def build_ingest_prompt(route: BorderoRoute) -> str:
    """Build bounded operator context for one silent Borderô turn."""

    return (
        "[ROTA BORDERÔ AUTOMÁTICA]\n"
        f"Origem: grupo WhatsApp {route.group_jid}; loja {route.location}; "
        f"empresa {route.store}.\n"
        "Processar esta entrada e anexos com a skill braza-operations e o "
        "runtime versionado do Borderô. Agrupar mensagens do mesmo borderô "
        "antes de qualquer conclusão; preservar idempotência e todos os gates "
        "documentais.\n"
        "Este leitor é somente leitura: não responder no WhatsApp, não enviar "
        "mídia, reação, presença ou comando para WhatsApp. Não usar clarify para "
        "conversar no WhatsApp; se faltar evidência ou houver dúvida material, "
        "bloquear e relatar a exceção.\n"
        f"O gateway encaminha automaticamente o relatório final ou bloqueio ao "
        f"tópico Telegram exato {route.telegram_target}; não tente escolher outro "
        "transporte nem fazer envio manual."
    )


def is_configured_bordero_group(chat_id: Any, config: BorderoReaderConfig) -> bool:
    """Return whether ``chat_id`` is one of the exact configured group JIDs."""

    return isinstance(chat_id, str) and config.enabled and chat_id in config.routes


def is_allowed_telegram_egress_target(
    target: Any, route: BorderoRoute | None
) -> bool:
    """Return whether a Borderô turn may use ``send_message`` for egress."""

    return isinstance(target, str) and route is not None and target == route.telegram_target


def bordero_send_message_block(
    tool_name: Any,
    args: Mapping[str, Any] | None,
    *,
    platform: Any,
    chat_id: Any,
    config: BorderoReaderConfig,
) -> str | None:
    """Return a block reason for unsafe cross-platform sends from Borderô.

    This is called by Hermes' ``pre_tool_call`` hook, before the generic
    ``send_message`` tool can resolve a target or touch a transport.
    """

    if tool_name != "send_message" or _platform_name(platform) != "whatsapp" or not config.enabled:
        return None

    route = config.routes.get(chat_id) if isinstance(chat_id, str) else None
    if route is None:
        return "BLOCKED: WhatsApp Borderô sessions may only use a configured Borderô group"

    action = args.get("action", "send") if isinstance(args, Mapping) else "send"
    if action != "send":
        return "BLOCKED: WhatsApp Borderô reader permits only Telegram report delivery"

    target = args.get("target") if isinstance(args, Mapping) else None
    if not is_allowed_telegram_egress_target(target, route):
        return (
            "BLOCKED: Borderô report target must be the exact configured Telegram topic "
            f"{route.telegram_target}"
        )
    return None
