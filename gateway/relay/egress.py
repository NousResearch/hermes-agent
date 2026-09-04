"""Gateway-side relay EGRESS AUTHORIZATION (P5).

Two halves of one security surface, kept together because they are the same
concern seen from both ends of the wire:

**(a) Target hygiene.** The ``send_message`` model tool takes a free-form
``target`` string (``'platform:chat_id'``). Nothing stopped a model from
naming an arbitrary chat id and having the gateway emit an outbound frame for
it. The connector now refuses such a frame (its egress-authorization floor),
but the gateway must not *silently* ask: a destination the gateway has no
record of is refused HERE, with a visible tool error naming the target.
:func:`authorize_relay_target` is that guard; :func:`attested_relay_targets`
is the set of destinations this gateway can show a provenance for (operator
home channel, channel directory, its own gateway session origins).

**(b) Decline visibility.** The connector answers an unauthorized destination
with a DEFINITE, non-ambiguous failure whose text is deliberately UNIFORM
(``"<platform> egress declined: target is not an approved destination for
this connection"``) — it must not leak whether the destination belongs to
another tenant or to nobody. :func:`is_egress_decline` recognises THAT a
decline happened; it never tries to parse WHY. Egress lanes that legitimately
degrade a *transport drop* (advisory progress, cosmetic reactions, media
falling back to a text notice) must NOT degrade a decline — a refusal that
turns into a different op, or into a wrong "op unavailable" reason, is a
security event laundered into an apparent success.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# The connector stamps this on a structured decline (preferred signal).
EGRESS_DECLINE_CODE = "egress_declined"

# Fallback signal: the connector's uniform decline sentence. Matched
# case-insensitively on this fragment ONLY — the rest of the sentence is
# deliberately uninformative (finding F-005) and must not be parsed.
EGRESS_DECLINE_MARKER = "egress declined:"


def is_egress_decline(result: Any) -> bool:
    """True when *result* is the connector REFUSING the destination.

    A decline is distinguished from every other outbound failure by three
    properties, all required:

    * it failed (``success`` is falsey),
    * it is DEFINITE — an ``ambiguous`` result (lost ack, mid-write drop) says
      the frame may well have been applied, so it is a transport outcome, not
      an authorization one,
    * it carries the connector's decline code, or its uniform decline text.
    """
    if not isinstance(result, dict):
        return False
    if result.get("success"):
        return False
    if result.get("ambiguous"):
        return False
    if str(result.get("code") or "") == EGRESS_DECLINE_CODE:
        return True
    return EGRESS_DECLINE_MARKER in str(result.get("error") or "").lower()


def decline_error(result: Any) -> str:
    """The connector's decline text, verbatim, for surfacing to the caller.

    Verbatim on purpose: the gateway's job is to report faithfully THAT a
    decline happened, not to explain or re-word it.
    """
    if isinstance(result, dict):
        error = result.get("error")
        if error:
            return str(error)
    return "relay egress declined"


def log_decline(op: str, chat_id: Any, result: Any) -> None:
    """Record a decline on a lane whose contract cannot carry an error.

    Cosmetic lanes (typing, reactions, delete, thread ops) return ``bool`` /
    ``None`` by contract and legitimately degrade. A decline there is still a
    security-relevant event, so it is logged at WARNING rather than vanishing
    into the lane's debug-level best-effort handling.
    """
    logger.warning(
        "relay %s DECLINED for %s: %s", op, chat_id, decline_error(result)
    )


# ---------------------------------------------------------------------------
# (a) target attestation
# ---------------------------------------------------------------------------

def _relay_fronted() -> Set[str]:
    try:
        from gateway.relay import relay_fronted_platforms

        return {str(p) for p in relay_fronted_platforms()}
    except Exception:  # noqa: BLE001 - env/config absence must never break a send
        return set()


def _has_live_native_adapter(platform_name: str) -> bool:
    """Whether THIS process runs a native (non-relay) adapter for the platform.

    Mirrors ``gateway/delivery.resolve_delivery_transport``'s precedence: a
    concrete native adapter always wins, so a platform served natively here is
    not a relay egress and this guard does not apply to it.
    """
    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        if runner is None:
            return False
        adapters = getattr(runner, "adapters", None) or {}
        return adapters.get(Platform(platform_name)) is not None
    except Exception:  # noqa: BLE001 - no runner (cron/CLI) ⇒ no native adapter
        return False


def relay_routed_platform(platform_name: str) -> bool:
    """Whether a send to *platform_name* would egress over the relay connector.

    True for the generic ``relay`` plane itself, and for any logical platform
    the connector fronts for this gateway that has no live native adapter in
    this process.
    """
    name = str(platform_name or "").strip().lower()
    if not name:
        return False
    if name == "relay":
        return True
    if name not in _relay_fronted():
        return False
    return not _has_live_native_adapter(name)


def _home_channel_id(platform_name: str) -> Optional[str]:
    try:
        from gateway.config import Platform, load_gateway_config

        home = load_gateway_config().get_home_channel(Platform(platform_name))
        return str(home.chat_id) if home and home.chat_id else None
    except Exception:  # noqa: BLE001 - config absence must never break a send
        return None


def _directory_ids(platform_name: str) -> Set[str]:
    try:
        from gateway.channel_directory import load_directory

        entries = load_directory().get("platforms", {}).get(platform_name) or []
    except Exception:  # noqa: BLE001
        return set()
    ids: Set[str] = set()
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id"):
            ids.add(str(entry["id"]))
    return ids


def _session_ids(platform_name: str) -> Set[str]:
    """Chat ids this gateway has actually held a session in for the platform."""
    try:
        from gateway.channel_directory import _build_from_sessions

        entries = _build_from_sessions(platform_name) or []
    except Exception:  # noqa: BLE001
        return set()
    ids: Set[str] = set()
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id"):
            # Session entry ids may be thread-qualified ("chat:thread"); the
            # destination the connector authorizes is the CHAT, so attest both.
            raw = str(entry["id"])
            ids.add(raw)
            ids.add(raw.split(":", 1)[0])
    return ids


def attested_relay_targets(platform_name: str) -> Set[str]:
    """Chat ids this gateway can show a provenance for on *platform_name*.

    Three provenances, all of them things the gateway already knows rather
    than things a model can invent:

    * the operator-configured home channel,
    * the channel directory built from live adapters/session origins,
    * this gateway's own gateway-session origins.

    For the generic ``relay`` plane the union spans every platform the
    connector fronts: a relay session is filed under its LOGICAL platform
    (``source = "discord"``), so attesting ``relay`` against only ``relay``
    would refuse chats the agent is demonstrably already talking in.
    """
    name = str(platform_name or "").strip().lower()
    names = {name}
    if name == "relay":
        names |= _relay_fronted()
    attested: Set[str] = set()
    for candidate in names:
        home = _home_channel_id(candidate)
        if home:
            attested.add(home)
        attested |= _directory_ids(candidate)
        attested |= _session_ids(candidate)
    return attested


def _is_unresolved_handle(platform_name: str, target: str) -> bool:
    """Whether *target* is a NAME the gateway cannot compare against an id.

    Provenance records RESOLVED destinations (numeric chat ids). A Telegram
    public `@username` is not a destination yet — the Bot API resolves it at
    send time — so comparing it to a set of numeric ids can only ever refuse,
    no matter how legitimately the user configured it.

    Deliberately narrow: `@`-prefixed Telegram targets only. A numeric id, a
    `-100…` supergroup, or any other platform's form is a resolved destination
    and stays fully guarded.
    """
    return platform_name == "telegram" and target.startswith("@")


def authorize_relay_target(platform_name: str, chat_id: Any) -> Optional[str]:
    """Return an error string when this relay destination may not be named.

    ``None`` means the send may proceed. Non-relay platforms are never
    restricted here — their own adapters own their authorization.
    """
    if not relay_routed_platform(platform_name):
        return None
    target = str(chat_id or "").strip()
    if not target:
        return None
    name = str(platform_name).strip().lower()
    if target in attested_relay_targets(name):
        return None
    # ── Telegram `@username`: authorized by the CONNECTOR, not here ─────────
    #
    # Checked AFTER attestation, so a handle that IS attested takes the normal
    # path; this only catches the case that would otherwise be a false refusal.
    #
    # WHY THE GATEWAY CANNOT ANSWER THIS: the guard fires only when there is no
    # live native adapter (`_has_live_native_adapter`), i.e. on relay-fronted
    # deployments — and on exactly those the CONNECTOR holds the bot token, not
    # this process. There is no local way to turn `@channel` into the numeric id
    # provenance stores, so refusing here is not "fail closed", it is "fail
    # always". It regressed the public-channel username support added in #53573.
    #
    # WHY THAT IS SAFE, NOT A HOLE: the destination is still authorized, one
    # layer out. The connector's Telegram egress floor (gg#238, merged 743a7c2)
    # classifies and refuses unauthorized destinations after ITS resolution,
    # which is the layer that closed the reported vulnerability in the first
    # place. This carve-out drops handles from two guards to one — the
    # authoritative one — rather than to zero.
    #
    # FOLLOW-UP (option 2, deliberately not done here): resolve the handle
    # before authorizing, so both layers apply. That needs a resolution
    # round-trip through the connector — new wire surface — so it belongs in
    # its own phase, not bolted onto this one.
    if _is_unresolved_handle(name, target):
        logger.debug(
            "relay target '%s:%s' is an unresolved handle — deferring "
            "authorization to the connector's egress floor",
            name,
            target,
        )
        return None
    return (
        f"Refusing to send to unattested relay target '{name}:{target}': "
        "this gateway has no record of that destination. Use "
        "send_message(action='list') to see the targets it can reach."
    )
