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

# Transport-ambiguity text. A lost ack means the frame MAY have been delivered,
# so it is never an authorization refusal. Needed because the text-only branch
# of `declined_send` sees no structured `ambiguous` flag to check.
_AMBIGUOUS_MARKER = "ack lost"


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


def declined_send(result: Any) -> bool:
    """True when a SEND RESULT object carries a destination refusal.

    ``is_egress_decline`` classifies a raw connector DICT. Callers hold a
    ``SendResult``, and eight call sites each hand-rolled the unwrapping — with
    two different answers. Six checked only ``raw_response``; two also checked
    the error text. That disagreement was a real defect, not mere repetition:
    a connector that answers with the uniform decline SENTENCE and no
    structured code (the documented contract for older connectors, see
    ``_approval_send_outcome``) was classified as an ordinary failure by those
    six, so the lane treated a refusal as "editing unavailable" and retried.

    Measured: text-only decline -> six-site check False, two-site check True.

    Both shapes are authoritative, so both are checked here, once:

    * ``raw_response`` when the connector sent a structured body,
    * otherwise the error text against the uniform decline marker.

    The latch in ``RelayAdapter`` already catches both (it classifies the
    transport dict directly), which is why the inconsistency cost verdicts and
    futile retries rather than leaked content. This helper makes the gateway
    lanes agree with the wire.
    """
    raw = getattr(result, "raw_response", None)
    if isinstance(raw, dict):
        return is_egress_decline(raw)
    error = getattr(result, "error", None)
    if not error:
        return False
    if _AMBIGUOUS_MARKER in str(error).lower():
        # DEFENCE IN DEPTH for a lost `ambiguous` flag. The text branch exists
        # only for connectors that send no structured body, so it cannot see
        # `ambiguous` — and a projection that drops `raw_response` therefore
        # turned "ack lost" into a definite refusal, terminating the run for a
        # frame that may well have been delivered. Any error text that says the
        # ack was lost is treated as a transport outcome, never authorization.
        return False
    return is_egress_decline({"success": False, "error": error})


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

class RelayRouteUnknown(RuntimeError):
    """Relay routing could not be determined — callers must FAIL CLOSED.

    Distinct from "no relay is configured", which is an empty set and means the
    guard does not apply. This is "we could not find out", and the two must not
    share a return value: an empty set here says "not relay-routed", which
    skips authorization entirely.
    """


def _is_missing_gateway_relay(exc: ImportError) -> bool:
    """True only when the gateway relay package ITSELF is absent.

    `ImportError.name` is the module that could not be found. A nested
    dependency failing is a broken installation, not "there is no relay here",
    and the two must not share a verdict.
    """
    # Absence is ModuleNotFoundError WITH a name; anything else is a fault.
    if not isinstance(exc, ModuleNotFoundError):
        return False
    return getattr(exc, "name", None) in (
        "gateway",
        "gateway.relay",
        "gateway.relay.egress",
    )


def _live_relay_fronted() -> Optional[Set[str]]:
    """The connected relay adapter's OWN fronted set.

    Returns None ONLY for genuine absence — no runner, or no relay adapter in
    it. A live adapter that cannot answer raises `RelayRouteUnknown`.

    ABSENCE vs FAULT, again. `None` here means "fall back to the config
    snapshot", so letting a FAULT return it re-created the bypass this function
    exists to close: with a live adapter whose `fronts_platform()` raised and an
    empty/stale config snapshot, the guard concluded "not relay-routed" and
    authorized an unattested destination, while `resolve_delivery_transport`
    asks that same adapter and still routes over the relay. Measured:
    `relay_routed=False`, verdict `None`.
    """
    # FAIL CLOSED BY DEFAULT. Five review rounds found this same defect at five
    # different boundaries — the call, the attribute lookup, a non-callable
    # attribute, the nested imports, and the adapter-registry `.get()` — because
    # the old shape asked "did something go wrong?" and answered `None`, which
    # MEANS "no live adapter, use the config snapshot". Every new statement was
    # a new chance to fail open, so the fix kept relocating instead of closing.
    #
    # Inverted here: every `return None` is guarded by an explicit narrow check
    # that cannot itself be the fault, and the outer handler turns anything else
    # into `RelayRouteUnknown`. So a statement added inside this function is
    # fail-CLOSED by default; previously it was fail-open by default.
    try:
        try:
            from gateway.config import Platform
            from gateway.run import _gateway_runner_ref
        except ImportError as exc:
            # Only the gateway package being genuinely absent is benign; a
            # nested dependency failing to load is a broken install, i.e. a
            # fault. Same rule as `_relay_fronted` below.
            if _is_missing_gateway_relay(exc):
                return None
            raise

        runner = _gateway_runner_ref()
        if runner is None:
            return None  # no gateway runner in this process (CLI, cron)

        registry = getattr(runner, "adapters", None)
        if not registry:
            return None  # a runner with no adapters at all

        relay = registry.get(Platform.RELAY)
        if relay is None:
            return None  # native-only deployment: nothing fronts anything

        # A PRESENT adapter must answer. Anything below this line that fails is
        # a fault, never an absence.
        fronts = relay.fronts_platform
        return {
            str(p.value).strip().lower()
            for p in Platform
            if str(getattr(p, "value", "")).lower() != "relay" and fronts(p)
        }
    except Exception as exc:  # noqa: BLE001 - anything unproven is UNKNOWN
        logger.exception("could not determine the live relay adapter's fronted set")
        raise RelayRouteUnknown(
            "the connected relay adapter could not report which platforms it "
            "fronts, so this destination's routing could not be determined"
        ) from exc


def _relay_fronted() -> Set[str]:
    """Platforms the connector fronts for this gateway.

    ABSENCE vs FAULT is the whole point of the split below. No gateway relay
    module means there is no relay egress to authorize, so an empty set is the
    honest answer. Any OTHER failure — a config read that raised, a broken
    dependency inside the module — means routing is UNKNOWN, and returning an
    empty set there silently reclassifies a relay platform as native and
    bypasses the guard. Review demonstrated exactly that: with discovery
    raising, an unattested target was authorized.
    """
    try:
        from gateway.relay import relay_fronted_platforms
    except ImportError as exc:
        # Only the ABSENCE of the relay module itself is benign. An ImportError
        # naming a nested dependency means an installed module failed to load —
        # a FAULT — and returning an empty set there reclassifies every relay
        # platform as native. Review probed this with
        # `ImportError.name = "gateway.relay.dependency"` and got an authorized
        # verdict, so `except ImportError` alone was not the fix it looked like.
        if not _is_missing_gateway_relay(exc):
            raise RelayRouteUnknown(
                f"relay module failed to import: {exc}"
            ) from exc
        return set()

    # PREFER THE LIVE ADAPTER'S OWN ANSWER. `resolve_delivery_transport` asks
    # the CONNECTED relay adapter (`fronts_platform`, from the identity set sent
    # at handshake); reconstructing routing from mutable environment state is a
    # DIFFERENT snapshot, and the two disagree whenever GATEWAY_RELAY_PLATFORMS
    # changes or is momentarily absent after connect. Review probed exactly
    # that: guard said "not relay-routed", delivery routed relay, and the guard
    # was skipped for a relay send. Same source, same answer.
    live = _live_relay_fronted()
    if live is not None:
        return live
    try:
        # CASE-NORMALIZED. `relay_routed_platform` lowercases the requested
        # name, so an un-normalized set made a mixed-case configured platform
        # ("Discord") miss the membership test and look native — an
        # attestation bypass on a string comparison.
        return {str(p).strip().lower() for p in relay_fronted_platforms()}
    except Exception as exc:  # noqa: BLE001 - routing unknown; never assume native
        raise RelayRouteUnknown(
            f"relay route discovery failed: {exc}"
        ) from exc


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
        platform = Platform(platform_name)
        if adapters.get(platform) is None:
            return False
        # MATCH `resolve_delivery_transport` EXACTLY. A native adapter that is
        # explicitly DISABLED never shadows Relay there, so treating mere
        # presence as "native" made the guard believe a send was native while
        # delivery actually routed it over the relay — the guard then skipped
        # authorization for a relay send. Two independent routing classifiers
        # are unsafe; this one now answers the same question the router does.
        from gateway.config import load_gateway_config

        # NO `except: return True` HERE. I wrote one, and it recreated the very
        # bypass this method was fixed for: a config read that fails would
        # declare the platform native while the router — reading the real
        # config — sends over the relay, so the guard is skipped for a relay
        # send. A routing question we cannot answer is UNKNOWN, and the caller
        # turns that into a refusal.
        try:
            native_config = load_gateway_config().platforms.get(platform)
        except Exception as exc:  # noqa: BLE001 - routing unknown; never assume native
            raise RelayRouteUnknown(
                f"native-adapter config lookup failed for {platform_name}: {exc}"
            ) from exc
        return native_config is None or bool(getattr(native_config, "enabled", True))
    except RelayRouteUnknown:
        # Routing is UNKNOWN, not "no native adapter" — must not be flattened
        # into False here, which would send the caller down the relay-guard
        # path on a guess. Propagate; authorize_relay_target turns it into a
        # refusal.
        raise
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
            raw = str(entry["id"])
            ids.add(raw)
            # RECOVER THE CHAT FROM THE STRUCTURED FIELD, NEVER BY SPLITTING.
            # `_session_entry_id` builds the id as f"{chat_id}:{thread_id}"
            # when a thread exists, and the entry still carries `thread_id`
            # separately — so the parent is knowable exactly. Splitting on ":"
            # was a guess that INVENTED attestations: a Matrix room id
            # `!room:server.org` attested a bare `!room` no session ever used.
            # An allow-list of "platforms whose ids have no native colon" would
            # only have narrowed the guess; the structured field removes it.
            if entry.get("thread_id"):
                suffix = f":{entry['thread_id']}"
                if raw.endswith(suffix):
                    ids.add(raw[: -len(suffix)])
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


_UNSET = object()


def _has_native_credential(platform_name: str) -> bool:
    """Whether the gateway itself holds a token that can send to *platform_name*.

    `_send_to_platform` sends Telegram natively via `pconfig.token`, entirely
    outside the connector. When that credential exists, no connector-side
    authorization happens, so the unresolved-handle exemption has no basis.
    Faults answer True — the SAFE direction, since it only ever withdraws an
    exemption and falls back to the ordinary attestation check.
    """
    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        pconfig = config.platforms.get(Platform(platform_name))
        return bool(pconfig and pconfig.enabled and getattr(pconfig, "token", None))
    except Exception:  # noqa: BLE001 - a fault must not GRANT the exemption
        logger.debug("native-credential probe failed; withdrawing the exemption", exc_info=True)
        return True


def _is_unresolved_handle(platform_name: str, target: str, native_token: Any = _UNSET) -> bool:
    """Whether *target* is a NAME the gateway cannot compare against an id.

    Provenance records RESOLVED destinations (numeric chat ids). A Telegram
    public `@username` is not a destination yet — the Bot API resolves it at
    send time — so comparing it to a set of numeric ids can only ever refuse,
    no matter how legitimately the user configured it.

    Deliberately narrow: `@`-prefixed Telegram targets only. A numeric id, a
    `-100…` supergroup, or any other platform's form is a resolved destination
    and stays fully guarded.
    """
    if platform_name != "telegram" or not target.startswith("@"):
        return False
    # AND THE SEND MUST ACTUALLY REACH THE CONNECTOR. The carve-out's entire
    # justification is "the connector holds the token and authorizes this".
    # That is false whenever the tool sends NATIVELY: `_send_to_platform`
    # calls `_send_telegram(pconfig.token, ...)` directly, so with a native
    # Telegram token configured, an unattested @handle went out with the
    # gateway's own credential and no connector ever saw it. Review probed it:
    # the numeric control was refused, the handle was delivered.
    #
    # So the exemption survives only when there is NO native credential able to
    # send this handle. `_send_to_platform` reaches for `pconfig.token`; if that
    # exists, delivery never involves the connector.
    if native_token is not _UNSET:
        # The caller passed the token from the SAME config snapshot it will
        # dispatch with; that is authoritative and race-free.
        return not native_token
    return not _has_native_credential(platform_name)


def authorize_relay_target(
    platform_name: str, chat_id: Any, thread_id: Any = None, *, native_token: Any = _UNSET
) -> Optional[str]:
    """Return an error string when this relay destination may not be named.

    ``None`` means the send may proceed. Non-relay platforms are never
    restricted here — their own adapters own their authorization.

    THREAD IDS ARE PART OF THE DESTINATION, not a formatting detail. On Discord
    the thread is the literal REST target
    (``POST /channels/{thread_id}/messages``), so authorizing only the parent
    let an attested channel vouch for an arbitrary caller-supplied thread. The
    thread must therefore carry its own attestation — a session in the parent
    is not evidence of a session in the thread.
    """
    try:
        routed = relay_routed_platform(platform_name)
    except RelayRouteUnknown as exc:
        # Routing could not be determined. "Not relay-routed" would skip this
        # guard entirely, so an unknown route must refuse rather than assume
        # the safe-looking default. Returned as a refusal string (not raised)
        # because every caller treats this function's output as the verdict.
        logger.warning(
            "relay routing unknown for %s — refusing the send: %s",
            platform_name,
            exc,
        )
        return (
            f"Refusing to send to '{platform_name}': relay routing could not "
            "be determined, so this destination could not be verified."
        )
    if not routed:
        return None
    target = str(chat_id or "").strip()
    if not target:
        return None
    name = str(platform_name).strip().lower()
    attested = attested_relay_targets(name)
    if target in attested:
        # The CHAT is attested. If the caller also named a thread, that thread
        # is the real destination on thread-addressed platforms, so it needs an
        # attestation OF ITS OWN, BOUND TO THIS PARENT.
        #
        # A bare `thread in attested` arm used to satisfy this, and it proved
        # nothing about parentage: any attested chat whose id happened to equal
        # the requested thread id vouched for it. Measured — attested
        # `{"-100A", "7"}`, request `(-100A, thread 7)` → authorized, though no
        # session ever existed in thread 7 of -100A.
        #
        # The bound form is always available, so nothing legitimate needs the
        # bare one: `_session_entry_id` records a threaded origin as
        # f"{chat_id}:{thread_id}", and a platform whose thread IS its own
        # channel (a Discord thread addressed directly) arrives as `chat_id`
        # and is authorized by the `target in attested` check above.
        thread = str(thread_id or "").strip()
        if thread and thread != target:
            if f"{target}:{thread}" in attested:
                return None
            return (
                f"Refusing to send to relay target '{name}:{target}:{thread}': "
                f"the parent chat is attested but this gateway has no record of "
                f"the thread, and on this platform the thread IS the delivery "
                f"destination."
            )
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
    if _is_unresolved_handle(name, target, native_token):
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
