"""Conversation location and kind, independent of session isolation and legacy key spelling."""

import re
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class ConversationAddress:
    platform: str
    chat_id: str
    thread_id: Optional[str] = None
    scope_id: Optional[str] = None
    profile: Optional[str] = None
    account: Optional[str] = None

    def __post_init__(self):
        for value in (self.platform, self.chat_id):
            if not isinstance(value, str):
                raise ValueError("Conversation platform and chat ID must be strings")
        for value in (self.thread_id, self.scope_id, self.profile, self.account):
            if value is not None and not isinstance(value, str):
                raise ValueError("Conversation address fields must be strings or null")


def conversation_kind(chat_type: str) -> str:
    # Old adapters used "thread" as a kind; topology now lives in the address.
    kind = {"thread": "group", "forum": "group", "room": "channel", "c2c": "dm", "guild": "group", "private": "dm"}.get(chat_type, chat_type)
    if kind not in {"dm", "group", "channel"}:
        raise ValueError(f"Unknown conversation kind: {chat_type!r}")
    return kind


def address_of(source) -> ConversationAddress:
    from gateway.session_identity import identity_of, transport_profile_of

    identity = identity_of(source)
    is_dm = (getattr(source, "conversation_kind", None) or source.chat_type) == "dm"
    thread = source.thread_id or (None if is_dm else getattr(source, "prospective_thread_id", None))
    parent = getattr(source, "parent_chat_id", None) or source.chat_id
    return ConversationAddress(
        source.platform.value, str(parent), str(thread) if thread else None,
        getattr(source, "scope_id", None), identity.runtime_profile if identity else getattr(source, "profile", None),
        transport_profile_of(source),
    )


def conversation_origin(source, *, thread_id: Optional[str]) -> dict:
    from dataclasses import replace

    address = replace(address_of(source), thread_id=thread_id)
    kind = conversation_kind(source.conversation_kind or source.chat_type)
    origin = {"version": 1, "address": asdict(address), "kind": kind}
    if source.chat_type != kind:
        origin["key_chat_type"] = source.chat_type
    return origin


def source_for_address(address: ConversationAddress, kind: str, *, key_chat_type=None, **fields):
    """Encode an address using existing adapter/key layouts, without guessing its kind."""
    from gateway.config import Platform
    from gateway.session import SessionSource

    kind = conversation_kind(kind)
    chat_type = key_chat_type or kind
    if (conversation_kind(chat_type) == "dm") != (kind == "dm"):
        raise ValueError("Legacy key spelling cannot change DM isolation")
    chat_id = address.chat_id
    if address.thread_id and kind != "dm" and address.platform == "discord":
        chat_type, chat_id = "thread", address.thread_id
    elif address.platform == "discord" and kind != "dm":
        chat_type = "group"
    return SessionSource(
        platform=Platform(address.platform), chat_id=chat_id, chat_type=chat_type,
        thread_id=address.thread_id, parent_chat_id=address.chat_id if address.thread_id else None,
        scope_id=address.scope_id, profile=address.profile, conversation_kind=kind, **fields,
    )


def receiving_adapter_for_origin(adapter, platform: str, chat_id: str, origin: dict):
    """A captured receiver takes precedence over the runtime profile's own bot."""
    captured = origin.get("conversation")
    if (not isinstance(captured, dict) or captured.get("version") != 1
            or origin.get("platform") != platform
            or str(origin.get("parent_chat_id") or origin.get("chat_id", "")) != str(chat_id)):
        return adapter
    address = ConversationAddress(**captured["address"])
    runner = getattr(adapter, "gateway_runner", None)
    if not address.account or not getattr(getattr(runner, "config", None), "multiplex_profiles", False):
        return adapter
    if (address.platform, address.chat_id) != (platform, str(chat_id)):
        raise PermissionError("Cron conversation address disagrees with its delivery target")
    from gateway.session_identity import restore_identity
    source = source_for_address(address, captured["kind"], user_id=origin.get("user_id"),
                                user_id_alt=origin.get("user_id_alt"), key_chat_type=captured.get("key_chat_type") or origin.get("chat_type"))
    restore_identity(source, runner=runner, transport_profile=address.account)
    receiving = runner._delivery_adapter_for(source)
    if receiving is None:
        raise PermissionError("The cron conversation's receiving bot is offline; another bot cannot replace it")
    return receiving


async def resolve_delivery_source(adapter, platform: str, chat_id: str, thread_id: Optional[str], origin: dict):
    """Resolve a reply-facing source through the receiving adapter; ambiguous legacy origins refuse."""
    same_chat = (str(origin.get("platform", "")).lower() == platform
                 and str(origin.get("parent_chat_id") or origin.get("chat_id", "")) == str(chat_id))
    resolved_thread_id = thread_id
    captured = origin.get("conversation") if same_chat else None
    if captured is not None:
        if not isinstance(captured, dict) or captured.get("version") != 1:
            raise PermissionError("Unsupported cron conversation origin; recreate or repair the job")
        stored = captured.get("address")
        if not isinstance(stored, dict) or stored.get("platform") != platform or stored.get("chat_id") != str(chat_id):
            raise PermissionError("Cron conversation address disagrees with its delivery target")
        ConversationAddress(**stored)
        kind = conversation_kind(captured.get("kind"))
        key_chat_type = captured.get("key_chat_type") or origin.get("chat_type")
    else:
        stored = {}
        import inspect
        scope = origin.get("scope_id") if same_chat else None
        parameters = inspect.signature(adapter.get_chat_info).parameters
        kwargs = {"scope_id": scope} if scope and "scope_id" in parameters else {}
        if "metadata" in parameters:
            kwargs["metadata"] = {key: origin[key] for key in ("scope_id", "profile", "user_id")
                                  if same_chat and origin.get(key)}
            kwargs["metadata"]["platform"] = platform
        info = await adapter.get_chat_info(str(chat_id), **kwargs)
        if not isinstance(info, dict) or info.get("error"):
            raise ValueError("Adapter could not resolve the legacy cron conversation")
        if info.get("type") not in (None, "unknown"):
            kind = conversation_kind(info.get("type"))
            key_chat_type = info.get("type")
            stored["scope_id"] = scope or info.get("scope_id")
            stored["scope_id"] = stored["scope_id"] or info.get("guild_id")
            stored["chat_id"] = info.get("parent_chat_id") or info.get("chat_id") or str(chat_id)
            resolved_thread_id = thread_id or info.get("thread_id")
            if thread_id and info.get("thread_id") and str(thread_id) != str(info["thread_id"]):
                raise PermissionError("Adapter conversation disagrees with the requested thread")
        else:
            store = getattr(adapter, "_session_store", None)
            entry = (existing_delivery_session(store, platform, chat_id, thread_id, origin)
                     if store is not None else None)
            if entry is None:
                raise ValueError("Adapter and durable state could not resolve the legacy cron conversation")
            stored = asdict(address_of(entry.origin))
            stored["account"] = entry.transport_profile
            kind = conversation_kind(entry.origin.conversation_kind or entry.origin.chat_type)
            key_chat_type = entry.origin.chat_type
    if platform == "slack" and not stored.get("scope_id"):
        raise ValueError("Cannot seed a Slack conversation without its workspace scope")
    address = ConversationAddress(
        platform, stored.get("chat_id") or str(chat_id), str(resolved_thread_id) if resolved_thread_id else None,
        stored.get("scope_id") or (origin.get("scope_id") if same_chat else None),
        stored.get("profile") or (origin.get("profile") if same_chat else None),
        stored.get("account"),
    )
    return _canonical_delivery_source(
        adapter,
        address, kind, key_chat_type=key_chat_type, user_id=origin.get("user_id") if same_chat else None,
        user_id_alt=origin.get("user_id_alt") if same_chat else None,
        chat_name=origin.get("chat_name") if same_chat else None,
    )


def _canonical_delivery_source(adapter, address, kind, *, key_chat_type=None, **fields):
    from dataclasses import replace
    from gateway.session_identity import identity_of

    # A persisted runtime is an expectation, never an input that bypasses current routes.
    source = source_for_address(replace(address, profile=None), kind, key_chat_type=key_chat_type, **fields)
    adapter._canonicalize(source)
    if source.profile_route_rejected:
        raise PermissionError("Cron conversation routes to an unserved profile")
    identity = identity_of(source)
    if address.account and (identity is None or identity.transport_profile != address.account):
        raise PermissionError("Cron conversation belongs to another bot account")
    if address.profile and identity and identity.runtime_profile != address.profile:
        raise PermissionError("Cron conversation belongs to another runtime profile")
    return source


def resolve_reply_source(adapter, source, thread_id):
    """Resolve the final reply address anew; changing topology cannot carry a pinned identity."""
    from dataclasses import replace

    return _canonical_delivery_source(
        adapter, replace(address_of(source), thread_id=thread_id),
        source.conversation_kind or source.chat_type,
        key_chat_type=source.chat_type, user_id=source.user_id, user_id_alt=source.user_id_alt, chat_name=source.chat_name)


def persisted_session_entry(store, session_key, platform):
    """Read an exact durable lane without requiring the launch gateway's routing index."""
    import json
    from dataclasses import replace
    from datetime import datetime
    from gateway.session import SessionSource

    entry = store.lookup_by_session_key(session_key)
    if entry is not None and entry.origin is not None:
        return entry if entry.origin.platform.value == platform else None
    row = store._peer_row(store._db, source=platform, session_key=session_key, raise_on_lookup_error=True)
    if not row or row.get("session_key") != session_key or not row.get("origin_json"):
        return None
    source = SessionSource.from_dict(json.loads(row["origin_json"]))
    if source.platform.value != platform:
        return None
    entry = store._create_entry_from_recovered_row(row=row, session_key=session_key, source=source, now=datetime.now())
    return replace(entry, transport_profile=row.get("transport_profile"))


def _delivery_session_key(store, source, participant, participant_alt, account, profile):
    from dataclasses import replace
    from gateway.session import build_session_key

    key = build_session_key(
        replace(source, user_id=participant, user_id_alt=participant_alt),
        store.config.group_sessions_per_user, store.config.thread_sessions_per_user,
        profile=profile if store.config.multiplex_profiles else None)
    if store.config.multiplex_profiles and account and profile and account != profile:
        key += f":account~:{account}"
    return key


def existing_delivery_session(store, platform: str, chat_id: str, thread_id: Optional[str], origin: dict):
    """Compatibility for sends without a live adapter: one exact address and allowed participant."""
    from gateway.session import is_shared_multi_user_session

    same_chat = (origin.get("platform") == platform
                 and str(origin.get("parent_chat_id") or origin.get("chat_id", "")) == str(chat_id))
    captured = origin.get("conversation") if same_chat else None
    if captured is not None and (not isinstance(captured, dict) or captured.get("version") != 1):
        return None
    expected = captured.get("address", {}) if captured else (origin if same_chat else {})
    if captured:
        parsed = ConversationAddress(**expected)
        if (parsed.platform, parsed.chat_id) != (platform, str(chat_id)):
            return None
    participant = (origin.get("user_id_alt") or origin.get("user_id")) if same_chat else None
    kind = captured.get("kind") if captured else (origin.get("chat_type") if same_chat else None)
    entries = store.list_sessions()
    if kind:
        source = source_for_address(
            ConversationAddress(platform, str(chat_id), thread_id, expected.get("scope_id"), expected.get("profile")),
            kind, key_chat_type=captured.get("key_chat_type") if captured else origin.get("chat_type") if same_chat else None,
            user_id=origin.get("user_id") if same_chat else None,
            user_id_alt=origin.get("user_id_alt") if same_chat else None)
        key = _delivery_session_key(store, source, source.user_id, source.user_id_alt,
                                    expected.get("account"), expected.get("profile") or store._active_profile_name())
        durable = persisted_session_entry(store, key, platform)
        if durable is not None and not any(e.session_id == durable.session_id for e in entries):
            entries.append(durable)
    candidates = []
    for entry in entries:
        source = entry.origin
        if source is None:
            continue
        address = address_of(source)
        if (address.platform, address.chat_id, address.thread_id) != (platform, str(chat_id), thread_id):
            continue
        if kind and (conversation_kind(kind) == "dm") != (conversation_kind(source.conversation_kind or source.chat_type) == "dm"):
            continue
        if expected.get("account") and entry.transport_profile != expected["account"]:
            continue
        if expected.get("scope_id") and address.scope_id != expected["scope_id"]:
            continue
        profile = (store._profile_from_session_key(entry.session_key) if store.config.multiplex_profiles
                   else store._active_profile_name())
        if expected.get("profile") and profile != expected["profile"]:
            continue
        shared = is_shared_multi_user_session(
            source, group_sessions_per_user=store.config.group_sessions_per_user,
            thread_sessions_per_user=store.config.thread_sessions_per_user)
        if source.chat_type != "dm" and not shared:
            if not participant or participant != (source.user_id_alt or source.user_id):
                continue
        key_type = ((captured.get("key_chat_type") or origin.get("chat_type")) if captured
                    else origin.get("chat_type") if same_chat else None) or source.chat_type
        from dataclasses import replace
        reply_source = source_for_address(replace(address, profile=profile), kind or source.conversation_kind or source.chat_type,
                                          key_chat_type=key_type)
        key = _delivery_session_key(
            store, reply_source, origin.get("user_id") if same_chat else None,
            origin.get("user_id_alt") if same_chat else None, entry.transport_profile, profile)
        legacy_key = key.removesuffix(f":account~:{entry.transport_profile}") if entry.transport_profile else None
        if entry.session_key not in (key, legacy_key):
            # A changed isolation policy does not authorize writing into the old lane.
            continue
        candidates.append(entry)
    return candidates[0] if len(candidates) == 1 else None


_PROFILE_ID_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _parse_session_key(session_key: str) -> dict | None:
    """Parse a session key (``agent:{ns}:{platform}:{chat_type}:{chat_id}[:{extra}...]``).

    ``{ns}`` is ``main`` for the default profile, ``main~`` for a profile literally named ``main``
    (``gateway.session._session_key_namespace``), or a named-profile id (profile ids match
    ``[a-z0-9][a-z0-9_-]{0,63}`` — never contain ``:`` — so a plain split stays unambiguous).
    For group/channel sessions the suffix may be a user_id, not a thread_id, so ``thread_id``
    is omitted. Named profiles are reported as ``profile``; ``main`` keys keep their historical
    shape exactly (no ``profile`` key) so equality assertions on parsed dicts stay stable.
    """
    if ":account~:" in session_key:
        return None  # A receiving bot requires a persisted origin, never a raw-key guess.
    from gateway.session import profile_from_session_key_namespace
    parts = session_key.split(":")
    if (
        len(parts) >= 5
        and parts[0] == "agent"
        and (parts[1] in ("main", "main~") or _PROFILE_ID_KEY_RE.match(parts[1]))
    ):
        result = {"platform": parts[2], "chat_type": parts[3], "chat_id": parts[4]}
        if parts[1] != "main":
            result["profile"] = profile_from_session_key_namespace(parts[1])
        if len(parts) > 5 and parts[3] in {"dm", "thread"}:
            result["thread_id"] = parts[5]
        return result
    return None
