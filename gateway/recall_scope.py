"""Canonical recall identity shared by gateway recall and SessionDB SQL.

Live :class:`~gateway.session.SessionSource` metadata and durable
``sessions.origin_json`` describe the same routing location in several shapes.
This module is the one policy that collapses those shapes into a recall
identity and emits the matching SQLite predicate.  Recall intentionally omits
the sender id: per-user gateway sessions inside one chat/thread are peers.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import re
from typing import Any, List, Mapping, Optional, Tuple


_SAFE_IDENTITY_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SAFE_PROFILE_TOKEN = re.compile(r"^(?:default|[a-z0-9][a-z0-9_-]{0,63})$")
_NON_RECALL_PLATFORMS = frozenset(
    {
        "api_server",
        "cli",
        "codex",
        "desktop",
        "gateway",
        "kanban",
        "local",
        "msgraph_webhook",
        "tool",
        "tui",
        "webhook",
    }
)
_RECALL_KINDS = frozenset({"chat", "dm", "dm_thread", "thread"})

GATEWAY_PROXY_MARKER_HEADER = "X-Hermes-Gateway-Proxy"
GATEWAY_PROXY_ORIGIN_HEADER = "X-Hermes-Gateway-Origin"
GATEWAY_PROXY_SESSION_KEY_HEADER = "X-Hermes-Gateway-Session-Key"
GATEWAY_PROXY_CHAT_COMPLETIONS_PATH = (
    "/internal/gateway-proxy/v1/chat/completions"
)
# aiohttp's HTTP parser defaults ``max_field_size`` to 8190 bytes.  Keep the
# wire encoders at that exact accepted value so the client never emits a
# header that our own default server rejects before route dispatch.
_MAX_PROXY_HEADER_VALUE_BYTES = 8190
_PROXY_ORIGIN_FIELDS = (
    "platform",
    "chat_id",
    "chat_type",
    "thread_id",
    "prospective_thread_id",
    "parent_chat_id",
    "user_id",
    "user_id_alt",
    "scope_id",
    "guild_id",
    "profile",
)


def _identity_field_types_valid(
    origin: Mapping[str, Any], fields: Tuple[str, ...] = _PROXY_ORIGIN_FIELDS
) -> bool:
    """Reject typed JSON values that Python would otherwise stringify loosely.

    Missing fields and explicit JSON nulls both mean "not supplied".  Every
    supplied identity value must be a string; booleans, numbers, containers,
    and objects are malformed rather than aliases for their string rendering.
    """
    return all(
        key not in origin or origin.get(key) is None or isinstance(origin.get(key), str)
        for key in fields
    )


def _valid_proxy_header_value(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    try:
        return len(value.encode("ascii")) <= _MAX_PROXY_HEADER_VALUE_BYTES
    except UnicodeEncodeError:
        return False


def _valid_gateway_session_key(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and value
        and value == value.strip()
        and not any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)
    )


@dataclass(frozen=True)
class RecallIdentity:
    """One logical gateway chat/thread recall boundary."""

    platform: str
    chat_kind: str
    chat_type: str
    chat_id: str
    thread_id: str
    scope_id: str
    profile: str
    # Aliases affect the durable SQL match but not live-vs-durable identity
    # equality.  Two WhatsApp JID forms are equal only when both canonicalize
    # to the same chat_id; aliases merely let old origin_json rows prove that
    # equivalence without broadening to another chat.
    chat_aliases: Tuple[str, ...] = field(default=(), compare=False)

    def as_mapping(self) -> Mapping[str, Any]:
        return {
            "platform": self.platform,
            "chat_kind": self.chat_kind,
            "chat_type": self.chat_type,
            "chat_id": self.chat_id,
            "thread_id": self.thread_id,
            "scope_id": self.scope_id,
            "profile": self.profile,
            "chat_aliases": self.chat_aliases,
        }


def is_recall_gateway_platform(platform: object) -> bool:
    """Whether a live source surface requires current-chat recall scoping."""
    value = str(getattr(platform, "value", platform) or "").strip().lower()
    return bool(
        value
        and _SAFE_IDENTITY_TOKEN.fullmatch(value)
        and value not in _NON_RECALL_PLATFORMS
    )


def _minimal_origin(origin: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in _PROXY_ORIGIN_FIELDS:
        value = _text(origin, key)
        if value:
            result[key] = value
    return result


def encode_gateway_proxy_origin(origin: Mapping[str, Any]) -> str:
    """Encode a minimal, validated gateway origin for the proxy wire."""
    if not isinstance(origin, Mapping) or not _identity_field_types_valid(origin):
        raise ValueError("gateway proxy origin contains malformed identity fields")
    minimal = _minimal_origin(origin)
    if canonical_recall_identity(minimal) is None:
        raise ValueError("gateway proxy origin is incomplete or ambiguous")
    payload = json.dumps(
        {"version": 1, "origin": minimal},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    if len(encoded) > _MAX_PROXY_HEADER_VALUE_BYTES:
        raise ValueError("gateway proxy origin is too large")
    return encoded


def decode_gateway_proxy_origin(value: object) -> dict[str, str]:
    """Decode an untrusted proxy origin header, rejecting every loose shape."""
    if not _valid_proxy_header_value(value):
        raise ValueError("gateway proxy origin is missing or too large")
    try:
        assert isinstance(value, str)
        padded = value + "=" * (-len(value) % 4)
        raw = base64.b64decode(padded, altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("gateway proxy origin is malformed") from exc
    if (
        not isinstance(payload, Mapping)
        or type(payload.get("version")) is not int
        or payload.get("version") != 1
    ):
        raise ValueError("gateway proxy origin version is unsupported")
    raw_origin = payload.get("origin")
    if (
        not isinstance(raw_origin, Mapping)
        or set(raw_origin).difference(_PROXY_ORIGIN_FIELDS)
        or not _identity_field_types_valid(raw_origin)
    ):
        raise ValueError("gateway proxy origin is missing")
    origin = _minimal_origin(raw_origin)
    if canonical_recall_identity(origin) is None:
        raise ValueError("gateway proxy origin is incomplete or ambiguous")
    if encode_gateway_proxy_origin(origin) != value:
        raise ValueError("gateway proxy origin is not canonically encoded")
    return origin


def encode_gateway_proxy_session_key(session_key: str) -> str:
    """Encode an exact gateway session key into one parser-safe header."""
    if not _valid_gateway_session_key(session_key):
        raise ValueError("gateway proxy session key is missing or malformed")
    encoded = base64.urlsafe_b64encode(session_key.encode("utf-8")).decode(
        "ascii"
    ).rstrip("=")
    if len(encoded) > _MAX_PROXY_HEADER_VALUE_BYTES:
        raise ValueError("gateway proxy session key is too large")
    return encoded


def decode_gateway_proxy_session_key(value: object) -> str:
    """Decode the exact session key, accepting only canonical wire encoding."""
    if not _valid_proxy_header_value(value):
        raise ValueError("gateway proxy session key is missing or too large")
    try:
        assert isinstance(value, str)
        padded = value + "=" * (-len(value) % 4)
        session_key = base64.b64decode(
            padded, altchars=b"-_", validate=True
        ).decode("utf-8")
    except Exception as exc:
        raise ValueError("gateway proxy session key is malformed") from exc
    if (
        not _valid_gateway_session_key(session_key)
        or encode_gateway_proxy_session_key(session_key) != value
    ):
        raise ValueError("gateway proxy session key is malformed")
    return session_key


def _text(origin: Mapping[str, Any], key: str) -> str:
    value = origin.get(key)
    if not isinstance(value, str):
        return ""
    return value.strip()


def _whatsapp_dm_identity(value: str) -> tuple[str, Tuple[str, ...]]:
    """Return a canonical DM id plus the exact aliases proving equivalence."""
    try:
        from gateway.whatsapp_identity import (
            canonical_whatsapp_identifier,
            expand_whatsapp_aliases,
            normalize_whatsapp_identifier,
        )

        normalized = normalize_whatsapp_identifier(value)
        if not normalized:
            return "", ()
        aliases = {
            normalize_whatsapp_identifier(alias)
            for alias in expand_whatsapp_aliases(normalized)
        }
        aliases.discard("")
        canonical = canonical_whatsapp_identifier(normalized)
        if canonical:
            aliases.add(canonical)
        return canonical or normalized, tuple(sorted(aliases or {normalized}))
    except Exception:
        # Alias lookup is best-effort, but failure narrows to the exact bare
        # identifier.  It must never turn into an unscoped/fuzzy match.
        bare = str(value or "").strip().replace("+", "", 1)
        bare = bare.split(":", 1)[0].split("@", 1)[0]
        return bare, (bare,) if bare else ()


def canonical_recall_identity(origin: object) -> Optional[RecallIdentity]:
    """Normalize live or durable routing metadata, failing closed on ambiguity."""
    if not isinstance(origin, Mapping) or not _identity_field_types_valid(origin):
        return None

    platform = _text(origin, "platform").lower()
    chat_type = _text(origin, "chat_type").lower()
    if (
        not is_recall_gateway_platform(platform)
        or not chat_type
        or not _SAFE_IDENTITY_TOKEN.fullmatch(chat_type)
    ):
        return None

    scope_id = _text(origin, "scope_id") or _text(origin, "guild_id")
    # Slack channel and DM ids are only workspace-local.  An old row without
    # the workspace cannot prove membership in a current recall boundary.
    if platform == "slack" and not scope_id:
        return None

    chat_id = _text(origin, "chat_id")
    if chat_type == "dm" and not chat_id:
        chat_id = _text(origin, "user_id_alt") or _text(origin, "user_id")
    if not chat_id:
        return None

    prospective_thread_id = _text(origin, "prospective_thread_id")
    thread_id = _text(origin, "thread_id") or prospective_thread_id
    if chat_type == "thread" and not thread_id:
        # Supported adapters may identify a thread solely by chat_id.
        thread_id = chat_id

    parent_chat_id = _text(origin, "parent_chat_id")
    if thread_id:
        chat_kind = "dm_thread" if chat_type == "dm" else "thread"
        # A Discord auto-thread starts as parent chat + prospective id, then
        # arrives as thread chat + parent id.  Both normalize to parent+thread.
        # An old actual-thread row carrying only thread==chat cannot prove its
        # parent and is therefore excluded from current scope.
        if (
            platform == "discord"
            and chat_type == "thread"
            and not parent_chat_id
            and chat_id == thread_id
        ):
            return None
        if parent_chat_id and chat_kind == "thread":
            chat_id = parent_chat_id
    else:
        chat_kind = "dm" if chat_type == "dm" else "chat"

    canonical_chat_type = chat_type if chat_kind == "chat" else chat_kind

    aliases: Tuple[str, ...] = ()
    if platform == "whatsapp" and chat_kind in {"dm", "dm_thread"}:
        chat_id, aliases = _whatsapp_dm_identity(chat_id)
        if not chat_id:
            return None

    profile = (_text(origin, "profile") or "default").lower()
    if not _SAFE_PROFILE_TOKEN.fullmatch(profile):
        return None
    return RecallIdentity(
        platform=platform,
        chat_kind=chat_kind,
        chat_type=canonical_chat_type,
        chat_id=chat_id,
        thread_id=thread_id if thread_id else "",
        scope_id=scope_id,
        profile=profile,
        chat_aliases=aliases,
    )


def coerce_recall_identity(value: object) -> RecallIdentity:
    """Accept a canonical identity or its mapping representation."""
    if isinstance(value, RecallIdentity):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("malformed recall scope")

    canonical_fields = (
        "platform",
        "chat_kind",
        "chat_type",
        "chat_id",
        "thread_id",
        "scope_id",
        "profile",
    )
    if not _identity_field_types_valid(value, canonical_fields):
        raise ValueError("malformed recall scope")

    platform = _text(value, "platform").lower()
    chat_kind = _text(value, "chat_kind").lower()
    chat_type = _text(value, "chat_type").lower()
    chat_id = _text(value, "chat_id")
    thread_id = _text(value, "thread_id")
    scope_id = _text(value, "scope_id")
    profile = (_text(value, "profile") or "default").lower()
    if (
        not is_recall_gateway_platform(platform)
        or chat_kind not in _RECALL_KINDS
        or not chat_type
        or not _SAFE_IDENTITY_TOKEN.fullmatch(chat_type)
        or (chat_kind == "chat" and chat_type in {"dm", "dm_thread", "thread"})
        or (chat_kind != "chat" and chat_type != chat_kind)
        or not chat_id
        or (chat_kind in {"thread", "dm_thread"}) != bool(thread_id)
        or (platform == "slack" and not scope_id)
        or not _SAFE_PROFILE_TOKEN.fullmatch(profile)
    ):
        raise ValueError("malformed recall scope")

    raw_aliases = value.get("chat_aliases")
    aliases = (
        tuple(sorted({str(alias).strip() for alias in raw_aliases if str(alias).strip()}))
        if isinstance(raw_aliases, (list, tuple, set, frozenset))
        else ()
    )
    return RecallIdentity(
        platform=platform,
        chat_kind=chat_kind,
        chat_type=chat_type,
        chat_id=chat_id,
        thread_id=thread_id,
        scope_id=scope_id,
        profile=profile,
        chat_aliases=aliases,
    )


def _whatsapp_bare_sql(expression: str) -> str:
    trimmed = f"TRIM({expression})"
    no_plus = (
        f"(CASE WHEN SUBSTR({trimmed}, 1, 1) = '+' "
        f"THEN SUBSTR({trimmed}, 2) ELSE {trimmed} END)"
    )
    before_colon = (
        f"(CASE WHEN INSTR({no_plus}, ':') > 0 "
        f"THEN SUBSTR({no_plus}, 1, INSTR({no_plus}, ':') - 1) "
        f"ELSE {no_plus} END)"
    )
    return (
        f"(CASE WHEN INSTR({before_colon}, '@') > 0 "
        f"THEN SUBSTR({before_colon}, 1, INSTR({before_colon}, '@') - 1) "
        f"ELSE {before_colon} END)"
    )


def recall_scope_sql(
    alias: str,
    recall_scope: object,
) -> tuple[str, List[Any]]:
    """Emit the fail-closed ``origin_json`` predicate for an identity."""
    identity = coerce_recall_identity(recall_scope)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", alias):
        raise ValueError("invalid SQL alias for recall scope")

    origin = f"{alias}.origin_json"
    safe_origin = f"(CASE WHEN json_valid({origin}) THEN {origin} ELSE '{{}}' END)"

    def json_string(path: str) -> str:
        return (
            f"(CASE WHEN json_type({safe_origin}, '{path}') = 'text' "
            f"THEN TRIM(CAST(json_extract({safe_origin}, '{path}') AS TEXT)) "
            "ELSE '' END)"
        )

    def json_string_type_guard(path: str) -> str:
        json_kind = f"json_type({safe_origin}, '{path}')"
        return f"({json_kind} IS NULL OR {json_kind} IN ('text', 'null'))"

    candidate_platform = f"LOWER({json_string('$.platform')})"
    candidate_chat_type = f"LOWER({json_string('$.chat_type')})"
    raw_chat_id = json_string("$.chat_id")
    raw_parent_chat_id = (
        json_string("$.parent_chat_id")
    )
    raw_thread_id = (
        "COALESCE("
        f"NULLIF({json_string('$.thread_id')}, ''), "
        f"NULLIF({json_string('$.prospective_thread_id')}, ''), "
        f"CASE WHEN {candidate_chat_type} = 'thread' THEN {raw_chat_id} ELSE '' END, "
        "''"
        ")"
    )
    raw_dm_chat_id = (
        "COALESCE("
        f"NULLIF({raw_chat_id}, ''), "
        f"NULLIF({json_string('$.user_id_alt')}, ''), "
        f"NULLIF({json_string('$.user_id')}, ''), ''"
        ")"
    )
    candidate_kind = (
        "CASE "
        f"WHEN {candidate_chat_type} = 'dm' AND {raw_thread_id} <> '' THEN 'dm_thread' "
        f"WHEN {candidate_chat_type} = 'dm' THEN 'dm' "
        f"WHEN {raw_thread_id} <> '' THEN 'thread' "
        "ELSE 'chat' END"
    )
    candidate_canonical_chat_type = (
        "CASE "
        f"WHEN {candidate_kind} = 'chat' THEN {candidate_chat_type} "
        f"ELSE {candidate_kind} END"
    )
    candidate_chat_id = (
        "CASE "
        f"WHEN {candidate_kind} IN ('dm', 'dm_thread') THEN {raw_dm_chat_id} "
        f"WHEN {candidate_kind} = 'thread' THEN COALESCE("
        f"NULLIF({raw_parent_chat_id}, ''), {raw_chat_id}) "
        f"ELSE {raw_chat_id} END"
    )
    candidate_scope_id = (
        "COALESCE("
        f"NULLIF({json_string('$.scope_id')}, ''), "
        f"NULLIF({json_string('$.guild_id')}, ''), '')"
    )
    candidate_profile = (
        f"LOWER(COALESCE(NULLIF({json_string('$.profile')}, ''), 'default'))"
    )
    candidate_is_ambiguous_discord_thread = (
        f"({candidate_platform} = 'discord' "
        f"AND {candidate_chat_type} = 'thread' "
        f"AND {raw_parent_chat_id} = '' "
        f"AND {raw_chat_id} = {raw_thread_id})"
    )

    params: List[Any] = [
        identity.platform,
        identity.chat_kind,
        identity.chat_type,
    ]
    if identity.platform == "whatsapp" and identity.chat_kind in {
        "dm",
        "dm_thread",
    }:
        aliases = tuple(sorted(set(identity.chat_aliases or (identity.chat_id,))))
        placeholders = ",".join("?" for _ in aliases)
        chat_clause = f"{_whatsapp_bare_sql(candidate_chat_id)} IN ({placeholders})"
        params.extend(aliases)
    else:
        chat_clause = f"{candidate_chat_id} = ?"
        params.append(identity.chat_id)
    params.extend(
        [identity.thread_id, identity.scope_id, identity.profile]
    )

    type_guards = " AND ".join(
        json_string_type_guard(f"$.{field}") for field in _PROXY_ORIGIN_FIELDS
    )
    clause = (
        f"json_valid({origin}) = 1 "
        f"AND {type_guards} "
        f"AND {candidate_chat_type} <> '' "
        f"AND {candidate_chat_type} GLOB '[a-z0-9]*' "
        f"AND {candidate_chat_type} NOT GLOB '*[^a-z0-9_.-]*' "
        f"AND {candidate_platform} = ? "
        f"AND {candidate_kind} = ? "
        f"AND {candidate_canonical_chat_type} = ? "
        f"AND {chat_clause} "
        f"AND {raw_thread_id} = ? "
        f"AND {candidate_scope_id} = ? "
        f"AND {candidate_profile} = ? "
        f"AND NOT {candidate_is_ambiguous_discord_thread}"
    )
    return f"({clause})", params
