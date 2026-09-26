"""Immutable Telegram menu projection of the canonical command catalog.

This is the bounded PR-8 projection seam under #96692. It consumes a catalog
snapshot and owns only Telegram presentation: command-name sanitization, native
visibility, native limits, and one deterministic projection fingerprint.
Command identity, aliases, policy, arguments, availability, and execution remain
catalog/dispatcher concerns.

Current main does not yet expose the versioned ``CommandCatalog`` ABI, so this
module accepts catalog-shaped objects without importing a second schema. A
non-blank ``command_id`` is authoritative when present; canonical name is the
explicit v1 compatibility fallback until the stable-id slice lands.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


TELEGRAM_BOT_API_MAX_COMMANDS = 100
TELEGRAM_COMMAND_NAME_MAX_LENGTH = 32

_MISSING = object()
_TELEGRAM_TYPED_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
_TELEGRAM_NATIVE_INVALID_RE = re.compile(r"[^a-z0-9_]")
_TELEGRAM_MULTI_UNDERSCORE_RE = re.compile(r"_{2,}")


class TelegramMenuOmissionReason(str, Enum):
    """Why a catalog command is absent from Telegram's native menu."""

    HIDDEN = "hidden"
    NATIVE_NAME_INVALID = "native_name_invalid"
    NATIVE_LIMIT = "native_limit"


@dataclass(frozen=True, slots=True)
class TelegramCommandBinding:
    """Telegram syntax binding for one canonical catalog command."""

    command_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    description: str
    typed_tokens: tuple[str, ...]
    native_name: str | None
    native_visible: bool


@dataclass(frozen=True, slots=True)
class TelegramNativeCommand:
    """One exact Telegram ``BotCommand`` projection row."""

    command_id: str
    command: str
    description: str


@dataclass(frozen=True, slots=True)
class TelegramMenuOmission:
    """One catalog command intentionally absent from the native menu."""

    command_id: str
    canonical_name: str
    reason: TelegramMenuOmissionReason


@dataclass(frozen=True, slots=True)
class TelegramCommandProjection:
    """Immutable Telegram projection of one exact catalog snapshot."""

    catalog_revision: str
    projection_fingerprint: str
    bindings: tuple[TelegramCommandBinding, ...]
    native_commands: tuple[TelegramNativeCommand, ...]
    omissions: tuple[TelegramMenuOmission, ...]

    @property
    def native_payload(self) -> tuple[tuple[str, str], ...]:
        """Return the exact ordered payload used by Telegram's Bot API."""

        return tuple(
            (command.command, command.description)
            for command in self.native_commands
        )


def _value(source: object, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _command_value(source: object, name: str, default: Any = None) -> Any:
    value = _value(source, name, _MISSING)
    if value is not _MISSING:
        return value
    legacy = _value(source, "legacy", None)
    if isinstance(legacy, Mapping):
        return legacy.get(name, default)
    return default


def _nonblank_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Iterable):
        return ()
    result: list[str] = []
    for item in value:
        text = _nonblank_text(item)
        if text is not None:
            result.append(text)
    return tuple(result)


def _normalized_tokens(value: object) -> frozenset[str]:
    return frozenset(
        item.casefold().replace("_", "-") for item in _string_tuple(value)
    )


def _catalog_commands(catalog: object) -> tuple[object, ...]:
    commands = _value(catalog, "commands", None)
    if commands is None:
        commands = catalog
    if isinstance(commands, (str, bytes, Mapping)):
        raise TypeError("Telegram projection requires an iterable of catalog commands")
    try:
        return tuple(commands)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "Telegram projection requires an iterable of catalog commands"
        ) from exc


def _catalog_revision(catalog: object, explicit_revision: str | None) -> str | None:
    if explicit_revision is not None:
        revision = _nonblank_text(explicit_revision)
        if revision is None:
            raise ValueError("catalog revision must not be blank")
        return revision
    for field in ("revision", "catalog_revision", "fingerprint"):
        revision = _nonblank_text(_value(catalog, field, None))
        if revision is not None:
            return revision
    return None


def _canonical_name(command: object) -> str:
    name = _nonblank_text(_value(command, "name", None))
    if name is None:
        name = _nonblank_text(_value(command, "canonical_name", None))
    if name is None:
        raise ValueError("Telegram catalog command name must not be blank")
    if not _TELEGRAM_TYPED_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid Telegram typed command name: {name}")
    return name


def _command_id(command: object, canonical_name: str) -> str:
    command_id = _nonblank_text(_value(command, "command_id", None))
    return command_id or canonical_name


def _aliases(command: object) -> tuple[str, ...]:
    aliases: list[str] = []
    for alias in _string_tuple(_value(command, "aliases", ())):
        normalized = alias.removeprefix("/")
        if _TELEGRAM_TYPED_NAME_RE.fullmatch(normalized):
            aliases.append(normalized)
    return tuple(aliases)


def _telegram_presentation_override(command: object) -> Mapping[str, Any]:
    overrides = _value(command, "presentation_overrides", None)
    if not isinstance(overrides, Mapping):
        return {}
    telegram = overrides.get("telegram")
    return telegram if isinstance(telegram, Mapping) else {}


def _description(command: object, canonical_name: str) -> str:
    telegram = _telegram_presentation_override(command)
    candidates = (
        telegram.get("description"),
        telegram.get("description_fallback"),
        telegram.get("label"),
        _command_value(command, "telegram_label", None),
        _value(command, "description_fallback", None),
        _value(command, "description", None),
    )
    for candidate in candidates:
        text = _nonblank_text(candidate)
        if text is not None:
            return re.sub(r"\s+", " ", text)
    return f"Run /{canonical_name}"


def _visibility_tokens(visibility: object) -> frozenset[str]:
    if isinstance(visibility, str):
        return frozenset({visibility.casefold().replace("_", "-")})
    if isinstance(visibility, Mapping):
        return frozenset()
    if isinstance(visibility, Sequence):
        return frozenset(
            str(item).strip().casefold().replace("_", "-")
            for item in visibility
            if str(item).strip()
        )
    return frozenset()


def _native_visible(command: object) -> bool:
    if bool(_command_value(command, "hidden", False)) or bool(
        _command_value(command, "debug", False)
    ):
        return False

    telegram = _telegram_presentation_override(command)
    if bool(telegram.get("hidden")) or bool(telegram.get("debug")):
        return False
    for key in ("native_menu", "native-menu"):
        if key in telegram and not bool(telegram[key]):
            return False

    visibility = _command_value(command, "visibility", None)
    if isinstance(visibility, Mapping):
        if bool(visibility.get("hidden")) or bool(visibility.get("debug")):
            return False
        for key in ("native_menu", "native-menu"):
            if key in visibility:
                return bool(visibility[key])
        telegram_visibility = visibility.get("telegram")
        if isinstance(telegram_visibility, bool):
            return telegram_visibility
        if isinstance(telegram_visibility, Mapping):
            if bool(telegram_visibility.get("hidden")) or bool(
                telegram_visibility.get("debug")
            ):
                return False
            for key in ("native_menu", "native-menu"):
                if key in telegram_visibility:
                    return bool(telegram_visibility[key])
        return True

    tokens = _visibility_tokens(visibility)
    if tokens & {"hidden", "debug"}:
        return False
    presentation_tokens = tokens & {"help", "completion", "native-menu"}
    return not presentation_tokens or "native-menu" in presentation_tokens


def _availability_mapping_supports_telegram(availability: Mapping[str, Any]) -> bool:
    if availability.get("available") is False:
        return False

    unsupported = _normalized_tokens(
        availability.get("unsupported_platforms")
        or availability.get("unsupported_surfaces")
    )
    if unsupported & {"telegram", "gateway", "messaging"}:
        return False

    supported_value = (
        availability.get("supported_platforms")
        or availability.get("supported_surfaces")
    )
    supported = _normalized_tokens(supported_value)
    if supported and not supported & {"telegram", "gateway", "messaging"}:
        return False
    return True


def _supports_telegram(command: object) -> bool:
    if _command_value(command, "available", True) is False:
        return False

    unsupported = _normalized_tokens(
        _command_value(command, "unsupported_platforms", None)
        or _command_value(command, "unsupported_surfaces", None)
    )
    if unsupported & {"telegram", "gateway", "messaging"}:
        return False

    supported_value = (
        _command_value(command, "supported_platforms", None)
        or _command_value(command, "supported_surfaces", None)
    )
    supported = _normalized_tokens(supported_value)
    if supported and not supported & {"telegram", "gateway", "messaging"}:
        return False

    availability = _command_value(command, "availability", None)
    if isinstance(availability, Mapping) and not _availability_mapping_supports_telegram(
        availability
    ):
        return False

    # Compatibility for a raw current-v1/PR-1 catalog instead of the future
    # context-filtered snapshot. Unresolved CLI-only rows fail closed.
    if bool(_command_value(command, "cli_only", False)) and not bool(
        _command_value(command, "gateway_only", False)
    ):
        return False
    return True


def _sanitize_native_name(raw: str) -> str | None:
    name = raw.casefold().replace("-", "_")
    name = _TELEGRAM_NATIVE_INVALID_RE.sub("", name)
    name = _TELEGRAM_MULTI_UNDERSCORE_RE.sub("_", name).strip("_")
    if not name or len(name) > TELEGRAM_COMMAND_NAME_MAX_LENGTH:
        return None
    return name


def _binding_tokens(canonical_name: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for name in (canonical_name, *aliases):
        for candidate in (name, _sanitize_native_name(name)):
            if candidate is None:
                continue
            key = candidate.casefold()
            if key not in seen:
                seen.add(key)
                result.append(key)
    return tuple(result)


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_telegram_command_projection(
    catalog: object,
    *,
    max_commands: int = TELEGRAM_BOT_API_MAX_COMMANDS,
    catalog_revision: str | None = None,
) -> TelegramCommandProjection:
    """Project one exact catalog snapshot into Telegram syntax and menu rows.

    Catalog order is preserved. Native clipping never removes a command from
    typed resolution. Duplicate IDs, aliases, and sanitized Telegram names fail
    closed instead of allowing catalog order to select authority.
    """

    if isinstance(max_commands, bool) or not isinstance(max_commands, int):
        raise TypeError("max_commands must be an integer")
    if not 0 <= max_commands <= TELEGRAM_BOT_API_MAX_COMMANDS:
        raise ValueError(
            f"max_commands must be between 0 and {TELEGRAM_BOT_API_MAX_COMMANDS}"
        )

    bindings: list[TelegramCommandBinding] = []
    seen_ids: dict[str, str] = {}
    seen_tokens: dict[str, str] = {}
    seen_native_names: dict[str, str] = {}

    for command in _catalog_commands(catalog):
        if not _supports_telegram(command):
            continue

        canonical_name = _canonical_name(command)
        command_id = _command_id(command, canonical_name)
        aliases = _aliases(command)
        typed_tokens = _binding_tokens(canonical_name, aliases)
        native_name = _sanitize_native_name(canonical_name)
        binding = TelegramCommandBinding(
            command_id=command_id,
            canonical_name=canonical_name,
            aliases=aliases,
            description=_description(command, canonical_name),
            typed_tokens=typed_tokens,
            native_name=native_name,
            native_visible=_native_visible(command),
        )

        id_key = command_id.casefold()
        if id_key in seen_ids:
            raise ValueError(
                "duplicate Telegram command identity: "
                f"{command_id} ({seen_ids[id_key]} and {canonical_name})"
            )
        seen_ids[id_key] = canonical_name

        for token in typed_tokens:
            owner = seen_tokens.get(token)
            if owner is not None and owner != command_id:
                raise ValueError(
                    f"Telegram token collision for {token!r}: {owner} vs {command_id}"
                )
            seen_tokens[token] = command_id

        if native_name is not None:
            owner = seen_native_names.get(native_name)
            if owner is not None and owner != command_id:
                raise ValueError(
                    "Telegram native-name collision for "
                    f"{native_name!r}: {owner} vs {command_id}"
                )
            seen_native_names[native_name] = command_id

        bindings.append(binding)

    semantic_payload = [
        {
            "command_id": binding.command_id,
            "canonical_name": binding.canonical_name,
            "aliases": binding.aliases,
            "description": binding.description,
            "typed_tokens": binding.typed_tokens,
            "native_name": binding.native_name,
            "native_visible": binding.native_visible,
        }
        for binding in bindings
    ]
    revision = _catalog_revision(catalog, catalog_revision)
    if revision is None:
        revision = f"compat-v1:{_stable_digest(semantic_payload)}"

    candidates: list[tuple[TelegramCommandBinding, TelegramNativeCommand]] = []
    omissions: list[TelegramMenuOmission] = []
    for binding in bindings:
        if not binding.native_visible:
            omissions.append(
                TelegramMenuOmission(
                    binding.command_id,
                    binding.canonical_name,
                    TelegramMenuOmissionReason.HIDDEN,
                )
            )
        elif binding.native_name is None:
            omissions.append(
                TelegramMenuOmission(
                    binding.command_id,
                    binding.canonical_name,
                    TelegramMenuOmissionReason.NATIVE_NAME_INVALID,
                )
            )
        else:
            candidates.append(
                (
                    binding,
                    TelegramNativeCommand(
                        binding.command_id,
                        binding.native_name,
                        binding.description,
                    ),
                )
            )

    native_commands = tuple(command for _binding, command in candidates[:max_commands])
    omissions.extend(
        TelegramMenuOmission(
            binding.command_id,
            binding.canonical_name,
            TelegramMenuOmissionReason.NATIVE_LIMIT,
        )
        for binding, _command in candidates[max_commands:]
    )

    projection_payload = {
        "catalog_revision": revision,
        "max_commands": max_commands,
        "bindings": semantic_payload,
        "native_commands": [
            {
                "command_id": command.command_id,
                "command": command.command,
                "description": command.description,
            }
            for command in native_commands
        ],
        "omissions": [
            {
                "command_id": omission.command_id,
                "canonical_name": omission.canonical_name,
                "reason": omission.reason.value,
            }
            for omission in omissions
        ],
    }
    return TelegramCommandProjection(
        catalog_revision=revision,
        projection_fingerprint=_stable_digest(projection_payload),
        bindings=tuple(bindings),
        native_commands=native_commands,
        omissions=tuple(omissions),
    )


def resolve_telegram_command_binding(
    projection: TelegramCommandProjection,
    token: str,
) -> TelegramCommandBinding | None:
    """Resolve one canonical, alias, or sanitized Telegram token."""

    key = str(token or "").lstrip("/").casefold()
    if not key:
        return None
    for binding in projection.bindings:
        if key in binding.typed_tokens:
            return binding
    return None
