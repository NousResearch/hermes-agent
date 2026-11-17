"""The secrets plane's rules for a write, applied to the exact request bytes (contract §2, §6).

:func:`validate_patch` mirrors gateway-gateway ``src/secretsPlane`` (``router.ts``: the body limit,
``JSON.parse``, the store's entry count; ``patch.ts`` ``parseProfilePatch``; ``reservedNames.ts``;
``types.ts``): a body of at most :data:`MAX_BODY_BYTES` bytes that is UTF-8 JSON; env names and auth
provider ids in the plane's grammar and outside its reserved names (the vendored
``conformance/secrets-reserved-names.json`` the plane's list is generated from); an env value a
string of at most :data:`MAX_ENTRY_BYTES` UTF-8 bytes; an auth record a JSON object whose
``JSON.stringify`` is at most :data:`MAX_ENTRY_BYTES` UTF-8 bytes; at most
:data:`MAX_ENTRIES_PER_LEVEL` entries; versions integers by ``Number.isInteger`` (``0.0``, ``1e0`` and
``-0.0`` count). The bytes are read as the plane reads them (:mod:`.plane_json`: a leading BOM
dropped, any nesting depth). Stricter than the plane in two ways, where the plane would store
something else: a number a JavaScript number cannot hold (``1e400``, a 400-digit integer; the plane
stores null), and bytes that are not UTF-8 or a lone surrogate (the plane stores U+FFFD).
A body of any JSON shape gets violations, never an exception.

:func:`check` encodes a body with the transport's own encoder
(:func:`plugins.config_backends.remote.client.encode_body`, which refuses NaN / Infinity and lone
surrogates) and validates those bytes. Each problem is described by name and number, never by value.
"""
from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

from plugins.config_backends.remote.client import encode_body
from plugins.secret_store.remote import plane_json

MAX_ENTRY_BYTES = 64 * 1024
MAX_ENTRIES_PER_LEVEL = 1000
MAX_BODY_BYTES = 1024 * 1024
MAX_VERSION = 2147483647
#: The plane's name rules (contract §2; gateway-gateway refuses others with ``secret_name_invalid``).
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
AUTH_ID = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}")
RESERVED_NAMES = Path(__file__).parent / "vendor" / "conformance" / "secrets-reserved-names.json"
_BODY_KEYS = frozenset({"expectedProfileVersion", "set", "unset", "auth"})


class Violation(NamedTuple):
    """One rule the request breaks: the entry's *kind* (``"env"`` / ``"auth"``; None for the
    request as a whole) and *name*, a description (*text*: names escaped, never a value) and the
    plane's error code."""
    kind: str | None
    name: str | None
    text: str
    code: str


def import_body(env: dict[str, Any], records: dict[str, Any]) -> dict[str, Any]:
    """The one-time import's ``PATCH /self`` body for a profile the plane has never stored."""
    return {"expectedProfileVersion": 0, "set": env,
            "auth": {rid: {"expectedVersion": 0, "record": r} for rid, r in records.items()}}


def check(body: dict[str, Any]) -> list[Violation]:
    """Every rule the plane would refuse *body* by, as the transport encodes it; [] when it would
    take it. A body the encoder refuses is blamed on the entries it refuses."""
    try:
        data = encode_body(body)
    except (ValueError, TypeError, RecursionError):   # UnicodeEncodeError is a ValueError
        return _unencodable(body)
    return validate_patch(data)


def validate_patch(data: bytes) -> list[Violation]:
    """Every rule of the plane's ``PATCH /self`` the request bytes *data* break, in the order the
    plane checks them (its first is the plane's answer); [] when the plane would take them."""
    out = []
    if len(data) > MAX_BODY_BYTES:
        out.append(_whole(f"the import request would be {len(data)} bytes (at most {MAX_BODY_BYTES}): "
                          "too much at once", "secret_body_too_large"))
    try:
        raw = plane_json.parse(data)
    except ValueError:                            # not UTF-8, not JSON, or a NaN / Infinity token
        return [*out, _whole("the import request is not JSON", "secret_request_invalid")]
    # The plane's order (patch.ts parseProfilePatch): the body's keys, `set`, `unset`, `auth`, then
    # the version. A part of the wrong JSON type ends the check, as the plane's first throw does.
    not_a_write = _whole("the import request is not a profile write", "secret_request_invalid")
    if not isinstance(raw, dict) or not set(raw) <= _BODY_KEYS:
        return [*out, not_a_write]
    env = raw.get("set", {})
    if not isinstance(env, dict):
        return [*out, not_a_write]
    out += _env_violations(env)
    unset = raw.get("unset", [])
    if not isinstance(unset, list):
        return [*out, not_a_write]
    out += _unset_violations(unset, env, not_a_write)
    auth = raw.get("auth", {})
    if not isinstance(auth, dict):
        return [*out, not_a_write]
    out += _auth_violations(auth, not_a_write)
    if not (env or unset or auth):
        out.append(_whole("the import request writes nothing", "secret_request_invalid"))
    if "expectedProfileVersion" in raw:
        if not _version(raw["expectedProfileVersion"]):
            out.append(not_a_write)
    elif env or unset:
        out.append(not_a_write)
    if (count := len(env) + sum(isinstance(c, dict) and c.get("record") is not None
                                for c in auth.values())) > MAX_ENTRIES_PER_LEVEL:
        out.append(_whole(f"{count} entries (a profile holds at most {MAX_ENTRIES_PER_LEVEL})",
                          "secret_level_too_large"))
    return out


def _env_violations(env: dict) -> list[Violation]:
    out = []
    for name, value in env.items():
        out += _name("env", name)
        size = _utf8_size(value) if isinstance(value, str) else None
        if size is None:
            out.append(Violation("env", name, f"{shown(name)} (the value is not UTF-8 text)", "secret_value_invalid"))
        elif size > MAX_ENTRY_BYTES:
            out.append(Violation("env", name, f"{shown(name)} (a {size}-byte value; at most {MAX_ENTRY_BYTES})",
                                 "secret_value_invalid"))
    return out


def _unset_violations(unset: list, env: dict, not_a_write: Violation) -> list[Violation]:
    out = []
    for name in unset:
        if not isinstance(name, str):             # `unset` holds names: [{}], [[]], [1], [null] are not
            out.append(not_a_write)
            continue
        out += _name("env", name)
        if name in env:
            out.append(not_a_write)
    return out


def _auth_violations(auth: dict, not_a_write: Violation) -> list[Violation]:
    out = []
    for rid, change in auth.items():
        out += _name("auth", rid)
        if not (isinstance(change, dict) and set(change) == {"expectedVersion", "record"}
                and _version(change["expectedVersion"])):
            out.append(not_a_write)
            continue
        record = change["record"]
        if record is None:
            continue
        size = _utf8_size(plane_json.stringify(record)) if isinstance(record, dict) else None
        if size is None:
            out.append(Violation("auth", rid, f"auth record {shown(rid)} (not a JSON object of UTF-8 text, or it "
                                 "holds a number JavaScript cannot hold: NaN, Infinity or beyond 1.8e308)",
                                 "secret_value_invalid"))
        elif size > MAX_ENTRY_BYTES:
            out.append(Violation("auth", rid, f"auth record {shown(rid)} ({size} bytes; at most {MAX_ENTRY_BYTES})",
                                 "secret_value_invalid"))
    return out


def shown(name) -> str:
    """*name* for a message: ASCII-escaped and quoted (no control or bidi character reaches a
    terminal), cut at 80 characters."""
    text = ascii(str(name))
    return text if len(text) <= 80 else text[:77] + "..."


def _whole(text: str, code: str) -> Violation:
    return Violation(None, None, text, code)


def _name(kind: str, name: str) -> list[Violation]:
    if not (ENV_NAME if kind == "env" else AUTH_ID).fullmatch(name):
        return [Violation(kind, name, f"invalid name {shown(name)} (env names: letters, digits and _, not "
                          "starting with a digit; auth provider ids: lower-case letters, digits and _ . : -)",
                          "secret_name_invalid")]
    if _reserved(kind, name):
        return [Violation(kind, name, f"{shown(name)} (a name the secrets plane never stores)",
                          "secret_name_reserved")]
    return []


@functools.cache
def _reserved_sets() -> tuple[frozenset, tuple, frozenset, frozenset]:
    names = json.loads(RESERVED_NAMES.read_text(encoding="utf-8-sig"))
    return (frozenset(names["globalEnvExact"] + names["bootstrapEnvExact"] + names["bootstrapContract"]["envExact"]),
            tuple(names["globalEnvPrefixes"] + names["bootstrapEnvPrefixes"]),
            frozenset(n.upper() for n in names["launchOnlyAnyCase"]),
            frozenset(names["bootstrapContract"]["authProviders"]))


def _reserved(kind: str, name: str) -> bool:
    exact, prefixes, any_case, auth = _reserved_sets()
    if kind == "auth":
        return name in auth
    return name in exact or name.startswith(prefixes) or name.upper() in any_case


def _utf8_size(text: str | None) -> int | None:
    """*text*'s UTF-8 size; None for None or a lone surrogate (not UTF-8 text: the plane would
    store U+FFFD in its place)."""
    try:
        return None if text is None else len(text.encode("utf-8"))
    except UnicodeEncodeError:
        return None


def _version(v: Any) -> bool:
    """The plane's ``versionOk``: a JSON number ``Number.isInteger`` takes (``1.0``, ``1e0`` and
    ``-0.0`` are integers; true is not a number) from 0 to :data:`MAX_VERSION`."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return (isinstance(v, int) or v.is_integer()) and 0 <= v <= MAX_VERSION


def _unencodable(body: dict[str, Any]) -> list[Violation]:
    """The entries of *body* the transport's encoder refuses (a lone surrogate, NaN / Infinity)."""
    def fails(*values) -> bool:
        try:
            encode_body({"v": values})
        except (ValueError, TypeError, RecursionError):
            return True
        return False
    def entries(part: str) -> list:
        value = body.get(part)
        return list(value.items()) if isinstance(value, dict) else []
    out = [Violation("env", n, f"{shown(n)} (the name or value is not UTF-8 text)", "secret_value_invalid")
           for n, v in entries("set") if fails(n, v)]
    out += [Violation("auth", r, f"auth record {shown(r)} (not JSON of UTF-8 text: a lone surrogate, NaN or "
                      "Infinity)", "secret_value_invalid") for r, c in entries("auth") if fails(r, c)]
    return out or [_whole("the import request cannot be encoded", "secret_request_invalid")]
