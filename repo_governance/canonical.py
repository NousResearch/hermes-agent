"""Closed canonical byte grammar for the disposable D1 prototype."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import unicodedata
from typing import Any


class CanonicalEncodingError(ValueError):
    """One fail-closed error class for malformed/noncanonical values."""


_UNSIGNED_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_SIGNED_DECIMAL = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
_DOMAIN = re.compile(r"[a-z0-9][a-z0-9-]*/[0-9]+\.[0-9]+\Z")
_LOWER_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
_BASE64URL_64 = re.compile(r"[A-Za-z0-9_-]{86}\Z")
_UUID7 = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
UINT64_MAX = (1 << 64) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

_OUTER_FIELDS = (
    "witnessPayload",
    "recordDigest",
    "witnessSignatureBase64url",
)
_PAYLOAD_FIELDS = (
    "witnessSchemaVersion",
    "recordType",
    "previousRecordDigest",
    "hostInstanceId",
    "governanceStoreInstanceId",
    "witnessChainId",
    "witnessKeyId",
    "witnessKeyVersionUnsignedDecimal",
    "witnessKeyStatusEpochUnsignedDecimal",
    "witnessPublicKeySha256",
    "witnessTransactionIdOrNull",
    "preparedRecordDigestOrNull",
    "terminalOutcomeOrNull",
    "sqliteEvaluatorProfileDigest",
    "schemaManifestDigest",
    "installationIdentityDigest",
    "installationIdentityOrNull",
    "authoritativeStateDigest",
    "governanceCommitSequenceUnsignedDecimal",
)
_INSTALLATION_FIELDS = (
    "identityVersion",
    "ownerApprovedRootCanonicalUtf8",
    "rootDeviceUnsignedDecimal",
    "rootInodeUnsignedDecimal",
    "rootUidUnsignedDecimal",
    "rootModeOctal",
    "witnessRelativeName",
    "witnessDeviceUnsignedDecimal",
    "witnessInodeUnsignedDecimal",
    "witnessUidUnsignedDecimal",
    "witnessModeOctal",
)


def require_unsigned_decimal(value: object, *, maximum: int = UINT64_MAX) -> int:
    """Return a canonical unsigned decimal string's bounded integer value."""
    if not isinstance(value, str) or _UNSIGNED_DECIMAL.fullmatch(value) is None:
        raise CanonicalEncodingError("invalid canonical unsigned decimal")
    if type(maximum) is not int or not 0 <= maximum <= UINT64_MAX:
        raise CanonicalEncodingError("invalid unsigned decimal bound")
    limit = str(maximum)
    if len(value) > len(limit) or (len(value) == len(limit) and value > limit):
        raise CanonicalEncodingError("unsigned decimal is out of range")
    return int(value)


def require_signed64_decimal(value: object) -> int:
    """Return a canonical signed-64 decimal string's integer value."""
    if not isinstance(value, str) or _SIGNED_DECIMAL.fullmatch(value) is None:
        raise CanonicalEncodingError("invalid canonical signed decimal")
    negative = value.startswith("-")
    magnitude = value[1:] if negative else value
    limit = str(1 << 63) if negative else str(INT64_MAX)
    if len(magnitude) > len(limit) or (
        len(magnitude) == len(limit) and magnitude > limit
    ):
        raise CanonicalEncodingError("signed decimal is out of range")
    return int(value)


def _require_unicode_scalar_string(value: str) -> None:
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise CanonicalEncodingError("invalid Unicode scalar value")


def require_nfc_text(value: object, *, nonempty: bool = False) -> str:
    """Return an exact NFC Unicode scalar string or fail closed."""
    if not isinstance(value, str):
        raise CanonicalEncodingError("text value is not a string")
    _require_unicode_scalar_string(value)
    if nonempty and not value:
        raise CanonicalEncodingError("text value is empty")
    if unicodedata.normalize("NFC", value) != value:
        raise CanonicalEncodingError("text value is not NFC")
    return value


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        _require_unicode_scalar_string(value)
        return value
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalEncodingError("object key is not a string")
            _require_unicode_scalar_string(key)
            items.append((key, _canonical_value(item)))
        items.sort(key=lambda pair: pair[0].encode("utf-16-be"))
        return {key: item for key, item in items}
    raise CanonicalEncodingError("value is outside the closed canonical grammar")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize the closed no-JSON-number subset of RFC 8785 as UTF-8."""
    try:
        canonical = _canonical_value(value)
        text = json.dumps(
            canonical,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return text.encode("utf-8")
    except CanonicalEncodingError:
        raise
    except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise CanonicalEncodingError("canonical serialization failed") from exc


def require_closed_object(value: object, fields: tuple[str, ...]) -> dict[str, Any]:
    """Require a JSON object whose key set is exactly ``fields``."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CanonicalEncodingError("value is not a string-keyed object")
    if len(fields) != len(set(fields)) or set(value) != set(fields):
        raise CanonicalEncodingError("closed object fields do not match")
    return value


def _reject_json_number(_token: str) -> None:
    raise CanonicalEncodingError("raw JSON numbers are outside the closed grammar")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalEncodingError("duplicate object field")
        result[key] = value
    return result


def parse_canonical_json_bytes(encoded: bytes) -> Any:
    """Parse only exact canonical bytes in the closed no-number grammar."""
    if not isinstance(encoded, bytes):
        raise CanonicalEncodingError("canonical input is not bytes")
    try:
        text = encoded.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_reject_json_number,
            parse_float=_reject_json_number,
            parse_constant=_reject_json_number,
        )
        if canonical_json_bytes(value) != encoded:
            raise CanonicalEncodingError("input bytes are not canonical")
        return value
    except CanonicalEncodingError:
        raise
    except (
        RecursionError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise CanonicalEncodingError("canonical parse failed") from exc


def domain_separated_json_preimage(domain: str, value: Any) -> bytes:
    """Build ``ASCII(domain) + NUL + canonical JSON`` for a closed domain."""
    if not isinstance(domain, str) or _DOMAIN.fullmatch(domain) is None:
        raise CanonicalEncodingError("invalid domain separator")
    return domain.encode("ascii") + b"\x00" + canonical_json_bytes(value)


def domain_separated_json_digest(domain: str, value: Any) -> str:
    """Return lowercase SHA-256 hex for a domain-separated JSON preimage."""
    return hashlib.sha256(domain_separated_json_preimage(domain, value)).hexdigest()


def witness_signature_preimage(record_digest: object) -> bytes:
    """Build the normative /0.9 witness signature preimage without a live key."""
    if not isinstance(record_digest, str) or _LOWER_HEX_256.fullmatch(record_digest) is None:
        raise CanonicalEncodingError("invalid record digest")
    return (
        b"hermes-governance-commit-witness-signature/0.9\x00"
        + bytes.fromhex(record_digest)
    )


def _require_digest(value: object) -> str:
    if not isinstance(value, str) or _LOWER_HEX_256.fullmatch(value) is None:
        raise CanonicalEncodingError("invalid lowercase SHA-256 digest")
    return value


def _require_id(value: object) -> str:
    return require_nfc_text(value, nonempty=True)


def _require_absolute_canonical_path(value: object) -> str:
    path = require_nfc_text(value, nonempty=True)
    if "\x00" in path or not path.startswith("/"):
        raise CanonicalEncodingError("installation root is not absolute")
    if path != "/" and path.endswith("/"):
        raise CanonicalEncodingError("installation root has a trailing slash")
    if path == "/":
        return path
    components = path.split("/")[1:]
    if any(component in {"", ".", ".."} for component in components):
        raise CanonicalEncodingError("installation root has a forbidden component")
    return path


def _validate_installation_identity(value: object) -> dict[str, Any]:
    installation = require_closed_object(value, _INSTALLATION_FIELDS)
    if installation["identityVersion"] != "witness-installation-identity/0.9":
        raise CanonicalEncodingError("wrong installation identity version")
    _require_absolute_canonical_path(installation["ownerApprovedRootCanonicalUtf8"])
    for field in (
        "rootDeviceUnsignedDecimal",
        "rootInodeUnsignedDecimal",
        "rootUidUnsignedDecimal",
        "witnessDeviceUnsignedDecimal",
        "witnessInodeUnsignedDecimal",
        "witnessUidUnsignedDecimal",
    ):
        require_unsigned_decimal(installation[field])
    if installation["rootModeOctal"] != "0700":
        raise CanonicalEncodingError("wrong root mode")
    if installation["witnessModeOctal"] != "0600":
        raise CanonicalEncodingError("wrong witness mode")
    if installation["witnessRelativeName"] != "governance-commit-witness.v09.jsonl":
        raise CanonicalEncodingError("wrong witness relative name")
    return installation


def _validate_signature(value: object) -> str:
    if not isinstance(value, str) or _BASE64URL_64.fullmatch(value) is None:
        raise CanonicalEncodingError("invalid witness signature encoding")
    try:
        decoded = base64.b64decode(value + "==", altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CanonicalEncodingError("invalid witness signature encoding") from exc
    if len(decoded) != 64:
        raise CanonicalEncodingError("wrong witness signature length")
    if base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") != value:
        raise CanonicalEncodingError("noncanonical witness signature encoding")
    return value


def _validate_transaction_id(value: object) -> None:
    if value is not None and (
        not isinstance(value, str) or _UUID7.fullmatch(value) is None
    ):
        raise CanonicalEncodingError("invalid UUIDv7 transaction ID")


def _validate_witness_payload(value: object) -> dict[str, Any]:
    payload = require_closed_object(value, _PAYLOAD_FIELDS)
    if payload["witnessSchemaVersion"] != "governance-commit-witness/0.9":
        raise CanonicalEncodingError("wrong witness schema version")
    record_type = payload["recordType"]
    if not isinstance(record_type, str) or record_type not in {
        "genesis",
        "prepared",
        "terminal",
    }:
        raise CanonicalEncodingError("invalid witness record type")

    for field in (
        "hostInstanceId",
        "governanceStoreInstanceId",
        "witnessChainId",
        "witnessKeyId",
    ):
        _require_id(payload[field])
    for field in (
        "previousRecordDigest",
        "witnessPublicKeySha256",
        "sqliteEvaluatorProfileDigest",
        "schemaManifestDigest",
        "installationIdentityDigest",
        "authoritativeStateDigest",
    ):
        _require_digest(payload[field])
    require_unsigned_decimal(payload["witnessKeyVersionUnsignedDecimal"])
    require_unsigned_decimal(payload["witnessKeyStatusEpochUnsignedDecimal"])
    sequence = require_unsigned_decimal(
        payload["governanceCommitSequenceUnsignedDecimal"], maximum=INT64_MAX
    )
    _validate_transaction_id(payload["witnessTransactionIdOrNull"])
    prepared_digest = payload["preparedRecordDigestOrNull"]
    if prepared_digest is not None:
        _require_digest(prepared_digest)

    installation = payload["installationIdentityOrNull"]
    if record_type == "genesis":
        validated_installation = _validate_installation_identity(installation)
        expected_installation_digest = domain_separated_json_digest(
            "hermes-witness-installation-identity/0.9", validated_installation
        )
        if payload["installationIdentityDigest"] != expected_installation_digest:
            raise CanonicalEncodingError("installation identity digest mismatch")
        if (
            payload["previousRecordDigest"] != "0" * 64
            or payload["witnessTransactionIdOrNull"] is not None
            or prepared_digest is not None
            or payload["terminalOutcomeOrNull"] is not None
            or sequence != 0
        ):
            raise CanonicalEncodingError("invalid genesis state grammar")
    elif record_type == "prepared":
        if (
            payload["witnessTransactionIdOrNull"] is None
            or prepared_digest is not None
            or payload["terminalOutcomeOrNull"] is not None
            or installation is not None
        ):
            raise CanonicalEncodingError("invalid prepared state grammar")
    else:
        outcome = payload["terminalOutcomeOrNull"]
        if (
            payload["witnessTransactionIdOrNull"] is None
            or prepared_digest is None
            or not isinstance(outcome, str)
            or outcome not in {"committed", "aborted"}
            or installation is not None
        ):
            raise CanonicalEncodingError("invalid terminal state grammar")
    return payload


def witness_record_digest(payload: object) -> str:
    """Validate and digest one complete /0.9 witness payload."""
    validated = _validate_witness_payload(payload)
    return domain_separated_json_digest(
        "hermes-governance-commit-witness-digest/0.9", validated
    )


def _validate_witness_record(value: object) -> dict[str, Any]:
    record = require_closed_object(value, _OUTER_FIELDS)
    payload = _validate_witness_payload(record["witnessPayload"])
    digest = _require_digest(record["recordDigest"])
    if digest != domain_separated_json_digest(
        "hermes-governance-commit-witness-digest/0.9", payload
    ):
        raise CanonicalEncodingError("witness record digest mismatch")
    _validate_signature(record["witnessSignatureBase64url"])
    return record


def encode_witness_frame(record: object) -> bytes:
    """Encode one validated /0.9 witness record plus exactly one LF."""
    return canonical_json_bytes(_validate_witness_record(record)) + b"\x0a"


def parse_witness_frame(frame: bytes) -> dict[str, Any]:
    """Parse one exact /0.9 canonical witness frame."""
    if not isinstance(frame, bytes) or not frame.endswith(b"\x0a"):
        raise CanonicalEncodingError("witness frame is missing final LF")
    if b"\x0a" in frame[:-1]:
        raise CanonicalEncodingError("witness frame has an extra LF")
    body = frame[:-1]
    record = _validate_witness_record(parse_canonical_json_bytes(body))
    if encode_witness_frame(record) != frame:
        raise CanonicalEncodingError("witness frame is not byte canonical")
    return record
