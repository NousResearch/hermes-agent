from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from repo_governance.canonical import (
    CanonicalEncodingError,
    canonical_json_bytes,
    domain_separated_json_digest,
    domain_separated_json_preimage,
    encode_witness_frame,
    parse_canonical_json_bytes,
    parse_witness_frame,
    require_closed_object,
    require_nfc_text,
    require_signed64_decimal,
    require_unsigned_decimal,
    witness_record_digest,
    witness_signature_preimage,
)


GOLDEN = json.loads(
    (Path(__file__).with_name("vectors") / "d1_canonical_golden.json").read_text()
)


def golden_record(name: str = "genesis") -> dict[str, Any]:
    return GOLDEN["records"][name]


def test_canonical_json_bytes_sorts_keys_by_utf16_and_preserves_scalars() -> None:
    value = {
        "דּ": "Hebrew Letter Dalet With Dagesh",
        "😀": "Emoji: Grinning Face",
        "€": "Euro Sign",
        "ö": "Latin Small Letter O With Diaeresis",
        "\u0080": "Control",
        "1": "One",
        "\r": "Carriage Return",
        "literals": [None, True, False],
    }

    assert canonical_json_bytes(value) == (
        '{"\\r":"Carriage Return","1":"One",'
        '"literals":[null,true,false],'
        '"\u0080":"Control",'
        '"ö":"Latin Small Letter O With Diaeresis",'
        '"€":"Euro Sign",'
        '"😀":"Emoji: Grinning Face",'
        '"דּ":"Hebrew Letter Dalet With Dagesh"}'
    ).encode("utf-8")


def test_canonical_json_uses_minimal_rfc8785_string_escapes() -> None:
    assert canonical_json_bytes({"s": "€$\u000f\nA'B\"\\\"/"}) == (
        "{\"s\":\"€$\\u000f\\nA'B\\\"\\\\\\\"/\"}"
    ).encode("utf-8")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        ("18446744073709551615", 18446744073709551615),
    ],
)
def test_unsigned_decimal_accepts_only_canonical_uint64_strings(
    value: str, expected: int
) -> None:
    assert require_unsigned_decimal(value) == expected


def test_unsigned_decimal_custom_maximum_is_narrowing_only() -> None:
    assert require_unsigned_decimal("9", maximum=9) == 9

    with pytest.raises(CanonicalEncodingError):
        require_unsigned_decimal("10", maximum=9)

    for widened_maximum in (
        18446744073709551616,
        2**100,
        10**4999,
    ):
        with pytest.raises(CanonicalEncodingError):
            require_unsigned_decimal("1", maximum=widened_maximum)


@pytest.mark.parametrize(
    "value",
    ["", "00", "01", "+1", "-0", "-1", "18446744073709551616", 1],
)
def test_unsigned_decimal_rejects_noncanonical_or_out_of_range_values(
    value: object,
) -> None:
    with pytest.raises(CanonicalEncodingError):
        require_unsigned_decimal(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("-9223372036854775808", -9223372036854775808),
        ("0", 0),
        ("9223372036854775807", 9223372036854775807),
    ],
)
def test_signed64_decimal_accepts_canonical_boundary_strings(
    value: str, expected: int
) -> None:
    assert require_signed64_decimal(value) == expected


@pytest.mark.parametrize(
    "value",
    ["-0", "00", "+1", "9223372036854775808", "-9223372036854775809", 0],
)
def test_signed64_decimal_rejects_noncanonical_or_out_of_range_values(
    value: object,
) -> None:
    with pytest.raises(CanonicalEncodingError):
        require_signed64_decimal(value)


@pytest.mark.parametrize(
    ("validator", "value"),
    [
        (require_unsigned_decimal, "9" * 5000),
        (require_signed64_decimal, "-" + "9" * 5000),
    ],
)
def test_decimal_validators_normalize_hostile_digit_lengths_to_one_error_class(
    validator: Any, value: str
) -> None:
    with pytest.raises(CanonicalEncodingError):
        validator(value)


def test_nfc_text_accepts_exact_composed_utf8_and_rejects_decomposition() -> None:
    assert require_nfc_text("Café", nonempty=True) == "Café"
    with pytest.raises(CanonicalEncodingError):
        require_nfc_text("Cafe\u0301", nonempty=True)


@pytest.mark.parametrize("value", [1, -1, 1.5, float("nan"), float("inf")])
def test_canonical_json_rejects_raw_json_numbers(value: object) -> None:
    with pytest.raises(CanonicalEncodingError):
        canonical_json_bytes({"value": value})


def test_canonical_json_rejects_surrogate_code_points() -> None:
    with pytest.raises(CanonicalEncodingError):
        canonical_json_bytes({"value": "\ud800"})


def test_canonical_json_rejects_recursive_container_with_single_error_class() -> None:
    recursive: list[object] = []
    recursive.append(recursive)
    with pytest.raises(CanonicalEncodingError):
        canonical_json_bytes(recursive)


@pytest.mark.parametrize("framed", [False, True])
def test_deep_canonical_bytes_use_the_single_fail_closed_error_class(
    framed: bool,
) -> None:
    depth = 2000
    encoded = b"[" * depth + b"null" + b"]" * depth
    with pytest.raises(CanonicalEncodingError):
        if framed:
            parse_witness_frame(encoded + b"\n")
        else:
            parse_canonical_json_bytes(encoded)


def test_strict_parser_accepts_only_byte_exact_canonical_json() -> None:
    encoded = b'{"a":null,"b":[true,false]}'
    assert parse_canonical_json_bytes(encoded) == {
        "a": None,
        "b": [True, False],
    }


@pytest.mark.parametrize(
    "encoded",
    [
        b'{"a":null,"a":null}',
        b'{"b":null,"a":null}',
        b'{"a": null}',
        b'{"a":1}',
        b'{"a":null}\n',
        b'\xef\xbb\xbf{"a":null}',
        b'{"a":"\xff"}',
    ],
)
def test_strict_parser_rejects_duplicate_or_noncanonical_bytes(encoded: bytes) -> None:
    with pytest.raises(CanonicalEncodingError):
        parse_canonical_json_bytes(encoded)


def test_closed_object_rejects_missing_and_unknown_fields() -> None:
    assert require_closed_object({"a": None, "b": True}, ("a", "b")) == {
        "a": None,
        "b": True,
    }
    with pytest.raises(CanonicalEncodingError):
        require_closed_object({"a": None}, ("a", "b"))
    with pytest.raises(CanonicalEncodingError):
        require_closed_object({"a": None, "b": True, "c": False}, ("a", "b"))


def test_domain_separated_json_preimage_and_digest_match_golden_bytes() -> None:
    value = {"a": None}
    assert domain_separated_json_preimage("hermes-governance-json/0.2", value) == (
        b'hermes-governance-json/0.2\x00{"a":null}'
    )
    assert domain_separated_json_digest(
        "hermes-governance-json/0.2", value
    ) == "8184023c841cb97e0af1cb112299884bf025ddee9c0c89c6ec030f64ff1d4611"
    assert domain_separated_json_digest(
        "hermes-governance-json/0.3", value
    ) == "7439c77b2e8f1802d4dbfda825e2fe288c1aed6268ebbc63c28487dfca656422"


@pytest.mark.parametrize("domain", ["", "contains space/0.1", "é/0.1", "bad\x00domain"])
def test_domain_separator_rejects_noncanonical_ascii_profiles(domain: str) -> None:
    with pytest.raises(CanonicalEncodingError):
        domain_separated_json_preimage(domain, {"a": None})


def test_witness_signature_preimage_uses_raw_digest_bytes() -> None:
    assert witness_signature_preimage("00" * 32) == (
        b"hermes-governance-commit-witness-signature/0.9\x00" + bytes(32)
    )
    with pytest.raises(CanonicalEncodingError):
        witness_signature_preimage("00" * 31)


@pytest.mark.parametrize("name", ["genesis", "prepared", "committed", "aborted"])
def test_witness_golden_vectors_match_exact_bytes_and_preimages(name: str) -> None:
    vector = golden_record(name)
    payload = vector["payload"]
    outer = vector["outerRecord"]
    frame = bytes.fromhex(vector["frameHex"])

    assert canonical_json_bytes(payload).hex() == vector["canonicalPayloadHex"]
    assert witness_record_digest(payload) == vector["recordDigest"]
    assert witness_signature_preimage(vector["recordDigest"]).hex() == vector[
        "signaturePreimageHex"
    ]
    assert encode_witness_frame(outer) == frame
    assert parse_witness_frame(frame) == outer


@pytest.mark.parametrize("mutation", ["missing-lf", "double-lf", "leading-space"])
def test_witness_frame_rejects_malformed_boundaries(mutation: str) -> None:
    frame = bytes.fromhex(golden_record()["frameHex"])
    malformed = {
        "missing-lf": frame[:-1],
        "double-lf": frame + b"\n",
        "leading-space": b" " + frame,
    }[mutation]
    with pytest.raises(CanonicalEncodingError):
        parse_witness_frame(malformed)


def test_witness_frame_rejects_duplicate_outer_field() -> None:
    frame = bytes.fromhex(golden_record()["frameHex"])
    duplicate = frame.replace(
        b'{"recordDigest":',
        b'{"recordDigest":"' + b"0" * 64 + b'","recordDigest":',
        1,
    )
    with pytest.raises(CanonicalEncodingError):
        parse_witness_frame(duplicate)


def test_witness_encoder_rejects_unknown_fields_and_digest_mismatch() -> None:
    outer = dict(golden_record()["outerRecord"])
    outer["unknown"] = None
    with pytest.raises(CanonicalEncodingError):
        encode_witness_frame(outer)

    outer = dict(golden_record()["outerRecord"])
    payload = dict(outer["witnessPayload"])
    payload["unknown"] = None
    outer["witnessPayload"] = payload
    with pytest.raises(CanonicalEncodingError):
        encode_witness_frame(outer)

    outer = dict(golden_record()["outerRecord"])
    outer["recordDigest"] = "0" * 64
    with pytest.raises(CanonicalEncodingError):
        encode_witness_frame(outer)


def test_witness_genesis_rejects_nonnull_transaction_state() -> None:
    outer = dict(golden_record()["outerRecord"])
    payload = dict(outer["witnessPayload"])
    payload["witnessTransactionIdOrNull"] = "018f47a0-0000-7000-8000-000000000000"
    outer["witnessPayload"] = payload
    outer["recordDigest"] = domain_separated_json_digest(
        "hermes-governance-commit-witness-digest/0.9", payload
    )
    with pytest.raises(CanonicalEncodingError):
        encode_witness_frame(outer)


def test_witness_genesis_accepts_owner_approved_root_slash() -> None:
    outer = dict(golden_record()["outerRecord"])
    payload = dict(outer["witnessPayload"])
    installation = dict(payload["installationIdentityOrNull"])
    installation["ownerApprovedRootCanonicalUtf8"] = "/"
    payload["installationIdentityOrNull"] = installation
    payload["installationIdentityDigest"] = domain_separated_json_digest(
        "hermes-witness-installation-identity/0.9", installation
    )
    outer["witnessPayload"] = payload
    outer["recordDigest"] = domain_separated_json_digest(
        "hermes-governance-commit-witness-digest/0.9", payload
    )
    assert parse_witness_frame(encode_witness_frame(outer)) == outer


def test_malformed_witness_types_use_the_single_fail_closed_error_class() -> None:
    payload = dict(golden_record()["payload"])
    payload["recordType"] = []
    with pytest.raises(CanonicalEncodingError):
        witness_record_digest(payload)


def test_witness_schema_version_change_fails_even_with_matching_digest() -> None:
    outer = dict(golden_record()["outerRecord"])
    payload = dict(outer["witnessPayload"])
    payload["witnessSchemaVersion"] = "governance-commit-witness/1.0"
    outer["witnessPayload"] = payload
    outer["recordDigest"] = domain_separated_json_digest(
        "hermes-governance-commit-witness-digest/0.9", payload
    )
    with pytest.raises(CanonicalEncodingError):
        encode_witness_frame(outer)


def test_all_golden_vectors_are_deterministic_across_fresh_processes() -> None:
    root = Path(__file__).parents[2]
    vector_path = root / "tests/repo_governance/vectors/d1_canonical_golden.json"
    program = """
import json
from pathlib import Path
from repo_governance.canonical import encode_witness_frame, witness_record_digest
vectors = json.loads(Path(%r).read_text())["records"]
for name in sorted(vectors):
    vector = vectors[name]
    print(name, witness_record_digest(vector["payload"]), encode_witness_frame(vector["outerRecord"]).hex())
""" % str(vector_path)

    outputs = []
    for seed in ("1", "8675309"):
        env = dict(os.environ)
        env.update(PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.encode("utf-8"))
    assert outputs[0] == outputs[1]
