"""Offline verifier for externally executed full Muncho canary evidence.

This module deliberately does not call a model, Discord, PostgreSQL, or
systemd.  A separately approved live driver runs through the sealed gateway
and services, then supplies the explicit fixture and service-generated
receipts validated here.  The split prevents a fixture or this verifier from
being mistaken for proof that a real external operation occurred.

The verifier contains no semantic classifier.  It checks only exact IDs,
cryptographic bindings, structured state transitions, and receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
)
from gateway.discord_edge_writer_authority import (
    derive_routeback_edge_idempotency_key,
)
from gateway.discord_edge_protocol import (
    DiscordEdgeAuthorityKind,
    DiscordEdgeErrorCode,
    DiscordEdgeOperation,
    DiscordEdgeReceiptOutcome,
    DiscordPublicTarget,
    SignedDiscordEdgeEnvelope,
    parse_request_for_reconciliation,
    verify_receipt,
    verify_request_capability_for_reconciliation,
)


FIXTURE_SCHEMA = "muncho-full-canary-e2e-fixture.v1"
EVIDENCE_SCHEMA = "muncho-full-canary-e2e-evidence.v1"
INVARIANT_RECEIPT_SCHEMA = "muncho-full-canary-e2e-verification.v1"
MODEL_CALL_RECEIPT_SCHEMA = "muncho-live-model-call-receipt.v1"
REASONING_RECEIPT_SCHEMA = "muncho-model-reasoning-directive-receipt.v1"
CANONICAL_TRUTH_RECEIPT_SCHEMA = "muncho-canonical-truth-readback-receipt.v1"
TASK_OUTCOME_RECEIPT_SCHEMA = "muncho-full-canary-task-outcome-receipt.v1"
PRIVATE_DENIAL_RECEIPT_SCHEMA = "muncho-discord-private-denial-receipt.v1"
SOURCE_RECEIPT_SCHEMA = "muncho-live-api-source-receipt.v1"

MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_STEPS = 32
MAX_CRITERIA = 32
LIVE_FIXTURE_PATH = Path("/etc/muncho/full-canary/fixture.json")
LIVE_EVIDENCE_ROOT = Path("/var/lib/muncho-full-canary/plans")
_ZERO_SHA256 = "0" * 64
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_CASE_ID_RE = re.compile(r"^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_SNOWFLAKE_RE = re.compile(r"^[0-9]{1,25}$")
_HEALTHY_TURN_EXIT_RE = re.compile(r"^text_response\(finish_reason=stop\)$")

_MODEL_ROUTE = {
    "provider": "openai-codex",
    "api_mode": "codex_responses",
    "base_url": "https://chatgpt.com/backend-api/codex",
    "model": "gpt-5.6-sol",
    "initial_effort": "high",
    "elevated_effort": "xhigh",
}
_INVARIANTS = (
    "live_provenance_bound",
    "canonical_writer_ready",
    "owner_preapproved_one_shot_scope_claimed_and_durably_revoked",
    "gpt56_model_authored_high_to_xhigh",
    "canonical_plan_event_verification_truth_complete",
    "public_discord_routeback_signed_and_readback_verified",
    "discord_dm_private_target_denied_without_dispatch",
    "sustained_multistep_task_completed_nonpartial",
)


class CanaryEvidenceError(ValueError):
    """A fixed-code failure which never reflects untrusted receipt content."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise CanaryEvidenceError(code)


def _strict_object(
    value: Any,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        _fail(code)
    result = dict(value)
    if set(result) - set(required) - set(optional) or set(required) - set(result):
        _fail(code)
    return result


def _sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail(code)
    return value


def _git_sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _GIT_SHA_RE.fullmatch(value):
        _fail(code)
    return value


def _safe_id(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        _fail(code)
    return value


def _case_id(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _CASE_ID_RE.fullmatch(value):
        _fail(code)
    return value


def _uuid(value: Any, code: str) -> str:
    if not isinstance(value, str):
        _fail(code)
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        _fail(code)
    if parsed.int == 0 or str(parsed) != value:
        _fail(code)
    return value


def _positive_int(value: Any, code: str) -> int:
    if type(value) is not int or value < 1 or value >= 1 << 63:
        _fail(code)
    return value


def _snowflake(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _SNOWFLAKE_RE.fullmatch(value):
        _fail(code)
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("non_canonical_json_value")


def _digest_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _decode_strict_json(body: bytes, code: str) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise ValueError("non-json numeric constant")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            body.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _fail(code)
    if not isinstance(value, dict):
        _fail(code)
    return value


def _read_bound_artifact(
    path: Path,
    expected_sha256: str,
    code: str,
    *,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    expected_mode: int | None = None,
) -> tuple[dict[str, Any], str]:
    expected = _sha256(expected_sha256, code)
    raw = os.fspath(path)
    if not path.is_absolute() or os.path.normpath(raw) != raw:
        _fail(code)
    try:
        observed = path.lstat()
    except OSError:
        _fail(code)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_mode & 0o022
        or observed.st_size < 2
        or observed.st_size > MAX_ARTIFACT_BYTES
        or expected_uid is not None
        and observed.st_uid != expected_uid
        or expected_gid is not None
        and observed.st_gid != expected_gid
        or expected_mode is not None
        and stat.S_IMODE(observed.st_mode) != expected_mode
    ):
        _fail(code)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError:
        _fail(code)
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(observed, name) != getattr(opened, name)
            for name in identity_fields
        ):
            _fail(code)
        while total <= MAX_ARTIFACT_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, MAX_ARTIFACT_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    except OSError:
        _fail(code)
    finally:
        os.close(descriptor)
    try:
        reachable = path.lstat()
    except OSError:
        _fail(code)
    if (
        total > MAX_ARTIFACT_BYTES
        or total != observed.st_size
        or any(
            getattr(observed, name) != getattr(after, name)
            for name in identity_fields
        )
        or any(
            getattr(observed, name) != getattr(reachable, name)
            for name in identity_fields
        )
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
    ):
        _fail(code)
    body = b"".join(chunks)
    digest = hashlib.sha256(body).hexdigest()
    if digest != expected:
        _fail(code)
    return _decode_strict_json(body, code), digest


def _validate_fixture(value: Any) -> dict[str, Any]:
    fixture = _strict_object(
        value,
        required=(
            "schema",
            "canary_run_id",
            "release_sha",
            "release_artifact_sha256",
            "valid_from_unix_ms",
            "valid_until_unix_ms",
            "case_id",
            "owner_discord_user_id",
            "source",
            "model_route",
            "task_policy",
            "public_routeback",
            "discord_public_keys",
        ),
        code="fixture_shape_invalid",
    )
    if fixture["schema"] != FIXTURE_SCHEMA:
        _fail("fixture_schema_invalid")
    _uuid(fixture["canary_run_id"], "fixture_run_id_invalid")
    _git_sha(fixture["release_sha"], "fixture_release_invalid")
    _sha256(fixture["release_artifact_sha256"], "fixture_release_invalid")
    valid_from = _positive_int(fixture["valid_from_unix_ms"], "fixture_window_invalid")
    valid_until = _positive_int(
        fixture["valid_until_unix_ms"], "fixture_window_invalid"
    )
    if valid_until <= valid_from or valid_until - valid_from > 3_600_000:
        _fail("fixture_window_invalid")
    _case_id(fixture["case_id"], "fixture_case_invalid")
    _snowflake(fixture["owner_discord_user_id"], "fixture_owner_invalid")

    source = _strict_object(
        fixture["source"],
        required=(
            "platform",
            "control_protocol",
            "host",
            "port",
            "session_create_endpoint",
            "chat_stream_endpoint_template",
        ),
        code="fixture_source_invalid",
    )
    if source != {
        "platform": "api_server",
        "control_protocol": "authenticated_loopback_api_server.v1",
        "host": "127.0.0.1",
        "port": 8642,
        "session_create_endpoint": "/api/sessions",
        "chat_stream_endpoint_template": "/api/sessions/{session_id}/chat/stream",
    }:
        _fail("fixture_source_invalid")

    route = _strict_object(
        fixture["model_route"],
        required=tuple(_MODEL_ROUTE),
        code="fixture_model_route_invalid",
    )
    if route != _MODEL_ROUTE:
        _fail("fixture_model_route_invalid")

    task_policy = _strict_object(
        fixture["task_policy"],
        required=("minimum_completed_steps", "prompt", "prompt_sha256"),
        code="fixture_task_invalid",
    )
    minimum_steps = task_policy["minimum_completed_steps"]
    if type(minimum_steps) is not int or not 3 <= minimum_steps <= MAX_STEPS:
        _fail("fixture_task_invalid")
    prompt = task_policy["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16_000:
        _fail("fixture_task_invalid")
    prompt_sha256 = _sha256(task_policy["prompt_sha256"], "fixture_task_invalid")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_sha256:
        _fail("fixture_task_invalid")

    routeback = _strict_object(
        fixture["public_routeback"],
        required=("target", "canonical_idempotency_key"),
        code="fixture_routeback_invalid",
    )
    try:
        target = DiscordPublicTarget.from_mapping(routeback["target"])
    except (TypeError, ValueError):
        _fail("fixture_routeback_invalid")
    _safe_id(routeback["canonical_idempotency_key"], "fixture_routeback_invalid")
    del target

    keys = _strict_object(
        fixture["discord_public_keys"],
        required=("writer_capability_ed25519_hex", "edge_receipt_ed25519_hex"),
        code="fixture_discord_keys_invalid",
    )
    for key in keys.values():
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            _fail("fixture_discord_keys_invalid")
    if len(set(keys.values())) != 2:
        _fail("fixture_discord_keys_invalid")
    return fixture


def _validate_live_provenance(
    fixture: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    start_receipt_sha256: str,
) -> None:
    provenance = _strict_object(
        evidence.get("runtime_provenance"),
        required=(
            "execution_mode",
            "synthetic",
            "release_sha",
            "canary_run_id",
            "owner_discord_user_id",
            "full_canary_start_receipt_sha256",
            "gateway_service_identity_sha256",
            "canonical_writer_service_identity_sha256",
            "discord_edge_service_identity_sha256",
            "collector_receipt_sha256",
        ),
        code="live_provenance_invalid",
    )
    if (
        provenance["execution_mode"] != "live_isolated_canary"
        or provenance["synthetic"] is not False
        or provenance["release_sha"] != fixture["release_sha"]
        or provenance["canary_run_id"] != fixture["canary_run_id"]
        or provenance["owner_discord_user_id"] != fixture["owner_discord_user_id"]
        or provenance["full_canary_start_receipt_sha256"] != start_receipt_sha256
    ):
        _fail("live_provenance_invalid")
    for key in (
        "full_canary_start_receipt_sha256",
        "gateway_service_identity_sha256",
        "canonical_writer_service_identity_sha256",
        "discord_edge_service_identity_sha256",
        "collector_receipt_sha256",
    ):
        _sha256(provenance[key], "live_provenance_invalid")
    if (
        len({provenance[key] for key in provenance if key.endswith("identity_sha256")})
        != 3
    ):
        _fail("live_provenance_invalid")


def _validate_source_receipt(
    fixture: Mapping[str, Any],
    value: Any,
) -> int:
    """Bind the authenticated loopback control request to the observed turn.

    This canary has no Discord ingress and this receipt must never be treated
    as evidence that Discord delivered an inbound message.
    """

    receipt = _strict_object(
        value,
        required=(
            "schema",
            "provenance",
            "release_sha",
            "canary_run_id",
            "session_id",
            "turn_id",
            "platform",
            "control_protocol",
            "host",
            "port",
            "session_create_endpoint",
            "chat_stream_endpoint",
            "session_create_request_id",
            "chat_stream_request_id",
            "api_run_id",
            "api_message_id",
            "loopback_peer_verified",
            "credential_provenance_receipt_sha256",
            "session_key_sha256",
            "capability_epoch_sha256",
            "message_content_sha256",
            "observed_at_unix_ms",
        ),
        code="source_receipt_invalid",
    )
    source = fixture["source"]
    if (
        receipt["schema"] != SOURCE_RECEIPT_SCHEMA
        or receipt["provenance"] != "live_gateway_authenticated_loopback_api"
        or receipt["release_sha"] != fixture["release_sha"]
        or receipt["canary_run_id"] != fixture["canary_run_id"]
        or receipt["platform"] != source["platform"]
        or receipt["control_protocol"] != source["control_protocol"]
        or receipt["host"] != source["host"]
        or receipt["port"] != source["port"]
        or receipt["session_create_endpoint"] != source["session_create_endpoint"]
        or receipt["chat_stream_endpoint"]
        != source["chat_stream_endpoint_template"].replace(
            "{session_id}", receipt["session_id"]
        )
        or receipt["loopback_peer_verified"] is not True
        or receipt["message_content_sha256"] != fixture["task_policy"]["prompt_sha256"]
    ):
        _fail("source_receipt_invalid")
    _safe_id(receipt["session_id"], "source_receipt_invalid")
    _safe_id(receipt["turn_id"], "source_receipt_invalid")
    _uuid(receipt["session_create_request_id"], "source_receipt_invalid")
    _uuid(receipt["chat_stream_request_id"], "source_receipt_invalid")
    if receipt["session_create_request_id"] == receipt["chat_stream_request_id"]:
        _fail("source_receipt_invalid")
    _safe_id(receipt["api_run_id"], "source_receipt_invalid")
    _safe_id(receipt["api_message_id"], "source_receipt_invalid")
    if not receipt["api_run_id"].startswith("run_"):
        _fail("source_receipt_invalid")
    if not receipt["api_message_id"].startswith("msg_"):
        _fail("source_receipt_invalid")
    _sha256(
        receipt["credential_provenance_receipt_sha256"],
        "source_receipt_invalid",
    )
    _sha256(receipt["session_key_sha256"], "source_receipt_invalid")
    _sha256(receipt["capability_epoch_sha256"], "source_receipt_invalid")
    _sha256(receipt["message_content_sha256"], "source_receipt_invalid")
    observed_at = _positive_int(
        receipt["observed_at_unix_ms"], "source_receipt_invalid"
    )
    if (
        not fixture["valid_from_unix_ms"]
        <= observed_at
        <= fixture["valid_until_unix_ms"]
    ):
        _fail("source_receipt_invalid")
    return observed_at


def _validate_writer_readiness(value: Any) -> None:
    receipt = _strict_object(
        value,
        required=(
            "version",
            "observed_at_unix",
            "observed_at_boottime_ns",
            "boot_id_sha256",
            "gateway_pid",
            "gateway_start_time_ticks",
            "writer_request_id",
            "writer_service",
            "writer_protocol",
            "database_identity",
            "gateway_module_origin",
            "gateway_module_sha256",
            "gateway_dumpable",
            "gateway_core_soft_limit",
            "gateway_core_hard_limit",
            "effective_import_paths",
            "unexpected_import_paths",
            "loaded_module_origins",
            "unexpected_import_origins",
            "loaded_module_origins_complete",
            "effective_environment_variable_names",
            "effective_environment_variable_value_sha256",
        ),
        code="writer_readiness_invalid",
    )
    if (
        receipt["version"] != "canonical-writer-readiness-v1"
        or receipt["writer_service"] != "canonical_writer"
        or receipt["writer_protocol"] != "v1"
        or receipt["database_identity"] != CANONICAL_WRITER_MIGRATION_OWNER
        or receipt["gateway_dumpable"] is not False
        or receipt["gateway_core_soft_limit"] != 0
        or receipt["gateway_core_hard_limit"] != 0
        or receipt["loaded_module_origins_complete"] is not True
        or receipt["unexpected_import_paths"] != []
        or receipt["unexpected_import_origins"] != []
    ):
        _fail("writer_readiness_invalid")
    _positive_int(receipt["observed_at_unix"], "writer_readiness_invalid")
    _positive_int(receipt["observed_at_boottime_ns"], "writer_readiness_invalid")
    _positive_int(receipt["gateway_pid"], "writer_readiness_invalid")
    _positive_int(receipt["gateway_start_time_ticks"], "writer_readiness_invalid")
    _uuid(receipt["writer_request_id"], "writer_readiness_invalid")
    _sha256(receipt["boot_id_sha256"], "writer_readiness_invalid")
    _sha256(receipt["gateway_module_sha256"], "writer_readiness_invalid")
    if not isinstance(receipt["gateway_module_origin"], str) or not receipt[
        "gateway_module_origin"
    ].startswith("/"):
        _fail("writer_readiness_invalid")
    for key in (
        "effective_import_paths",
        "loaded_module_origins",
        "effective_environment_variable_names",
    ):
        items = receipt[key]
        if (
            not isinstance(items, list)
            or items != sorted(set(items))
            or any(not isinstance(item, str) for item in items)
        ):
            _fail("writer_readiness_invalid")
    env_digests = receipt["effective_environment_variable_value_sha256"]
    if (
        not isinstance(env_digests, Mapping)
        or list(env_digests) != receipt["effective_environment_variable_names"]
    ):
        _fail("writer_readiness_invalid")
    for digest in env_digests.values():
        _sha256(digest, "writer_readiness_invalid")


def _validate_model_call(
    fixture: Mapping[str, Any],
    value: Any,
    *,
    ordinal: int,
    effort: str,
    source_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _strict_object(
        value,
        required=(
            "schema",
            "provenance",
            "release_sha",
            "canary_run_id",
            "session_id",
            "turn_id",
            "request_ordinal",
            "provider",
            "api_mode",
            "base_url",
            "model",
            "reasoning_effort",
            "api_request_sha256",
            "response_payload_sha256",
            "response_model",
            "response_observed_at_unix_ms",
            "assistant_tool_call_ids",
        ),
        code="model_call_receipt_invalid",
    )
    expected_route = fixture["model_route"]
    if (
        receipt["schema"] != MODEL_CALL_RECEIPT_SCHEMA
        or receipt["provenance"] != "live_gateway_model_adapter"
        or receipt["release_sha"] != fixture["release_sha"]
        or receipt["canary_run_id"] != fixture["canary_run_id"]
        or receipt["session_id"] != source_receipt["session_id"]
        or receipt["turn_id"] != source_receipt["turn_id"]
        or receipt["request_ordinal"] != ordinal
        or receipt["provider"] != expected_route["provider"]
        or receipt["api_mode"] != expected_route["api_mode"]
        or receipt["base_url"] != expected_route["base_url"]
        or receipt["model"] != expected_route["model"]
        or receipt["reasoning_effort"] != effort
        or receipt["response_model"] != expected_route["model"]
    ):
        _fail("model_call_receipt_invalid")
    _sha256(receipt["api_request_sha256"], "model_call_receipt_invalid")
    _sha256(receipt["response_payload_sha256"], "model_call_receipt_invalid")
    observed_at = _positive_int(
        receipt["response_observed_at_unix_ms"], "model_call_receipt_invalid"
    )
    if (
        observed_at < source_receipt["observed_at_unix_ms"]
        or not fixture["valid_from_unix_ms"]
        <= observed_at
        <= fixture["valid_until_unix_ms"]
    ):
        _fail("model_call_receipt_invalid")
    call_ids = receipt["assistant_tool_call_ids"]
    if not isinstance(call_ids, list) or len(set(call_ids)) != len(call_ids):
        _fail("model_call_receipt_invalid")
    for call_id in call_ids:
        _safe_id(call_id, "model_call_receipt_invalid")
    return receipt


def _validate_reasoning_transition(
    fixture: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    source_receipt: Mapping[str, Any],
) -> tuple[int, int]:
    calls = evidence.get("model_calls")
    if not isinstance(calls, list) or len(calls) < 2:
        _fail("adaptive_reasoning_evidence_invalid")
    first = _validate_model_call(
        fixture,
        calls[0],
        ordinal=1,
        effort="high",
        source_receipt=source_receipt,
    )
    second = _validate_model_call(
        fixture,
        calls[1],
        ordinal=2,
        effort="xhigh",
        source_receipt=source_receipt,
    )
    validated_calls = [first, second]
    for ordinal, call in enumerate(calls[2:], start=3):
        validated_calls.append(
            _validate_model_call(
                fixture,
                call,
                ordinal=ordinal,
                effort="xhigh",
                source_receipt=source_receipt,
            )
        )

    observed_times = [call["response_observed_at_unix_ms"] for call in validated_calls]
    all_tool_call_ids = [
        call_id
        for call in validated_calls
        for call_id in call["assistant_tool_call_ids"]
    ]
    if (
        observed_times != sorted(observed_times)
        or any(not call["assistant_tool_call_ids"] for call in validated_calls[:-1])
        or validated_calls[-1]["assistant_tool_call_ids"] != []
        or len(all_tool_call_ids) != len(set(all_tool_call_ids))
    ):
        _fail("model_call_receipt_invalid")

    directive = _strict_object(
        evidence.get("reasoning_directive"),
        required=(
            "schema",
            "provenance",
            "release_sha",
            "canary_run_id",
            "session_id",
            "turn_id",
            "tool_name",
            "tool_call_id",
            "model_authored",
            "directive",
            "reasoning_control",
            "produced_by_model_call_ordinal",
            "applied_before_model_call_ordinal",
            "todo_result_sha256",
        ),
        code="adaptive_reasoning_evidence_invalid",
    )
    control = _strict_object(
        directive.get("reasoning_control"),
        required=("status", "scope", "expires", "effective", "change_count"),
        code="adaptive_reasoning_evidence_invalid",
    )
    if (
        directive["schema"] != REASONING_RECEIPT_SCHEMA
        or directive["provenance"] != "live_gateway_assistant_tool_call"
        or directive["release_sha"] != fixture["release_sha"]
        or directive["canary_run_id"] != fixture["canary_run_id"]
        or directive["session_id"] != source_receipt["session_id"]
        or directive["turn_id"] != source_receipt["turn_id"]
        or directive["tool_name"] != "todo"
        or directive["model_authored"] is not True
        or directive["directive"] != {"effort": "xhigh"}
        or control.get("status") != "applied"
        or control.get("scope") != "current_turn"
        or control.get("expires") != "end_of_current_turn"
        or control.get("effective") != {"effort": "xhigh"}
        or control.get("change_count") != 1
        or directive["produced_by_model_call_ordinal"] != 1
        or directive["applied_before_model_call_ordinal"] != 2
        or second["response_observed_at_unix_ms"]
        < first["response_observed_at_unix_ms"]
        or directive["tool_call_id"] not in first["assistant_tool_call_ids"]
    ):
        _fail("adaptive_reasoning_evidence_invalid")
    _safe_id(directive["tool_call_id"], "adaptive_reasoning_evidence_invalid")
    _sha256(directive["todo_result_sha256"], "adaptive_reasoning_evidence_invalid")
    return len(validated_calls), len(all_tool_call_ids)


def _normalized_plan_event(
    fixture: Mapping[str, Any], value: Any, *, revision: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    event = _strict_object(
        value,
        required=(
            "event_id",
            "event_type",
            "case_id",
            "readback_verified",
            "canonical_content_sha256",
            "plan",
        ),
        code="canonical_plan_truth_invalid",
    )
    plan = _strict_object(
        event["plan"],
        required=(
            "plan_id",
            "revision",
            "state",
            "current_step_id",
            "resume_cursor",
            "steps",
            "criterion_ids",
            "verification_event_ids",
        ),
        code="canonical_plan_truth_invalid",
    )
    if (
        event["event_type"] != "task.plan.updated"
        or event["case_id"] != fixture["case_id"]
        or event["readback_verified"] is not True
        or plan["revision"] != revision
    ):
        _fail("canonical_plan_truth_invalid")
    _uuid(event["event_id"], "canonical_plan_truth_invalid")
    _sha256(event["canonical_content_sha256"], "canonical_plan_truth_invalid")
    _safe_id(plan["plan_id"], "canonical_plan_truth_invalid")
    if not isinstance(plan["steps"], list):
        _fail("canonical_plan_truth_invalid")
    steps: list[dict[str, Any]] = []
    for raw_step in plan["steps"]:
        step = _strict_object(
            raw_step,
            required=("id", "status", "depends_on"),
            code="canonical_plan_truth_invalid",
        )
        _safe_id(step["id"], "canonical_plan_truth_invalid")
        dependencies = step["depends_on"]
        if (
            step["status"] not in {"pending", "in_progress", "completed"}
            or not isinstance(dependencies, list)
            or len(set(dependencies)) != len(dependencies)
        ):
            _fail("canonical_plan_truth_invalid")
        for dependency in dependencies:
            _safe_id(dependency, "canonical_plan_truth_invalid")
        steps.append(step)
    criteria = plan["criterion_ids"]
    if (
        not isinstance(criteria, list)
        or not 1 <= len(criteria) <= MAX_CRITERIA
        or len(set(criteria)) != len(criteria)
    ):
        _fail("canonical_plan_truth_invalid")
    for criterion_id in criteria:
        _safe_id(criterion_id, "canonical_plan_truth_invalid")
    cursor = _strict_object(
        plan["resume_cursor"],
        required=("next_step_id",),
        optional=("summary",),
        code="canonical_plan_truth_invalid",
    )
    if plan["current_step_id"] is not None:
        _safe_id(plan["current_step_id"], "canonical_plan_truth_invalid")
    if cursor["next_step_id"] is not None:
        _safe_id(cursor["next_step_id"], "canonical_plan_truth_invalid")
    if not isinstance(plan["verification_event_ids"], list):
        _fail("canonical_plan_truth_invalid")
    return event, {**plan, "steps": steps, "resume_cursor": cursor}


def _validate_plan_sequence(
    fixture: Mapping[str, Any], plan_events: Any
) -> dict[str, Any]:
    minimum_steps = fixture["task_policy"]["minimum_completed_steps"]
    if not isinstance(plan_events, list) or len(plan_events) < minimum_steps + 1:
        _fail("canonical_plan_truth_invalid")
    normalized = [
        _normalized_plan_event(fixture, event, revision=revision)
        for revision, event in enumerate(plan_events, start=1)
    ]
    first_plan = normalized[0][1]
    step_ids = [step["id"] for step in first_plan["steps"]]
    if (
        not minimum_steps <= len(step_ids) <= MAX_STEPS
        or len(set(step_ids)) != len(step_ids)
        or len(normalized) != len(step_ids) + 1
    ):
        _fail("canonical_plan_truth_invalid")
    known_ids = set(step_ids)
    dependency_map = {
        step["id"]: tuple(step["depends_on"]) for step in first_plan["steps"]
    }
    if any(
        dependency not in known_ids or dependency == step_id
        for step_id, dependencies in dependency_map.items()
        for dependency in dependencies
    ):
        _fail("canonical_plan_truth_invalid")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visiting:
            _fail("canonical_plan_truth_invalid")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in dependency_map[step_id]:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in step_ids:
        visit(step_id)

    plan_id = first_plan["plan_id"]
    criterion_ids = list(first_plan["criterion_ids"])
    previous_status = {step["id"]: step["status"] for step in first_plan["steps"]}
    if any(status == "completed" for status in previous_status.values()):
        _fail("canonical_plan_truth_invalid")
    rank = {"pending": 0, "in_progress": 1, "completed": 2}
    completion_order: list[str] = []
    seen_event_ids: set[str] = set()
    previous_current = first_plan["current_step_id"]

    for index, (event, plan) in enumerate(normalized):
        final = index == len(normalized) - 1
        if event["event_id"] in seen_event_ids:
            _fail("canonical_plan_truth_invalid")
        seen_event_ids.add(event["event_id"])
        current_steps = plan["steps"]
        current_ids = [step["id"] for step in current_steps]
        current_dependencies = {
            step["id"]: tuple(step["depends_on"]) for step in current_steps
        }
        current_status = {step["id"]: step["status"] for step in current_steps}
        if (
            plan["plan_id"] != plan_id
            or current_ids != step_ids
            or current_dependencies != dependency_map
            or plan["criterion_ids"] != criterion_ids
        ):
            _fail("canonical_plan_truth_invalid")
        if index:
            newly_completed = [
                step_id
                for step_id in step_ids
                if previous_status[step_id] != "completed"
                and current_status[step_id] == "completed"
            ]
            if newly_completed != [previous_current] or any(
                rank[current_status[step_id]] < rank[previous_status[step_id]]
                for step_id in step_ids
            ):
                _fail("canonical_plan_truth_invalid")
            completion_order.extend(newly_completed)
        completed = {
            step_id
            for step_id, status in current_status.items()
            if status == "completed"
        }
        in_progress = [
            step_id
            for step_id, status in current_status.items()
            if status == "in_progress"
        ]
        if len(in_progress) > 1 or any(
            not set(dependency_map[step_id]) <= completed
            for step_id, status in current_status.items()
            if status in {"in_progress", "completed"}
        ):
            _fail("canonical_plan_truth_invalid")
        if final:
            verification_ids = plan["verification_event_ids"]
            if (
                plan["state"] != "completed"
                or completed != known_ids
                or in_progress
                or plan["current_step_id"] is not None
                or plan["resume_cursor"]["next_step_id"] is not None
                or not 1 <= len(verification_ids) <= MAX_CRITERIA
                or len(set(verification_ids)) != len(verification_ids)
            ):
                _fail("canonical_plan_truth_invalid")
            for event_id in verification_ids:
                _uuid(event_id, "canonical_plan_truth_invalid")
        else:
            current = plan["current_step_id"]
            if (
                plan["state"] != "active"
                or current not in known_ids
                or plan["resume_cursor"]["next_step_id"] != current
                or current_status[current] not in {"pending", "in_progress"}
                or (in_progress and in_progress != [current])
                or plan["verification_event_ids"] != []
            ):
                _fail("canonical_plan_truth_invalid")
        previous_status = current_status
        previous_current = plan["current_step_id"]
    if len(completion_order) != len(step_ids):
        _fail("canonical_plan_truth_invalid")
    return {
        "plan_id": plan_id,
        "step_ids": step_ids,
        "criterion_ids": criterion_ids,
        "verification_event_ids": list(normalized[-1][1]["verification_event_ids"]),
        "verification_revision": len(normalized) - 1,
        "completion_order": completion_order,
    }


def _validate_scope_lifecycle(
    fixture: Mapping[str, Any],
    events_value: Any,
    retirement_value: Any,
    *,
    fixture_sha256: str,
    source_receipt: Mapping[str, Any],
) -> None:
    """Prove that the one-shot owner scope existed only for this API run."""

    code = "canonical_scope_truth_invalid"
    if not isinstance(events_value, list) or len(events_value) != 3:
        _fail(code)
    expected_types = (
        "canary.scope.preapproved",
        "canary.scope.claimed",
        "canary.scope.revoked",
    )
    normalized_events: list[tuple[dict[str, Any], dict[str, Any]]] = []
    occurred_at_values: list[int] = []
    for event_value, expected_type in zip(
        events_value, expected_types, strict=True
    ):
        event = _strict_object(
            event_value,
            required=(
                "event_id",
                "event_type",
                "case_id",
                "occurred_at_unix_ms",
                "readback_verified",
                "canonical_content_sha256",
                "scope",
            ),
            code=code,
        )
        occurred_at = _positive_int(event["occurred_at_unix_ms"], code)
        if (
            event["event_type"] != expected_type
            or event["case_id"] != fixture["case_id"]
            or event["readback_verified"] is not True
            or not fixture["valid_from_unix_ms"]
            <= occurred_at
            <= fixture["valid_until_unix_ms"]
        ):
            _fail(code)
        _uuid(event["event_id"], code)
        _sha256(event["canonical_content_sha256"], code)
        scope_required = {
            "canary.scope.preapproved": (
                "grant_id",
                "case_id",
                "release_sha256",
                "fixture_sha256",
                "run_id",
                "session_key_sha256",
                "expires_at_unix_ms",
                "approved_by",
                "approval_source_sha256",
                "state",
            ),
            "canary.scope.claimed": (
                "grant_id",
                "case_id",
                "release_sha256",
                "fixture_sha256",
                "run_id",
                "approval_source_sha256",
                "session_key_sha256",
                "capability_epoch_sha256",
                "expires_at_unix_ms",
                "state",
            ),
            "canary.scope.revoked": (
                "grant_id",
                "session_key_sha256",
                "capability_epoch_sha256",
                "reason",
                "session_tombstone_recorded",
                "state",
            ),
        }[expected_type]
        scope = _strict_object(event["scope"], required=scope_required, code=code)
        normalized_events.append((event, scope))
        occurred_at_values.append(occurred_at)

    if occurred_at_values != sorted(occurred_at_values):
        _fail(code)
    if len({event["event_id"] for event, _scope in normalized_events}) != 3:
        _fail(code)

    preapproval = normalized_events[0][1]
    claim = normalized_events[1][1]
    revocation = normalized_events[2][1]
    grant_id = _safe_id(preapproval["grant_id"], code)
    approval_source = _sha256(preapproval["approval_source_sha256"], code)
    session_key = _sha256(preapproval["session_key_sha256"], code)
    capability_epoch = _sha256(claim["capability_epoch_sha256"], code)
    expires_at = _positive_int(preapproval["expires_at_unix_ms"], code)
    if (
        preapproval["case_id"] != fixture["case_id"]
        or preapproval["release_sha256"]
        != fixture["release_artifact_sha256"]
        or preapproval["fixture_sha256"] != fixture_sha256
        or preapproval["run_id"] != fixture["canary_run_id"]
        or preapproval["approved_by"] != fixture["owner_discord_user_id"]
        or preapproval["state"] != "preapproved"
        or expires_at != fixture["valid_until_unix_ms"]
        or session_key != source_receipt["session_key_sha256"]
    ):
        _fail(code)

    for name in (
        "release_sha256",
        "fixture_sha256",
        "approval_source_sha256",
        "session_key_sha256",
    ):
        _sha256(claim[name], code)
    if (
        claim["grant_id"] != grant_id
        or claim["case_id"] != fixture["case_id"]
        or claim["release_sha256"] != preapproval["release_sha256"]
        or claim["fixture_sha256"] != fixture_sha256
        or claim["run_id"] != fixture["canary_run_id"]
        or claim["approval_source_sha256"] != approval_source
        or claim["session_key_sha256"] != session_key
        or claim["capability_epoch_sha256"]
        != source_receipt["capability_epoch_sha256"]
        or claim["expires_at_unix_ms"] != expires_at
        or claim["state"] != "claimed"
    ):
        _fail(code)

    if (
        revocation["grant_id"] != grant_id
        or _sha256(revocation["session_key_sha256"], code) != session_key
        or _sha256(revocation["capability_epoch_sha256"], code)
        != capability_epoch
        or revocation["reason"] != "api_server_run_finished"
        or revocation["session_tombstone_recorded"] is not True
        or revocation["state"] != "revoked"
    ):
        _fail(code)

    retirement = _strict_object(
        retirement_value,
        required=(
            "grant_id",
            "session_key_sha256",
            "capability_epoch_sha256",
            "authority_active",
            "revocation_event_id",
            "session_tombstone_commit_receipt_verified",
            "observed_at_unix_ms",
        ),
        code=code,
    )
    retirement_observed_at = _positive_int(retirement["observed_at_unix_ms"], code)
    if (
        retirement["grant_id"] != grant_id
        or _sha256(retirement["session_key_sha256"], code) != session_key
        or _sha256(retirement["capability_epoch_sha256"], code)
        != capability_epoch
        or retirement["authority_active"] is not False
        or retirement["revocation_event_id"] != normalized_events[2][0]["event_id"]
        or retirement["session_tombstone_commit_receipt_verified"] is not True
        or not occurred_at_values[-1]
        <= retirement_observed_at
        <= fixture["valid_until_unix_ms"]
    ):
        _fail(code)


def _validate_canonical_truth(
    fixture: Mapping[str, Any],
    value: Any,
    *,
    fixture_sha256: str,
    source_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _strict_object(
        value,
        required=(
            "schema",
            "provenance",
            "release_sha",
            "canary_run_id",
            "writer_query_request_id",
            "observed_at_unix_ms",
            "query_status",
            "query_view",
            "case_id",
            "support_incomplete",
            "plan_projection_complete",
            "completion_receipts_satisfied",
            "missing_verification_event_ids",
            "scope_events",
            "scope_retirement",
            "plan_events",
            "verification_events",
            "routeback_event",
        ),
        code="canonical_truth_invalid",
    )
    if (
        receipt["schema"] != CANONICAL_TRUTH_RECEIPT_SCHEMA
        or receipt["provenance"] != "canonical_writer_live_readback"
        or receipt["release_sha"] != fixture["release_sha"]
        or receipt["canary_run_id"] != fixture["canary_run_id"]
        or receipt["query_status"] != "CANONICAL_BRAIN_QUERY_PASS"
        or receipt["query_view"] != "resume_bundle"
        or receipt["case_id"] != fixture["case_id"]
        or receipt["support_incomplete"] is not False
        or receipt["plan_projection_complete"] is not True
        or receipt["completion_receipts_satisfied"] is not True
        or receipt["missing_verification_event_ids"] != []
    ):
        _fail("canonical_truth_invalid")
    _uuid(receipt["writer_query_request_id"], "canonical_truth_invalid")
    observed_at = _positive_int(
        receipt["observed_at_unix_ms"], "canonical_truth_invalid"
    )
    if (
        not fixture["valid_from_unix_ms"]
        <= observed_at
        <= fixture["valid_until_unix_ms"]
    ):
        _fail("canonical_truth_invalid")
    _validate_scope_lifecycle(
        fixture,
        receipt["scope_events"],
        receipt["scope_retirement"],
        fixture_sha256=fixture_sha256,
        source_receipt=source_receipt,
    )
    plan_summary = _validate_plan_sequence(fixture, receipt["plan_events"])
    verification_events = receipt["verification_events"]
    expected_event_ids = plan_summary["verification_event_ids"]
    if not isinstance(verification_events, list) or len(verification_events) != len(
        expected_event_ids
    ):
        _fail("canonical_verification_truth_invalid")
    covered_criteria: list[str] = []
    for event, expected_event_id in zip(
        verification_events, expected_event_ids, strict=True
    ):
        normalized = _strict_object(
            event,
            required=(
                "event_id",
                "event_type",
                "case_id",
                "readback_verified",
                "canonical_content_sha256",
                "verification",
            ),
            code="canonical_verification_truth_invalid",
        )
        verification = _strict_object(
            normalized["verification"],
            required=(
                "verification_id",
                "plan_id",
                "plan_revision",
                "criterion_ids",
                "outcome",
                "receipt",
            ),
            code="canonical_verification_truth_invalid",
        )
        verification_receipt = _strict_object(
            verification["receipt"],
            required=("kind", "sha256"),
            code="canonical_verification_truth_invalid",
        )
        if (
            normalized["event_id"] != expected_event_id
            or normalized["event_type"] != "task.verification.recorded"
            or normalized["case_id"] != fixture["case_id"]
            or normalized["readback_verified"] is not True
            or verification["plan_id"] != plan_summary["plan_id"]
            or verification["plan_revision"] != plan_summary["verification_revision"]
            or verification["outcome"] != "passed"
            or not isinstance(verification["criterion_ids"], list)
            or not verification["criterion_ids"]
            or not isinstance(verification_receipt["kind"], str)
            or not verification_receipt["kind"]
        ):
            _fail("canonical_verification_truth_invalid")
        _safe_id(
            verification["verification_id"], "canonical_verification_truth_invalid"
        )
        _sha256(
            normalized["canonical_content_sha256"],
            "canonical_verification_truth_invalid",
        )
        _sha256(verification_receipt["sha256"], "canonical_verification_truth_invalid")
        covered_criteria.extend(verification["criterion_ids"])
    if sorted(covered_criteria) != sorted(plan_summary["criterion_ids"]):
        _fail("canonical_verification_truth_invalid")
    return receipt, plan_summary


def _public_key(value: str, code: str) -> Ed25519PublicKey:
    try:
        return Ed25519PublicKey.from_public_bytes(bytes.fromhex(value))
    except (ValueError, TypeError):
        _fail(code)


def _validate_public_routeback(
    fixture: Mapping[str, Any],
    evidence: Mapping[str, Any],
    canonical_truth: Mapping[str, Any],
) -> None:
    routeback = _strict_object(
        evidence.get("public_routeback"),
        required=("provenance", "discord_edge_request", "discord_edge_receipt"),
        code="public_routeback_invalid",
    )
    if routeback["provenance"] != "live_discord_edge_signed_receipt":
        _fail("public_routeback_invalid")
    keys = fixture["discord_public_keys"]
    writer_key = _public_key(
        keys["writer_capability_ed25519_hex"], "public_routeback_invalid"
    )
    edge_key = _public_key(keys["edge_receipt_ed25519_hex"], "public_routeback_invalid")
    try:
        request = parse_request_for_reconciliation(routeback["discord_edge_request"])
        capability = verify_request_capability_for_reconciliation(request, writer_key)
        envelope = SignedDiscordEdgeEnvelope.from_mapping(
            routeback["discord_edge_receipt"],
            code=DiscordEdgeErrorCode.INVALID_RECEIPT,
            label="canary Discord edge receipt",
        )
        receipt = verify_receipt(
            envelope,
            edge_key,
            expected_request=request,
            expected_capability=capability,
            now_unix_ms=int(envelope.payload.get("occurred_at_unix_ms") or 0),
        )
        expected_target = DiscordPublicTarget.from_mapping(
            fixture["public_routeback"]["target"]
        )
    except (TypeError, ValueError):
        _fail("public_routeback_invalid")
    if (
        request.intent.operation is not DiscordEdgeOperation.PUBLIC_MESSAGE_SEND
        or request.intent.target != expected_target
        or request.intent.idempotency_key
        != derive_routeback_edge_idempotency_key(
            case_id=fixture["case_id"],
            canonical_idempotency_key=fixture["public_routeback"][
                "canonical_idempotency_key"
            ],
        )
        or capability.authority_kind is not DiscordEdgeAuthorityKind.CANONICAL_ROUTEBACK
        or receipt.outcome is not DiscordEdgeReceiptOutcome.VERIFIED
        or receipt.adapter_accepted is not True
        or receipt.readback_verified is not True
        or receipt.discord_object_id is None
        or receipt.content_sha256 != request.intent.content_sha256
        or not fixture["valid_from_unix_ms"]
        <= capability.issued_at_unix_ms
        <= receipt.occurred_at_unix_ms
        <= fixture["valid_until_unix_ms"]
        or capability.expires_at_unix_ms < receipt.occurred_at_unix_ms
        or request.deadline_unix_ms < receipt.occurred_at_unix_ms
    ):
        _fail("public_routeback_invalid")

    event = _strict_object(
        canonical_truth["routeback_event"],
        required=(
            "event_id",
            "event_type",
            "case_id",
            "readback_verified",
            "canonical_content_sha256",
            "canonical_idempotency_key",
            "authorization_id",
            "target_ref",
            "receipt",
        ),
        code="canonical_routeback_truth_invalid",
    )
    canonical_receipt = _strict_object(
        event["receipt"],
        required=(
            "platform",
            "adapter_receipt",
            "receipt_readback_verified",
            "message_id",
            "channel_id",
            "content_sha256",
        ),
        code="canonical_routeback_truth_invalid",
    )
    target_channel = expected_target.channel_id
    if (
        event["event_type"] != "route_back.sent"
        or event["case_id"] != fixture["case_id"]
        or event["readback_verified"] is not True
        or event["canonical_idempotency_key"]
        != fixture["public_routeback"]["canonical_idempotency_key"]
        or capability.authority_ref != event["authorization_id"]
        or canonical_receipt
        != {
            "platform": "discord",
            "adapter_receipt": True,
            "receipt_readback_verified": True,
            "message_id": receipt.discord_object_id,
            "channel_id": target_channel,
            "content_sha256": receipt.content_sha256,
        }
        or not isinstance(event["target_ref"], Mapping)
        or str(
            event["target_ref"].get("thread_id")
            or event["target_ref"].get("channel_id")
            or ""
        )
        != target_channel
    ):
        _fail("canonical_routeback_truth_invalid")
    _uuid(event["event_id"], "canonical_routeback_truth_invalid")
    _safe_id(event["authorization_id"], "canonical_routeback_truth_invalid")
    _sha256(event["canonical_content_sha256"], "canonical_routeback_truth_invalid")


def _validate_private_denial(
    fixture: Mapping[str, Any],
    evidence: Mapping[str, Any],
    service_identity_sha256: str,
    source_receipt: Mapping[str, Any],
) -> None:
    denial = _strict_object(
        evidence.get("private_denial"),
        required=(
            "schema",
            "provenance",
            "release_sha",
            "canary_run_id",
            "session_id",
            "turn_id",
            "observed_at_unix_ms",
            "discord_edge_service_identity_sha256",
            "socket_identity_sha256",
            "attempt_frame_sha256",
            "attempted_operation",
            "attempted_target_type",
            "connection_closed_without_response",
            "signed_receipt_observed",
            "journal_snapshot_before",
            "journal_snapshot_after",
        ),
        code="private_denial_invalid",
    )
    if (
        denial["schema"] != PRIVATE_DENIAL_RECEIPT_SCHEMA
        or denial["provenance"] != "live_gateway_to_discord_edge_probe"
        or denial["release_sha"] != fixture["release_sha"]
        or denial["canary_run_id"] != fixture["canary_run_id"]
        or denial["session_id"] != source_receipt["session_id"]
        or denial["turn_id"] != source_receipt["turn_id"]
        or denial["discord_edge_service_identity_sha256"] != service_identity_sha256
        or denial["attempted_operation"] != "public.message.send"
        or denial["attempted_target_type"]
        not in {"dm", "direct_message", "private_channel"}
        or denial["connection_closed_without_response"] is not True
        or denial["signed_receipt_observed"] is not False
    ):
        _fail("private_denial_invalid")
    observed_at = _positive_int(denial["observed_at_unix_ms"], "private_denial_invalid")
    if (
        not fixture["valid_from_unix_ms"]
        <= observed_at
        <= fixture["valid_until_unix_ms"]
    ):
        _fail("private_denial_invalid")
    _sha256(denial["socket_identity_sha256"], "private_denial_invalid")
    _sha256(denial["attempt_frame_sha256"], "private_denial_invalid")
    before = _strict_object(
        denial["journal_snapshot_before"],
        required=("record_count", "logical_sha256"),
        code="private_denial_invalid",
    )
    after = _strict_object(
        denial["journal_snapshot_after"],
        required=("record_count", "logical_sha256"),
        code="private_denial_invalid",
    )
    if (
        type(before["record_count"]) is not int
        or before["record_count"] < 0
        or before != after
    ):
        _fail("private_denial_invalid")
    _sha256(before["logical_sha256"], "private_denial_invalid")


def _validate_task_outcome(
    fixture: Mapping[str, Any],
    value: Any,
    *,
    source_receipt: Mapping[str, Any],
    plan_summary: Mapping[str, Any],
    expected_model_call_count: int,
    expected_tool_call_count: int,
) -> None:
    receipt = _strict_object(
        value,
        required=(
            "schema",
            "provenance",
            "release_sha",
            "canary_run_id",
            "session_id",
            "turn_id",
            "api_run_id",
            "api_message_id",
            "case_id",
            "plan_id",
            "stream_terminal_event",
            "completed",
            "partial",
            "interrupted",
            "failed",
            "turn_exit_reason",
            "completed_at_unix_ms",
            "completed_steps",
            "model_call_count",
            "tool_call_count",
            "final_response_sha256",
        ),
        code="task_outcome_invalid",
    )
    if (
        receipt["schema"] != TASK_OUTCOME_RECEIPT_SCHEMA
        or receipt["provenance"] != "live_gateway_turn_completion"
        or receipt["release_sha"] != fixture["release_sha"]
        or receipt["canary_run_id"] != fixture["canary_run_id"]
        or receipt["session_id"] != source_receipt["session_id"]
        or receipt["turn_id"] != source_receipt["turn_id"]
        or receipt["api_run_id"] != source_receipt["api_run_id"]
        or receipt["api_message_id"] != source_receipt["api_message_id"]
        or receipt["case_id"] != fixture["case_id"]
        or receipt["plan_id"] != plan_summary["plan_id"]
        or receipt["stream_terminal_event"] != "run.completed"
        or receipt["completed"] is not True
        or receipt["partial"] is not False
        or receipt["interrupted"] is not False
        or receipt["failed"] is not False
        or not isinstance(receipt["turn_exit_reason"], str)
        or _HEALTHY_TURN_EXIT_RE.fullmatch(receipt["turn_exit_reason"]) is None
    ):
        _fail("task_outcome_invalid")
    completed_at = _positive_int(
        receipt["completed_at_unix_ms"], "task_outcome_invalid"
    )
    if (
        not fixture["valid_from_unix_ms"]
        <= completed_at
        <= fixture["valid_until_unix_ms"]
    ):
        _fail("task_outcome_invalid")
    completed = receipt["completed_steps"]
    expected = plan_summary["completion_order"]
    if not isinstance(completed, list) or len(completed) != len(expected):
        _fail("task_outcome_invalid")
    for ordinal, (item, step_id) in enumerate(
        zip(completed, expected, strict=True), start=1
    ):
        normalized = _strict_object(
            item,
            required=("step_id", "completion_ordinal", "tool_receipt_sha256"),
            code="task_outcome_invalid",
        )
        if (
            normalized["step_id"] != step_id
            or normalized["completion_ordinal"] != ordinal
        ):
            _fail("task_outcome_invalid")
        _sha256(normalized["tool_receipt_sha256"], "task_outcome_invalid")
    if (
        type(receipt["model_call_count"]) is not int
        or receipt["model_call_count"] != expected_model_call_count
        or type(receipt["tool_call_count"]) is not int
        or receipt["tool_call_count"] != expected_tool_call_count
        or receipt["tool_call_count"] < len(expected)
    ):
        _fail("task_outcome_invalid")
    _sha256(receipt["final_response_sha256"], "task_outcome_invalid")


def verify_evidence(
    fixture_value: Any,
    evidence_value: Any,
    *,
    start_receipt_sha256: str,
    fixture_sha256: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    """Validate one digest-bound live evidence bundle without side effects."""

    start_digest = _sha256(start_receipt_sha256, "start_receipt_digest_invalid")
    fixture_digest = _sha256(fixture_sha256, "fixture_digest_invalid")
    evidence_digest = _sha256(evidence_sha256, "evidence_digest_invalid")
    fixture = _validate_fixture(fixture_value)
    if _digest_mapping(fixture) != fixture_digest:
        _fail("fixture_digest_invalid")
    evidence = _strict_object(
        evidence_value,
        required=(
            "schema",
            "fixture_sha256",
            "collected_at_unix_ms",
            "runtime_provenance",
            "writer_readiness",
            "source_receipt",
            "model_calls",
            "reasoning_directive",
            "canonical_truth",
            "public_routeback",
            "private_denial",
            "task_outcome",
        ),
        code="evidence_shape_invalid",
    )
    if _digest_mapping(evidence) != evidence_digest:
        _fail("evidence_digest_invalid")
    collected_at = _positive_int(
        evidence["collected_at_unix_ms"], "evidence_collection_time_invalid"
    )
    if (
        evidence["schema"] != EVIDENCE_SCHEMA
        or evidence["fixture_sha256"] != fixture_digest
        or not fixture["valid_from_unix_ms"]
        <= collected_at
        <= fixture["valid_until_unix_ms"]
    ):
        _fail("evidence_binding_invalid")

    _validate_live_provenance(
        fixture,
        evidence,
        start_receipt_sha256=start_digest,
    )
    _validate_writer_readiness(evidence["writer_readiness"])
    _validate_source_receipt(fixture, evidence["source_receipt"])
    source_receipt = evidence["source_receipt"]
    model_call_count, tool_call_count = _validate_reasoning_transition(
        fixture,
        evidence,
        source_receipt=source_receipt,
    )
    canonical_truth, plan_summary = _validate_canonical_truth(
        fixture,
        evidence["canonical_truth"],
        fixture_sha256=fixture_digest,
        source_receipt=source_receipt,
    )
    _validate_public_routeback(fixture, evidence, canonical_truth)
    service_identity = evidence["runtime_provenance"][
        "discord_edge_service_identity_sha256"
    ]
    _validate_private_denial(
        fixture,
        evidence,
        service_identity,
        source_receipt,
    )
    _validate_task_outcome(
        fixture,
        evidence["task_outcome"],
        source_receipt=source_receipt,
        plan_summary=plan_summary,
        expected_model_call_count=model_call_count,
        expected_tool_call_count=tool_call_count,
    )

    receipt: dict[str, Any] = {
        "schema": INVARIANT_RECEIPT_SCHEMA,
        "ok": True,
        "fixture_sha256": fixture_digest,
        "evidence_sha256": evidence_digest,
        "full_canary_start_receipt_sha256": start_digest,
        "release_sha": fixture["release_sha"],
        "canary_run_id": fixture["canary_run_id"],
        "invariants": list(_INVARIANTS),
    }
    receipt["invariant_receipt_sha256"] = _digest_mapping(receipt)
    return receipt


def _failure_receipt(
    *,
    fixture_sha256: str,
    evidence_sha256: str,
    failure_code: str,
) -> dict[str, Any]:
    fixture_digest = (
        fixture_sha256 if _SHA256_RE.fullmatch(fixture_sha256) else _ZERO_SHA256
    )
    evidence_digest = (
        evidence_sha256 if _SHA256_RE.fullmatch(evidence_sha256) else _ZERO_SHA256
    )
    receipt: dict[str, Any] = {
        "schema": INVARIANT_RECEIPT_SCHEMA,
        "ok": False,
        "fixture_sha256": fixture_digest,
        "evidence_sha256": evidence_digest,
        "failure_code": failure_code,
    }
    receipt["invariant_receipt_sha256"] = _digest_mapping(receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify externally generated live Muncho canary evidence"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--start-receipt-sha256", required=True)
    verify.add_argument("--fixture", required=True, type=Path)
    verify.add_argument("--fixture-sha256", required=True)
    verify.add_argument("--fixture-gid", type=int)
    verify.add_argument("--evidence", required=True, type=Path)
    verify.add_argument("--evidence-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    start_receipt_digest = str(args.start_receipt_sha256 or "")
    fixture_digest = str(args.fixture_sha256 or "")
    evidence_digest = str(args.evidence_sha256 or "")
    try:
        live_fixture = args.fixture == LIVE_FIXTURE_PATH
        try:
            live_evidence = args.evidence.is_relative_to(LIVE_EVIDENCE_ROOT)
        except (AttributeError, ValueError):  # pragma: no cover - py<3.9 fallback.
            live_evidence = str(args.evidence).startswith(
                str(LIVE_EVIDENCE_ROOT) + os.sep
            )
        if live_fixture and (
            type(args.fixture_gid) is not int or args.fixture_gid < 0
        ):
            _fail("fixture_artifact_invalid")
        fixture, observed_fixture = _read_bound_artifact(
            args.fixture,
            fixture_digest,
            "fixture_artifact_invalid",
            expected_uid=0 if live_fixture else None,
            expected_gid=args.fixture_gid if live_fixture else None,
            expected_mode=0o440 if live_fixture else None,
        )
        evidence, observed_evidence = _read_bound_artifact(
            args.evidence,
            evidence_digest,
            "evidence_artifact_invalid",
            expected_uid=0 if live_evidence else None,
            expected_gid=0 if live_evidence else None,
            expected_mode=0o400 if live_evidence else None,
        )
        if args.fixture == args.evidence:
            _fail("artifact_paths_not_distinct")
        result = verify_evidence(
            fixture,
            evidence,
            start_receipt_sha256=start_receipt_digest,
            fixture_sha256=observed_fixture,
            evidence_sha256=observed_evidence,
        )
        exit_code = 0
    except CanaryEvidenceError as exc:
        result = _failure_receipt(
            fixture_sha256=fixture_digest,
            evidence_sha256=evidence_digest,
            failure_code=exc.code,
        )
        exit_code = 2
    except Exception:
        # Never reflect paths, receipt content, crypto errors, or secrets.
        result = _failure_receipt(
            fixture_sha256=fixture_digest,
            evidence_sha256=evidence_digest,
            failure_code="unexpected_verification_failure",
        )
        exit_code = 2
    print(_canonical_bytes(result).decode("utf-8"))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_TRUTH_RECEIPT_SCHEMA",
    "CanaryEvidenceError",
    "EVIDENCE_SCHEMA",
    "FIXTURE_SCHEMA",
    "INVARIANT_RECEIPT_SCHEMA",
    "MODEL_CALL_RECEIPT_SCHEMA",
    "PRIVATE_DENIAL_RECEIPT_SCHEMA",
    "REASONING_RECEIPT_SCHEMA",
    "SOURCE_RECEIPT_SCHEMA",
    "TASK_OUTCOME_RECEIPT_SCHEMA",
    "main",
    "verify_evidence",
]
