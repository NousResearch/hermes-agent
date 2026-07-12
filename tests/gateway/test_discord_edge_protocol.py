import copy
import uuid

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.discord_edge_protocol import (
    MAX_REQUEST_BYTES,
    PROTOCOL_VERSION,
    RECONCILIATION_PROTOCOL_VERSION,
    DiscordEdgeAuthorityKind,
    DiscordEdgeErrorCode,
    DiscordEdgeIntent,
    DiscordEdgeOperation,
    DiscordEdgeProtocolError,
    DiscordEdgeReceiptOutcome,
    DiscordEdgeReconciliationQuery,
    DiscordEdgeThreadReadback,
    DiscordPublicTarget,
    DiscordPublicTargetType,
    SignedDiscordEdgeEnvelope,
    decode_request_json,
    make_request,
    parse_request,
    parse_reconciliation_query,
    sign_capability,
    sign_receipt,
    verify_receipt,
    verify_request_capability,
)


NOW_MS = 2_000_000_000_000
GUILD_ID = "100000000000000001"
CHANNEL_ID = "100000000000000002"
OTHER_CHANNEL_ID = "100000000000000003"
THREAD_ID = "100000000000000004"
MESSAGE_ID = "100000000000000005"
BOT_USER_ID = "100000000000000006"


@pytest.fixture
def writer_key():
    return Ed25519PrivateKey.generate()


@pytest.fixture
def edge_key():
    return Ed25519PrivateKey.generate()


def _target(channel_id=CHANNEL_ID):
    return DiscordPublicTarget.from_mapping(
        {
            "target_type": "public_guild_channel",
            "guild_id": GUILD_ID,
            "channel_id": channel_id,
        }
    )


def _intent(
    *,
    content="Exact public response",
    channel_id=CHANNEL_ID,
    idempotency_key="case-1:routeback:1",
):
    return DiscordEdgeIntent(
        operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
        target=_target(channel_id),
        payload={"content": content},
        idempotency_key=idempotency_key,
    )


def _request(writer_key, intent=None, *, authority=None, request_id=None):
    intent = intent or _intent()
    capability = sign_capability(
        writer_key,
        intent,
        authority_kind=authority or DiscordEdgeAuthorityKind.CANONICAL_ROUTEBACK,
        authority_ref="routeauth:case-1:1",
        issued_at_unix_ms=NOW_MS,
        expires_at_unix_ms=NOW_MS + 60_000,
        capability_id="20000000-0000-4000-8000-000000000001",
    )
    request = make_request(
        intent,
        capability,
        request_id=request_id or "30000000-0000-4000-8000-000000000001",
        now_unix_ms=NOW_MS,
    )
    return request, capability


def _reconciliation_query(request):
    return DiscordEdgeReconciliationQuery(
        idempotency_key=request.intent.idempotency_key,
        operation=request.intent.operation,
        target=request.intent.target,
        request_sha256=request.intent.request_sha256,
        content_sha256=request.intent.content_sha256,
    )


def test_reconciliation_query_has_one_fixed_mutation_free_shape(writer_key):
    request, _capability = _request(writer_key)
    query = _reconciliation_query(request)

    message = query.to_message()
    parsed = parse_reconciliation_query(message)

    assert message == {
        "protocol": RECONCILIATION_PROTOCOL_VERSION,
        "idempotency_key": request.intent.idempotency_key,
        "operation": request.intent.operation.value,
        "target": request.intent.target.to_dict(),
        "request_sha256": request.intent.request_sha256,
        "content_sha256": request.intent.content_sha256,
    }
    assert parsed == query
    assert parsed.matches_request(request)


@pytest.mark.parametrize(
    "escape_field",
    ["payload", "content", "capability", "action", "url", "method", "deadline_unix_ms"],
)
def test_reconciliation_query_rejects_mutation_escape_hatches(
    writer_key,
    escape_field,
):
    request, _capability = _request(writer_key)
    message = _reconciliation_query(request).to_message()
    message[escape_field] = {}

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        parse_reconciliation_query(message)

    assert exc.value.code is DiscordEdgeErrorCode.INVALID_SHAPE


def test_reconciliation_query_is_bound_to_exact_target_and_digests(writer_key):
    request, _capability = _request(writer_key)
    query = _reconciliation_query(request)
    changed_request, _ = _request(
        writer_key,
        _intent(channel_id=OTHER_CHANNEL_ID),
        request_id="30000000-0000-4000-8000-000000000002",
    )

    assert not query.matches_request(changed_request)
    message = query.to_message()
    message["request_sha256"] = "A" * 64
    with pytest.raises(DiscordEdgeProtocolError) as exc:
        parse_reconciliation_query(message)
    assert exc.value.code is DiscordEdgeErrorCode.INVALID_SHAPE


@pytest.mark.parametrize(
    "target_type",
    [
        "dm",
        "direct_message",
        "group_dm",
        "private",
        "private_channel",
        "private_thread",
    ],
)
def test_dm_and_private_target_types_are_rejected(target_type):
    with pytest.raises(DiscordEdgeProtocolError) as exc:
        DiscordPublicTarget.from_mapping(
            {
                "target_type": target_type,
                "guild_id": GUILD_ID,
                "channel_id": CHANNEL_ID,
            }
        )
    assert exc.value.code == DiscordEdgeErrorCode.FORBIDDEN_TARGET


def test_public_thread_requires_an_exact_distinct_parent():
    with pytest.raises(DiscordEdgeProtocolError) as missing:
        DiscordPublicTarget.from_mapping(
            {
                "target_type": "public_guild_thread",
                "guild_id": GUILD_ID,
                "channel_id": THREAD_ID,
            }
        )
    assert missing.value.code == DiscordEdgeErrorCode.INVALID_TARGET

    target = DiscordPublicTarget.from_mapping(
        {
            "target_type": "public_guild_thread",
            "guild_id": GUILD_ID,
            "channel_id": THREAD_ID,
            "parent_channel_id": CHANNEL_ID,
        }
    )
    assert target.to_dict()["parent_channel_id"] == CHANNEL_ID


def test_direct_public_thread_constructor_rejects_same_parent():
    with pytest.raises(DiscordEdgeProtocolError) as exc:
        DiscordPublicTarget(
            DiscordPublicTargetType.PUBLIC_GUILD_THREAD,
            GUILD_ID,
            THREAD_ID,
            THREAD_ID,
        )

    assert exc.value.code == DiscordEdgeErrorCode.INVALID_TARGET


def test_thread_create_rejects_session_reply_authority(writer_key):
    intent = DiscordEdgeIntent(
        operation=DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
        target=_target(),
        payload={"name": "Model-authored public thread"},
        idempotency_key="case-1:thread:1",
    )
    with pytest.raises(DiscordEdgeProtocolError) as exc:
        sign_capability(
            writer_key,
            intent,
            authority_kind=DiscordEdgeAuthorityKind.SESSION_REPLY,
            authority_ref="session:event:1",
            issued_at_unix_ms=NOW_MS,
            expires_at_unix_ms=NOW_MS + 60_000,
        )
    assert exc.value.code == DiscordEdgeErrorCode.INVALID_CAPABILITY


@pytest.mark.parametrize("initial_message", [None, ""])
def test_forum_thread_create_requires_non_empty_initial_message(initial_message):
    payload = {"name": "Model-authored public forum thread"}
    if initial_message is not None:
        payload["initial_message"] = initial_message
    with pytest.raises(DiscordEdgeProtocolError) as exc:
        DiscordEdgeIntent(
            operation=DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            target=DiscordPublicTarget(
                DiscordPublicTargetType.PUBLIC_GUILD_FORUM,
                GUILD_ID,
                CHANNEL_ID,
            ),
            payload=payload,
            idempotency_key="case-1:forum-thread:1",
        )
    assert exc.value.code == DiscordEdgeErrorCode.INVALID_PAYLOAD


def test_public_channel_thread_content_requires_separate_receipted_send():
    with pytest.raises(DiscordEdgeProtocolError) as exc:
        DiscordEdgeIntent(
            operation=DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            target=_target(),
            payload={
                "name": "Non-atomic thread",
                "initial_message": "must be a separate send",
            },
            idempotency_key="case-1:text-thread:non-atomic",
        )
    assert exc.value.code == DiscordEdgeErrorCode.INVALID_PAYLOAD


@pytest.mark.parametrize("escape_field", ["url", "method", "path", "headers", "token", "action"])
def test_request_schema_rejects_generic_http_escape_hatches(writer_key, escape_field):
    request, _ = _request(writer_key)
    message = request.to_message()
    message[escape_field] = "forbidden"

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        parse_request(message, now_unix_ms=NOW_MS)
    assert exc.value.code == DiscordEdgeErrorCode.INVALID_SHAPE


def test_unknown_operation_is_not_a_generic_dispatch_escape_hatch(writer_key):
    request, _ = _request(writer_key)
    message = request.to_message()
    message["operation"] = "http.request"

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        parse_request(message, now_unix_ms=NOW_MS)
    assert exc.value.code == DiscordEdgeErrorCode.UNKNOWN_OPERATION


def test_valid_capability_is_bound_to_exact_request(writer_key):
    request, _ = _request(writer_key)
    parsed = parse_request(request.to_message(), now_unix_ms=NOW_MS)

    capability = verify_request_capability(
        parsed,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )

    assert capability.operation == DiscordEdgeOperation.PUBLIC_MESSAGE_SEND
    assert capability.target.channel_id == CHANNEL_ID
    assert capability.idempotency_key == "case-1:routeback:1"
    assert capability.request_sha256 == parsed.intent.request_sha256
    assert capability.content_sha256 == parsed.intent.content_sha256


def test_capability_payload_tampering_breaks_signature(writer_key):
    request, capability = _request(writer_key)
    tampered = capability.to_message()
    tampered["payload"]["content_sha256"] = "0" * 64
    message = request.to_message()
    message["capability"] = tampered
    parsed = parse_request(message, now_unix_ms=NOW_MS)

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        verify_request_capability(
            parsed,
            writer_key.public_key(),
            now_unix_ms=NOW_MS,
        )
    assert exc.value.code == DiscordEdgeErrorCode.SIGNATURE_INVALID


@pytest.mark.parametrize(
    ("replacement", "expected_field"),
    [
        ({"content": "Different content"}, "content"),
        ({"channel_id": OTHER_CHANNEL_ID}, "target"),
        ({"idempotency_key": "case-1:routeback:2"}, "idempotency"),
    ],
)
def test_capability_cannot_be_reused_for_other_binding(
    writer_key,
    replacement,
    expected_field,
):
    original = _intent()
    _, capability = _request(writer_key, original)
    changed = _intent(
        content=replacement.get("content", original.content),
        channel_id=replacement.get("channel_id", original.target.channel_id),
        idempotency_key=replacement.get("idempotency_key", original.idempotency_key),
    )
    changed_request = make_request(
        changed,
        capability,
        request_id=str(uuid.uuid4()),
        now_unix_ms=NOW_MS,
    )

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        verify_request_capability(
            changed_request,
            writer_key.public_key(),
            now_unix_ms=NOW_MS,
        )
    assert exc.value.code == DiscordEdgeErrorCode.CAPABILITY_BINDING_MISMATCH
    assert expected_field in exc.value.detail


def test_capability_signed_by_another_key_is_rejected(writer_key):
    request, _ = _request(writer_key)
    other_key = Ed25519PrivateKey.generate()

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        verify_request_capability(
            request,
            other_key.public_key(),
            now_unix_ms=NOW_MS,
        )
    assert exc.value.code == DiscordEdgeErrorCode.SIGNATURE_INVALID


def test_expired_capability_is_rejected(writer_key):
    intent = _intent()
    capability = sign_capability(
        writer_key,
        intent,
        authority_kind=DiscordEdgeAuthorityKind.CANONICAL_ROUTEBACK,
        authority_ref="routeauth:case-1:expired",
        issued_at_unix_ms=NOW_MS,
        expires_at_unix_ms=NOW_MS + 1_000,
    )
    request = make_request(intent, capability, now_unix_ms=NOW_MS)

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        verify_request_capability(
            request,
            writer_key.public_key(),
            now_unix_ms=NOW_MS + 1_001,
        )
    assert exc.value.code == DiscordEdgeErrorCode.CAPABILITY_EXPIRED


def test_verified_receipt_is_signed_and_exactly_bound(writer_key, edge_key):
    request, _ = _request(writer_key)
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    envelope = sign_receipt(
        edge_key,
        request,
        capability,
        outcome=DiscordEdgeReceiptOutcome.VERIFIED,
        discord_object_id=MESSAGE_ID,
        bot_user_id=BOT_USER_ID,
        adapter_accepted=True,
        readback_verified=True,
        readback_content_sha256=request.intent.content_sha256,
        occurred_at_unix_ms=NOW_MS + 1_000,
        receipt_id="40000000-0000-4000-8000-000000000001",
    )

    receipt = verify_receipt(
        envelope,
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS + 1_000,
    )

    assert receipt.outcome == DiscordEdgeReceiptOutcome.VERIFIED
    assert receipt.discord_object_id == MESSAGE_ID
    assert receipt.adapter_accepted is True
    assert receipt.readback_verified is True
    assert receipt.request_sha256 == request.intent.request_sha256


def test_receipt_tampering_breaks_signature(writer_key, edge_key):
    request, _ = _request(writer_key)
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    envelope = sign_receipt(
        edge_key,
        request,
        capability,
        outcome=DiscordEdgeReceiptOutcome.VERIFIED,
        discord_object_id=MESSAGE_ID,
        bot_user_id=BOT_USER_ID,
        adapter_accepted=True,
        readback_verified=True,
        readback_content_sha256=request.intent.content_sha256,
        occurred_at_unix_ms=NOW_MS,
    )
    tampered = envelope.to_message()
    tampered["payload"]["discord_object_id"] = "999999999999999999"
    tampered_envelope = SignedDiscordEdgeEnvelope.from_mapping(
        tampered,
        code=DiscordEdgeErrorCode.INVALID_RECEIPT,
        label="receipt",
    )

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        verify_receipt(
            tampered_envelope,
            edge_key.public_key(),
            expected_request=request,
            expected_capability=capability,
            now_unix_ms=NOW_MS,
        )
    assert exc.value.code == DiscordEdgeErrorCode.SIGNATURE_INVALID


def test_receipt_cannot_be_rebound_to_another_request(writer_key, edge_key):
    request, _ = _request(writer_key)
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    envelope = sign_receipt(
        edge_key,
        request,
        capability,
        outcome=DiscordEdgeReceiptOutcome.VERIFIED,
        discord_object_id=MESSAGE_ID,
        bot_user_id=BOT_USER_ID,
        adapter_accepted=True,
        readback_verified=True,
        readback_content_sha256=request.intent.content_sha256,
        occurred_at_unix_ms=NOW_MS,
    )
    other_request, _ = _request(
        writer_key,
        _intent(content="Another exact response"),
        request_id="30000000-0000-4000-8000-000000000002",
    )

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        verify_receipt(
            envelope,
            edge_key.public_key(),
            expected_request=other_request,
            now_unix_ms=NOW_MS,
        )
    assert exc.value.code == DiscordEdgeErrorCode.RECEIPT_BINDING_MISMATCH


def test_verified_receipt_requires_acceptance_readback_and_object_id(writer_key, edge_key):
    request, _ = _request(writer_key)
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        sign_receipt(
            edge_key,
            request,
            capability,
            outcome=DiscordEdgeReceiptOutcome.VERIFIED,
            discord_object_id=None,
            bot_user_id=BOT_USER_ID,
            adapter_accepted=True,
            readback_verified=False,
            readback_content_sha256=None,
            occurred_at_unix_ms=NOW_MS,
        )
    assert exc.value.code == DiscordEdgeErrorCode.INVALID_RECEIPT


def test_request_payload_and_capability_shapes_are_strict(writer_key):
    request, _ = _request(writer_key)
    message = request.to_message()
    message["payload"]["method"] = "POST"

    with pytest.raises(DiscordEdgeProtocolError) as payload_exc:
        parse_request(message, now_unix_ms=NOW_MS)
    assert payload_exc.value.code == DiscordEdgeErrorCode.INVALID_PAYLOAD

    message = request.to_message()
    message["capability"]["token"] = "ambient-authority"
    with pytest.raises(DiscordEdgeProtocolError) as capability_exc:
        parse_request(message, now_unix_ms=NOW_MS)
    assert capability_exc.value.code == DiscordEdgeErrorCode.INVALID_CAPABILITY


def test_message_content_uses_discord_character_limit():
    with pytest.raises(DiscordEdgeProtocolError) as exc:
        _intent(content="a" * 2_001)

    assert exc.value.code == DiscordEdgeErrorCode.INVALID_PAYLOAD


def test_thread_name_uses_discord_character_limit():
    with pytest.raises(DiscordEdgeProtocolError) as exc:
        DiscordEdgeIntent(
            operation=DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            target=_target(),
            payload={"name": "a" * 101},
            idempotency_key="case-1:thread:too-long",
        )

    assert exc.value.code == DiscordEdgeErrorCode.INVALID_PAYLOAD


def test_protocol_version_is_exact(writer_key):
    request, _ = _request(writer_key)
    message = copy.deepcopy(request.to_message())
    message["protocol"] = f"{PROTOCOL_VERSION}.future"

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        parse_request(message, now_unix_ms=NOW_MS)
    assert exc.value.code == DiscordEdgeErrorCode.UNSUPPORTED_VERSION


@pytest.mark.parametrize(
    "body",
    [
        b'{"operation":"first","operation":"second"}',
        b'{"target":{},"target":{}}',
    ],
)
def test_strict_json_decode_rejects_duplicate_keys(body):
    with pytest.raises(DiscordEdgeProtocolError) as exc:
        decode_request_json(body)

    assert exc.value.code is DiscordEdgeErrorCode.INVALID_JSON


@pytest.mark.parametrize(
    "body",
    [b'{"value":NaN}', b'{"value":Infinity}', b'{"value":-Infinity}'],
)
def test_strict_json_decode_rejects_non_finite_constants(body):
    with pytest.raises(DiscordEdgeProtocolError) as exc:
        decode_request_json(body)

    assert exc.value.code is DiscordEdgeErrorCode.INVALID_JSON


def test_strict_json_decode_rejects_invalid_utf8_and_non_object_root():
    with pytest.raises(DiscordEdgeProtocolError) as invalid_utf8:
        decode_request_json(b'{"value":"\xff"}')
    assert invalid_utf8.value.code is DiscordEdgeErrorCode.INVALID_JSON

    with pytest.raises(DiscordEdgeProtocolError) as non_object:
        decode_request_json(b"[]")
    assert non_object.value.code is DiscordEdgeErrorCode.INVALID_JSON


def test_strict_json_decode_has_a_fixed_size_bound():
    with pytest.raises(DiscordEdgeProtocolError) as exc:
        decode_request_json(b"{" + b" " * MAX_REQUEST_BYTES + b"}")

    assert exc.value.code is DiscordEdgeErrorCode.REQUEST_TOO_LARGE


def test_dispatch_uncertain_receipt_uses_unknown_acceptance_shape(writer_key, edge_key):
    request, _ = _request(writer_key)
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    envelope = sign_receipt(
        edge_key,
        request,
        capability,
        outcome=DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN,
        discord_object_id=None,
        bot_user_id=None,
        adapter_accepted=None,
        readback_verified=False,
        readback_content_sha256=None,
        blocker_code="dispatch_outcome_uncertain",
        occurred_at_unix_ms=NOW_MS,
    )

    receipt = verify_receipt(
        envelope,
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS,
    )

    assert receipt.outcome is DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN
    assert receipt.adapter_accepted is None
    assert receipt.discord_object_id is None
    assert receipt.bot_user_id is None
    assert receipt.readback_verified is False
    assert receipt.blocker_code == "dispatch_outcome_uncertain"


def test_dispatch_uncertain_cannot_claim_false_acceptance(writer_key, edge_key):
    request, _ = _request(writer_key)
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        sign_receipt(
            edge_key,
            request,
            capability,
            outcome=DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN,
            discord_object_id=None,
            bot_user_id=None,
            adapter_accepted=False,
            readback_verified=False,
            readback_content_sha256=None,
            blocker_code="dispatch_outcome_uncertain",
            occurred_at_unix_ms=NOW_MS,
        )

    assert exc.value.code is DiscordEdgeErrorCode.INVALID_RECEIPT


@pytest.mark.parametrize(
    ("name", "archive"),
    [("Wrong thread name", 1_440), ("Exact public thread", 60)],
)
def test_verified_thread_receipt_requires_exact_external_metadata(
    writer_key,
    edge_key,
    name,
    archive,
):
    intent = DiscordEdgeIntent(
        operation=DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
        target=_target(),
        payload={
            "name": "Exact public thread",
            "auto_archive_minutes": 1_440,
        },
        idempotency_key="case-1:thread:receipt",
    )
    capability_envelope = sign_capability(
        writer_key,
        intent,
        authority_kind=DiscordEdgeAuthorityKind.CANONICAL_PLAN,
        authority_ref="plan:case-1:thread",
        issued_at_unix_ms=NOW_MS,
        expires_at_unix_ms=NOW_MS + 60_000,
    )
    request = make_request(intent, capability_envelope, now_unix_ms=NOW_MS)
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    thread_target = DiscordPublicTarget(
        target_type=DiscordPublicTargetType.PUBLIC_GUILD_THREAD,
        guild_id=GUILD_ID,
        channel_id=THREAD_ID,
        parent_channel_id=CHANNEL_ID,
    )
    evidence = DiscordEdgeThreadReadback(
        target=thread_target,
        name=name,
        auto_archive_minutes=archive,
    )

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        sign_receipt(
            edge_key,
            request,
            capability,
            outcome=DiscordEdgeReceiptOutcome.VERIFIED,
            discord_object_id=THREAD_ID,
            bot_user_id=BOT_USER_ID,
            adapter_accepted=True,
            readback_verified=True,
            readback_content_sha256=request.intent.content_sha256,
            readback_thread=evidence,
            occurred_at_unix_ms=NOW_MS,
        )

    assert exc.value.code is DiscordEdgeErrorCode.RECEIPT_BINDING_MISMATCH
