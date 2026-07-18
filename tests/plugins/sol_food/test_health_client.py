"""Health v3 transport client: token, envelope, loopback, verification."""

import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from plugins.sol_food.health_client import (
    HealthClientError,
    HealthFoodClient,
    build_commit_envelope,
    FrozenEnvelope,
    validate_commit_token,
)

VALID_TOKEN = base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
OCCURRED = "2026-07-17T12:00:00.000000Z"


def items(n: int = 1):
    return [{"plant_key": f"synthetic_item_{i}", "is_plant": True} for i in range(n)]


class TestToken:
    def test_valid_token(self):
        assert len(VALID_TOKEN) == 43
        assert validate_commit_token(VALID_TOKEN) == VALID_TOKEN

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            "",
            VALID_TOKEN[:-1],          # 42 chars
            VALID_TOKEN + "A",         # 44 chars
            VALID_TOKEN[:-1] + "=",    # padding
            VALID_TOKEN[:-1] + "+",    # non-url alphabet
            "B" * 43,                  # noncanonical trailing bits
            " " + VALID_TOKEN[:-1],    # whitespace
        ],
    )
    def test_bad_tokens_fail_closed(self, bad):
        with pytest.raises(HealthClientError) as excinfo:
            validate_commit_token(bad)
        assert excinfo.value.reason_code == "health_client_bad_token"

    def test_noncanonical_last_char_alias_rejected(self):
        # Flip low bits of the final character: same decoded prefix,
        # different string — canonical re-encode check must reject.
        token = VALID_TOKEN[:-1] + ("B" if VALID_TOKEN[-1] != "B" else "C")
        decoded = base64.urlsafe_b64decode(token + "=")
        if base64.urlsafe_b64encode(decoded).decode().rstrip("=") != token:
            with pytest.raises(HealthClientError):
                validate_commit_token(token)


class TestEnvelope:
    def test_create_shape_and_canonical_bytes(self):
        envelope = build_commit_envelope(
            operation="create",
            occurred_at=OCCURRED,
            items=items(2),
            expected_revision=0,
        )
        parsed = json.loads(envelope.request_bytes.decode("utf-8"))
        assert set(parsed.keys()) == {
            "schema",
            "mutation_id",
            "entry_id",
            "operation",
            "expected_revision",
            "occurred_at",
            "payload_version",
            "payload",
        }
        assert parsed["schema"] == "health.food_commit.v1"
        assert parsed["payload_version"] == "health.food_meal.v1"
        # No owner/user identity anywhere in the request, by construction.
        assert "user_id" not in envelope.request_bytes.decode("utf-8")
        assert "owner" not in parsed
        # Canonical: sorted keys, no whitespace.
        assert b" " not in envelope.request_bytes
        assert (
            hashlib.sha256(envelope.request_bytes).hexdigest()
            == envelope.request_sha256
        )

    def test_delete_takes_no_payload(self):
        envelope = build_commit_envelope(
            operation="delete",
            occurred_at=OCCURRED,
            items=None,
            expected_revision=3,
            entry_id="0" * 8 + "-0000-4000-8000-" + "0" * 12,
        )
        parsed = json.loads(envelope.request_bytes)
        assert "payload" not in parsed
        with pytest.raises(HealthClientError):
            build_commit_envelope(
                operation="delete",
                occurred_at=OCCURRED,
                items=items(),
                expected_revision=3,
            )

    def test_replace_requires_positive_revision(self):
        with pytest.raises(HealthClientError):
            build_commit_envelope(
                operation="replace",
                occurred_at=OCCURRED,
                items=items(),
                expected_revision=0,
            )

    def test_create_requires_revision_zero(self):
        with pytest.raises(HealthClientError):
            build_commit_envelope(
                operation="create",
                occurred_at=OCCURRED,
                items=items(),
                expected_revision=1,
            )

    @pytest.mark.parametrize("count,ok", [(23, True), (24, True), (25, False)])
    def test_item_count_boundary(self, count, ok):
        if ok:
            build_commit_envelope(
                operation="create",
                occurred_at=OCCURRED,
                items=items(count),
                expected_revision=0,
            )
        else:
            with pytest.raises(HealthClientError):
                build_commit_envelope(
                    operation="create",
                    occurred_at=OCCURRED,
                    items=items(count),
                    expected_revision=0,
                )

    def test_zero_items_rejected(self):
        with pytest.raises(HealthClientError):
            build_commit_envelope(
                operation="create",
                occurred_at=OCCURRED,
                items=[],
                expected_revision=0,
            )

    @pytest.mark.parametrize(
        "bad_item",
        [
            {"plant_key": "Synthetic_Item", "is_plant": True},  # uppercase
            {"plant_key": "bad-key", "is_plant": True},          # hyphen
            {"plant_key": "a" * 65, "is_plant": True},           # 65 bytes
            {"plant_key": "ok_key", "is_plant": 1},              # non-bool
            {"plant_key": "ok_key", "is_plant": True, "note": "x"},
            {"plant_key": "", "is_plant": True},
        ],
    )
    def test_bad_items_rejected(self, bad_item):
        with pytest.raises(HealthClientError):
            build_commit_envelope(
                operation="create",
                occurred_at=OCCURRED,
                items=[bad_item],
                expected_revision=0,
            )

    def test_plant_key_64_bytes_accepted(self):
        build_commit_envelope(
            operation="create",
            occurred_at=OCCURRED,
            items=[{"plant_key": "a" * 64, "is_plant": False}],
            expected_revision=0,
        )

    @pytest.mark.parametrize(
        "stamp",
        [
            "2026-07-17T12:00:00Z",           # no microseconds
            "2026-07-17 12:00:00.000000Z",    # space
            "2026-07-17T12:00:00.000000+00:00",
            "not-a-time",
        ],
    )
    def test_occurred_at_format_strict(self, stamp):
        with pytest.raises(HealthClientError):
            build_commit_envelope(
                operation="create",
                occurred_at=stamp,
                items=items(),
                expected_revision=0,
            )

    def test_frozen_roundtrip(self):
        envelope = build_commit_envelope(
            operation="create",
            occurred_at=OCCURRED,
            items=items(),
            expected_revision=0,
        )
        clone = FrozenEnvelope.from_json(envelope.to_json())
        assert clone.request_bytes == envelope.request_bytes
        assert clone.request_sha256 == envelope.request_sha256

    def test_frozen_roundtrip_detects_tamper(self):
        envelope = build_commit_envelope(
            operation="create",
            occurred_at=OCCURRED,
            items=items(),
            expected_revision=0,
        )
        data = envelope.to_json()
        data["request_sha256"] = "0" * 64
        with pytest.raises(HealthClientError):
            FrozenEnvelope.from_json(data)


class TestEndpointGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "http://192.168.1.10:8899/food",
            "http://health.example.com/food",
            "https://127.0.0.1:8899/food",  # non-http scheme refused
            "http://127.0.0.1:8899/food?x=1",
            "http://user:pw@127.0.0.1:8899/food",
            "http://127.0.0.1:8899",  # no path
        ],
    )
    def test_non_loopback_or_malformed_refused(self, url):
        with pytest.raises(HealthClientError) as excinfo:
            HealthFoodClient(url, VALID_TOKEN)
        assert excinfo.value.reason_code == "health_client_bad_endpoint"

    def test_loopback_accepted(self):
        HealthFoodClient("http://127.0.0.1:8899/food", VALID_TOKEN)

    def test_bad_token_refused_at_construction(self):
        with pytest.raises(HealthClientError):
            HealthFoodClient("http://127.0.0.1:8899/food", "short")


# ── Round-trip tests against a real loopback server ─────────────────────

def make_receipt(envelope: FrozenEnvelope) -> bytes:
    receipt = {
        "schema": "health.food_commit_receipt.v1",
        "mutation_id": envelope.mutation_id,
        "entry_id": envelope.entry_id,
        "operation": envelope.operation,
        "revision": 1,
        "request_sha256": envelope.request_sha256,
        "current_entry_sha256": "1" * 64,
        "affected_food_log_items_sha256": "2" * 64,
        "affected_date": "2026-07-17",
        "committed_at": "2026-07-17T12:00:00.000000Z",
        "status": "applied",
        "source_commit": "3" * 40,
        "runtime_fingerprint": "4" * 64,
        "tracked_clean": True,
        "same_transaction_readback_match": True,
    }
    return json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()


def make_response(envelope: FrozenEnvelope, *, replayed=False, mutate=None) -> bytes:
    receipt_bytes = make_receipt(envelope)
    body = {
        "schema": "health.food_commit_response.v1",
        "receipt_schema": "health.food_commit_receipt.v1",
        "receipt_canonical_json_b64": base64.b64encode(receipt_bytes).decode(),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "replayed": replayed,
        "post_commit_readback_sha256": "5" * 64,
        "historical_commit_verified": True,
        "latest_projection_consistent": True,
    }
    if mutate:
        mutate(body)
    return json.dumps(body).encode()


class _Server:
    """Loopback HTTP test double that records the raw request."""

    def __init__(self, response_factory, status=200):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                outer.requests.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get_all("Authorization"),
                        "body": body,
                    }
                )
                payload = response_factory(body)
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def envelope():
    return build_commit_envelope(
        operation="create",
        occurred_at=OCCURRED,
        items=items(2),
        expected_revision=0,
    )


class TestRoundTrip:
    def test_happy_path_verifies(self, envelope):
        server = _Server(lambda body: make_response(envelope))
        try:
            client = HealthFoodClient(
                f"http://127.0.0.1:{server.port}/food", VALID_TOKEN
            )
            verified = client.commit(envelope)
        finally:
            server.close()
        assert verified.replayed is False
        assert verified.receipt["mutation_id"] == envelope.mutation_id
        # Exactly one Authorization header with strict Bearer grammar.
        request = server.requests[0]
        assert request["auth"] == ["Bearer " + VALID_TOKEN]
        assert request["body"] == envelope.request_bytes

    def test_exact_retry_returns_replay(self, envelope):
        server = _Server(lambda body: make_response(envelope, replayed=True))
        try:
            client = HealthFoodClient(
                f"http://127.0.0.1:{server.port}/food", VALID_TOKEN
            )
            verified = client.commit(envelope)
        finally:
            server.close()
        assert verified.replayed is True

    @pytest.mark.parametrize(
        "mutator,reason",
        [
            (lambda b: b.update(receipt_sha256="9" * 64), "health_client_receipt_mismatch"),
            (lambda b: b.update(extra_key=1), "health_client_bad_response"),
            (lambda b: b.pop("replayed"), "health_client_bad_response"),
            (lambda b: b.update(replayed="false"), "health_client_bad_response"),
            (lambda b: b.update(schema="wrong.v9"), "health_client_bad_response"),
        ],
    )
    def test_bad_responses_fail_closed(self, envelope, mutator, reason):
        server = _Server(lambda body: make_response(envelope, mutate=mutator))
        try:
            client = HealthFoodClient(
                f"http://127.0.0.1:{server.port}/food", VALID_TOKEN
            )
            with pytest.raises(HealthClientError) as excinfo:
                client.commit(envelope)
        finally:
            server.close()
        assert excinfo.value.reason_code == reason
        assert excinfo.value.retryable is True

    def test_receipt_identity_mismatch_fails(self, envelope):
        other = build_commit_envelope(
            operation="create",
            occurred_at=OCCURRED,
            items=items(1),
            expected_revision=0,
        )
        server = _Server(lambda body: make_response(other))
        try:
            client = HealthFoodClient(
                f"http://127.0.0.1:{server.port}/food", VALID_TOKEN
            )
            with pytest.raises(HealthClientError) as excinfo:
                client.commit(envelope)
        finally:
            server.close()
        assert excinfo.value.reason_code == "health_client_receipt_mismatch"
        assert excinfo.value.retryable is True

    def test_failed_readback_blocks_ack_retryably(self, envelope):
        server = _Server(
            lambda body: make_response(
                envelope,
                mutate=lambda b: b.update(latest_projection_consistent=False),
            )
        )
        try:
            client = HealthFoodClient(
                f"http://127.0.0.1:{server.port}/food", VALID_TOKEN
            )
            with pytest.raises(HealthClientError) as excinfo:
                client.commit(envelope)
        finally:
            server.close()
        assert excinfo.value.reason_code == "health_client_readback_failed"
        assert excinfo.value.retryable is True

    def test_5xx_is_retryable(self, envelope):
        server = _Server(lambda body: b"oops", status=503)
        try:
            client = HealthFoodClient(
                f"http://127.0.0.1:{server.port}/food", VALID_TOKEN
            )
            with pytest.raises(HealthClientError) as excinfo:
                client.commit(envelope)
        finally:
            server.close()
        assert excinfo.value.retryable is True

    def test_connection_refused_is_retryable(self, envelope):
        client = HealthFoodClient("http://127.0.0.1:1/food", VALID_TOKEN, timeout=0.5)
        with pytest.raises(HealthClientError) as excinfo:
            client.commit(envelope)
        assert excinfo.value.reason_code == "health_client_transport_error"
        assert excinfo.value.retryable is True
