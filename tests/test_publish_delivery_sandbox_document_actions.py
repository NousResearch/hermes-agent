import json
import sys
import types
import hashlib
import hmac
import time
from io import BytesIO
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "scripts" / "runtime"
sys.path.insert(0, str(RUNTIME_DIR))
from scripts.runtime import publish_delivery_sandbox as publisher


def _server_module(tmp_path):
    runtime_dir = Path(__file__).resolve().parents[1] / "scripts" / "runtime"
    sys.path.insert(0, str(runtime_dir))
    module = types.ModuleType("generated_delivery_server")
    exec(publisher.SERVER_PY, module.__dict__)
    module.EVENT_DIR = tmp_path / "events"
    module.PUBLIC_DIR = tmp_path / "public"
    module.USER_DATA_DIR = tmp_path / "user-data"
    module.ALLOWED_HOSTS = set()
    return module


def test_generated_server_document_action_policy_and_private_recipient(tmp_path):
    server = _server_module(tmp_path)
    token = "A" * 24
    workspace = server.PUBLIC_DIR / "w" / token
    workspace.mkdir(parents=True)
    (workspace / "index.html").write_text("quote-1", encoding="utf-8")
    private = server.USER_DATA_DIR / "workspace-recipients"
    private.mkdir(parents=True)
    (private / f"{token}.json").write_text(
        json.dumps({"recipient": {"channel_id": "whatsapp", "target": "+13050000000", "label": "WhatsApp"}}),
        encoding="utf-8",
    )

    assert server.document_action_requires_otp("commented") is False
    assert server.document_action_requires_otp("approved") is True
    assert server.document_action_requires_otp("rejected") is True
    assert server.document_action_requires_otp("signed") is True
    assert server._workspace_matches_token(token, "quote-1", {"quote_id": "quote-1"}) is True
    assert server._document_action_recipient(token)["target"] == "+13050000000"


def test_generated_server_queue_otp_uses_document_action_message(tmp_path):
    server = _server_module(tmp_path)
    server._queue_otp(
        {
            "challenge_id": "challenge-1",
            "user_id": "client@example.com",
            "channel_id": "email",
            "target": "client@example.com",
            "purpose": "document_action",
            "event_type": "approved",
            "deliverable_id": "quote-1",
            "token_ref": "abc...",
            "message": "Código para aprobar: 123456",
        },
        "123456",
    )

    outbox = server._outbox_path().read_text(encoding="utf-8").splitlines()
    payload = json.loads(outbox[-1])
    assert payload["message"] == "Código para aprobar: 123456"
    assert payload["purpose"] == "document_action"
    assert payload["event_type"] == "approved"


class _FakeStripeHandler:
    def __init__(self, raw_body: bytes, signature: str):
        self.headers = {"Content-Length": str(len(raw_body)), "Stripe-Signature": signature, "User-Agent": "stripe-test"}
        self.rfile = BytesIO(raw_body)
        self.wfile = BytesIO()
        self.client_address = ("127.0.0.1", 12345)
        self.status = None
        self.response_headers = []

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.response_headers.append((key, value))

    def end_headers(self):
        pass


def test_generated_server_stripe_webhook_queues_signed_event(tmp_path):
    server = _server_module(tmp_path)
    server.STRIPE_WEBHOOK_SECRET = "whsec_test"
    payload = {
        "id": "evt_test_123",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_test_123",
                "object": "checkout.session",
                "client_reference_id": "pay-1",
                "metadata": {"payment_request_id": "pay-1", "invoice_id": "invoice-1"},
            }
        },
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    timestamp = int(time.time())
    digest = hmac.new(b"whsec_test", str(timestamp).encode() + b"." + raw, hashlib.sha256).hexdigest()
    handler = _FakeStripeHandler(raw, f"t={timestamp},v1={digest}")

    server._handle_stripe_webhook(handler)

    assert handler.status == 200
    response = json.loads(handler.wfile.getvalue().split(b"\r\n\r\n")[-1] or handler.wfile.getvalue())
    assert response["status"] == "queued"
    events = server._audit_path().read_text(encoding="utf-8").splitlines()
    queued = json.loads(events[-1])
    assert queued["event_type"] == "stripe_webhook"
    assert queued["deliverable_id"] == "invoice-1"
    assert queued["metadata"]["stripe_event_id"] == "evt_test_123"
    assert queued["metadata"]["stripe_event"]["type"] == "checkout.session.completed"


def test_generated_server_stripe_webhook_rejects_bad_signature(tmp_path):
    server = _server_module(tmp_path)
    server.STRIPE_WEBHOOK_SECRET = "whsec_test"
    handler = _FakeStripeHandler(b'{"id":"evt_bad"}', "t=123,v1=bad")

    server._handle_stripe_webhook(handler)

    assert handler.status == 400
