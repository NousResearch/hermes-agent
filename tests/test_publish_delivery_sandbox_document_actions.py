import json
import sys
import types
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
