import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "runtime" / "delivery_document_actions.py"
    spec = importlib.util.spec_from_file_location("delivery_document_actions", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalizes_document_actions_and_otp_policy():
    actions = _load_module()

    assert actions.normalize_document_action("comment") == "commented"
    assert actions.normalize_document_action("approve") == "approved"
    assert actions.normalize_document_action("reject") == "rejected"
    assert actions.normalize_document_action("sign") == "signed"
    assert actions.document_action_requires_otp("commented") is False
    assert actions.document_action_requires_otp("approved") is True
    assert actions.document_action_requires_otp("rejected") is True
    assert actions.document_action_requires_otp("signed") is True


def test_build_document_event_preserves_payload_and_metadata():
    actions = _load_module()

    event = actions.build_document_event(
        {
            "action": "approve",
            "deliverable_id": "quote-1",
            "actor_type": "customer",
            "actor_ref": "client@example.com",
            "comment": " Aprobado ",
            "metadata": {"workspace_id": "workspace-1"},
        },
        token_ref="abc123...",
        ip_address="127.0.0.1",
        user_agent="pytest",
    )

    assert event["event_type"] == "approved"
    assert event["deliverable_id"] == "quote-1"
    assert event["comment"] == "Aprobado"
    assert event["metadata"] == {"workspace_id": "workspace-1"}
    assert event["token_ref"] == "abc123..."
    assert event["status"] == "pending_agent_ingest"
