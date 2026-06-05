import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "runtime" / "ingest_delivery_events.py"
    spec = importlib.util.spec_from_file_location("ingest_delivery_events", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comment_does_not_regress_final_workspace_status():
    ingest = _load_module()

    assert ingest._status_for_event("commented", "approved") is None
    assert ingest._status_for_event("opened", "rejected") is None
    assert ingest._status_for_event("commented", "viewed") == "commented"
    assert ingest._status_for_event("approved", "commented") == "approved"


def test_ingest_uses_canonical_document_action_sets():
    ingest = _load_module()

    assert "commented" in ingest.SALES_EVENT_TYPES
    assert "approved" in ingest.FINAL_SALES_STATUS
    assert "rejected" in ingest.FINAL_RECEIPT_STATUS
    assert "signed" in ingest.EVENT_TYPES_WITH_OWNER_ACTION
