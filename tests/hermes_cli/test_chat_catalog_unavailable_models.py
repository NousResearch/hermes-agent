"""chat_catalog_ids drops provider rows declared shut down or scheduled for removal.

The picker probe and the plugin/canonical fetch chain both call
note_catalog_item, so the same statuses leave both lists.
"""

from hermes_cli.chat_catalog import chat_catalog_ids, is_known_non_chat_model


def test_chat_catalog_ids_drops_unavailable_statuses_and_empty_ids():
    ids = chat_catalog_ids([
        {"id": "live"},
        {"id": "shutdown-mixed", "status": "Shutdown"},
        {"id": "shutdown-upper", "status": "SHUTDOWN"},
        {"id": "retiring-padded", "status": " retiring "},
        {"id": "retired-mixed", "status": "Retired"},
        {"id": "retired-padded", "status": "  ReTiReD  "},
        {"id": ""},
        {"id": "no-status"},
        {"id": "null-status", "status": None},
        {"id": "unknown-status", "status": "Preview"},
    ])
    assert ids == ["live", "no-status", "null-status", "unknown-status"]


def test_shutdown_generation_row_is_still_recorded():
    """A generation row that is also Shutdown is still remembered before it is dropped."""
    model_id = "catalog-only-row"  # no generation token in the name; capabilities decide
    ids = chat_catalog_ids([
        {"id": model_id, "status": "Shutdown", "capabilities": {"type": "image"}},
    ])
    assert ids == []
    assert is_known_non_chat_model(model_id) is True
