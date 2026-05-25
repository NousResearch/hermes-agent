import json
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "hindsight_cleanup_persona_duplicates.py"
SPEC = importlib.util.spec_from_file_location("hindsight_cleanup_persona_duplicates", SCRIPT_PATH)
cleanup = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(cleanup)


def test_choose_duplicate_actions_keeps_newest_persona_state():
    records = [
        cleanup._record_from_mapping({
            "id": "old",
            "document_id": "doc-old",
            "text": "old state",
            "tags": ["persona-state"],
            "metadata": {"source": "inner_state"},
            "updated_at": "2026-01-01T00:00:00Z",
        }),
        cleanup._record_from_mapping({
            "id": "new",
            "document_id": "doc-new",
            "text": "new state",
            "tags": ["persona-state"],
            "metadata": {"source": "inner_state"},
            "updated_at": "2026-01-01T01:00:00Z",
        }),
    ]

    actions = cleanup.choose_duplicate_actions(records, ["persona-state"])

    assert len(actions) == 1
    assert actions[0]["memory_id"] == "old"
    assert actions[0]["superseded_by_memory_id"] == "new"
    assert actions[0]["correction"] == "new state"


def test_choose_duplicate_actions_covers_priority_tags_by_document_id():
    records = [
        cleanup._record_from_mapping({
            "id": "old",
            "document_id": "persona:pouls-current",
            "text": "Pouls 20h",
            "tags": ["pouls"],
            "updated_at": "2026-01-01T00:00:00Z",
        }),
        cleanup._record_from_mapping({
            "id": "new",
            "document_id": "persona:pouls-current",
            "text": "Pouls 21h",
            "tags": ["pouls"],
            "updated_at": "2026-01-01T01:00:00Z",
        }),
    ]

    actions = cleanup.choose_duplicate_actions(records, cleanup.DEFAULT_DEDUPE_CONTEXT_TAGS)

    assert len(actions) == 1
    assert actions[0]["memory_id"] == "old"
    assert actions[0]["superseded_by_memory_id"] == "new"
    assert actions[0]["tags"] == ["hygiene-memoire", "pouls"]


def test_apply_hygiene_actions_is_idempotent(tmp_path):
    hygiene_path = tmp_path / "memory_hygiene.jsonl"
    actions = [{
        "status": "superseded",
        "memory_id": "old",
        "document_id": "doc-old",
        "query": "old state",
        "reason": "duplicate",
        "correction": "new state",
        "tags": ["persona-state"],
        "created_at": "2026-01-01T00:00:00Z",
    }]

    assert cleanup.apply_hygiene_actions(hygiene_path, actions) == 1
    assert cleanup.apply_hygiene_actions(hygiene_path, actions) == 0

    lines = hygiene_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["memory_id"] == "old"
