"""Dry-run tests for lean Kanban graph_metadata sidecar helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_graph_metadata import (
    classify_durable_evidence,
    extract_graph_metadata_from_completion_metadata,
    extract_graph_metadata_from_text,
    validate_graph_metadata,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def lean_metadata(**overrides):
    data = {
        "workflow_id": "graph_kanban_metadata_v0_1_1",
        "node_id": "adapter_spike",
        "node_type": "implementation_spike",
        "owner_profile": "ax0s",
        "depends_on": ["validator_review"],
        "objective": "Implement dry-run parser only.",
        "deliverable": "Reviewable helper and tests.",
        "acceptance_criteria": ["No dispatcher mutation", "Legacy compatible"],
        "validation_required": True,
        "mutation_level": "helper_only",
        "artifact_ref": "/root/.hermes/kanban/boards/demo/durable-artifacts/packet.md",
        "source_of_truth": {"state": "candidate"},
    }
    data.update(overrides)
    return data


def test_extracts_valid_lean_graph_metadata_from_yaml_section():
    body = """
# Task body

```yaml
graph_metadata:
  workflow_id: graph_kanban_metadata_v0_1_1
  node_id: adapter_spike
  node_type: implementation_spike
  owner_profile: ax0s
  depends_on:
    - validator_review
  objective: Implement dry-run parser only.
  deliverable: Reviewable helper and tests.
  acceptance_criteria:
    - No dispatcher mutation
    - Legacy compatible
  validation_required: true
  mutation_level: helper_only
  artifact_ref: /root/.hermes/kanban/boards/demo/durable-artifacts/packet.md
  source_of_truth:
    state: candidate
```
"""

    metadata = extract_graph_metadata_from_text(body)
    result = validate_graph_metadata(metadata)

    assert metadata is not None
    assert metadata["workflow_id"] == "graph_kanban_metadata_v0_1_1"
    assert result.valid is True
    assert result.legacy is False
    assert result.missing_fields == []


def test_extracts_graph_metadata_from_markdown_section():
    body = """
## graph_metadata
workflow_id: graph_kanban_metadata_v0_1_1
node_id: adapter_spike
node_type: implementation_spike
owner_profile: ax0s
depends_on:
  - validator_review
objective: Implement dry-run parser only.
deliverable: Reviewable helper and tests.
acceptance_criteria:
  - No dispatcher mutation
validation_required: true
mutation_level: helper_only
artifact_ref: /root/.hermes/kanban/boards/demo/durable-artifacts/packet.md
source_of_truth:
  state: candidate

## Next section
Plain markdown resumes.
"""

    metadata = extract_graph_metadata_from_text(body)

    assert metadata is not None
    assert metadata["node_id"] == "adapter_spike"
    assert validate_graph_metadata(metadata).valid is True


def test_extracts_graph_metadata_from_completion_metadata_json():
    metadata = extract_graph_metadata_from_completion_metadata(
        {"graph_metadata": lean_metadata()}
    )

    assert metadata is not None
    assert metadata["node_id"] == "adapter_spike"
    assert validate_graph_metadata(metadata).valid is True


def test_missing_required_fields_are_reported():
    metadata = lean_metadata()
    del metadata["owner_profile"]
    del metadata["source_of_truth"]

    result = validate_graph_metadata(metadata)

    assert result.valid is False
    assert result.legacy is False
    assert result.missing_fields == ["owner_profile", "source_of_truth.state"]


def test_optional_advisory_fields_are_allowed_but_not_required():
    metadata = lean_metadata(
        unlocks=["validator_review"],
        non_goals=["dispatcher enforcement"],
        return_schema={"summary": "string"},
        attempt=1,
        edge_type="advisory",
        human_approval_required=True,
    )

    result = validate_graph_metadata(metadata)

    assert result.valid is True
    assert result.missing_fields == []


def test_legacy_absent_graph_metadata_is_valid_noop():
    assert extract_graph_metadata_from_text("# Legacy handoff\nNo sidecar here.") is None
    assert extract_graph_metadata_from_completion_metadata({"summary": "legacy"}) is None

    result = validate_graph_metadata(None)

    assert result.valid is True
    assert result.legacy is True
    assert result.missing_fields == []


def test_scratch_artifact_alone_is_insufficient():
    metadata = lean_metadata(
        artifact_ref="/root/.hermes/kanban/boards/demo/workspaces/t_1234/output.md"
    )

    evidence = classify_durable_evidence(metadata)

    assert evidence.state == "insufficient"
    assert "scratch" in evidence.reason


def test_custom_scratch_root_with_scratch_workspace_kind_is_insufficient():
    metadata = lean_metadata(
        artifact_ref="/tmp/custom-kanban-workspaces/t_123/output.md"
    )

    evidence = classify_durable_evidence(metadata, workspace_kind="scratch")

    assert evidence.state == "insufficient"
    assert "scratch" in evidence.reason


def test_scratch_artifact_with_mirror_pointer_only_is_insufficient():
    metadata = lean_metadata(
        artifact_ref="/root/.hermes/kanban/boards/demo/workspaces/t_1234/output.md",
        kanban_mirror_comment_id=19,
    )

    evidence = classify_durable_evidence(metadata)

    assert evidence.state == "insufficient"
    assert "mirror evidence" in evidence.reason


def test_scratch_artifact_with_validation_ready_mirror_comment_is_mirrored():
    metadata = lean_metadata(
        artifact_ref="/root/.hermes/kanban/boards/demo/workspaces/t_1234/output.md",
        kanban_mirror_comment_id=19,
    )

    evidence = classify_durable_evidence(
        metadata,
        comments=["## Validation-ready mirror\nFull packet excerpt for downstream review."],
    )

    assert evidence.state == "mirrored"


def test_validation_ready_mirror_comment_accepts_missing_artifact_as_mirrored():
    metadata = lean_metadata(artifact_ref=None, save_location=None)

    evidence = classify_durable_evidence(
        metadata,
        comments=["Validation-ready mirror: self-contained validator packet excerpt."],
    )

    assert evidence.state == "mirrored"


def test_durable_artifact_path_is_durable():
    metadata = lean_metadata(
        artifact_ref="/root/.hermes/kanban/boards/demo/durable-artifacts/output.md"
    )

    evidence = classify_durable_evidence(metadata)

    assert evidence.state == "durable"


def test_non_artifact_validator_node_is_not_applicable():
    metadata = lean_metadata(
        node_type="validator_review",
        deliverable="Validation verdict only",
        artifact_ref=None,
        save_location=None,
    )

    evidence = classify_durable_evidence(metadata)

    assert evidence.state == "not_applicable"


def test_graph_metadata_sidecar_does_not_change_dispatch_or_parent_child_behavior(
    kanban_home, monkeypatch
):
    body = """
```yaml
graph_metadata:
  workflow_id: graph_kanban_metadata_v0_1_1
  node_id: child
  node_type: implementation_spike
  owner_profile: other-profile
  depends_on: [some_graph_node]
  objective: This metadata is advisory only.
  deliverable: Helper tests.
  acceptance_criteria: [No dispatch mutation]
  validation_required: true
  mutation_level: helper_only
  artifact_ref: /root/.hermes/kanban/boards/demo/durable-artifacts/output.md
  source_of_truth:
    state: promoted
```
"""
    spawned = []
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "ax0s")

    def spawn_fn(task, *_args, **_kwargs):
        spawned.append(task.id)
        return 12345

    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(
            conn,
            title="child",
            assignee="ax0s",
            body=body,
            parents=[parent],
        )
        assert kb.get_task(conn, child).status == "todo"
        assert kb.dispatch_once(conn, spawn_fn=spawn_fn).spawned == []

        kb.complete_task(conn, parent, summary="done")
        assert kb.get_task(conn, child).status == "ready"
        result = kb.dispatch_once(conn, spawn_fn=spawn_fn)

    assert [item[0] for item in result.spawned] == [child]
    assert spawned == [child]
