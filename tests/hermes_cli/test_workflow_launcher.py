from pathlib import Path

from hermes_cli.workflow_launcher import (
    WORKFLOW_DIRNAME,
    init_workflow,
    inspect_workflow,
    inventory_artifact_repository,
)


def _write_artifact(root: Path, slug: str, files: dict[str, str]) -> Path:
    artifact_dir = root / "artifacts" / slug
    artifact_dir.mkdir(parents=True)
    for name, content in files.items():
        (artifact_dir / name).write_text(content, encoding="utf-8")
    return artifact_dir


def test_inventory_artifact_repository_counts_required_files(tmp_path):
    _write_artifact(
        tmp_path,
        "complete",
        {
            "metadata.json": '{"title":"Complete Artifact","updated_at":"2026-05-20"}',
            "preview.html": "<html></html>",
            "artifact.md": "# Artifact",
            "notes.md": "Notes",
            "thumbnail.png": "fake",
            "source.jsx": "export default function App() { return null }",
        },
    )
    _write_artifact(
        tmp_path,
        "missing-source",
        {
            "metadata.json": '{"title":"Missing Source"}',
            "preview.html": "<html></html>",
            "artifact.md": "# Artifact",
            "notes.md": "Notes",
        },
    )

    inventory = inventory_artifact_repository(tmp_path)

    assert inventory.total == 2
    assert inventory.with_preview == 2
    assert inventory.with_thumbnail == 1
    assert inventory.missing_required_count == 1
    missing = next(record for record in inventory.records if record.slug == "missing-source")
    assert "thumbnail.png" in missing.missing
    assert "source.html|source.jsx" in missing.missing


def test_init_workflow_dry_run_does_not_write(tmp_path):
    writes = init_workflow(tmp_path, workflow_name="Artifact Repo", dry_run=True)

    assert writes
    assert all(write.action in {"create", "write", "exists"} for write in writes)
    assert not (tmp_path / WORKFLOW_DIRNAME).exists()


def test_init_workflow_writes_and_refuses_overwrite_without_force(tmp_path):
    init_workflow(tmp_path, workflow_name="Artifact Repo")
    state_file = tmp_path / WORKFLOW_DIRNAME / "WORKFLOW_STATE.md"
    state_file.write_text("custom state\n", encoding="utf-8")

    writes = init_workflow(tmp_path, workflow_name="Artifact Repo")

    assert state_file.read_text(encoding="utf-8") == "custom state\n"
    assert any(write.path == state_file and write.action == "exists" for write in writes)


def test_inspect_workflow_reports_inventory(tmp_path):
    _write_artifact(
        tmp_path,
        "one",
        {
            "metadata.json": '{"title":"One"}',
            "preview.html": "<html></html>",
            "artifact.md": "# Artifact",
            "notes.md": "Notes",
            "thumbnail.png": "fake",
            "source.html": "<html></html>",
        },
    )

    output = inspect_workflow(tmp_path)

    assert "Artifacts: 1" in output
    assert "With preview: 1" in output
    assert "`one`" in output
