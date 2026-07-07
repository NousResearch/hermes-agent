from pathlib import Path

from hermes_cli.context_continuity import ContinuitySettings, ContinuityStore


def test_dropoff_ledger_is_written_and_searchable(tmp_path):
    store = ContinuityStore(
        db_path=tmp_path / "continuity.db",
        ledger_path=tmp_path / "dropoffs.jsonl",
        mirror_dir=tmp_path / "mirror",
    )

    record = store.record_dropoff(
        {
            "child_session_id": "child-1",
            "cwd": str(tmp_path / "repo"),
            "error": "Context length exceeded and cannot compress further",
            "files": ["src/app.ts"],
            "parent_session_id": "parent-1",
            "pending_prompt": "finish the logging task",
            "recent_summary": "We decided to add desktop diagnostics.",
            "trigger": "compression exhausted",
        }
    )

    assert (tmp_path / "dropoffs.jsonl").read_text(encoding="utf-8").count(record["id"]) == 1

    hits = store.search("desktop diagnostics src/app.ts", cwd=str(tmp_path / "repo"), limit=3)

    assert hits
    assert hits[0]["id"] == record["id"]
    assert "diagnostics" in hits[0]["snippet"]


def test_obsidian_index_is_allowlisted_and_skips_private_paths(tmp_path):
    vault = tmp_path / "vault"
    allowed = vault / "Projects"
    private = vault / ".obsidian"
    secret = vault / "Projects" / "secrets"
    allowed.mkdir(parents=True)
    private.mkdir()
    secret.mkdir()
    (allowed / "Hermes.md").write_text("# Hermes\n\nContinuation note about context recovery.", encoding="utf-8")
    (private / "workspace.json").write_text("hidden", encoding="utf-8")
    (secret / "Keys.md").write_text("# Keys\n\nDo not index.", encoding="utf-8")

    store = ContinuityStore(
        db_path=tmp_path / "continuity.db",
        ledger_path=tmp_path / "dropoffs.jsonl",
        mirror_dir=tmp_path / "mirror",
    )

    result = store.index_obsidian(
        ContinuitySettings(
            obsidian_allowlisted_folders=["Projects"],
            obsidian_read_enabled=True,
            obsidian_vault_path=str(vault),
        )
    )

    assert result["indexed"] == 1
    assert store.search("context recovery", limit=3)[0]["source_path"].endswith("Hermes.md")
    assert store.search("Do not index", limit=3) == []


def test_obsidian_disabled_does_not_index(tmp_path):
    store = ContinuityStore(
        db_path=tmp_path / "continuity.db",
        ledger_path=tmp_path / "dropoffs.jsonl",
        mirror_dir=tmp_path / "mirror",
    )

    result = store.index_obsidian(ContinuitySettings(obsidian_read_enabled=False, obsidian_vault_path=str(Path("/tmp"))))

    assert result == {"enabled": False, "indexed": 0, "skipped": 0}

