from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "notes" / "data-migration" / "path-contract-inventory.md"


def test_had_1147_path_contract_inventory_covers_required_regions_and_decisions():
    text = MANIFEST.read_text(encoding="utf-8")

    required_fragments = [
        "# HAD-1147 path-contract inventory",
        "No migration commands were executed for this issue.",
        "## Evidence inventory",
        "### Hermes source path resolution",
        "### State and logging",
        "### Cron storage",
        "### Gateway runtime",
        "### Terminal backend",
        "### Codex delegate and ctx",
        "### Tool storage",
        "### Docker and Compose",
        "### Systemd user services and timers",
        "## Path-contract table",
        "Logical path | Source/evidence | Owner | Mutability | Backup class | Migration strategy | Decision | Notes",
        "## Contradictions and unknowns",
        "## Follow-up checklist for the migration issue",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in text]
    assert missing == []

    required_logical_paths = [
        "/home/david/.hermes",
        "/home/david/stacks",
        "/home/david/.codex",
        "/home/david/.ctx-data",
        "/opt/data",
        "/tmp/hermes-results",
    ]
    missing_paths = [path for path in required_logical_paths if path not in text]
    assert missing_paths == []

    assert "| bind-mount |" in text
    assert "| no-move |" in text
    assert "Move behind bind-mount" in text
    assert "Docker named volumes" in text
