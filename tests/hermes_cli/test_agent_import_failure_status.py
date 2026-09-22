"""Import failures must be observable by shell automation and remain retryable."""

from types import SimpleNamespace

import pytest

from hermes_cli.agent_import import import_agent_command
from hermes_cli.agent_import_sync import (
    load_sync_manifest, sync_imported_agents, update_sync_manifest,
)


def test_failed_sync_exits_nonzero_and_retries_after_source_repair(tmp_path, monkeypatch):
    home, source = tmp_path / "home", tmp_path / "source"
    home.mkdir()
    source.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    settings = source / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    update_sync_manifest("claude-code", source, home, False, {"items": []})
    original_digest = load_sync_manifest(home)["agents"]["claude-code"]["digest"]
    settings.write_text("{broken", encoding="utf-8")
    with pytest.raises(SystemExit) as failed:
        sync_imported_agents(SimpleNamespace(dry_run=False))
    assert failed.value.code == 1
    assert load_sync_manifest(home)["agents"]["claude-code"]["digest"] == original_digest

    settings.write_text('{"permissions": {"allow": ["Bash(git status)"]}}', encoding="utf-8")
    sync_imported_agents(SimpleNamespace(dry_run=False))
    assert load_sync_manifest(home)["agents"]["claude-code"]["digest"] != original_digest


@pytest.mark.parametrize("dry_run", [True, False])
def test_partial_import_reports_failure_without_marking_source_fully_synced(tmp_path, monkeypatch, dry_run):
    home, source = tmp_path / "home", tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (source / "settings.json").write_text("{broken", encoding="utf-8")
    (source / "CLAUDE.md").write_text("- Prefer concise replies.\n", encoding="utf-8")
    args = SimpleNamespace(agent="claude-code", source=str(source), overwrite=False,
                           dry_run=dry_run, yes=True, sync=False)
    with pytest.raises(SystemExit) as failed:
        import_agent_command(args)
    assert failed.value.code == 1
    entry = load_sync_manifest(home)["agents"].get("claude-code", {})
    assert "digest" not in entry
    if not dry_run:
        assert "Prefer concise replies" in (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
