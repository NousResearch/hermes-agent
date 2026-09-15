"""Regression coverage for the bounded Knowledge Sync CLI surface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def test_knowledge_help_uses_existing_sync_and_verify_parser(monkeypatch, capsys):
    from hermes_cli import main as main_module

    monkeypatch.setattr(sys, "argv", ["hermes", "knowledge", "--help"])
    with pytest.raises(SystemExit) as exc:
        main_module.main()
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "sync" in output and "verify" in output


def test_knowledge_cli_is_registered_and_fails_closed_for_missing_manifest(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli import main as main_module

    monkeypatch.setattr(main_module, "_plugin_cli_discovery_needed", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "knowledge",
            "sync",
            "--manifest",
            str(tmp_path / "missing.json"),
            "--dry-run",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert captured.out, captured.err
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert "manifest" in payload["error"].lower()
    assert "invalid choice" not in captured.err.lower()


@pytest.fixture
def knowledge_cli_home(tmp_path, monkeypatch):
    from hermes_cli import main as main_module

    home = tmp_path / "profile"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("existing memory", encoding="utf-8")
    (home / "memories" / "USER.md").write_text("keep user", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(main_module, "_plugin_cli_discovery_needed", lambda: False)
    return home


def _invoke_knowledge_main(monkeypatch, capsys, *arguments):
    from hermes_cli import main as main_module

    monkeypatch.setattr(sys, "argv", ["hermes", "knowledge", *arguments, "--json"])
    try:
        result = main_module.main()
    except SystemExit as exc:
        result = exc.code
    captured = capsys.readouterr()
    assert captured.out, captured.err
    return 0 if result is None else result, json.loads(captured.out)


@pytest.mark.parametrize("tamper", [False, True], ids=["verify-success", "verify-mismatch"])
def test_knowledge_main_applies_real_memory_and_propagates_verify_status(
    knowledge_cli_home, tmp_path, monkeypatch, capsys, tamper
):
    home = knowledge_cli_home
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "run_id": "cli-public-001",
            "verified_at": "2026-08-28T06:00:00Z",
            "sources": [
                {"id": "local-doc", "kind": "local", "locator": "local", "revision": "r1"},
                {"id": "drive-index", "kind": "drive", "locator": "https://drive.example/doc", "revision": "r1"},
                {"id": "github-fork", "kind": "github", "locator": "https://github.com/example/test", "revision": "abc123"},
            ],
            "records": [{
                "id": "fact-1",
                "fact_key": "test.boundary",
                "domain": "routing",
                "statement": "The CLI fixture retains its existing memory.",
                "source_id": "local-doc",
            }],
        }),
        encoding="utf-8",
    )
    status, applied = _invoke_knowledge_main(
        monkeypatch, capsys, "sync", "--manifest", str(manifest), "--apply"
    )
    assert status == 0
    assert applied["memory"]["status"] == "applied"
    assert Path(applied["memory"]["path"]).resolve() == (home / "memories" / "MEMORY.md").resolve()
    assert Path(applied["memory"]["backup_path"]).read_text(encoding="utf-8") == "existing memory"
    memory_path = home / "memories" / "MEMORY.md"
    written = memory_path.read_text(encoding="utf-8")
    assert "existing memory" in written
    assert "The CLI fixture retains its existing memory." in written
    assert (home / "memories" / "USER.md").read_text(encoding="utf-8") == "keep user"
    audit_path = home / "knowledge" / "knowledge-sync.jsonl"
    audit_before = audit_path.read_bytes()
    if tamper:
        memory_path.write_text(written + "\nexternal change", encoding="utf-8")
    memory_before_verify = memory_path.read_bytes()

    status, verified = _invoke_knowledge_main(
        monkeypatch, capsys, "verify", "--run-id", applied["run_id"]
    )
    assert status == (1 if tamper else 0)
    assert verified["ok"] is (not tamper)
    assert verified["memory_sha256_matches"] is (not tamper)
    assert memory_path.read_bytes() == memory_before_verify
    assert audit_path.read_bytes() == audit_before


@pytest.mark.parametrize("malformed", [False, True], ids=["missing-audit", "malformed-audit"])
def test_knowledge_main_returns_two_for_unusable_audit(
    knowledge_cli_home, monkeypatch, capsys, malformed
):
    home = knowledge_cli_home
    if malformed:
        audit_path = home / "knowledge" / "knowledge-sync.jsonl"
        audit_path.parent.mkdir()
        audit_path.write_text("{invalid json\n", encoding="utf-8")

    status, refused = _invoke_knowledge_main(
        monkeypatch, capsys, "verify", "--run-id", "cli-public-001"
    )
    assert status == 2
    assert refused["ok"] is False
    assert "audit" in refused["error"]
    assert (home / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "existing memory"
