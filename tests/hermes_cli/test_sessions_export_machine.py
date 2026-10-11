"""Tests for --machine flag on hermes sessions export + import roundtrip.

Design: machine_id is a CLI-level provenance stamp, not a DB-layer concern.
The CLI export handler stamps each session dict with machine_id before
writing the JSONL. The DB API (export_session) stays clean.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _run_cli(args, env, cwd):
    """Run hermes CLI in a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main"] + args,
        capture_output=True, text=True, env=env,
        cwd=str(cwd), timeout=60,
    )


def _hermes_env(hermes_home):
    """Build an env dict with HERMES_HOME pointed at a temp dir."""
    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_PROFILE", None)
    return env


class TestExportMachineFlag:
    """--machine flag on CLI export stamps provenance."""

    def test_export_stamps_machine_id(self, tmp_path):
        """'hermes sessions export --machine X' stamps each line with machine_id."""
        from hermes_state import SessionDB

        # Create a session in a temp DB (simulate "Machine A")
        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        db = SessionDB(src_dir / "state.db")
        db.create_session("test-machine-1", source="cli")
        db.append_message("test-machine-1", role="user", content="Hello from A")
        db.append_message("test-machine-1", role="assistant", content="Hello from B")

        # Export via CLI with --machine
        export_path = tmp_path / "export.jsonl"
        repo_root = Path(__file__).parent.parent.parent
        result = _run_cli(
            ["sessions", "export", str(export_path), "--session-id", "test-machine-1",
             "--machine", "machine-a", "--format", "jsonl"],
            _hermes_env(src_dir), repo_root,
        )
        db.close()
        assert result.returncode == 0, f"CLI export failed: {result.stderr}"
        assert export_path.exists(), f"Export file not created: {export_path}"

        # Verify the stamped machine_id
        with open(export_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"
        assert lines[0].get("machine_id") == "machine-a", f"Expected machine_id=machine-a, got {lines[0].get('machine_id')}"
        assert lines[0].get("id") == "test-machine-1"
        assert len(lines[0].get("messages", [])) >= 2

    def test_export_defaults_machine_to_hostname(self, tmp_path):
        """'hermes sessions export' defaults --machine to hostname when omitted."""
        import socket
        from hermes_state import SessionDB

        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        db = SessionDB(src_dir / "state.db")
        db.create_session("test-default-machine", source="cli")
        db.append_message("test-default-machine", role="user", content="Test")
        db.close()

        export_path = tmp_path / "export-default.jsonl"
        repo_root = Path(__file__).parent.parent.parent
        result = _run_cli(
            ["sessions", "export", str(export_path), "--session-id", "test-default-machine",
             "--format", "jsonl"],
            _hermes_env(src_dir), repo_root,
        )
        assert result.returncode == 0, f"CLI export failed: {result.stderr}"

        with open(export_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1
        expected = socket.gethostname()
        assert lines[0].get("machine_id") == expected, f"Expected machine_id={expected}, got {lines[0].get('machine_id')}"


class TestImportRoundtrip:
    """Export from machine A, import into machine B, verify roundtrip."""

    def test_roundtrip_export_import(self, tmp_path):
        """Full roundtrip: export JSONL → import → verify session present."""
        from hermes_state import SessionDB

        # Source DB ("machine A")
        src = SessionDB(tmp_path / "src-state.db")
        src.create_session("roundtrip-1", source="cli")
        src.append_message("roundtrip-1", role="user", content="Hello from source")
        src.append_message("roundtrip-1", role="assistant", content="Hello from dest")
        exported = src.export_session("roundtrip-1")
        src.close()
        assert exported is not None

        # Add machine_id (simulate the CLI stamping)
        exported["machine_id"] = "machine-a"

        # Write JSONL
        export_path = tmp_path / "export.jsonl"
        with open(export_path, "w") as f:
            f.write(json.dumps(exported) + "\n")

        # Dest DB ("machine B", fresh)
        dest = SessionDB(tmp_path / "dest-state.db")
        with open(export_path) as f:
            sessions = [json.loads(line) for line in f if line.strip()]
        result = dest.import_sessions(sessions)
        assert result.get("imported", 0) >= 1, f"Expected import, got: {result}"

        # Verify
        imported = dest.get_session("roundtrip-1")
        assert imported is not None, "Session not found after import"
        msgs = dest.get_messages("roundtrip-1")
        assert len(msgs) >= 2, f"Expected >=2 messages, got {len(msgs)}"
        dest.close()

    def test_import_skips_existing(self, tmp_path):
        """Import skips sessions that already exist (ID dedup)."""
        from hermes_state import SessionDB

        src = SessionDB(tmp_path / "src-state.db")
        src.create_session("dedup-test", source="cli")
        src.append_message("dedup-test", role="user", content="Original")
        exported = src.export_session("dedup-test")
        src.close()

        # Dest: create the session first
        dest = SessionDB(tmp_path / "dest-state.db")
        dest.create_session("dedup-test", source="cli")
        dest.append_message("dedup-test", role="user", content="Existing")

        # Import
        result = dest.import_sessions([exported])
        assert result.get("skipped", 0) >= 1, f"Expected skip, got: {result}"
        assert result.get("imported", 0) == 0, f"Expected 0 imported, got: {result}"

        # Original content preserved
        msgs = dest.get_messages("dedup-test")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Existing"
        dest.close()


class TestImportCLI:
    """CLI end-to-end: hermes sessions import <file>."""

    def test_import_cli_end_to_end(self, tmp_path):
        """Export → file → CLI import → verify."""
        from hermes_state import SessionDB

        # Source
        src = SessionDB(tmp_path / "src-state.db")
        src.create_session("cli-test-1", source="cli")
        src.append_message("cli-test-1", role="user", content="Hello CLI")
        src.append_message("cli-test-1", role="assistant", content="World")
        exported = src.export_session("cli-test-1")
        src.close()
        exported["machine_id"] = "machine-x"

        export_path = tmp_path / "export.jsonl"
        with open(export_path, "w") as f:
            f.write(json.dumps(exported) + "\n")

        # Dest (fresh HERMES_HOME)
        dest_dir = tmp_path / "dest-hermes"
        dest_dir.mkdir()
        repo_root = Path(__file__).parent.parent.parent
        result = _run_cli(
            ["sessions", "import-hermes", str(export_path)],
            _hermes_env(dest_dir), repo_root,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "Imported" in result.stdout, f"Expected 'Imported', got: {result.stdout}"

        # Verify
        dest_db = SessionDB(dest_dir / "state.db")
        imported = dest_db.get_session("cli-test-1")
        assert imported is not None, "Session not found after CLI import"
        msgs = dest_db.get_messages("cli-test-1")
        assert len(msgs) >= 2, f"Expected >=2 messages, got {len(msgs)}"
        dest_db.close()

    def test_import_cli_machine_filter(self, tmp_path):
        """--machine on import filters by provenance."""
        from hermes_state import SessionDB

        # Two sessions from different machines
        src = SessionDB(tmp_path / "src-state.db")
        src.create_session("machine-a-session", source="cli")
        src.create_session("machine-b-session", source="cli")
        exported_a = src.export_session("machine-a-session")
        exported_b = src.export_session("machine-b-session")
        src.close()
        exported_a["machine_id"] = "machine-a"
        exported_b["machine_id"] = "machine-b"

        export_path = tmp_path / "multi.jsonl"
        with open(export_path, "w") as f:
            f.write(json.dumps(exported_a) + "\n")
            f.write(json.dumps(exported_b) + "\n")

        # Import only machine-a
        dest_dir = tmp_path / "dest-hermes"
        dest_dir.mkdir()
        repo_root = Path(__file__).parent.parent.parent
        result = _run_cli(
            ["sessions", "import-hermes", str(export_path), "--machine", "machine-a"],
            _hermes_env(dest_dir), repo_root,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Verify: machine-a present, machine-b absent
        dest_db = SessionDB(dest_dir / "state.db")
        assert dest_db.get_session("machine-a-session") is not None, "machine-a session should be present"
        assert dest_db.get_session("machine-b-session") is None, "machine-b session should be absent"
        dest_db.close()

    def test_import_cli_dry_run(self, tmp_path):
        """--dry-run previews without side effects."""
        from hermes_state import SessionDB

        src = SessionDB(tmp_path / "src-state.db")
        src.create_session("dryrun-test", source="cli")
        src.append_message("dryrun-test", role="user", content="Test")
        exported = src.export_session("dryrun-test")
        src.close()

        export_path = tmp_path / "dryrun-export.jsonl"
        with open(export_path, "w") as f:
            f.write(json.dumps(exported) + "\n")

        dest_dir = tmp_path / "dest-hermes"
        dest_dir.mkdir()
        repo_root = Path(__file__).parent.parent.parent
        result = _run_cli(
            ["sessions", "import-hermes", str(export_path), "--dry-run"],
            _hermes_env(dest_dir), repo_root,
        )
        assert result.returncode == 0, f"CLI dry-run failed: {result.stderr}"
        assert "Would import" in result.stdout, f"Expected 'Would import', got: {result.stdout}"

        # Verify NO session was actually imported
        dest_db = SessionDB(dest_dir / "state.db")
        imported = dest_db.get_session("dryrun-test")
        assert imported is None, "Session should NOT be present after --dry-run"
        dest_db.close()

    def test_import_cli_missing_file_rc1(self, tmp_path):
        """Importing a non-existent file returns rc=1."""
        dest_dir = tmp_path / "dest-hermes"
        dest_dir.mkdir()
        repo_root = Path(__file__).parent.parent.parent
        result = _run_cli(
            ["sessions", "import-hermes", str(tmp_path / "no-such-file.jsonl")],
            _hermes_env(dest_dir), repo_root,
        )
        assert result.returncode == 1, f"Expected rc=1, got {result.returncode}: {result.stdout}"
        assert "File not found" in result.stdout

    def test_import_cli_no_sessions_rc1(self, tmp_path):
        """Importing a file with no valid sessions returns rc=1."""
        dest_dir = tmp_path / "dest-hermes"
        dest_dir.mkdir()
        repo_root = Path(__file__).parent.parent.parent

        # File with only blank lines
        export_path = tmp_path / "empty.jsonl"
        export_path.write_text("\n\n\n")

        result = _run_cli(
            ["sessions", "import-hermes", str(export_path)],
            _hermes_env(dest_dir), repo_root,
        )
        assert result.returncode == 1, f"Expected rc=1, got {result.returncode}: {result.stdout}"
        assert "No sessions found" in result.stdout

    def test_import_cli_empty_filter_rc1(self, tmp_path):
        """--machine filter that matches nothing returns rc=1."""
        from hermes_state import SessionDB

        src = SessionDB(tmp_path / "src-state.db")
        src.create_session("only-a", source="cli")
        exported = src.export_session("only-a")
        src.close()
        exported["machine_id"] = "machine-a"

        export_path = tmp_path / "only-a.jsonl"
        with open(export_path, "w") as f:
            f.write(json.dumps(exported) + "\n")

        dest_dir = tmp_path / "dest-hermes"
        dest_dir.mkdir()
        repo_root = Path(__file__).parent.parent.parent
        result = _run_cli(
            ["sessions", "import-hermes", str(export_path), "--machine", "machine-b"],
            _hermes_env(dest_dir), repo_root,
        )
        assert result.returncode == 1, f"Expected rc=1, got {result.returncode}: {result.stdout}"
        assert "No sessions from machine" in result.stdout