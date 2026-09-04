"""Unit tests for gate-append-only-writes.py hook."""
import json
import subprocess
import tempfile
from pathlib import Path

import pytest


HOOK_PATH = Path("agent-hooks/gate-append-only-writes.py")


def run_gate(payload: dict) -> dict:
    """Run the gate hook with given payload, return parsed result."""
    result = subprocess.run(
        ["python3", str(HOOK_PATH)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Gate failed: {result.stderr}")
    return json.loads(result.stdout.strip())


def test_patch_without_path_extracts_v4a_targets():
    """When patch tool has no explicit path, extract from V4A headers and fail-closed."""
    patch = """*** Update File: trading-arena/trader-1/journal.md

--- trading-arena/trader-1/journal.md
+++ trading-arena/trader-1/journal.md
@@ -1,3 +1,4 @@
 - Existing line 1
 - Existing line 2
+- New appended line
"""
    payload = {
        "tool_name": "apply_patch",
        "args": {"patch": patch}
    }
    
    # Should detect the journal.md target even without explicit path arg
    # Since file doesn't exist to verify, should block (fail-closed)
    result = run_gate(payload)
    assert result.get("decision") == "block", f"Expected block, got {result}"


def test_patch_delete_operation_blocks():
    """Delete operations on protected journals must be blocked."""
    patch = """*** Delete File: trading-arena/trader-1/journal.md

--- trading-arena/trader-1/journal.md
+++ /dev/null
@@ -1,3 +0,0 @@
-- Line 1
-- Line 2
-- Line 3
"""
    payload = {
        "tool_name": "apply_patch",
        "args": {"patch": patch}
    }
    
    result = run_gate(payload)
    assert result.get("decision") == "block"
    assert "drop or shrink" in result.get("reason", "").lower() or "delete" in result.get("reason", "").lower()


def test_patch_move_operation_blocks():
    """Move operations on protected journals must be blocked."""
    patch = """*** Move File: trading-arena/trader-1/journal.md -> trading-arena/trader-1/journal-old.md

--- trading-arena/trader-1/journal.md
+++ trading-arena/trader-1/journal-old.md
"""
    payload = {
        "tool_name": "apply_patch",
        "args": {"patch": patch}
    }
    
    result = run_gate(payload)
    assert result.get("decision") == "block"


def test_patch_with_deletions_blocks():
    """Patches that remove lines from protected journals must be blocked."""
    patch = """*** Update File: trading-arena/trader-1/journal.md

--- trading-arena/trader-1/journal.md
+++ trading-arena/trader-1/journal.md
@@ -1,4 +1,3 @@
 - Line 1
-- Line 2 DELETED
 - Line 3
+- Line 4 added
"""
    payload = {
        "tool_name": "apply_patch",
        "args": {"patch": patch}
    }
    
    result = run_gate(payload)
    # Will block because it has deletion lines (- Line 2)
    assert result.get("decision") == "block"


def test_patch_append_only_allows():
    """Pure append patches (no deletions) should be allowed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = Path(tmpdir) / "trading-arena" / "trader-1" / "journal.md"
        journal.parent.mkdir(parents=True)
        journal.write_text("- Existing line 1\n- Existing line 2\n")
        
        patch = f"""*** Update File: {journal}

--- {journal}
+++ {journal}
@@ -1,2 +1,3 @@
 - Existing line 1
 - Existing line 2
+- New appended line
"""
        payload = {
            "tool_name": "apply_patch",
            "args": {"patch": patch, "path": str(journal)}
        }
        
        result = run_gate(payload)
        # Should allow pure append
        assert result.get("decision") != "block"


def test_terminal_truncate_blocks():
    """Terminal commands with > (truncate) on protected journals must block."""
    payload = {
        "tool_name": "terminal",
        "args": {"command": "echo 'new content' > trading-arena/trader-1/journal.md"}
    }
    
    result = run_gate(payload)
    assert result.get("decision") == "block"
    assert "rewrite" in result.get("reason", "").lower()


def test_terminal_append_allows():
    """Terminal commands with >> (append) on protected journals should allow."""
    payload = {
        "tool_name": "terminal",
        "args": {"command": "echo '- New entry' >> trading-arena/trader-1/journal.md"}
    }
    
    result = run_gate(payload)
    # Should allow append
    assert result.get("decision") != "block"


def test_write_file_on_existing_journal_blocks():
    """write_file on existing protected journal must block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = Path(tmpdir) / "trading-arena" / "trader-1" / "journal.md"
        journal.parent.mkdir(parents=True)
        journal.write_text("- Existing content\n")
        
        payload = {
            "tool_name": "write_file",
            "args": {"path": str(journal), "content": "- New content\n"}
        }
        
        result = run_gate(payload)
        assert result.get("decision") == "block"
        assert "write_file" in result.get("reason", "").lower()


def test_non_protected_path_allows():
    """Operations on non-protected paths should allow."""
    payload = {
        "tool_name": "write_file",
        "args": {"path": "/tmp/test.txt", "content": "anything"}
    }
    
    result = run_gate(payload)
    assert result.get("decision") != "block"


def test_improvements_md_protection():
    """IMPROVEMENTS.md under trading-arena should be protected."""
    payload = {
        "tool_name": "terminal",
        "args": {"command": "echo 'truncate' > trading-arena/IMPROVEMENTS.md"}
    }
    
    result = run_gate(payload)
    assert result.get("decision") == "block"


def test_relative_path_with_workdir_blocks():
    """Relative paths + workdir should resolve correctly and block protected journals."""
    payload = {
        "tool_name": "terminal",
        "args": {
            "command": "echo x | tee journal.md",
            "workdir": "/workspace/trading-arena/trader-1"
        }
    }
    
    result = run_gate(payload)
    assert result.get("decision") == "block", f"Expected block, got {result}"


def test_write_file_with_workdir_blocks():
    """write_file with relative path + workdir should block if file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = Path(tmpdir) / "trading-arena" / "trader-1" / "journal.md"
        journal.parent.mkdir(parents=True)
        journal.write_text("- Existing content\n")
        
        payload = {
            "tool_name": "write_file",
            "args": {
                "path": "journal.md",
                "content": "- New content\n",
                "workdir": str(journal.parent)
            }
        }
        
        result = run_gate(payload)
        assert result.get("decision") == "block", f"Expected block, got {result}"


def test_truncate_with_workdir_blocks():
    """truncate with relative path + workdir should block."""
    payload = {
        "tool_name": "terminal",
        "args": {
            "command": "truncate -s 0 journal.md",
            "workdir": "/workspace/trading-arena/trader-1"
        }
    }
    
    result = run_gate(payload)
    assert result.get("decision") == "block", f"Expected block, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
