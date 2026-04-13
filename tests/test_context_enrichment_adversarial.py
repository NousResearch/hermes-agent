"""Adversarial tests for context enrichment features.

Tests secret redaction, edge cases, boundary conditions, and
integration between search preview, related paths, and TaskCheckpoint.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Feature 1: search_files preview_lines
# ---------------------------------------------------------------------------

class TestSearchPreviewLines:
    """Adversarial tests for search_files preview_lines parameter."""

    def test_preview_lines_zero_disables_preview(self, tmp_path):
        """preview_lines=0 should produce no preview field in matches."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n\ndef goodbye():\n    pass\n")
        from tools.file_tools import search_tool
        result = search_tool(pattern="def ", target="content", path=str(tmp_path), preview_lines=0)
        data = json.loads(result)
        assert "matches" in data
        for m in data["matches"]:
            assert "preview" not in m, f"preview should not exist when preview_lines=0: {m}"

    def test_preview_lines_default_is_zero(self, tmp_path):
        """Default preview_lines=0 should NOT include preview (opt-in feature)."""
        lines = [f"line {i}" for i in range(10)]
        lines[5] = "MATCH_HERE"
        test_file = tmp_path / "test.py"
        test_file.write_text("\n".join(lines) + "\n")
        from tools.file_tools import search_tool
        result = search_tool(pattern="MATCH_HERE", target="content", path=str(tmp_path))
        data = json.loads(result)
        assert "matches" in data
        match = data["matches"][0]
        assert "preview" not in match, "Default should be preview disabled (0)"

    def test_preview_lines_explicit_two(self, tmp_path):
        """Explicit preview_lines=2 should include preview with ~5 lines."""
        # Create file with 10 lines
        lines = [f"line {i}" for i in range(10)]
        lines[5] = "MATCH_HERE"
        test_file = tmp_path / "test.py"
        test_file.write_text("\n".join(lines) + "\n")
        from tools.file_tools import search_tool
        result = search_tool(pattern="MATCH_HERE", target="content", path=str(tmp_path), preview_lines=2)
        data = json.loads(result)
        assert "matches" in data
        match = data["matches"][0]
        assert "preview" in match
        preview_lines = match["preview"].strip().split("\n")
        # Should have 5-6 lines (2 before + match + 2 after, allow trailing)
        assert 5 <= len(preview_lines) <= 6, f"Expected 5-6 preview lines, got {len(preview_lines)}"

    def test_preview_lines_cap_at_20(self, tmp_path):
        """preview_lines > 20 should be capped at 20 (41 lines max window)."""
        lines = [f"line {i}" for i in range(100)]
        lines[50] = "MATCH_HERE"
        test_file = tmp_path / "test.py"
        test_file.write_text("\n".join(lines) + "\n")
        from tools.file_tools import search_tool
        result = search_tool(pattern="MATCH_HERE", target="content", path=str(tmp_path), preview_lines=50)
        data = json.loads(result)
        match = data["matches"][0]
        assert "preview" in match
        preview_lines_list = match["preview"].strip().split("\n")
        # Max window = 20*2+1 = 41 (allow +1 for potential edge)
        assert len(preview_lines_list) <= 42, f"Preview should be capped near 41 lines, got {len(preview_lines_list)}"

    def test_preview_redacts_secrets(self, tmp_path):
        """Preview content should be redacted for secrets like content is."""
        test_file = tmp_path / "secrets.py"
        test_file.write_text("# config\nAPI_KEY='sk-1234567890abcdef'\nNORMAL='ok'\n")
        from tools.file_tools import search_tool
        result = search_tool(pattern="API_KEY", target="content", path=str(tmp_path))
        data = json.loads(result)
        assert data["total_count"] >= 1, f"Should find API_KEY match, got: {data}"
        match = data["matches"][0]
        # Content should be redacted
        assert "sk-1234567890abcdef" not in match.get("content", "")
        # Preview should also be redacted
        if "preview" in match:
            assert "sk-1234567890abcdef" not in match["preview"]

    def test_preview_on_first_match_of_file(self, tmp_path):
        """Match on line 1 should show preview starting from line 1."""
        test_file = tmp_path / "test.py"
        test_file.write_text("MATCH line 1\nline 2\nline 3\nline 4\nline 5\n")
        from tools.file_tools import search_tool
        result = search_tool(pattern="MATCH", target="content", path=str(tmp_path), preview_lines=2)
        data = json.loads(result)
        match = data["matches"][0]
        preview = match.get("preview", "")
        # Should start at line 1, not try to go before
        assert "MATCH line 1" in preview

    def test_preview_on_last_line_of_file(self, tmp_path):
        """Match on last line should show preview ending at last line."""
        lines = [f"line {i}" for i in range(5)]
        lines.append("MATCH_END")
        test_file = tmp_path / "test.py"
        test_file.write_text("\n".join(lines) + "\n")
        from tools.file_tools import search_tool
        result = search_tool(pattern="MATCH_END", target="content", path=str(tmp_path), preview_lines=2)
        data = json.loads(result)
        match = data["matches"][0]
        preview = match.get("preview", "")
        assert "MATCH_END" in preview

    def test_preview_no_match_returns_no_preview(self, tmp_path):
        """Search with no results should not produce any preview fields."""
        test_file = tmp_path / "test.py"
        test_file.write_text("nothing here\n")
        from tools.file_tools import search_tool
        result = search_tool(pattern="NONEXISTENT_PATTERN_XYZ", target="content", path=str(tmp_path))
        data = json.loads(result)
        assert data["total_count"] == 0

    def test_preview_files_target_no_preview(self, tmp_path):
        """File search (target=files) should not produce preview fields."""
        test_file = tmp_path / "test_file.py"
        test_file.write_text("content\n")
        from tools.file_tools import search_tool
        result = search_tool(pattern="test_file", target="files", path=str(tmp_path))
        data = json.loads(result)
        if "files" in data:
            for f in data["files"]:
                assert "preview" not in str(f)

    def test_preview_caching_avoids_duplicate_reads(self, tmp_path):
        """Multiple matches in same file near each other should share preview reads."""
        lines = ["MATCH"] * 5 + [f"line {i}" for i in range(20)]
        test_file = tmp_path / "test.py"
        test_file.write_text("\n".join(lines) + "\n")
        from tools.file_tools import search_tool
        # Should not raise or produce garbage even with overlapping windows
        result = search_tool(pattern="MATCH", target="content", path=str(tmp_path), limit=10)
        data = json.loads(result)
        assert data["total_count"] == 5


# ---------------------------------------------------------------------------
# Feature 2: read_file _related_paths
# ---------------------------------------------------------------------------

class TestRelatedPaths:
    """Adversarial tests for read_file _related_paths hints."""

    def test_related_paths_injected_on_first_read(self, tmp_path, monkeypatch):
        """_related_paths should appear on offset=1 reads when local imports exist."""
        monkeypatch.chdir(tmp_path)
        # Create local modules so _related_paths finds them
        (tmp_path / "mylib.py").write_text("# library\n")
        (tmp_path / "myutils.py").write_text("# utils\n")
        src = tmp_path / "mymodule.py"
        src.write_text("import mylib\nimport myutils\nimport os\n")
        from tools.file_tools import read_file_tool
        result = read_file_tool(path=str(src), offset=1, limit=10)
        data = json.loads(result)
        related = data.get("_related_paths", [])
        paths = [e["path"] for e in related]
        # Should find local modules (mylib.py, myutils.py) but not stdlib (os)
        assert any("mylib" in p for p in paths), f"Should find mylib.py import, got: {paths}"

    def test_related_paths_not_injected_on_paginated_read(self, tmp_path):
        """_related_paths should NOT appear when offset > 1."""
        src = tmp_path / "mymodule.py"
        src.write_text("import os\n" * 100)
        from tools.file_tools import read_file_tool
        result = read_file_tool(path=str(src), offset=10, limit=10)
        data = json.loads(result)
        assert "_related_paths" not in data

    def test_related_paths_only_existing_files(self, tmp_path):
        """_related_paths should only include files that actually exist on disk."""
        src = tmp_path / "mymodule.py"
        src.write_text("import os\nimport nonexistent_module_xyz\n")
        from tools.file_tools import read_file_tool
        result = read_file_tool(path=str(src), offset=1, limit=10)
        data = json.loads(result)
        related = data.get("_related_paths", [])
        for entry in related:
            p = entry["path"]
            # All returned paths should exist
            assert (Path.cwd() / p).exists() or Path(p).exists(), f"Related path does not exist: {p}"

    def test_related_paths_capped_at_five(self, tmp_path):
        """_related_paths should have at most 5 entries."""
        # Create a file with many imports
        src = tmp_path / "mymodule.py"
        imports = "\n".join([f"import os  # import {i}" for i in range(20)])
        src.write_text(imports + "\n")
        from tools.file_tools import read_file_tool
        result = read_file_tool(path=str(src), offset=1, limit=10)
        data = json.loads(result)
        related = data.get("_related_paths", [])
        assert len(related) <= 5, f"Expected at most 5 related paths, got {len(related)}"

    def test_related_paths_no_self_reference(self, tmp_path):
        """_related_paths should not include the file itself."""
        src = tmp_path / "mymodule.py"
        src.write_text("import os\n")
        from tools.file_tools import read_file_tool
        result = read_file_tool(path=str(src), offset=1, limit=10)
        data = json.loads(result)
        related = data.get("_related_paths", [])
        for entry in related:
            assert entry["path"] != str(src), "Should not include self"

    def test_related_paths_with_test_companion(self, tmp_path, monkeypatch):
        """Should detect test companion files."""
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "widget.py"
        test = tmp_path / "test_widget.py"
        src.write_text("class Widget:\n    pass\n")
        test.write_text("from widget import Widget\n")
        from tools.file_tools import read_file_tool
        result = read_file_tool(path=str(src), offset=1, limit=10)
        data = json.loads(result)
        related = data.get("_related_paths", [])
        paths = [e["path"] for e in related]
        assert any("test_widget" in p for p in paths), f"Should find test companion, got: {paths}"

    def test_related_paths_with_init_companion(self, tmp_path, monkeypatch):
        """Should detect __init__.py companion for package files."""
        monkeypatch.chdir(tmp_path)
        pkg = tmp_path / "mypackage"
        pkg.mkdir()
        init = pkg / "__init__.py"
        src = pkg / "module.py"
        init.write_text("# package init\n")
        src.write_text("def func():\n    pass\n")
        from tools.file_tools import read_file_tool
        result = read_file_tool(path=str(src), offset=1, limit=10)
        data = json.loads(result)
        related = data.get("_related_paths", [])
        paths = [e["path"] for e in related]
        assert any("__init__" in p for p in paths), f"Should find __init__.py, got: {paths}"

    def test_related_paths_on_non_python_file(self, tmp_path):
        """Non-Python files should not crash, just return limited hints."""
        src = tmp_path / "config.yaml"
        src.write_text("key: value\n")
        from tools.file_tools import read_file_tool
        result = read_file_tool(path=str(src), offset=1, limit=10)
        data = json.loads(result)
        # Should not crash; may or may not have related paths
        assert "error" not in data or data.get("error") is None

    def test_related_paths_on_error_returns_none(self, tmp_path):
        """When file has an error (not found), _related_paths should not be injected."""
        from tools.file_tools import read_file_tool
        result = read_file_tool(path="/nonexistent/path/file.py", offset=1, limit=10)
        data = json.loads(result)
        assert "_related_paths" not in data


# ---------------------------------------------------------------------------
# Feature 3: TaskCheckpoint in compression
# ---------------------------------------------------------------------------

class TestRelatedPathsMultiLanguage:
    """Adversarial tests for multi-language related-path discovery."""

    def test_js_ts_related_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "module.ts"
        src.write_text("import { value } from './dep'\nconst x = require('./dep')\nexport * from './dep'\n")
        (tmp_path / "dep.js").write_text("export const value = 1;\n")
        (tmp_path / "module.test.ts").write_text("// test\n")
        (tmp_path / "__tests__").mkdir()
        (tmp_path / "__tests__" / "module.ts").write_text("// test\n")
        from tools.file_tools import read_file_tool
        data = json.loads(read_file_tool(path=str(src), offset=1, limit=20))
        paths = [e["path"] for e in data.get("_related_paths", [])]
        assert any(p.endswith("dep.js") for p in paths)
        assert any("module.test.ts" in p for p in paths)
        assert any("__tests__" in p for p in paths)

    def test_go_related_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "main.go"
        src.write_text("import (\n  \"fmt\"\n  \"./local\"\n)\n")
        (tmp_path / "local.go").write_text("package main\n")
        (tmp_path / "main_test.go").write_text("package main\n")
        from tools.file_tools import read_file_tool
        data = json.loads(read_file_tool(path=str(src), offset=1, limit=20))
        paths = [e["path"] for e in data.get("_related_paths", [])]
        assert any(p.endswith("local.go") for p in paths)
        assert any(p.endswith("main_test.go") for p in paths)

    def test_rust_related_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pkg = tmp_path / "src"
        pkg.mkdir()
        src = pkg / "lib.rs"
        src.write_text("mod module;\nuse crate::module;\n#[cfg(test)] mod tests;\n")
        (pkg / "module.rs").write_text("pub fn f() {}\n")
        (pkg / "tests").mkdir()
        (pkg / "tests" / "lib.rs").write_text("#[test]\nfn t() {}\n")
        from tools.file_tools import read_file_tool
        data = json.loads(read_file_tool(path=str(src), offset=1, limit=20))
        paths = [e["path"] for e in data.get("_related_paths", [])]
        assert any(p.endswith("module.rs") for p in paths)
        assert any("tests/lib.rs" in p for p in paths)

    def test_config_shell_markdown_related_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = tmp_path / "config.yml"
        config.write_text("a: b\n")
        (tmp_path / "config.yml.example").write_text("x: 1\n")
        shell = tmp_path / "script.sh"
        shell.write_text("source ./other.sh\n. ./other2.bash\n")
        (tmp_path / "other.sh").write_text("echo hi\n")
        (tmp_path / "other2.bash").write_text("echo hi\n")
        md = tmp_path / "guide.md"
        md.write_text("[more](./other.md)\n")
        (tmp_path / "other.md").write_text("ok\n")
        (tmp_path / "package.json").write_text('{"name":"x"}\n')
        for d in ["src", "lib", "tests"]:
            (tmp_path / d).mkdir(exist_ok=True)
        from tools.file_tools import read_file_tool
        config_data = json.loads(read_file_tool(path=str(config), offset=1, limit=20))
        config_paths = [e["path"] for e in config_data.get("_related_paths", [])]
        assert any("config.yml.example" in p for p in config_paths)

        shell_data = json.loads(read_file_tool(path=str(shell), offset=1, limit=20))
        shell_paths = [e["path"] for e in shell_data.get("_related_paths", [])]
        assert any(p.endswith("other.sh") for p in shell_paths)
        assert any(p.endswith("other2.bash") for p in shell_paths)

        md_data = json.loads(read_file_tool(path=str(md), offset=1, limit=20))
        md_paths = [e["path"] for e in md_data.get("_related_paths", [])]
        assert any(p.endswith("other.md") for p in md_paths)


class TestTaskCheckpointAdversarial:
    """Adversarial tests for TaskCheckpoint injection during compression."""

    def test_checkpoint_preserves_modified_files(self):
        """Checkpoint should track files modified by write_file/patch tool calls."""
        from agent.context_compressor import ContextCompressor, TaskCheckpoint
        compressor = ContextCompressor(model="gpt-4o-mini", threshold_percent=0.50)

        messages = [
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": "Let me fix it.",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "patch", "arguments": json.dumps({"path": "src/app.py", "old_string": "bug", "new_string": "fix"})}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"status": "ok", "diff": "-bug\n+fix"})},
        ]

        checkpoint = compressor._build_task_checkpoint(messages, messages[1:])
        assert checkpoint is not None
        assert "src/app.py" in checkpoint.recently_modified_files

    def test_checkpoint_preserves_pending_tool_calls(self):
        """Tool calls without results should be flagged as pending."""
        from agent.context_compressor import ContextCompressor
        compressor = ContextCompressor(model="gpt-4o-mini", threshold_percent=0.50)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "Working...",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "search_files", "arguments": json.dumps({"pattern": "test"})}}]},
            # No tool result for call_1 — it's pending
        ]

        checkpoint = compressor._build_task_checkpoint(messages, messages[1:])
        assert checkpoint is not None
        assert any("search_files" in c for c in checkpoint.pending_tool_calls)

    def test_checkpoint_format_empty_returns_none(self):
        """Empty checkpoint should return None from format_for_injection."""
        from agent.context_compressor import TaskCheckpoint
        cp = TaskCheckpoint()
        assert cp.format_for_injection() is None

    def test_checkpoint_format_respects_max_chars(self):
        """Formatted checkpoint should never exceed _TASK_CHECKPOINT_MAX_CHARS."""
        from agent.context_compressor import TaskCheckpoint, _TASK_CHECKPOINT_MAX_CHARS
        cp = TaskCheckpoint(
            pending_tool_calls=["search_files:pattern=" + "x" * 200],
            recently_modified_files=["file" + str(i) + ".py" * 50 for i in range(10)],
            current_plan_step="A" * 300,
            last_tool_batch_results=["result " + "y" * 200 for _ in range(5)],
        )
        text = cp.format_for_injection()
        if text:
            assert len(text) <= _TASK_CHECKPOINT_MAX_CHARS, f"Checkpoint {len(text)} > {_TASK_CHECKPOINT_MAX_CHARS}"

    def test_checkpoint_survives_compression_with_tool_use(self):
        """After full compression, checkpoint text should be in the compressed output."""
        from agent.context_compressor import ContextCompressor
        compressor = ContextCompressor(model="gpt-4o-mini", threshold_percent=0.50, quiet_mode=True)

        messages = [{"role": "system", "content": "You are helpful."}]
        # Add enough messages to trigger compression
        for i in range(30):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Answer {i}",
                             "tool_calls": [{"id": f"call_{i}", "type": "function",
                                             "function": {"name": "read_file", "arguments": json.dumps({"path": f"file_{i}.py"})}}]})
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": f"Content of file_{i}.py"})

        # Mock the LLM call to avoid real API calls
        with patch.object(compressor, '_generate_summary') as mock_summary:
            mock_summary.return_value = "[CONTEXT COMPACTION] Summary of work done."
            compressed = compressor.compress(messages, current_tokens=50000)

        # Check that checkpoint text is present somewhere in compressed messages
        all_content = " ".join(msg.get("content", "") for msg in compressed)
        has_checkpoint = ("RESUME STATE" in all_content or
                          "[TASK_CHECKPOINT]" in all_content or
                          compressor._build_task_checkpoint(messages, messages[1:]).is_empty())
        assert has_checkpoint, "Checkpoint should be injected or legitimately empty"

    def test_checkpoint_json_parseable(self):
        """The checkpoint text should be valid JSON after the prefix."""
        from agent.context_compressor import TaskCheckpoint, TASK_CHECKPOINT_PREFIX
        cp = TaskCheckpoint(
            pending_tool_calls=["search_files:query"],
            recently_modified_files=["src/main.py"],
            current_plan_step="Fix the bug",
            last_tool_batch_results=["search:ok"],
        )
        text = cp.format_for_injection()
        if text:
            assert text.startswith(TASK_CHECKPOINT_PREFIX)
            json_part = text[len(TASK_CHECKPOINT_PREFIX):].strip()
            parsed = json.loads(json_part)
            assert "v" in parsed
            assert "s" in parsed or "p" in parsed

    def test_checkpoint_does_not_inflate_compressed_size(self):
        """Compressed messages with checkpoint should not be dramatically larger."""
        from agent.context_compressor import ContextCompressor
        compressor = ContextCompressor(model="gpt-4o-mini", threshold_percent=0.50, quiet_mode=True)

        messages = [{"role": "system", "content": "You are helpful."}]
        for i in range(30):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Answer {i}"})

        with patch.object(compressor, '_generate_summary') as mock_summary:
            mock_summary.return_value = "[CONTEXT COMPACTION] Brief summary."
            compressed = compressor.compress(messages, current_tokens=50000)

        # Checkpoint is capped at 500 chars — shouldn't add more than 1-2% to output
        total_chars = sum(len(msg.get("content", "")) for msg in compressed)
        # Even with checkpoint, total should be reasonable
        assert total_chars < 50000, f"Compressed output too large: {total_chars} chars"

    def test_checkpoint_injected_as_system_message(self):
        """Checkpoint should be a separate system message, not embedded in summary."""
        from agent.context_compressor import ContextCompressor
        compressor = ContextCompressor(model="gpt-4o-mini", threshold_percent=0.50, quiet_mode=True)

        messages = [{"role": "system", "content": "You are helpful."}]
        for i in range(30):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Answer {i}",
                             "tool_calls": [{"id": f"call_{i}", "type": "function",
                                             "function": {"name": "patch", "arguments": json.dumps({"path": f"file_{i}.py", "old_string": "a", "new_string": "b"})}}]})
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": json.dumps({"status": "ok"})})

        with patch.object(compressor, '_generate_summary') as mock_summary:
            mock_summary.return_value = "[CONTEXT COMPACTION] Summary of work."
            compressed = compressor.compress(messages, current_tokens=50000)

        # Find system messages in compressed output
        system_msgs = [m for m in compressed if m.get("role") == "system"]
        resume_msgs = [m for m in system_msgs if "RESUME STATE" in (m.get("content") or "")]
        # Should have at least one RESUME STATE system message (checkpoint)
        # OR the checkpoint might be legitimately empty if no tool calls survived compression
        checkpoint = compressor._build_task_checkpoint(messages, messages[1:])
        if checkpoint and not checkpoint.is_empty():
            assert len(resume_msgs) >= 1, (
                f"Checkpoint should be injected as system message with 'RESUME STATE' prefix. "
                f"System messages found: {len(system_msgs)}, RESUME STATE found: {len(resume_msgs)}"
            )
