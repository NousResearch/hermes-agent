"""Tests for agent.execution_verifier — post-tool-call verification."""

import json
import os
import tempfile

import pytest

from agent.execution_verifier import (
    VerificationResult,
    verify_tool_result,
    _verify_terminal,
    _verify_write_file,
    _verify_patch,
    _verify_read_file,
    _verify_browser_navigate,
    _verify_web_extract,
    VERIFIED,
    WARNING,
    MISMATCH,
)


# ===================================================================
# VerificationResult
# ===================================================================

class TestVerificationResult:
    def test_to_dict_verified(self):
        vr = VerificationResult(
            status=VERIFIED, tool_name="terminal", check="git_clone_dir_exists",
        )
        d = vr.to_dict()
        assert d["status"] == "verified"
        assert d["tool"] == "terminal"
        assert d["check"] == "git_clone_dir_exists"
        assert "message" not in d  # empty message excluded

    def test_to_dict_mismatch_includes_message(self):
        vr = VerificationResult(
            status=MISMATCH, tool_name="write_file", check="file_written",
            message="written file does not exist: /tmp/foo.py",
            details={"expected_path": "/tmp/foo.py", "exists": False},
        )
        d = vr.to_dict()
        assert d["status"] == "mismatch"
        assert "does not exist" in d["message"]
        assert d["details"]["exists"] is False

    def test_to_dict_warning(self):
        vr = VerificationResult(
            status=WARNING, tool_name="write_file", check="file_written",
            message="file exists but is empty despite non-empty content arg: /tmp/empty.txt",
        )
        d = vr.to_dict()
        assert d["status"] == "warning"
        assert "empty" in d["message"]


# ===================================================================
# Terminal verifier
# ===================================================================

class TestTerminalVerifier:
    # --- git clone ---

    def test_git_clone_with_explicit_dir_exists(self, tmp_path):
        target = tmp_path / "myrepo"
        target.mkdir()
        args = {"command": f"git clone https://github.com/user/repo.git {target}"}
        result_data = {"output": "Cloning...", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.check == "git_clone_dir_exists"

    def test_git_clone_with_explicit_dir_missing(self, tmp_path):
        target = tmp_path / "nonexistent"
        args = {"command": f"git clone https://github.com/user/repo.git {target}"}
        result_data = {"output": "Cloning...", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH
        assert "does not exist" in vr.message

    def test_git_clone_inferred_dir_exists(self, tmp_path):
        target = tmp_path / "my-project"
        target.mkdir()
        args = {
            "command": "git clone https://github.com/user/my-project.git",
            "workdir": str(tmp_path),
        }
        result_data = {"output": "Cloning...", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED

    def test_git_clone_inferred_dir_missing(self, tmp_path):
        args = {
            "command": "git clone https://github.com/user/my-project.git",
            "workdir": str(tmp_path),
        }
        result_data = {"output": "Cloning...", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH

    def test_git_clone_nonzero_exit_skipped(self):
        args = {"command": "git clone https://github.com/user/repo.git /tmp/x"}
        result_data = {"output": "fatal: ...", "exit_code": 128, "error": "clone failed"}
        vr = _verify_terminal(args, result_data)
        assert vr is None  # no verification on failure

    def test_git_clone_with_flags(self, tmp_path):
        target = tmp_path / "repo"
        target.mkdir()
        args = {"command": f"git clone --depth 1 --branch main https://github.com/user/repo.git {target}"}
        result_data = {"output": "Cloning...", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED

    # --- git init ---

    def test_git_init_success(self, tmp_path):
        target = tmp_path / "myproject"
        target.mkdir()
        (target / ".git").mkdir()
        args = {"command": f"git init {target}"}
        result_data = {"output": "Initialized empty Git repository", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.check == "git_init_dir_exists"

    def test_git_init_missing(self, tmp_path):
        target = tmp_path / "myproject"
        target.mkdir()  # no .git subdir
        args = {"command": f"git init {target}"}
        result_data = {"output": "Initialized empty Git repository", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH
        assert ".git" in vr.message

    def test_git_init_no_arg_uses_workdir(self, tmp_path):
        (tmp_path / ".git").mkdir()
        args = {"command": "git init", "workdir": str(tmp_path)}
        result_data = {"output": "Initialized empty Git repository", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED

    def test_git_init_bare_skipped(self):
        args = {"command": "git init --bare myrepo"}
        result_data = {"output": "Initialized empty Git repository", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is None

    # --- mkdir ---

    def test_mkdir_exists(self, tmp_path):
        target = tmp_path / "newdir"
        target.mkdir()
        args = {"command": f"mkdir -p {target}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.check == "mkdir_dir_exists"

    def test_mkdir_missing(self, tmp_path):
        target = tmp_path / "ghost"
        args = {"command": f"mkdir {target}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH

    # --- cp ---

    def test_cp_dest_exists(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("data")
        dest = tmp_path / "dest.txt"
        dest.write_text("data")
        args = {"command": f"cp {src} {dest}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.check == "cp_dest_exists"

    def test_cp_dest_missing(self, tmp_path):
        src = tmp_path / "src.txt"
        dest = tmp_path / "dest.txt"
        args = {"command": f"cp {src} {dest}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH
        assert "does not exist" in vr.message

    def test_cp_with_flags(self, tmp_path):
        src = tmp_path / "srcdir"
        src.mkdir()
        dest = tmp_path / "destdir"
        dest.mkdir()
        args = {"command": f"cp -r {src} {dest}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED

    def test_compound_command_cp(self, tmp_path):
        src = tmp_path / "a.txt"
        src.write_text("hi")
        dest = tmp_path / "b.txt"
        dest.write_text("hi")
        args = {"command": f"cp {src} {dest} && echo done"}
        result_data = {"output": "done", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.check == "cp_dest_exists"

    # --- mv ---

    def test_mv_dest_exists(self, tmp_path):
        dest = tmp_path / "moved.txt"
        dest.write_text("data")
        args = {"command": f"mv {tmp_path / 'orig.txt'} {dest}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.check == "mv_dest_exists"

    def test_mv_dest_missing(self, tmp_path):
        dest = tmp_path / "gone.txt"
        args = {"command": f"mv {tmp_path / 'orig.txt'} {dest}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH

    # --- rm ---

    def test_rm_targets_removed(self, tmp_path):
        target = tmp_path / "deleteme.txt"
        # target does NOT exist — rm succeeded
        args = {"command": f"rm {target}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.check == "rm_targets_removed"

    def test_rm_targets_still_exist(self, tmp_path):
        target = tmp_path / "stubborn.txt"
        target.write_text("still here")
        args = {"command": f"rm {target}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH
        assert "still exist" in vr.message

    def test_rm_with_rf_flag(self, tmp_path):
        target = tmp_path / "somedir"
        # target does NOT exist — rm -rf succeeded
        args = {"command": f"rm -rf {target}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED

    def test_rm_glob_skipped(self):
        args = {"command": "rm *.tmp"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is None

    # --- touch ---

    def test_touch_file_exists(self, tmp_path):
        target = tmp_path / "newfile.txt"
        target.write_text("")
        args = {"command": f"touch {target}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.check == "touch_file_exists"

    def test_touch_file_missing(self, tmp_path):
        target = tmp_path / "phantom.txt"
        args = {"command": f"touch {target}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH

    def test_touch_with_value_flags(self, tmp_path):
        target = tmp_path / "dated.txt"
        target.write_text("")
        args = {"command": f"touch -t 202301010000 {target}"}
        result_data = {"output": "", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED

    # --- unrelated ---

    def test_unrelated_command_returns_none(self):
        args = {"command": "ls -la"}
        result_data = {"output": "total 0", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is None


# ===================================================================
# write_file verifier
# ===================================================================

class TestWriteFileVerifier:
    def test_file_exists_and_nonempty(self, tmp_path):
        f = tmp_path / "hello.py"
        f.write_text("print('hello')")
        args = {"path": str(f), "content": "print('hello')"}
        result_data = {"bytes_written": 14}
        vr = _verify_write_file(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.details["size_bytes"] == 14

    def test_file_missing(self, tmp_path):
        args = {"path": str(tmp_path / "missing.py"), "content": "x = 1"}
        result_data = {"bytes_written": 10}
        vr = _verify_write_file(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH
        assert "does not exist" in vr.message

    def test_file_exists_but_empty_intentional(self, tmp_path):
        """Empty file with empty content arg → VERIFIED (intentional)."""
        f = tmp_path / "empty.txt"
        f.write_text("")
        args = {"path": str(f), "content": ""}
        result_data = {"bytes_written": 0}
        vr = _verify_write_file(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED

    def test_file_exists_but_empty_no_content_key(self, tmp_path):
        """Empty file with no content key in args → VERIFIED (no expectation)."""
        f = tmp_path / "empty2.txt"
        f.write_text("")
        args = {"path": str(f)}
        result_data = {"bytes_written": 0}
        vr = _verify_write_file(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED

    def test_file_exists_but_empty_unexpected(self, tmp_path):
        """Empty file but content arg was non-empty → WARNING."""
        f = tmp_path / "empty3.txt"
        f.write_text("")
        args = {"path": str(f), "content": "some data that should be here"}
        result_data = {"bytes_written": 0}
        vr = _verify_write_file(args, result_data)
        assert vr is not None
        assert vr.status == WARNING
        assert "empty" in vr.message.lower()
        assert "non-empty content" in vr.message.lower()

    def test_skipped_on_error_result(self):
        args = {"path": "/some/file.py"}
        result_data = {"error": "Permission denied"}
        vr = _verify_write_file(args, result_data)
        assert vr is None

    def test_skipped_on_no_path(self):
        args = {}
        result_data = {"bytes_written": 10}
        vr = _verify_write_file(args, result_data)
        assert vr is None


# ===================================================================
# patch verifier
# ===================================================================

class TestPatchVerifier:
    def test_patched_file_exists(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("x = 1")
        args = {"path": str(f), "old_string": "x = 1", "new_string": "x = 2"}
        result_data = {"success": True, "diff": "...", "files_modified": [str(f)]}
        vr = _verify_patch(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED

    def test_patched_file_missing(self, tmp_path):
        missing = str(tmp_path / "gone.py")
        args = {"path": missing}
        result_data = {"success": True, "diff": "...", "files_modified": [missing]}
        vr = _verify_patch(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH
        assert "missing" in vr.message.lower()

    def test_replace_mode_fallback_to_path_arg(self, tmp_path):
        f = tmp_path / "app.js"
        f.write_text("const x = 1;")
        args = {"path": str(f), "old_string": "const x = 1;", "new_string": "const x = 2;"}
        result_data = {"success": True, "diff": "..."}
        vr = _verify_patch(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED

    def test_skipped_on_failure(self):
        args = {"path": "/tmp/foo.py"}
        result_data = {"success": False, "error": "Could not find old_string"}
        vr = _verify_patch(args, result_data)
        assert vr is None


# ===================================================================
# read_file verifier
# ===================================================================

class TestReadFileVerifier:
    def test_success_with_content(self):
        args = {"path": "/some/file.py"}
        result_data = {"content": "1|hello\n2|world", "total_lines": 2, "file_size": 11, "error": None}
        vr = _verify_read_file(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.check == "file_read"
        assert vr.details["file_size"] == 11

    def test_error_with_similar_files(self):
        args = {"path": "/tmp/foo.py"}
        result_data = {"error": "File not found: /tmp/foo.py", "similar_files": ["/tmp/foo2.py", "/tmp/foobar.py"]}
        vr = _verify_read_file(args, result_data)
        assert vr is not None
        assert vr.status == WARNING
        assert "similar files available" in vr.message
        assert vr.details["similar_files"] == ["/tmp/foo2.py", "/tmp/foobar.py"]

    def test_error_without_similar_files(self):
        args = {"path": "/tmp/foo.py"}
        result_data = {"error": "File not found: /tmp/foo.py"}
        vr = _verify_read_file(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH
        assert "no alternatives" in vr.message

    def test_no_content_no_error(self):
        """Empty file read — no opinion."""
        args = {"path": "/some/empty.txt"}
        result_data = {"content": "", "total_lines": 0, "file_size": 0, "error": None}
        vr = _verify_read_file(args, result_data)
        assert vr is None


# ===================================================================
# browser_navigate verifier
# ===================================================================

class TestBrowserNavigateVerifier:
    def test_navigation_success(self):
        args = {"url": "https://example.com"}
        result_data = {"success": True, "url": "https://example.com", "title": "Example Domain"}
        vr = _verify_browser_navigate(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.check == "navigation_success"

    def test_navigation_failed(self):
        args = {"url": "https://example.com"}
        result_data = {"success": False, "error": "Connection refused"}
        vr = _verify_browser_navigate(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH
        assert "Connection refused" in vr.message

    def test_navigation_bot_detection(self):
        args = {"url": "https://example.com"}
        result_data = {
            "success": True,
            "url": "https://example.com",
            "title": "Verify you are human",
            "bot_detection_warning": "Page title suggests bot detection",
        }
        vr = _verify_browser_navigate(args, result_data)
        assert vr is not None
        assert vr.status == WARNING
        assert "bot detection" in vr.message.lower()

    def test_no_success_field(self):
        args = {"url": "https://example.com"}
        result_data = {"url": "https://example.com"}
        vr = _verify_browser_navigate(args, result_data)
        assert vr is None


# ===================================================================
# web_extract verifier
# ===================================================================

class TestWebExtractVerifier:
    def test_all_urls_succeeded(self):
        args = {"urls": ["https://a.com", "https://b.com"]}
        result_data = {"results": [
            {"url": "https://a.com", "title": "A", "content": "Page A content"},
            {"url": "https://b.com", "title": "B", "content": "Page B content"},
        ]}
        vr = _verify_web_extract(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.details["succeeded"] == 2

    def test_all_urls_failed(self):
        args = {"urls": ["https://a.com"]}
        result_data = {"results": [
            {"url": "https://a.com", "error": "timeout", "content": ""},
        ]}
        vr = _verify_web_extract(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH
        assert "failed to extract" in vr.message.lower()

    def test_partial_failure(self):
        args = {"urls": ["https://a.com", "https://b.com"]}
        result_data = {"results": [
            {"url": "https://a.com", "title": "A", "content": "Page A content"},
            {"url": "https://b.com", "error": "timeout", "content": ""},
        ]}
        vr = _verify_web_extract(args, result_data)
        assert vr is not None
        assert vr.status == WARNING
        assert "1 of 2" in vr.message

    def test_empty_results(self):
        args = {"urls": ["https://a.com"]}
        result_data = {"results": []}
        vr = _verify_web_extract(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH
        assert "no results" in vr.message.lower()

    def test_top_level_error(self):
        args = {"urls": ["https://a.com"]}
        result_data = {"error": "Content was inaccessible"}
        vr = _verify_web_extract(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH


# ===================================================================
# Integration: verify_tool_result
# ===================================================================

class TestVerifyToolResult:
    def test_augments_terminal_result(self, tmp_path):
        target = tmp_path / "repo"
        target.mkdir()
        args = {"command": f"git clone https://github.com/x/repo.git {target}"}
        original = json.dumps({"output": "done", "exit_code": 0, "error": None})
        result = verify_tool_result("terminal", args, original)
        parsed = json.loads(result)
        assert "_verification" in parsed
        assert parsed["_verification"]["status"] == "verified"
        assert "_warning" not in parsed  # no warning on verified

    def test_augments_write_file_result(self, tmp_path):
        f = tmp_path / "out.txt"
        f.write_text("data")
        args = {"path": str(f), "content": "data"}
        original = json.dumps({"bytes_written": 4})
        result = verify_tool_result("write_file", args, original)
        parsed = json.loads(result)
        assert "_verification" in parsed
        assert parsed["_verification"]["status"] == "verified"
        assert "_warning" not in parsed  # no warning on verified

    def test_augments_read_file_result(self):
        args = {"path": "/some/file.py"}
        original = json.dumps({"content": "1|hello", "total_lines": 1, "file_size": 5, "error": None})
        result = verify_tool_result("read_file", args, original)
        parsed = json.loads(result)
        assert "_verification" in parsed
        assert parsed["_verification"]["status"] == "verified"

    def test_augments_browser_navigate_result(self):
        args = {"url": "https://example.com"}
        original = json.dumps({"success": True, "url": "https://example.com", "title": "Example"})
        result = verify_tool_result("browser_navigate", args, original)
        parsed = json.loads(result)
        assert "_verification" in parsed
        assert parsed["_verification"]["status"] == "verified"

    def test_augments_web_extract_result(self):
        args = {"urls": ["https://example.com"]}
        original = json.dumps({"results": [{"url": "https://example.com", "content": "data", "title": "Ex"}]})
        result = verify_tool_result("web_extract", args, original)
        parsed = json.loads(result)
        assert "_verification" in parsed
        assert parsed["_verification"]["status"] == "verified"

    def test_no_verifier_returns_unchanged(self):
        original = json.dumps({"results": [1, 2, 3]})
        result = verify_tool_result("web_search", {}, original)
        assert result == original

    def test_non_json_returns_unchanged(self):
        raw = "not valid json at all"
        result = verify_tool_result("terminal", {"command": "ls"}, raw)
        assert result == raw

    def test_unrelated_terminal_command_returns_unchanged(self):
        original = json.dumps({"output": "hello", "exit_code": 0, "error": None})
        result = verify_tool_result("terminal", {"command": "echo hello"}, original)
        parsed = json.loads(result)
        assert "_verification" not in parsed

    # --- mismatch cases: VERIFICATION FAILED ---

    def test_mismatch_terminal_has_failed_warning(self, tmp_path):
        missing = str(tmp_path / "nope")
        args = {"command": f"git clone https://github.com/x/repo.git {missing}"}
        original = json.dumps({"output": "done", "exit_code": 0, "error": None})
        result = verify_tool_result("terminal", args, original)
        parsed = json.loads(result)
        assert parsed["_verification"]["status"] == "mismatch"
        assert "_warning" in parsed
        assert "VERIFICATION FAILED" in parsed["_warning"]
        assert "conflicts with environment state" in parsed["_warning"]

    def test_mismatch_write_file_has_failed_warning(self, tmp_path):
        args = {"path": str(tmp_path / "gone.py"), "content": "x = 1"}
        original = json.dumps({"bytes_written": 10})
        result = verify_tool_result("write_file", args, original)
        parsed = json.loads(result)
        assert parsed["_verification"]["status"] == "mismatch"
        assert "_warning" in parsed
        assert "VERIFICATION FAILED" in parsed["_warning"]
        assert "conflicts with environment state" in parsed["_warning"]

    def test_mismatch_patch_has_failed_warning(self, tmp_path):
        missing = str(tmp_path / "deleted.py")
        args = {"path": missing}
        original = json.dumps({"success": True, "diff": "...", "files_modified": [missing]})
        result = verify_tool_result("patch", args, original)
        parsed = json.loads(result)
        assert parsed["_verification"]["status"] == "mismatch"
        assert "_warning" in parsed
        assert "VERIFICATION FAILED" in parsed["_warning"]

    def test_mismatch_browser_navigate_has_failed_warning(self):
        args = {"url": "https://example.com"}
        original = json.dumps({"success": False, "error": "Connection refused"})
        result = verify_tool_result("browser_navigate", args, original)
        parsed = json.loads(result)
        assert parsed["_verification"]["status"] == "mismatch"
        assert "_warning" in parsed
        assert "VERIFICATION FAILED" in parsed["_warning"]

    # --- warning cases: VERIFICATION WARNING ---

    def test_warning_empty_file_has_warning_text(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        args = {"path": str(f), "content": "data that should be here"}
        original = json.dumps({"bytes_written": 0})
        result = verify_tool_result("write_file", args, original)
        parsed = json.loads(result)
        assert parsed["_verification"]["status"] == "warning"
        assert "_warning" in parsed
        assert "VERIFICATION WARNING" in parsed["_warning"]
        assert "Result may be incomplete" in parsed["_warning"]
        # Must NOT contain the mismatch text
        assert "VERIFICATION FAILED" not in parsed["_warning"]

    def test_warning_web_extract_partial_has_warning_text(self):
        args = {"urls": ["https://a.com", "https://b.com"]}
        original = json.dumps({"results": [
            {"url": "https://a.com", "content": "data", "title": "A"},
            {"url": "https://b.com", "error": "timeout", "content": ""},
        ]})
        result = verify_tool_result("web_extract", args, original)
        parsed = json.loads(result)
        assert parsed["_verification"]["status"] == "warning"
        assert "_warning" in parsed
        assert "VERIFICATION WARNING" in parsed["_warning"]
        assert "VERIFICATION FAILED" not in parsed["_warning"]

    def test_warning_read_file_similar_files_has_warning_text(self):
        args = {"path": "/tmp/foo.py"}
        original = json.dumps({"error": "File not found", "similar_files": ["/tmp/foo2.py"]})
        result = verify_tool_result("read_file", args, original)
        parsed = json.loads(result)
        assert parsed["_verification"]["status"] == "warning"
        assert "_warning" in parsed
        assert "VERIFICATION WARNING" in parsed["_warning"]
