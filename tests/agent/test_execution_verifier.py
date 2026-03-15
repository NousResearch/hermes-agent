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
            message="file exists but is empty; result may be incomplete or intentional: /tmp/empty.txt",
        )
        d = vr.to_dict()
        assert d["status"] == "warning"
        assert "empty" in d["message"]


# ===================================================================
# Terminal verifier
# ===================================================================

class TestTerminalVerifier:
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

    def test_unrelated_command_returns_none(self):
        args = {"command": "ls -la"}
        result_data = {"output": "total 0", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is None

    def test_git_clone_with_flags(self, tmp_path):
        target = tmp_path / "repo"
        target.mkdir()
        args = {"command": f"git clone --depth 1 --branch main https://github.com/user/repo.git {target}"}
        result_data = {"output": "Cloning...", "exit_code": 0, "error": None}
        vr = _verify_terminal(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED


# ===================================================================
# write_file verifier
# ===================================================================

class TestWriteFileVerifier:
    def test_file_exists_and_nonempty(self, tmp_path):
        f = tmp_path / "hello.py"
        f.write_text("print('hello')")
        args = {"path": str(f)}
        result_data = {"bytes_written": 14}
        vr = _verify_write_file(args, result_data)
        assert vr is not None
        assert vr.status == VERIFIED
        assert vr.details["size_bytes"] == 14

    def test_file_missing(self, tmp_path):
        args = {"path": str(tmp_path / "missing.py")}
        result_data = {"bytes_written": 10}
        vr = _verify_write_file(args, result_data)
        assert vr is not None
        assert vr.status == MISMATCH
        assert "does not exist" in vr.message

    def test_file_exists_but_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        args = {"path": str(f)}
        result_data = {"bytes_written": 0}
        vr = _verify_write_file(args, result_data)
        assert vr is not None
        assert vr.status == WARNING
        assert "empty" in vr.message.lower()

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
        args = {"path": str(f)}
        original = json.dumps({"bytes_written": 4})
        result = verify_tool_result("write_file", args, original)
        parsed = json.loads(result)
        assert "_verification" in parsed
        assert parsed["_verification"]["status"] == "verified"
        assert "_warning" not in parsed  # no warning on verified

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

    # --- mismatch cases: ❌ VERIFICATION FAILED ---

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
        args = {"path": str(tmp_path / "gone.py")}
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

    # --- warning cases: ⚠️ VERIFICATION WARNING ---

    def test_warning_empty_file_has_warning_text(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        args = {"path": str(f)}
        original = json.dumps({"bytes_written": 0})
        result = verify_tool_result("write_file", args, original)
        parsed = json.loads(result)
        assert parsed["_verification"]["status"] == "warning"
        assert "_warning" in parsed
        assert "VERIFICATION WARNING" in parsed["_warning"]
        assert "Result may be incomplete" in parsed["_warning"]
        # Must NOT contain the mismatch text
        assert "VERIFICATION FAILED" not in parsed["_warning"]
