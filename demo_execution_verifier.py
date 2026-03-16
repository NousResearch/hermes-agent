#!/usr/bin/env python3
"""
Demo: Execution Integrity Layer

Shows how verify_tool_result() augments tool outputs with post-call
verification metadata.  Run this standalone — no LLM or server needed.

Usage:
    python3 demo_execution_verifier.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.execution_verifier import verify_tool_result


def _pp(label: str, result_json: str):
    """Pretty-print a verification result."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        data = json.loads(result_json)
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError:
        print(result_json)


def main():
    print("Execution Integrity Layer — Demo")
    print("Demonstrates post-tool-call verification for terminal, write_file,")
    print("patch, read_file, browser_navigate, and web_extract.\n")

    with tempfile.TemporaryDirectory(prefix="hermes-demo-") as tmpdir:

        # ── 1. git clone — SUCCESS ─────────────────────────────────────
        clone_dir = os.path.join(tmpdir, "my-project")
        os.makedirs(clone_dir)  # simulate successful clone

        result = verify_tool_result(
            "terminal",
            {"command": f"git clone https://github.com/user/my-project.git {clone_dir}"},
            json.dumps({"output": "Cloning into 'my-project'...\ndone.", "exit_code": 0, "error": None}),
        )
        _pp("1. git clone — directory EXISTS (pass)", result)

        # ── 2. git clone — FAILURE ─────────────────────────────────────
        missing_dir = os.path.join(tmpdir, "ghost-repo")
        result = verify_tool_result(
            "terminal",
            {"command": f"git clone https://github.com/user/ghost-repo.git {missing_dir}"},
            json.dumps({"output": "Cloning into 'ghost-repo'...\ndone.", "exit_code": 0, "error": None}),
        )
        _pp("2. git clone — directory MISSING (fail)", result)

        # ── 3. write_file — SUCCESS ────────────────────────────────────
        written_file = os.path.join(tmpdir, "hello.py")
        Path(written_file).write_text("print('hello world')\n")

        result = verify_tool_result(
            "write_file",
            {"path": written_file, "content": "print('hello world')\n"},
            json.dumps({"bytes_written": 21}),
        )
        _pp("3. write_file — file EXISTS and non-empty (pass)", result)

        # ── 4. write_file — EMPTY FILE (intentional) ──────────────────
        empty_file = os.path.join(tmpdir, "empty.txt")
        Path(empty_file).write_text("")

        result = verify_tool_result(
            "write_file",
            {"path": empty_file, "content": ""},
            json.dumps({"bytes_written": 0}),
        )
        _pp("4. write_file — file EXISTS, intentionally empty (pass)", result)

        # ── 5. write_file — MISSING FILE ───────────────────────────────
        result = verify_tool_result(
            "write_file",
            {"path": os.path.join(tmpdir, "vanished.py"), "content": "x=1"},
            json.dumps({"bytes_written": 3}),
        )
        _pp("5. write_file — file MISSING after write (fail)", result)

        # ── 6. patch — SUCCESS ─────────────────────────────────────────
        patched_file = os.path.join(tmpdir, "app.py")
        Path(patched_file).write_text("x = 2\n")

        result = verify_tool_result(
            "patch",
            {"path": patched_file, "old_string": "x = 1", "new_string": "x = 2"},
            json.dumps({"success": True, "diff": "- x = 1\n+ x = 2", "files_modified": [patched_file]}),
        )
        _pp("6. patch — modified file EXISTS (pass)", result)

        # ── 7. patch — MISSING AFTER PATCH ──────────────────────────────
        result = verify_tool_result(
            "patch",
            {"path": os.path.join(tmpdir, "deleted.py")},
            json.dumps({"success": True, "diff": "...", "files_modified": [os.path.join(tmpdir, "deleted.py")]}),
        )
        _pp("7. patch — modified file MISSING (fail)", result)

        # ── 8. Unrelated tool — passthrough ─────────────────────────────
        result = verify_tool_result(
            "web_search",
            {"query": "python asyncio"},
            json.dumps({"results": [{"title": "asyncio docs", "url": "https://..."}]}),
        )
        _pp("8. web_search — no verifier, unchanged passthrough", result)

        # ── 9. terminal cp — destination EXISTS ─────────────────────────
        cp_src = os.path.join(tmpdir, "src.txt")
        cp_dest = os.path.join(tmpdir, "dest.txt")
        Path(cp_src).write_text("data")
        Path(cp_dest).write_text("data")

        result = verify_tool_result(
            "terminal",
            {"command": f"cp {cp_src} {cp_dest}"},
            json.dumps({"output": "", "exit_code": 0, "error": None}),
        )
        _pp("9. terminal cp — destination EXISTS (pass)", result)

        # ── 10. terminal rm — target REMOVED ────────────────────────────
        rm_target = os.path.join(tmpdir, "to_delete.txt")
        # target does NOT exist — rm succeeded
        result = verify_tool_result(
            "terminal",
            {"command": f"rm {rm_target}"},
            json.dumps({"output": "", "exit_code": 0, "error": None}),
        )
        _pp("10. terminal rm — target REMOVED (pass)", result)

        # ── 11. terminal touch — file EXISTS ────────────────────────────
        touch_file = os.path.join(tmpdir, "touched.txt")
        Path(touch_file).write_text("")

        result = verify_tool_result(
            "terminal",
            {"command": f"touch {touch_file}"},
            json.dumps({"output": "", "exit_code": 0, "error": None}),
        )
        _pp("11. terminal touch — file EXISTS (pass)", result)

        # ── 12. terminal git init — .git EXISTS ─────────────────────────
        init_dir = os.path.join(tmpdir, "new-repo")
        os.makedirs(os.path.join(init_dir, ".git"))

        result = verify_tool_result(
            "terminal",
            {"command": f"git init {init_dir}"},
            json.dumps({"output": "Initialized empty Git repository", "exit_code": 0, "error": None}),
        )
        _pp("12. terminal git init — .git EXISTS (pass)", result)

        # ── 13. read_file — content returned ────────────────────────────
        result = verify_tool_result(
            "read_file",
            {"path": "/some/file.py"},
            json.dumps({"content": "1|print('hello')\n2|print('world')", "total_lines": 2, "file_size": 28, "error": None}),
        )
        _pp("13. read_file — content returned (pass)", result)

        # ── 14. read_file — error with similar_files ────────────────────
        result = verify_tool_result(
            "read_file",
            {"path": "/tmp/foo.py"},
            json.dumps({"error": "File not found: /tmp/foo.py", "similar_files": ["/tmp/foo2.py", "/tmp/foobar.py"]}),
        )
        _pp("14. read_file — error with similar_files (warning)", result)

        # ── 15. browser_navigate — success ──────────────────────────────
        result = verify_tool_result(
            "browser_navigate",
            {"url": "https://example.com"},
            json.dumps({"success": True, "url": "https://example.com", "title": "Example Domain"}),
        )
        _pp("15. browser_navigate — success (pass)", result)

        # ── 16. browser_navigate — bot detection ────────────────────────
        result = verify_tool_result(
            "browser_navigate",
            {"url": "https://protected-site.com"},
            json.dumps({
                "success": True,
                "url": "https://protected-site.com",
                "title": "Verify you are human",
                "bot_detection_warning": "Page title suggests bot detection",
            }),
        )
        _pp("16. browser_navigate — bot detection triggered (warning)", result)

        # ── 17. web_extract — partial failure ───────────────────────────
        result = verify_tool_result(
            "web_extract",
            {"urls": ["https://a.com", "https://b.com"]},
            json.dumps({"results": [
                {"url": "https://a.com", "title": "A", "content": "Page A content extracted successfully."},
                {"url": "https://b.com", "error": "Connection timeout", "content": ""},
            ]}),
        )
        _pp("17. web_extract — partial failure (warning)", result)

    print(f"\n{'='*60}")
    print("  Demo complete. Verification checks executed across pass,")
    print("  warning, and mismatch scenarios for 6 tool types.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
