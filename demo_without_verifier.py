#!/usr/bin/env python3
"""
Demo: Raw tool output WITHOUT Execution Integrity Layer.

Shows the same 17 scenarios as demo_execution_verifier.py but returns
raw tool output with NO _verification or _warning fields — this is
what the model sees when the verifier is disabled.

Compare side-by-side:
    python3 demo_without_verifier.py   # raw output (before)
    python3 demo_execution_verifier.py # augmented output (after)
"""

import json
import os
import tempfile
from pathlib import Path


def _pp(label: str, result_json: str):
    """Pretty-print a raw tool result."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        data = json.loads(result_json)
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError:
        print(result_json)


def main():
    print("Raw Tool Output — No Verification (Before)")
    print("Same 17 scenarios, but the model gets ONLY the tool's own JSON.\n")

    with tempfile.TemporaryDirectory(prefix="hermes-demo-") as tmpdir:

        # ── 1. git clone — SUCCESS ─────────────────────────────────────
        clone_dir = os.path.join(tmpdir, "my-project")
        os.makedirs(clone_dir)  # simulate successful clone

        result = json.dumps({"output": "Cloning into 'my-project'...\ndone.", "exit_code": 0, "error": None})
        _pp("1. git clone — directory EXISTS (no verification)", result)

        # ── 2. git clone — FAILURE ─────────────────────────────────────
        result = json.dumps({"output": "Cloning into 'ghost-repo'...\ndone.", "exit_code": 0, "error": None})
        _pp("2. git clone — directory MISSING (no verification)", result)

        # ── 3. write_file — SUCCESS ────────────────────────────────────
        written_file = os.path.join(tmpdir, "hello.py")
        Path(written_file).write_text("print('hello world')\n")

        result = json.dumps({"bytes_written": 21})
        _pp("3. write_file — file EXISTS and non-empty (no verification)", result)

        # ── 4. write_file — EMPTY FILE (intentional) ──────────────────
        empty_file = os.path.join(tmpdir, "empty.txt")
        Path(empty_file).write_text("")

        result = json.dumps({"bytes_written": 0})
        _pp("4. write_file — file EXISTS, intentionally empty (no verification)", result)

        # ── 5. write_file — MISSING FILE ───────────────────────────────
        result = json.dumps({"bytes_written": 3})
        _pp("5. write_file — file MISSING after write (no verification)", result)

        # ── 6. patch — SUCCESS ─────────────────────────────────────────
        patched_file = os.path.join(tmpdir, "app.py")
        Path(patched_file).write_text("x = 2\n")

        result = json.dumps({"success": True, "diff": "- x = 1\n+ x = 2", "files_modified": [patched_file]})
        _pp("6. patch — modified file EXISTS (no verification)", result)

        # ── 7. patch — MISSING AFTER PATCH ──────────────────────────────
        deleted_path = os.path.join(tmpdir, "deleted.py")
        result = json.dumps({"success": True, "diff": "...", "files_modified": [deleted_path]})
        _pp("7. patch — modified file MISSING (no verification)", result)

        # ── 8. Unrelated tool — passthrough ─────────────────────────────
        result = json.dumps({"results": [{"title": "asyncio docs", "url": "https://..."}]})
        _pp("8. web_search — passthrough (no verification)", result)

        # ── 9. terminal cp — destination EXISTS ─────────────────────────
        result = json.dumps({"output": "", "exit_code": 0, "error": None})
        _pp("9. terminal cp — destination EXISTS (no verification)", result)

        # ── 10. terminal rm — target REMOVED ────────────────────────────
        result = json.dumps({"output": "", "exit_code": 0, "error": None})
        _pp("10. terminal rm — target REMOVED (no verification)", result)

        # ── 11. terminal touch — file EXISTS ────────────────────────────
        result = json.dumps({"output": "", "exit_code": 0, "error": None})
        _pp("11. terminal touch — file EXISTS (no verification)", result)

        # ── 12. terminal git init — .git EXISTS ─────────────────────────
        result = json.dumps({"output": "Initialized empty Git repository", "exit_code": 0, "error": None})
        _pp("12. terminal git init — .git EXISTS (no verification)", result)

        # ── 13. read_file — content returned ────────────────────────────
        result = json.dumps({"content": "1|print('hello')\n2|print('world')", "total_lines": 2, "file_size": 28, "error": None})
        _pp("13. read_file — content returned (no verification)", result)

        # ── 14. read_file — error with similar_files ────────────────────
        result = json.dumps({"error": "File not found: /tmp/foo.py", "similar_files": ["/tmp/foo2.py", "/tmp/foobar.py"]})
        _pp("14. read_file — error with similar_files (no verification)", result)

        # ── 15. browser_navigate — success ──────────────────────────────
        result = json.dumps({"success": True, "url": "https://example.com", "title": "Example Domain"})
        _pp("15. browser_navigate — success (no verification)", result)

        # ── 16. browser_navigate — bot detection ────────────────────────
        result = json.dumps({
            "success": True,
            "url": "https://protected-site.com",
            "title": "Verify you are human",
            "bot_detection_warning": "Page title suggests bot detection",
        })
        _pp("16. browser_navigate — bot detection triggered (no verification)", result)

        # ── 17. web_extract — partial failure ───────────────────────────
        result = json.dumps({"results": [
            {"url": "https://a.com", "title": "A", "content": "Page A content extracted successfully."},
            {"url": "https://b.com", "error": "Connection timeout", "content": ""},
        ]})
        _pp("17. web_extract — partial failure (no verification)", result)

    print(f"\n{'='*60}")
    print("  Demo complete. Notice: NO _verification or _warning fields.")
    print("  The model has no signal that scenarios 2, 5, 7, 14, 16, 17")
    print("  had issues requiring attention.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
