import json
import subprocess
import sys
from pathlib import Path

import scripts.context_file_brief as brief


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "context_file_brief.py"


def test_scan_classifies_context_files_and_skips_vendor_dirs(tmp_path):
    (tmp_path / "AGENTS.md").write_text("a" * 9, encoding="utf-8")
    nested = tmp_path / "project"
    nested.mkdir()
    (nested / "CLAUDE.md").write_text("b" * 10, encoding="utf-8")
    (nested / "SOUL.md").write_text("c" * 15, encoding="utf-8")
    vendor = tmp_path / "node_modules"
    vendor.mkdir()
    (vendor / "AGENTS.md").write_text("d" * 100, encoding="utf-8")

    results = brief.scan_context_files(tmp_path, warn_bytes=10, over_bytes=15)

    by_name = {item.path.name: item for item in results}
    assert by_name["AGENTS.md"].status == "ok"
    assert by_name["CLAUDE.md"].status == "warn"
    assert by_name["SOUL.md"].status == "over"
    assert all("node_modules" not in item.path.parts for item in results)


def test_markdown_silent_when_everything_is_ok(tmp_path):
    (tmp_path / "AGENTS.md").write_text("small", encoding="utf-8")

    results = brief.scan_context_files(tmp_path, warn_bytes=100, over_bytes=200)

    assert brief.render_markdown(results, tmp_path, silent_if_ok=True) == "[SILENT]"


def test_markdown_lists_only_attention_items_by_default(tmp_path):
    ok = brief.ContextFileReport(tmp_path / "AGENTS.md", 5, "ok")
    warn = brief.ContextFileReport(tmp_path / "CLAUDE.md", 11, "warn")
    over = brief.ContextFileReport(tmp_path / "SOUL.md", 20, "over")

    output = brief.render_markdown([ok, warn, over], tmp_path)

    assert "CLAUDE.md" in output
    assert "SOUL.md" in output
    assert "AGENTS.md" not in output
    assert "Next action" in output


def test_json_cli_outputs_machine_readable_summary(tmp_path):
    (tmp_path / "AGENTS.md").write_text("a" * 10, encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("b" * 20, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--warn-bytes",
            "15",
            "--over-bytes",
            "20",
            "--format",
            "json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["root"] == str(tmp_path)
    assert payload["counts"] == {"ok": 1, "warn": 0, "over": 1}
    assert [item["status"] for item in payload["files"]] == ["ok", "over"]
