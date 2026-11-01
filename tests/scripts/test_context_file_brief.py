import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "context_file_brief.py"


def load_module():
    spec = importlib.util.spec_from_file_location("context_file_brief", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_discovers_known_context_files_in_root_and_hermes_dir(tmp_path):
    module = load_module()
    (tmp_path / "AGENTS.md").write_text("repo rules\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("claude rules\n", encoding="utf-8")
    (tmp_path / ".hermes").mkdir()
    (tmp_path / ".hermes" / "SOUL.md").write_text("persona\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("not a context file\n", encoding="utf-8")

    found = [path.relative_to(tmp_path).as_posix() for path in module.discover_context_files(tmp_path)]

    assert found == ["AGENTS.md", "CLAUDE.md", ".hermes/SOUL.md"]


def test_analyzes_oversized_context_file_with_actionable_suggestion(tmp_path):
    module = load_module()
    path = tmp_path / "AGENTS.md"
    path.write_text("x" * 125, encoding="utf-8")

    result = module.analyze_file(path, max_chars=100, warn_ratio=0.8)

    assert result.status == "over"
    assert result.chars == 125
    assert result.lines == 1
    assert "split stable procedures into skills" in result.suggestion


def test_render_brief_silent_ok_only_when_no_attention_needed(tmp_path):
    module = load_module()
    ok_path = tmp_path / "AGENTS.md"
    ok_path.write_text("small\n", encoding="utf-8")
    warn_path = tmp_path / "CLAUDE.md"
    warn_path.write_text("x" * 85, encoding="utf-8")

    ok = module.analyze_file(ok_path, max_chars=100, warn_ratio=0.8)
    warn = module.analyze_file(warn_path, max_chars=100, warn_ratio=0.8)

    assert module.render_brief([ok], silent_ok=True) == "[SILENT]"
    brief = module.render_brief([ok, warn], silent_ok=True)
    assert "CLAUDE.md" in brief
    assert "warn" in brief
    assert brief != "[SILENT]"


def test_cli_json_output_reports_context_file_status(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 101, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--root",
            str(tmp_path),
            "--max-chars",
            "100",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(result.stdout)
    assert data["summary"] == {"ok": 0, "warn": 0, "over": 1}
    assert data["files"][0]["path"] == "AGENTS.md"
    assert data["files"][0]["status"] == "over"
