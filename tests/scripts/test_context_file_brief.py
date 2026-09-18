from __future__ import annotations

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


def test_discovers_root_and_hermes_context_files(tmp_path: Path) -> None:
    module = load_module()
    (tmp_path / "AGENTS.md").write_text("root\n", encoding="utf-8")
    (tmp_path / ".hermes").mkdir()
    (tmp_path / ".hermes" / "SOUL.md").write_text("soul\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("ignored\n", encoding="utf-8")

    found = module.discover_context_files(tmp_path)

    assert [item.relative_path for item in found] == ["AGENTS.md", ".hermes/SOUL.md"]


def test_analyzes_oversized_file_with_actionable_suggestion(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "AGENTS.md"
    path.write_text("x" * 45, encoding="utf-8")

    report = module.analyze_file(path, tmp_path, max_chars=30, warn_ratio=0.5)

    assert report.status == "over"
    assert report.char_count == 45
    assert report.line_count == 1
    assert "split durable procedures" in report.suggestion


def test_silent_ok_is_exact_only_when_all_files_ok(tmp_path: Path, capsys) -> None:
    module = load_module()
    (tmp_path / "AGENTS.md").write_text("short\n", encoding="utf-8")

    code = module.main(["--root", str(tmp_path), "--max-chars", "100", "--silent-ok"])

    assert code == 0
    assert capsys.readouterr().out == "[SILENT]\n"


def test_cli_json_uses_stable_relative_paths(tmp_path: Path) -> None:
    (tmp_path / ".hermes").mkdir()
    (tmp_path / ".hermes" / "CLAUDE.md").write_text("hello\nworld\n", encoding="utf-8")

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
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["root"] == str(tmp_path)
    assert payload["files"][0]["path"] == ".hermes/CLAUDE.md"
    assert payload["files"][0]["status"] == "ok"


def test_silent_ok_does_not_suppress_warning(tmp_path: Path, capsys) -> None:
    module = load_module()
    (tmp_path / "SOUL.md").write_text("x" * 75, encoding="utf-8")

    code = module.main(["--root", str(tmp_path), "--max-chars", "100", "--silent-ok"])

    output = capsys.readouterr().out
    assert code == 1
    assert "SOUL.md" in output
    assert "warn" in output
    assert output != "[SILENT]\n"
