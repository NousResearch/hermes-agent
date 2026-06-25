from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(
    "/Users/rattanasak/ObsidianVault/HermesAgent/99-System/automation/scan_project_ai_adapters.py"
)


def load_scanner():
    spec = importlib.util.spec_from_file_location("scan_project_ai_adapters", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_use_pair_ai_alias_resolves_to_use_ai_pair():
    scanner = load_scanner()

    assert scanner.canonical_shortcut("Use Pair AI") == "Use AI Pair"
    assert scanner.canonical_shortcut("use pair ai") == "Use AI Pair"
    assert scanner.canonical_shortcut("Use AI Pair") == "Use AI Pair"


def test_scan_project_reports_missing_adapter_and_runtime_levels(tmp_path):
    scanner = load_scanner()
    project = tmp_path / "Sample Project"
    project.mkdir()
    (project / "AGENTS.md").write_text(
        "Read /Users/rattanasak/ObsidianVault/HermesAgent/ai-context/prompt-shortcut-registry.md\n"
        "Shortcut: Use AI Pair or Use Pair AI\n",
        encoding="utf-8",
    )

    result = scanner.scan_project(project)

    assert result["project"] == "Sample Project"
    assert result["prompt_shortcut_ready"] is False
    assert result["handoff_ready"] is True
    assert result["auto_runtime_ready"] is False
    assert "CLAUDE.md" in result["missing_top_level_adapters"]
    assert "Use Pair AI" not in result["missing_direct_shortcuts"]

