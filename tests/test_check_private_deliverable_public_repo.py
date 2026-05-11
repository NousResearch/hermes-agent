from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_private_deliverable_public_repo.py"
spec = importlib.util.spec_from_file_location("check_private_deliverable_public_repo", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_sensitive_markers_are_detected_in_text() -> None:
    assert module.matches_sensitive("Revenue Lab sample deliverables for RAN-1381") == "Revenue Lab"
    assert module.matches_sensitive("Aligned Insights private strategy") == "Aligned Insights"


def test_benign_text_is_allowed() -> None:
    assert module.matches_sensitive("docs: clarify GitHub pull request workflow") is None


def test_public_visibility_blocks_sensitive_pr_text(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert module.run(["git", "init"], tmp_path).returncode == 0
    assert module.run(["git", "remote", "add", "origin", "https://github.com/NousResearch/hermes-agent.git"], tmp_path).returncode == 0

    status = module.main([
        "--visibility",
        "public",
        "--text",
        "Add Revenue Lab sample deliverables",
        "--path",
        "README.md",
    ])

    assert status == 2
    assert "BLOCKED" in capsys.readouterr().err


def test_private_visibility_allows_sensitive_pr_text(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert module.run(["git", "init"], tmp_path).returncode == 0

    status = module.main([
        "--visibility",
        "private",
        "--text",
        "Add Revenue Lab sample deliverables",
        "--path",
        "README.md",
    ])

    assert status == 0
