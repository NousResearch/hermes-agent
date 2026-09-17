"""Integration coverage for the two-phase updater startup preflight."""
from __future__ import annotations

from pathlib import Path

from hermes_cli.update_preflight import run_gateway_startup_preflight


def _minimal_candidate(root: Path, *, file_signature: bool) -> None:
    (root / "utils.py").write_text(
        ("def file_signature(value): return (1,2,3,4)\n" if file_signature else "VALUE=1\n"),
        encoding="utf-8",
    )
    for package in ("hermes_cli", "gateway", "plugins", "plugins/platforms", "plugins/platforms/slack"):
        path = root / package
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").write_text("", encoding="utf-8")
    (root / "hermes_cli/main.py").write_text("", encoding="utf-8")
    (root / "gateway/run.py").write_text("", encoding="utf-8")
    (root / "gateway/status.py").write_text("", encoding="utf-8")
    (root / "gateway/config.py").write_text("def load_gateway_config(): return object()\n", encoding="utf-8")
    (root / "plugins/platforms/slack/adapter.py").write_text("", encoding="utf-8")


def test_preflight_passes_candidate_with_required_import_contract(tmp_path):
    _minimal_candidate(tmp_path, file_signature=True)
    result = run_gateway_startup_preflight(tmp_path)
    assert result["ok"] is True and result["returncode"] == 0


def test_preflight_blocks_candidate_missing_new_cross_module_symbol(tmp_path):
    _minimal_candidate(tmp_path, file_signature=False)
    result = run_gateway_startup_preflight(tmp_path)
    assert result["ok"] is False and result["returncode"] != 0
    assert any(row[0] == "gateway-startup-preflight" and row[1] == "ImportError" for row in result["errors"])
