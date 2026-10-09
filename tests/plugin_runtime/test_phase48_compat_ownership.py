"""Phase 4.8 ratchet after the scheduled Sep 2026 plugin-compat removal."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_STUB = ROOT / "hermes_cli" / "plugin_compat.py"


def _top_level_defs(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def test_phase48_scheduled_compat_layer_remains_removed() -> None:
    assert not (ROOT / "COMPAT_MANIFEST.md").exists()
    assert not (ROOT / "compat_manifest.json").exists()
    assert not (ROOT / "plugin_runtime" / "compat.py").exists()
    assert not (ROOT / "plugin_runtime" / "compat_warning.py").exists()


def test_phase48_old_updater_stub_is_frozen_and_inert() -> None:
    import hermes_cli.plugin_compat as compat

    assert _top_level_defs(LEGACY_STUB) == [
        "compat_report",
        "removal_in_effect",
        "summary_lines",
    ]
    assert compat.compat_report(force=True) == {}
    assert compat.removal_in_effect() is True
    assert compat.summary_lines({"ignored": []}) == []


def test_phase48_first_party_has_no_removed_compat_runtime_imports() -> None:
    offenders = []
    for root_name in (
        "acp_adapter", "agent", "cron", "gateway", "hermes_cli",
        "plugin_runtime", "plugins", "providers", "tools", "tui_gateway",
    ):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path == LEGACY_STUB:
                continue
            source = path.read_text(encoding="utf-8")
            if "plugin_runtime.compat" in source:
                offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_phase48_contributor_docs_record_scheduled_removal() -> None:
    docs = (ROOT / "plugins" / "AGENTS.md").read_text(encoding="utf-8")
    assert "removed after its 2026-09-14 window" in docs
    assert "survives only as three inert stubs" in docs
