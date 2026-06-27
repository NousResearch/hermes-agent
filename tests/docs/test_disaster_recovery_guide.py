"""Regression coverage for the disaster-recovery runbook.

The guide is operational documentation, but it protects important behavior
contracts: which commands are for full sensitive backup vs portable profile
export, which secrets must never go to git, and which restore surfaces must be
verified after a machine migration.
"""

from pathlib import Path


DOC = Path(__file__).resolve().parents[2] / "website" / "docs" / "user-guide" / "disaster-recovery.md"
SIDEBAR = Path(__file__).resolve().parents[2] / "website" / "sidebars.ts"


def _doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def test_disaster_recovery_guide_is_in_sidebar():
    assert "user-guide/disaster-recovery" in SIDEBAR.read_text(encoding="utf-8")


def test_disaster_recovery_guide_covers_backup_commands_and_scope():
    text = _doc_text()
    lower = text.lower()
    for required in [
        "hermes backup",
        "hermes import <zip>",
        "hermes backup --quick",
        "hermes profile export",
        "hermes profile import",
        "full machine migration",
        "portable profile snapshots",
    ]:
        assert required in lower


def test_disaster_recovery_guide_covers_secret_boundaries():
    text = _doc_text()
    for required in [
        "Do not commit secrets",
        ".env",
        "auth.json",
        "OAuth refresh tokens",
        "full `hermes backup` zip files",
        "encrypted backup system",
        "private git repo",
    ]:
        assert required in text


def test_disaster_recovery_guide_covers_restore_verification_surfaces():
    text = _doc_text()
    for required in [
        "hermes doctor",
        "hermes status --all",
        "hermes mcp list",
        "hermes tools list",
        "hermes skills list",
        "hermes cron list",
        "hermes sessions list",
        "hermes gateway status",
        "voice",
        "session search",
    ]:
        assert required in text
