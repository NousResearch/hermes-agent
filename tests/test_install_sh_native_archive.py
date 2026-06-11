"""Regression tests for install.sh native repository archive bootstrap behavior."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _run_stage_body() -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    start = text.index("run_stage_body() {")
    end = text.index("\nrun_stage_protocol()", start)
    return text[start:end]


def test_prerequisites_can_skip_git_for_native_repository_archive() -> None:
    body = _run_stage_body()

    assert "HERMES_NATIVE_REPOSITORY_ARCHIVE" in body
    assert "Skipping Git check; native repository archive will fetch source" in body


def test_repository_stage_can_be_native_archive_noop() -> None:
    body = _run_stage_body()

    assert "Repository stage handled by native bootstrap archive" in body
    assert body.index("Repository stage handled by native bootstrap archive") < body.index("clone_repo")
