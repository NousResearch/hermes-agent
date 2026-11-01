"""Operator marker refuses every in-place update path."""
from pathlib import Path

from hermes_cli import update_contract as uc


def test_marker_refuses_with_operator_note(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    marker = tmp_path / uc.EXTERNAL_UPDATE_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text("Update with: ops/update-release.sh <tag>\n", encoding="utf-8")
    refusal = uc.evaluate_update_admission(tmp_path)
    assert refusal is not None
    assert refusal.code == "managed-externally"
    assert "managed outside Hermes" in refusal.message
    assert "ops/update-release.sh" in refusal.message


def test_no_marker_leaves_git_checkout_admitted(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    assert uc._external_update_refusal(tmp_path) is None


def test_directory_named_like_marker_is_ignored(tmp_path: Path):
    (tmp_path / uc.EXTERNAL_UPDATE_MARKER).mkdir(parents=True)
    assert uc._external_update_refusal(tmp_path) is None
