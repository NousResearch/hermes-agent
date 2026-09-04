"""Steward detection reads the install stamp beside the code, not markers.

``hermes_cli.steward`` decides whether a steward owns this tree (and
therefore whether ``hermes update`` and the uninstaller must refuse code
mutation). That is a fact about the INSTALL, so it comes from the two
facts the tree itself carries: ``.git`` (a checkout we own) and the
install stamp that ships with the code (``install-stamp.json``, written by
``scripts/write_install_stamp.py`` with a required ``updateMechanism`` of
``self``/``electron-updater``/``external``).
"""

import json

import pytest

from hermes_cli.steward import (
    STEWARD_DESKTOP,
    STEWARD_DOCKER,
    STEWARD_NIX,
    classify_install,
    read_install_stamp,
    sealed_steward,
)


def _stamp(root, *, mechanism="external", **fields):
    root.mkdir(exist_ok=True)
    path = root / "install-stamp.json"
    payload = {"commit": "a" * 40, "updateMechanism": mechanism, **fields}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestStampDrivenDetection:
    @pytest.mark.parametrize(
        ("distribution", "mechanism"),
        [
            (STEWARD_NIX, "external"),
            (STEWARD_DOCKER, "external"),
            (STEWARD_DESKTOP, "electron-updater"),
        ],
    )
    def test_sealed_tree_names_its_steward(self, tmp_path, distribution, mechanism):
        root = tmp_path / "install"
        _stamp(root, mechanism=mechanism, distribution=distribution)
        assert sealed_steward(root) == distribution
        assert classify_install(root) == (distribution, False)

    def test_git_checkout_is_not_steward_owned(self, tmp_path):
        root = tmp_path / "checkout"
        root.mkdir()
        (root / ".git").mkdir()
        assert sealed_steward(root) is None
        assert classify_install(root) == ("git", True)

    def test_git_beats_stamp(self, tmp_path):
        """A checkout carrying a baked stamp (self-updating source install)
        is still a checkout — the uninstaller may remove it."""
        root = tmp_path / "checkout"
        root.mkdir()
        (root / ".git").mkdir()
        _stamp(root, mechanism="self", distribution=None)
        assert sealed_steward(root) is None

    def test_worktree_gitfile_counts_as_checkout(self, tmp_path):
        root = tmp_path / "worktree"
        root.mkdir()
        (root / ".git").write_text("gitdir: ../elsewhere\n", encoding="utf-8")
        assert sealed_steward(root) is None

    def test_missing_stamp_gives_unknown_steward(self, tmp_path):
        """No .git and no stamp: we cannot prove the tree is ours — sealed
        under an unknown steward, so code removal refuses."""
        root = tmp_path / "bare"
        root.mkdir()
        assert sealed_steward(root) == "unknown"
        assert classify_install(root) == ("unknown", False)

    def test_malformed_stamp_degrades_to_unknown(self, tmp_path):
        root = tmp_path / "install"
        root.mkdir()
        (root / "install-stamp.json").write_text("{not json", encoding="utf-8")
        assert read_install_stamp(root) == {}
        assert sealed_steward(root) == "unknown"


class TestUpdateAdmission:
    """The update admission gate refuses sealed trees with the steward's text."""

    @pytest.fixture(autouse=True)
    def _no_image_marker(self, tmp_path, monkeypatch):
        import hermes_cli.image_provenance as ip

        monkeypatch.setattr(ip, "IMAGE_PROVENANCE_PATH", tmp_path / "absent.json")

    def test_desktop_app_stamp_refuses_update(self, tmp_path, monkeypatch):
        from hermes_cli.update_contract import evaluate_update_admission

        root = tmp_path / "payload"
        _stamp(root, mechanism="electron-updater", distribution=STEWARD_DESKTOP)
        refusal = evaluate_update_admission(root)
        assert refusal is not None
        assert refusal.code == STEWARD_DESKTOP
        assert "desktop app" in refusal.message

    def test_nix_stamp_refuses_update(self, tmp_path):
        from hermes_cli.update_contract import evaluate_update_admission

        root = tmp_path / "store-tree"
        _stamp(root, mechanism="external", distribution=STEWARD_NIX)
        refusal = evaluate_update_admission(root)
        assert refusal is not None
        assert refusal.code == "nix"
        assert "Nix store" in refusal.message

    def test_docker_stamp_refuses_with_docker_message(self, tmp_path):
        from hermes_cli.update_contract import evaluate_update_admission

        root = tmp_path / "image-tree"
        _stamp(root, mechanism="external", distribution=STEWARD_DOCKER)
        refusal = evaluate_update_admission(root)
        assert refusal is not None
        assert refusal.code == "docker"
        assert "docker pull" in refusal.update_command

    def test_git_checkout_is_admitted(self, tmp_path, monkeypatch):
        from hermes_cli.update_contract import evaluate_update_admission

        root = tmp_path / "checkout"
        root.mkdir()
        (root / ".git").mkdir()
        monkeypatch.setattr(
            "hermes_cli.config.detect_install_method", lambda *a, **k: "git"
        )
        assert evaluate_update_admission(root) is None

    def test_unknown_steward_falls_through_to_heuristics(self, tmp_path, monkeypatch):
        """A stampless gitless tree is not refused by the steward rung —
        the pre-existing heuristics keep the final say (an unknown tree used
        to be admitted, and stays admitted)."""
        from hermes_cli.update_contract import evaluate_update_admission

        root = tmp_path / "bare"
        root.mkdir()
        monkeypatch.setattr(
            "hermes_cli.config.detect_install_method", lambda *a, **k: "unknown"
        )
        assert evaluate_update_admission(root) is None
