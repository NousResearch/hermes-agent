"""Official Homebrew formula install detection (#101676).

The homebrew/core ``hermes-agent`` formula installs under
``<prefix>/Cellar/hermes-agent/<version>/libexec/...``.  The CLI must classify
that layout as ``homebrew`` (package-manager-owned) so it never offers the
in-place self-update path inside the Cellar and directs users to
``brew upgrade hermes-agent`` instead of comparing git commit distance to
upstream ``main``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import config as config_mod
from hermes_cli.update_contract import evaluate_update_admission


@pytest.fixture(autouse=True)
def isolate_hermes_home(tmp_path, monkeypatch):
    """Keep the home-scoped ``.install_method`` stamp probe out of the way."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)


@pytest.mark.parametrize(
    ("project_root", "expected"),
    [
        # Apple Silicon formula layout, reported by the issue.
        (
            "/opt/homebrew/Cellar/hermes-agent/2026.9.14/libexec/lib/python3.13/site-packages",
            "homebrew",
        ),
        # Intel prefix, same layout.
        (
            "/usr/local/Cellar/hermes-agent/2026.9.14/libexec/lib/python3.13/site-packages",
            "homebrew",
        ),
        # Root resolved a directory deeper than site-packages.
        (
            "/opt/homebrew/Cellar/hermes-agent/2026.9.14/libexec/lib/python3.13/site-packages/hermes_cli",
            "homebrew",
        ),
        # A different formula in the Cellar is not ours.
        (
            "/opt/homebrew/Cellar/some-other-formula/1.0/libexec",
            "unknown",
        ),
        # A name-prefix match is not the formula.
        (
            "/opt/homebrew/Cellar/hermes-agent-helper/1.0/libexec",
            "unknown",
        ),
        # A plain non-Cellar install keeps its old answer.
        (
            "/usr/local/lib/python3.13/site-packages",
            "unknown",
        ),
    ],
)
def test_cellar_layout_detects_homebrew(project_root, expected):
    assert config_mod.detect_install_method(Path(project_root)) == expected


def test_code_scoped_stamp_wins_over_cellar_path(tmp_path):
    """The stamp is authoritative (config.py resolution order, step 1)."""
    tree = tmp_path / "Cellar" / "hermes-agent" / "1.0" / "libexec"
    tree.mkdir(parents=True)
    (tree / ".install_method").write_text("git\n", encoding="utf-8")
    assert config_mod.detect_install_method(tree) == "git"


def test_homebrew_stamp_is_a_supported_method(tmp_path):
    """A formula install may self-identify via the stamp; the value must validate."""
    tree = tmp_path / "install"
    tree.mkdir()
    (tree / ".install_method").write_text("homebrew\n", encoding="utf-8")
    assert config_mod.detect_install_method(tree) == "homebrew"


def test_homebrew_update_command_is_brew_upgrade():
    assert config_mod.recommended_update_command_for_method("homebrew") == "brew upgrade hermes-agent"


def test_homebrew_recommended_update_command(monkeypatch):
    monkeypatch.setattr(config_mod, "detect_install_method", lambda *a, **k: "homebrew")
    assert config_mod.recommended_update_command() == "brew upgrade hermes-agent"


def test_admission_refuses_homebrew_install(tmp_path, monkeypatch):
    """``hermes update`` must not mutate a Cellar owned by Homebrew."""
    import hermes_cli.image_provenance as ip

    monkeypatch.setattr(ip, "IMAGE_PROVENANCE_PATH", tmp_path / "absent-marker.json")
    monkeypatch.setattr(config_mod, "detect_install_method", lambda *a, **k: "homebrew")

    refusal = evaluate_update_admission(tmp_path)

    assert refusal is not None
    assert refusal.code == "homebrew"
    assert "brew upgrade hermes-agent" in refusal.update_command
    assert "brew upgrade hermes-agent" in refusal.message


def test_admission_still_admits_unknown_installs(tmp_path, monkeypatch):
    """The refusal is homebrew-specific — mutable unknown installs keep the old contract."""
    import hermes_cli.image_provenance as ip

    monkeypatch.setattr(ip, "IMAGE_PROVENANCE_PATH", tmp_path / "absent-marker.json")
    monkeypatch.setattr(config_mod, "detect_install_method", lambda *a, **k: "unknown")

    assert evaluate_update_admission(tmp_path) is None
