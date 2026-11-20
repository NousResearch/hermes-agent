"""``tarball`` as an install method: a release archive with an external update owner.

A tree unpacked from a release archive has no ``.git`` and no package manager that
Hermes can name. Before this it fell through to ``unknown``, which the update plan
treats as eligible for an in-place update - so ``hermes update`` started a git pull
inside a tree that has no checkout and died with "Not a git repository" *after*
asking the user to confirm. The method gives that shape a name, refuses the update
up front and points at the installer that owns it.
"""

import json
from unittest.mock import patch


def _code_tree(tmp_path):
    """A tarball tree: code plus home, no ``.git``, external install stamp."""
    code = tmp_path / "code"
    home = tmp_path / "home"
    code.mkdir()
    home.mkdir()
    return code, home


def _write_stamp(tree, mechanism="external", **extra):
    payload = {"schemaVersion": 2, "source": "build", "updateMechanism": mechanism}
    payload.update(extra)
    (tree / "install-stamp.json").write_text(json.dumps(payload), encoding="utf-8")


def _detect(code, home):
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=home):
        from hermes_cli.config import detect_install_method

        return detect_install_method(project_root=code)


def test_external_install_stamp_without_git_is_a_tarball(tmp_path):
    """The shape a release archive has: no stamp, no .git, external update owner."""
    code, home = _code_tree(tmp_path)
    _write_stamp(code)
    assert _detect(code, home) == "tarball"


def test_git_checkout_wins_over_the_external_stamp(tmp_path):
    """A real checkout is ours to mutate, whatever its stamp says."""
    code, home = _code_tree(tmp_path)
    _write_stamp(code)
    (code / ".git").mkdir()
    assert _detect(code, home) == "git"


def test_self_updated_stamp_is_not_a_tarball(tmp_path):
    """Only ``external`` hands updates away; ``self`` keeps the in-place path."""
    code, home = _code_tree(tmp_path)
    _write_stamp(code, mechanism="self")
    assert _detect(code, home) == "unknown"


def test_tarball_is_an_accepted_stamp_value(tmp_path):
    """An installer that prefers the explicit stamp gets the same name."""
    code, home = _code_tree(tmp_path)
    (code / ".install_method").write_text("tarball\n", encoding="utf-8")
    assert _detect(code, home) == "tarball"


def test_update_command_is_guidance_not_hermes_update(tmp_path):
    """``recommended_update_command_for_method`` must not answer "hermes update"."""
    from hermes_cli.config import recommended_update_command_for_method

    command = recommended_update_command_for_method("tarball")
    assert command != "hermes update"
    assert "installer" in command


def test_update_admission_refuses_before_git_runs(tmp_path):
    """The refusal carries the guidance and never reaches the git path."""
    code, home = _code_tree(tmp_path)
    _write_stamp(code)
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=home), \
         patch("hermes_cli.image_provenance.read_image_provenance", return_value=None):
        from hermes_cli.update_contract import evaluate_update_admission

        refusal = evaluate_update_admission(code)

    assert refusal is not None
    assert refusal.code == "tarball"
    assert "release archive" in refusal.message
    assert refusal.update_command != "hermes update"


def test_dashboard_marks_tarball_non_applyable():
    """The dashboard's update check reports tarball as not applyable in place."""
    actions = __import__("pytest").importorskip("hermes_cli.web_routers.actions")
    assert "tarball" in actions._NON_APPLYABLE_MESSAGES
    assert actions._NON_APPLYABLE_MESSAGES["tarball"]()
