"""Recorded extras follow the source being built, not its installed identity.

Additional staged-update coverage for #135408 / #135347.
"""
from pathlib import Path

import pytest

from pm.install import _member_inputs, _target_selection
from pm.packages import Venv
from pm.plugin_inputs import Members


def _manifest(root: Path, name: str, extras: list[str]) -> None:
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "1.0.0"\n'
        + "[project.optional-dependencies]\n"
        + "".join(f'{extra} = []\n' for extra in extras),
        encoding="utf-8",
    )


@pytest.mark.parametrize("staged_declares_extra", [True, False], ids=["retain", "prune"])
@pytest.mark.parametrize("staged_mapping", [True, False], ids=["mapping", "sequence"])
def test_selection_uses_staged_member_declarations(tmp_path, staged_declares_extra, staged_mapping):
    core = tmp_path / "core"
    installed = tmp_path / "home" / "plugins" / "storage"
    staged = tmp_path / "staged" / "storage"
    _manifest(core, "core", ["core-feature"])
    _manifest(installed, "storage", [] if staged_declares_extra else ["postgres"])
    _manifest(staged, "storage", ["postgres"] if staged_declares_extra else [])
    (core / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    members = {installed: staged} if staged_mapping else [staged]
    inputs = _member_inputs(Members(members))
    package = Venv(core)
    enabled, stamp, selected_inputs = _target_selection(
        package, {"extras": ["core-feature", "postgres"]},
        extras=None, inputs=inputs, repair=False, shipped=None, frozen=None,
    )

    expected = ["core-feature", "postgres"] if staged_declares_extra else ["core-feature"]
    assert enabled == expected
    assert stamp == package.expected_stamp(expected, plugin_dirs=members)
    assert selected_inputs == inputs
