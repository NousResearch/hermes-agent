"""The pinned-source gate's confinement and field rules, on any POSIX host.

``test_plugin_catalog_ci_confinement.py`` runs the whole workflow step and is Linux-only; these
cases pin the two decisions it rests on: a resolved path must stay inside the pinned clone, and
PR-controlled entry fields are refused before anything uses them.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci"))

import catalog_entry_fields as fields
import catalog_validate_pinned_entries as gate

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def clone(tmp_path):
    root = tmp_path / "clone"
    (root / "plugin").mkdir(parents=True)
    (root / ".git").mkdir()
    (tmp_path / "outside").mkdir()
    return root


def test_symlink_escaping_the_clone_is_found_behind_an_in_clone_link(clone, tmp_path):
    # plugin/ext -> ../lib is confined; the walk must still reach lib/evil -> outside.
    (clone / "lib").mkdir()
    (clone / "lib" / "evil").symlink_to(tmp_path / "outside")
    (clone / "plugin" / "ext").symlink_to("../lib")
    root = os.path.realpath(clone)
    assert gate.escaping_symlink(root) == f"{root}/lib/evil"
    assert gate.is_confined(f"{root}/plugin/ext", root)
    assert not gate.is_confined(f"{root}/plugin/ext/evil", root)


def test_in_clone_links_and_git_internals_pass(clone, tmp_path):
    (clone / "plugin" / "alias.yaml").symlink_to("plugin.yaml")
    (clone / ".git" / "hook").symlink_to(tmp_path / "outside")
    root = os.path.realpath(clone)
    assert gate.escaping_symlink(root) is None
    assert gate.is_confined(root, root)
    # A sibling sharing the clone's name as a prefix is outside it.
    assert not gate.is_confined(root + "-other", root)


SHA = "a" * 40


@pytest.mark.parametrize("entry, bad", [
    ({"repo": "https://x/y", "sha": SHA, "subdir": "plugin"}, ""),
    ({"repo": "https://x/y", "sha": SHA, "subdir": None}, ""),
    ({"repo": "https://x/y", "sha": SHA, "subdir": "../x"}, "subdir must be a relative path"),
    ({"repo": "https://x/y", "sha": SHA, "subdir": 0}, "subdir must be a string"),
    ({"repo": "file:///x", "sha": SHA}, "repo must be an https:// git URL"),
    ({"repo": "https://x/y\n::error::x", "sha": SHA}, "repo must be an https:// git URL"),
    ({"repo": "https://x/y", "sha": SHA + "\n"}, "sha must be exactly 40 lowercase hex"),
    ({"repo": "https://x/y", "sha": "main"}, "sha must be exactly 40 lowercase hex"),
    (["not", "a", "mapping"], "entry yaml must be a mapping"),
    (None, "entry yaml is empty or unparseable"),
])
def test_entry_fields_are_refused_before_use(entry, bad):
    verdict = fields.source_fields(entry)["BAD"]
    assert verdict.startswith(bad) if bad else verdict == ""
