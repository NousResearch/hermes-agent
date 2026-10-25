"""A buildable plugin enabled in two profiles declares ONE [project].name twice.

uv identifies a workspace member by that name, so "Two workspace members are both
named ..." fails `uv lock` for the WHOLE workspace — admission fails for every
profile that shares the environment, and the resolver text names hashed member
dirs no user can act on.

A collision is a fact about the SELECTION, not about one plugin: the refusal must
say which homes hold the copies, and what reconciles it.
"""
import tomllib
from pathlib import Path

import pytest


def _buildable(plugin: Path, name: str) -> Path:
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "pyproject.toml").write_text(
        "[project]\n"
        f'name = "{name}"\n'
        'version = "1.0.0"\n'
        "[build-system]\n"
        "requires = []\n"
        'build-backend = "backend"\n',
        encoding="utf-8",
    )
    return plugin


def _virtual(plugin: Path, name: str) -> Path:
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "pyproject.toml").write_text(
        "[project]\n" f'name = "{name}"\n' 'version = "1.0.0"\n'
        'dependencies = ["requests>=2"]\n' "[tool.uv]\npackage = false\n",
        encoding="utf-8",
    )
    return plugin


def _entries(*paths: Path):
    return [(p.parent, p.name, p) for p in paths]


def test_two_buildable_members_with_one_name_are_reported_with_homes(tmp_path):
    from pm.plugin_eviction import _colliding_members

    a = _buildable(tmp_path / "home-a" / "plugins" / "shared", "hermes-shared")
    b = _buildable(tmp_path / "home-b" / "plugins" / "shared", "hermes-shared")

    collisions = _colliding_members(_entries(a, b))

    assert set(collisions) == {a.resolve(), b.resolve()}
    reason_a = collisions[a.resolve()]
    assert "shares the build-metadata name" in reason_a
    assert "disable the duplicate in all but one profile" in reason_a
    # The reason must name the OTHER home -- the plugin key is the same string twice,
    # so telling the operator to disable "the duplicate" without naming where it lives
    # leaves them grepping. This is a message an operator acts on.
    assert str(b.parent.parent.resolve()) in reason_a
    assert str(a.parent.parent.resolve()) not in reason_a


def test_a_single_copy_never_collides(tmp_path):
    from pm.plugin_eviction import _colliding_members

    a = _buildable(tmp_path / "home" / "plugins" / "shared", "hermes-shared")
    b = _buildable(tmp_path / "home" / "plugins" / "other", "hermes-other")

    assert _colliding_members(_entries(a, b)) == {}


def test_two_copies_with_different_names_never_collide(tmp_path):
    from pm.plugin_eviction import _colliding_members

    a = _buildable(tmp_path / "home" / "plugins" / "shared", "hermes-shared-one")
    b = _buildable(tmp_path / "home" / "profiles" / "other" / "plugins" / "shared", "hermes-shared-two")

    assert _colliding_members(_entries(a, b)) == {}


def test_virtual_members_cannot_collide_because_they_are_renamed(tmp_path):
    """A metadata-only member is renamed to its unique key, so a shared declared
    name is not a collision there -- and must not be reported as one."""
    from pm.plugin_eviction import _colliding_members
    from pm.workspace import _is_member_candidate

    a = _virtual(tmp_path / "home" / "plugins" / "shared", "same-name")
    b = _virtual(tmp_path / "home" / "profiles" / "other" / "plugins" / "shared", "same-name")

    assert _is_member_candidate(a) and _is_member_candidate(b)
    assert _colliding_members(_entries(a, b)) == {}


def test_admission_refuses_a_selection_two_homes_cannot_build(tmp_path, monkeypatch):
    """Admission must refuse with the reconcile instruction instead of uv's
    resolver noise: uv refuses the WHOLE workspace, and the resolver text names
    hashed member dirs no user can act on."""
    from pm.package import InstallError
    from pm.publication import candidate_members

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    current = home / "plugins" / "shared"
    other_home = home / "profiles" / "other"
    other = other_home / "plugins" / "shared"
    for member in (current, other):
        member.mkdir(parents=True)
        _buildable(member, "hermes-shared")
    (home / "config.yaml").write_text("plugins:\n  enabled: [shared]\n  disabled: []\n")
    (other_home / "config.yaml").write_text("plugins:\n  enabled: [shared]\n")

    with pytest.raises(InstallError) as excinfo:
        candidate_members(proposed_home=home, enabled=["shared"], disabled=[])
    message = str(excinfo.value)
    assert "hermes-shared" in message
    assert "shares the build-metadata name" in message
    assert "disable the duplicate in all but one profile" in message
