"""One plugin installed in two homes must produce exactly one uv workspace member.

A member key is ``<plugin-dir-name>-<sha256(identity path)[:16]>``, so the same plugin
enabled in the default home *and* in a live profile arrives twice with two different keys.
uv identifies a member by ``[project].name`` and refuses the whole workspace, which used to
abort every plugin change in that profile with "Two workspace members are both named ...".
"""

from pathlib import Path

from pm import workspace


def _plugin(root: Path, *, buildable: bool) -> Path:
    plugin = root / "apify"
    plugin.mkdir(parents=True)
    body = (
        '[project]\nname = "apify-hermes-agent-plugin"\nversion = "0.1.5"\n'
        'requires-python = ">=3.11"\ndependencies = ["httpx>=0.28.1,<1"]\n'
    )
    if buildable:
        body += '[build-system]\nrequires = []\nbuild-backend = "backend"\n'
    (plugin / "pyproject.toml").write_text(body, encoding="utf-8")
    (plugin / "plugin.yaml").write_text("name: apify\n", encoding="utf-8")
    return plugin


def test_same_project_from_two_homes_yields_one_member(tmp_path):
    """The profile's install shadows the default home's instead of failing the lock."""
    default_home = _plugin(tmp_path / "home" / "plugins", buildable=True)
    profile_home = _plugin(tmp_path / "profiles" / "fuar-dubai" / "plugins", buildable=True)

    pairs = workspace._unique_member_sources({default_home: default_home, profile_home: profile_home})

    assert len(pairs) == 1
    assert pairs[0][1] == profile_home.resolve()


def test_distinct_plugins_are_both_kept(tmp_path):
    default_home = _plugin(tmp_path / "home" / "plugins", buildable=True)
    other = default_home.parent / "linkup"
    other.mkdir()
    (other / "pyproject.toml").write_text(
        '[project]\nname = "linkup-plugin"\nversion = "1.0"\ndependencies = ["httpx>=0.28.1,<1"]\n',
        encoding="utf-8",
    )
    (other / "plugin.yaml").write_text("name: linkup\n", encoding="utf-8")

    pairs = workspace._unique_member_sources({default_home: default_home, other: other})

    assert len(pairs) == 2


def test_seed_lock_member_path_is_repointed(tmp_path):
    """A seed lock naming yesterday's key must not abort uv with "Distribution not found"."""
    member = tmp_path / "plugin-sources" / "apify-46cbef8e7f4ac30b"
    member.mkdir(parents=True)
    seed = (
        "[[package]]\n"
        'name = "apify-hermes-agent-plugin"\n'
        'source = { editable = "plugin-sources/apify-bccd6f10a461723b" }\n'
    )

    repaired = workspace._repair_seed_lock(seed, tmp_path)

    assert "apify-46cbef8e7f4ac30b" in repaired
    assert "bccd6f10a461723b" not in repaired


def test_seed_lock_keeps_present_member_paths(tmp_path):
    member = tmp_path / "plugin-sources" / "apify-46cbef8e7f4ac30b"
    member.mkdir(parents=True)
    seed = 'source = { editable = "plugin-sources/apify-46cbef8e7f4ac30b" }\n'

    assert workspace._repair_seed_lock(seed, tmp_path) == seed
