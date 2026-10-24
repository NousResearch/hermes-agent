"""AGENTS.md directory-chain resolution outside git repositories.

A non-git workspace tree can declare its root with a ``.hermes-root`` marker file; ancestor AGENTS.md
files between the marker root and the cwd are then merged like the git-root chain does.
"""

from agent.prompt_builder import _agents_md_directory_chain


def test_non_git_cwd_without_marker_checks_only_cwd(tmp_path):
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert _agents_md_directory_chain(deep) == [deep.resolve()]


def test_marker_root_extends_the_chain_to_ancestors(tmp_path):
    root = tmp_path / "ws"
    sub = root / "profiles" / "engineer"
    sub.mkdir(parents=True)
    (root / ".hermes-root").write_text("")
    (root / "AGENTS.md").write_text("common rules")
    (sub / "AGENTS.md").write_text("engineer rules")

    chain = _agents_md_directory_chain(sub)
    assert chain == [root.resolve(), root / "profiles", sub.resolve()]

    # Sanity: per-directory first-non-empty selection picks both AGENTS.md files.
    from agent.prompt_builder import _agents_md_candidates

    found = _agents_md_candidates(sub)
    assert [(label, content) for label, _, content in found] == [
        ("../../AGENTS.md", "common rules"),
        ("AGENTS.md", "engineer rules"),
    ]


def test_marker_closer_than_tmp_root_stops_the_walk(tmp_path):
    root = tmp_path / "ws"
    sub = root / "nested" / "deep"
    sub.mkdir(parents=True)
    (root / ".hermes-root").write_text("")
    (sub / ".hermes-root").write_text("")  # nearest marker wins; nothing above it is included

    assert _agents_md_directory_chain(sub) == [sub.resolve()]


def test_git_behaviour_is_unchanged(tmp_path):
    root = tmp_path / "repo"
    sub = root / "pkg"
    sub.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "AGENTS.md").write_text("repo rules")
    (sub / "AGENTS.md").write_text("pkg rules")

    assert [d.name for d in _agents_md_directory_chain(sub)] == ["repo", "pkg"]
