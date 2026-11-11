"""Disk-persisted catalog rows: a fresh process boots warm.

The in-memory cache dies with the gateway process, so without persistence the
first keystroke after a restart re-parses every SKILL.md (~0.5 s at ~900
skills). The resolved rows are a pure function of the library the signature
already proves unchanged, so they persist keyed by the signature digest.
"""
from unittest.mock import patch

import tools.skills_tool as st


def _make_skill(skills_dir, name):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n---\n\nDo the thing.\n",
        encoding="utf-8",
    )
    return skill_dir


def _counting_parse(st_mod):
    parses = []
    real = st_mod._parse_frontmatter

    def counting(*args, **kwargs):
        parses.append(1)
        return real(*args, **kwargs)

    return parses, counting


def test_restart_serves_disk_catalog_without_reparse(tmp_path):
    """A cold in-memory cache with a signature-matching disk file must not
    re-parse the library — the restart boot path."""
    _make_skill(tmp_path, "alpha")
    parses, counting = _counting_parse(st)

    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch("tools.skills_tool._catalog_cache_path", return_value=tmp_path.parent / "cat.json"),
        patch.object(st, "_parse_frontmatter", side_effect=counting),
    ):
        st._SKILLS_CACHE.clear()
        first = sorted(s["name"] for s in st.find_all_skills())
        assert first == ["alpha"]
        assert len(parses) == 1

        # process restart: memory cache gone, disk file present
        st._SKILLS_CACHE.clear()
        second = sorted(s["name"] for s in st.find_all_skills())
        assert second == ["alpha"]
        assert len(parses) == 1, (
            "a fresh process must serve the persisted catalog rows, not re-parse the library"
        )


def test_library_change_invalidates_disk_catalog(tmp_path):
    """A skill installed after the file was written must force a re-scan and
    rewrite the file."""
    _make_skill(tmp_path, "alpha")
    parses, counting = _counting_parse(st)

    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch("tools.skills_tool._catalog_cache_path", return_value=tmp_path.parent / "cat.json"),
        patch.object(st, "_parse_frontmatter", side_effect=counting),
    ):
        st._SKILLS_CACHE.clear()
        assert "beta" not in [s["name"] for s in st.find_all_skills()]

        _make_skill(tmp_path, "beta")
        st._SKILLS_CACHE.clear()
        names = [s["name"] for s in st.find_all_skills()]
        assert "beta" in names, "the new skill must appear (signature changed -> rescan)"
        # parses count SKILL.md files, not scans: 1 (alpha's scan) + 2 (alpha+beta rescan)
        assert len(parses) == 3

        # and the rewritten file serves the new state to the next restart
        st._SKILLS_CACHE.clear()
        names = [s["name"] for s in st.find_all_skills()]
        assert "beta" in names
        assert len(parses) == 3


def test_corrupt_disk_catalog_falls_through_to_scan(tmp_path):
    """A garbage cache file must never break catalog reads."""
    _make_skill(tmp_path, "alpha")
    cache_file = tmp_path.parent / "cat.json"
    cache_file.write_text("{not json at all", encoding="utf-8")

    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch("tools.skills_tool._catalog_cache_path", return_value=cache_file),
    ):
        st._SKILLS_CACHE.clear()
        names = [s["name"] for s in st.find_all_skills()]
    assert names == ["alpha"]
