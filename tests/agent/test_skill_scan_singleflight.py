"""Single-flight skill scans: concurrent cold callers share one scan.

Without coalescing, every thread that arrives while caches are cold runs its own
full directory walk + SKILL.md parse pass; the gateway's completion RPCs run on
``asyncio.to_thread`` workers, so a keystroke burst on a cold palette spawns
several simultaneous scans that serialize on the GIL (measured: 6 concurrent
``complete.slash`` on an 855-skill install took 8-9 s wall, ~13x a single scan).
"""
import threading
from unittest.mock import patch


def _make_skill(skills_dir, name):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n---\n\nDo the thing.\n", encoding="utf-8"
    )
    return skill_dir


def test_concurrent_cold_command_map_calls_run_one_scan(tmp_path):
    """Eight threads hitting ``get_skill_commands`` cold parse every SKILL.md once, not 8x."""
    import agent.skill_commands as sc_mod
    import tools.skills_tool as st
    from agent.skill_commands import get_skill_commands

    for i in range(40):
        _make_skill(tmp_path, f"skill{i:02d}")

    parses = []
    real_parse = sc_mod._register_row_command

    def counting_parse(*args, **kwargs):
        parses.append(1)
        return real_parse(*args, **kwargs)

    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()
        results.append(get_skill_commands())

    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch.object(sc_mod, "_skill_commands_by_key", {}),
        patch.object(sc_mod, "_register_row_command", side_effect=counting_parse),
    ):
        st.clear_skills_cache()
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    expected = {f"/skill{i:02d}" for i in range(40)}
    assert len(parses) <= len(expected), (
        f"concurrent cold callers ran {len(parses)} skill-file parses "
        f"({len(parses) / len(expected):.1f}x the catalog); one scan must serve all of them"
    )
    assert all(set(r) == expected for r in results), "every caller must receive the complete scan result"


def test_concurrent_cold_catalog_reads_run_one_parse_pass(tmp_path):
    """Eight threads hitting the shared catalog cold parse each SKILL.md once, not 8x."""
    import tools.skills_tool as st

    for i in range(40):
        _make_skill(tmp_path, f"skill{i:02d}")

    parses = []
    real_parse = st._parse_frontmatter

    def counting_parse(*args, **kwargs):
        parses.append(1)
        return real_parse(*args, **kwargs)

    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()
        results.append([s["name"] for s in st.find_all_skills()])

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path), patch.object(
        st, "_parse_frontmatter", side_effect=counting_parse
    ):
        st.clear_skills_cache()
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    expected_names = [f"skill{i:02d}" for i in range(40)]
    assert len(parses) <= len(expected_names), (
        f"concurrent catalog readers ran {len(parses)} frontmatter parses "
        f"({len(parses) / len(expected_names):.1f}x the catalog); one scan must serve all of them"
    )
    assert all(sorted(r) == sorted(expected_names) for r in results)


def test_cache_freshness_without_ttl(tmp_path):
    """The scan signature covers SKILL.md files, not only directories: an
    in-place edit (dir mtimes + listings unchanged) invalidates the cache
    immediately, and an unchanged library never re-parses — no TTL re-scan."""
    import os
    import time as _time

    import tools.skills_tool as st

    _make_skill(tmp_path, "alpha")

    parses = []
    real_parse = st._parse_frontmatter

    def counting_parse(*args, **kwargs):
        parses.append(1)
        return real_parse(*args, **kwargs)

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path), patch.object(
        st, "_parse_frontmatter", side_effect=counting_parse
    ):
        st.clear_skills_cache()
        first = [s["name"] for s in st.find_all_skills()]
        assert first == ["alpha"]
        assert len(parses) == 1

        # nothing changed: a later call must serve the cache without re-parsing
        st.find_all_skills()
        assert len(parses) == 1, "an unchanged library must not re-parse the catalog"

        # in-place edit: file facts change, directory facts do NOT
        skill_md = tmp_path / "alpha" / "SKILL.md"
        text = skill_md.read_text(encoding="utf-8")
        skill_md.write_text(
            text.replace("Description for alpha.", "Edited in place."), encoding="utf-8"
        )
        os.utime(skill_md, ns=(_time.time_ns(), _time.time_ns() + 1_000_000_000))
        rows = st.find_all_skills()
        assert len(parses) == 2, "an in-place SKILL.md edit must invalidate the scan cache"
        assert any("Edited in place." in (s["description"] or "") for s in rows), (
            "the edited description must be visible immediately, not after a TTL"
        )
