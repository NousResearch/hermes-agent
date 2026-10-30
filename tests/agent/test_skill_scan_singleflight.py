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
    real_parse = sc_mod._scan_skill_md

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
        patch.object(sc_mod, "_scan_skill_md", side_effect=counting_parse),
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
