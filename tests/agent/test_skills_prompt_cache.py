"""The skills manifest is memoized: a repeat prompt build must not re-walk the skills tree.

Building ``<available_skills>`` needs a file-signature manifest of every SKILL.md, and on Windows
that walk costs ~900 directory opens plus one stat and one content digest per skill — measured
0.9 s warm and 8.5 s cold against a 644-skill tree, i.e. every prompt build paid it. The manifest
is now memoized per skills root, with two revalidation triggers:

* a cheap first-level probe on every build (an entry added/removed/renamed at the top of the tree),
* a time window, ``_SKILLS_MANIFEST_TTL_SECS``, for everything deeper — a skill directory added
  inside a category, or a SKILL.md whose content changed (which moves only that file's mtime).

A test that wants the exact detector rather than the accelerator sets the window to 0 ("the window
has expired"); that is the same path a long-lived process takes every ``_SKILLS_MANIFEST_TTL_SECS``.
"""

import time

import pytest

from agent import prompt_builder as pb


@pytest.fixture(autouse=True)
def _clean_skills_cache():
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    yield
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)


def _write_skill(home, rel, name, description):
    skill_dir = home / "skills" / rel
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody.\n", encoding="utf-8"
    )
    return skill_dir


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def scan_spy(monkeypatch):
    """Record every full manifest walk; the memo must make repeats avoid it entirely."""
    real = pb._scan_skills_manifest
    scans: list[str] = []
    monkeypatch.setattr(pb, "_scan_skills_manifest", lambda d: (scans.append(str(d)), real(d))[1])
    return scans


@pytest.fixture
def window_expired(monkeypatch):
    """Force the exact (walk + digest) revalidation on the next build."""
    monkeypatch.setattr(pb, "_SKILLS_MANIFEST_TTL_SECS", 0)


def test_repeat_build_answers_from_the_memo_without_rescanning(home, scan_spy):
    _write_skill(home, "tools/search", "search", "Search stuff")
    first = pb.build_skills_system_prompt()
    assert "search: Search stuff" in first
    assert len(scan_spy) == 1

    started = time.perf_counter()
    second = pb.build_skills_system_prompt()
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert second == first
    assert len(scan_spy) == 1, "a repeat build re-walked the skills tree"
    # The walk is what cost seconds; without it only prompt assembly remains. Generous bound — the
    # bench target is 50 ms, this only has to catch a regression that brings the walk back.
    assert elapsed_ms < 250, f"repeat build took {elapsed_ms:.1f} ms"


def test_new_top_level_skill_is_seen_without_waiting_for_the_window(home, scan_spy):
    _write_skill(home, "tools/search", "search", "Search stuff")
    assert "brand-new" not in pb.build_skills_system_prompt()

    # A new entry at the top of the tree is what the cheap probe sees, so no TTL is involved.
    _write_skill(home, "experimental", "brand-new", "Fresh skill")
    assert "brand-new: Fresh skill" in pb.build_skills_system_prompt()
    assert len(scan_spy) == 2


def test_content_edit_is_picked_up_once_the_window_expires(home, scan_spy, monkeypatch):
    _write_skill(home, "tools/search", "search", "Old description")
    assert "search: Old description" in pb.build_skills_system_prompt()

    _write_skill(home, "tools/search", "search", "New description")
    # Inside the window the memo answers: this is the documented staleness, and it is bounded.
    assert "search: Old description" in pb.build_skills_system_prompt()

    monkeypatch.setattr(pb, "_SKILLS_MANIFEST_TTL_SECS", 0)  # the window has now expired
    assert "search: New description" in pb.build_skills_system_prompt()
    assert len(scan_spy) == 2  # first build + revalidation, nothing in between


def test_skill_added_inside_a_category_is_picked_up_by_revalidation(home, scan_spy, window_expired):
    _write_skill(home, "tools/search", "search", "Search stuff")
    pb.build_skills_system_prompt()

    # Windows directory timestamps lag, so this shape is the TTL's job, not the probe's.
    _write_skill(home, "tools/ripgrep", "ripgrep", "Ripgrep helper")
    assert "ripgrep: Ripgrep helper" in pb.build_skills_system_prompt()
    assert len(scan_spy) == 2


def test_removed_skill_disappears(home, window_expired):
    import shutil

    _write_skill(home, "tools/search", "search", "Search stuff")
    doomed = _write_skill(home, "tools/doomed", "doomed", "Going away")
    assert "doomed" in pb.build_skills_system_prompt()

    shutil.rmtree(doomed)
    assert "doomed" not in pb.build_skills_system_prompt()


def test_timestamp_preserving_rewrite_still_reads_as_a_change(home, window_expired):
    """The manifest's content digest is what catches ``cp -p``/tar-restore style rewrites."""
    import os

    skill_dir = _write_skill(home, "tools/search", "search", "Old description")
    skill_file = skill_dir / "SKILL.md"
    stat = skill_file.stat()
    assert "search: Old description" in pb.build_skills_system_prompt()

    skill_file.write_text("---\nname: search\ndescription: Swapped description\n---\nBody.\n", "utf-8")
    os.utime(skill_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same mtime, same size
    assert "search: Swapped description" in pb.build_skills_system_prompt()


def test_clear_cache_forces_a_fresh_walk(home, scan_spy):
    _write_skill(home, "tools/search", "search", "Search stuff")
    pb.build_skills_system_prompt()
    assert len(scan_spy) == 1

    pb.clear_skills_system_prompt_cache()
    pb.build_skills_system_prompt()
    assert len(scan_spy) == 2


def test_callers_cannot_corrupt_the_memo(home):
    _write_skill(home, "tools/search", "search", "Search stuff")
    skills_dir = home / "skills"
    manifest = pb._build_skills_manifest(skills_dir)
    assert manifest

    manifest["injected/SKILL.md"] = [0, 0, 0, 0, "forged"]
    assert "injected/SKILL.md" not in pb._build_skills_manifest(skills_dir)
