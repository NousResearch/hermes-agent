"""`hermes curator archive` refuses pinned skills (with an `unpin` hint) and never archives them."""

from __future__ import annotations

from types import SimpleNamespace


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


# ─── archive ────────────────────────────────────────────────────────────────


def test_archive_refuses_pinned(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(skill_usage, "get_record", lambda name: {"pinned": True})
    called = []
    monkeypatch.setattr(
        skill_usage, "archive_skill",
        lambda name: called.append(name) or (True, "should not get here"),
    )

    rc = curator_cli._cmd_archive(_ns(skill="pinned-skill"))
    assert rc == 1
    assert called == []
    out = capsys.readouterr().out
    assert "pinned" in out.lower()
    assert "hermes curator unpin" in out


# ─── purge ──────────────────────────────────────────────────────────────────


def test_purge_ages_an_archive_from_when_it_was_archived(tmp_path, monkeypatch):
    """The TTL counts from archival, not from the skill's last edit: a long-idle skill the
    curator archives today must survive `purge --days 30`."""
    import os
    import time

    import hermes_cli.curator as curator_cli
    from tools import skill_usage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skill_dir = tmp_path / "skills" / "old-helper"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: old-helper\ndescription: x\n---\n# body\n", encoding="utf-8")
    long_ago = time.time() - 100 * 86400
    for path in (skill_dir / "SKILL.md", skill_dir):
        os.utime(path, (long_ago, long_ago))
    skill_usage.record_created("old-helper", agent_created=True)
    ok, msg = skill_usage.archive_skill("old-helper")
    assert ok, msg
    # A stale archived_at (left by a manual un-archive + re-archive) must not beat the fresh mtime.
    usage = skill_usage.load_usage()
    usage["old-helper"]["archived_at"] = "2000-01-01T00:00:00+00:00"
    assert skill_usage.save_usage(usage)

    assert curator_cli._cmd_purge(_ns(days=30, dry_run=False, yes=True)) == 0
    assert (tmp_path / "skills" / ".archive" / "old-helper" / "SKILL.md").is_file()


def test_purge_prefers_recorded_archived_at_over_a_stale_dir_mtime(tmp_path, monkeypatch):
    """An archive made before the mtime stamp keeps an old dir mtime; its usage record's
    archived_at (today) is what the TTL counts from."""
    import os
    import time

    import hermes_cli.curator as curator_cli
    from tools import skill_usage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skill_dir = tmp_path / "skills" / "legacy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: legacy\ndescription: x\n---\n# body\n", encoding="utf-8")
    skill_usage.record_created("legacy", agent_created=True)
    ok, msg = skill_usage.archive_skill("legacy")
    assert ok, msg
    archived = tmp_path / "skills" / ".archive" / "legacy"
    long_ago = time.time() - 100 * 86400
    os.utime(archived, (long_ago, long_ago))
    # A legacy archive flattened under its folder name: the record is keyed by the
    # SKILL.md frontmatter name, so that is where its archived_at must be read from.
    flattened = tmp_path / "skills" / ".archive" / "accelerate"
    flattened.mkdir()
    (flattened / "SKILL.md").write_text(
        "---\nname: huggingface-accelerate\ndescription: x\n---\n# body\n", encoding="utf-8")
    os.utime(flattened, (long_ago, long_ago))
    usage = skill_usage.load_usage()
    usage["huggingface-accelerate"] = {
        "state": skill_usage.STATE_ARCHIVED, "archived_at": usage["legacy"]["archived_at"]}
    assert skill_usage.save_usage(usage)

    assert curator_cli._cmd_purge(_ns(days=30, dry_run=False, yes=True)) == 0
    assert (archived / "SKILL.md").is_file()
    assert (flattened / "SKILL.md").is_file()


# ─── prune ──────────────────────────────────────────────────────────────────


# ─── argparse wiring ────────────────────────────────────────────────────────


