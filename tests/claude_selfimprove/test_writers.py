from __future__ import annotations

from claude_selfimprove import paths, writers


# --- CLAUDE.md managed block -------------------------------------------------


def test_creates_managed_block_in_fresh_claude_md(sandbox):
    result = writers.write_claude_md_block([{"title": "Rule one", "body": "Do the thing."}])
    assert result.success
    content = paths.claude_md_path().read_text()
    assert writers.CLAUDE_MD_BEGIN_MARKER in content
    assert writers.CLAUDE_MD_END_MARKER in content
    assert "Do the thing." in content


def test_preserves_human_content_around_managed_block(sandbox):
    sandbox.claude_home.mkdir(parents=True, exist_ok=True)
    human_text = "# My personal instructions\n\nAlways be nice.\n"
    paths.claude_md_path().write_text(human_text)

    result = writers.write_claude_md_block([{"title": "Rule one", "body": "Do the thing."}])
    assert result.success
    content = paths.claude_md_path().read_text()
    assert "My personal instructions" in content
    assert "Always be nice." in content
    assert "Do the thing." in content


def test_second_write_replaces_block_without_duplicating(sandbox):
    writers.write_claude_md_block([{"title": "A", "body": "First version."}])
    writers.write_claude_md_block([{"title": "A", "body": "Second version."}])
    content = paths.claude_md_path().read_text()
    assert content.count(writers.CLAUDE_MD_BEGIN_MARKER) == 1
    assert content.count(writers.CLAUDE_MD_END_MARKER) == 1
    assert "First version." not in content
    assert "Second version." in content


def test_refuses_write_when_markers_are_damaged(sandbox):
    sandbox.claude_home.mkdir(parents=True, exist_ok=True)
    # Only the begin marker present — damaged.
    paths.claude_md_path().write_text(f"{writers.CLAUDE_MD_BEGIN_MARKER}\nstuff\n")
    result = writers.write_claude_md_block([{"title": "A", "body": "New content."}])
    assert result.success is False
    assert "damaged" in result.reason.lower()
    # Original content must be untouched.
    assert "New content." not in paths.claude_md_path().read_text()


def test_managed_block_respects_size_cap(sandbox):
    entries = [{"title": f"Rule {i}", "body": "x" * 100} for i in range(30)]
    block = writers.render_managed_block(entries, max_chars=500)
    inner = block.split(writers.CLAUDE_MD_BEGIN_MARKER)[1].split(writers.CLAUDE_MD_END_MARKER)[0]
    assert len(inner) < 700  # markers + a handful of entries, not all 30


def test_read_managed_block_body_roundtrips(sandbox):
    writers.write_claude_md_block([{"title": "A", "body": "Body text here."}])
    body = writers.read_managed_block_body()
    assert "Body text here." in body


def test_write_refused_on_secret_leftover(sandbox, monkeypatch):
    # Force validate to pass so the secret-scan gate is what's exercised.
    result = writers.write_claude_md_block(
        [{"title": "A", "body": "token: abcdefabcdef123456"}]
    )
    assert result.success is False
    assert "secret" in result.reason.lower()
    assert not paths.claude_md_path().exists() or "abcdefabcdef123456" not in paths.claude_md_path().read_text()


# --- rule files --------------------------------------------------------


def test_creates_pipeline_owned_rule_file(sandbox):
    path = paths.rules_dir() / "never-force-push.md"
    result = writers.write_rule_file(path, title="Never force push", body="Never force push main.")
    assert result.success
    content = path.read_text()
    assert content.startswith(writers.RULE_OWNERSHIP_MARKER)
    assert "Never force push main." in content


def test_updates_existing_pipeline_owned_rule(sandbox):
    path = paths.rules_dir() / "some-rule.md"
    writers.write_rule_file(path, title="A", body="First body.")
    result = writers.write_rule_file(path, title="A", body="Second body.")
    assert result.success
    content = path.read_text()
    assert "Second body." in content
    assert "First body." not in content


def test_refuses_to_overwrite_human_authored_rule(sandbox):
    path = paths.rules_dir() / "human-rule.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Human-written rule\n\nDo not touch this.\n")

    result = writers.write_rule_file(path, title="A", body="Pipeline content.")
    assert result.success is False
    assert "does not own" in result.reason
    assert "Do not touch this." in path.read_text()


# --- skill files -------------------------------------------------------


def test_creates_pipeline_owned_skill_file(sandbox):
    path = paths.skills_dir() / "split-refactors" / "SKILL.md"
    result = writers.write_skill_file(
        path, slug="split-refactors", title="Split large refactors",
        body="Split a large refactor into several small PRs.",
        canonical_key="split-large-refactors",
    )
    assert result.success
    content = path.read_text()
    assert content.startswith("---\n")
    assert "name: split-refactors" in content
    assert "managed_by: claude-selfimprove-pipeline" in content


def test_refuses_to_overwrite_human_authored_skill(sandbox):
    path = paths.skills_dir() / "human-skill" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\nname: human-skill\ndescription: hand written\n---\n\nDo not touch.\n")

    result = writers.write_skill_file(
        path, slug="human-skill", title="A", body="b", canonical_key="a-b",
    )
    assert result.success is False
    assert "Do not touch." in path.read_text()


# --- backup / rollback --------------------------------------------------


def test_backup_and_rollback_restores_prior_content(sandbox):
    path = paths.rules_dir() / "a-rule.md"
    writers.write_rule_file(path, title="A", body="Version one.")
    result2 = writers.write_rule_file(path, title="A", body="Version two.")
    assert "Version two." in path.read_text()

    rollback_result = writers.rollback_backup(result2.backup_meta_path)
    assert rollback_result.success
    assert "Version one." in path.read_text()


def test_rollback_of_first_write_deletes_the_file(sandbox):
    path = paths.rules_dir() / "brand-new.md"
    result = writers.write_rule_file(path, title="A", body="First ever.")
    assert path.exists()

    rollback_result = writers.rollback_backup(result.backup_meta_path)
    assert rollback_result.success
    assert not path.exists()


def test_list_backups_and_latest_backup_for(sandbox):
    path = paths.rules_dir() / "tracked.md"
    writers.write_rule_file(path, title="A", body="v1")
    writers.write_rule_file(path, title="A", body="v2")

    backups = writers.list_backups()
    assert len(backups) == 2
    latest = writers.latest_backup_for(path)
    assert latest is not None
    assert latest["target_path"] == str(path)


def test_atomic_write_never_leaves_partial_file_on_crash(sandbox, monkeypatch):
    import os as _os

    path = paths.rules_dir() / "crash-test.md"
    writers.write_rule_file(path, title="A", body="Good content.")

    original_replace = _os.replace
    calls = {"n": 0}

    def boom_on_second_call(*a, **kw):
        # First call is the backup metadata write; let it through so the
        # failure being tested is specifically the *target* write crashing.
        calls["n"] += 1
        if calls["n"] >= 2:
            raise OSError("simulated crash mid-write")
        return original_replace(*a, **kw)

    monkeypatch.setattr(_os, "replace", boom_on_second_call)
    result = writers.write_rule_file(path, title="A", body="Should never land.")
    assert result.success is False
    monkeypatch.setattr(_os, "replace", original_replace)
    # Original content must still be intact.
    assert "Good content." in path.read_text()
    assert "Should never land." not in path.read_text()
