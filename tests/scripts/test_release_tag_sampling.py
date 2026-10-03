"""Release sampling follows tag chronology across CalVer and SemVer."""
import os
import subprocess

from scripts.releases.pick_tags import pick_tags


def test_sampling_includes_newest_stable_and_excludes_candidate(tmp_path):
    def git(*args, env=None):
        return subprocess.run(["git", *args], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)

    git("init", "-b", "main")
    git("config", "user.name", "fixture")
    git("config", "user.email", "fixture@example.invalid")
    for index, tag in enumerate(["v2026.7.20", "v0.27.0", "v0.28.0"]):
        env = {**os.environ, "GIT_AUTHOR_DATE": f"2026-08-{index + 1:02d}T00:00:00Z", "GIT_COMMITTER_DATE": f"2026-08-{index + 1:02d}T00:00:00Z"}
        git("commit", "--allow-empty", "-m", tag, env=env)
        git("tag", tag)
    assert pick_tags(tmp_path, 1, "v0.28.0") == ["v0.27.0"]
    assert pick_tags(tmp_path, 3, "v0.28.0") == ["v2026.7.20", "v0.27.0"]


def test_sampling_accepts_historical_canary_but_ignores_head_tag(tmp_path):
    def git(*args, env=None):
        return subprocess.run(["git", *args], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)

    git("init", "-b", "main")
    git("config", "user.name", "fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("commit", "--allow-empty", "-m", "old")
    git("tag", "v0.21.4+canary.20261001T120000Z")
    git("commit", "--allow-empty", "-m", "head")
    git("tag", "v0.21.4+canary.20261002T120000Z")
    assert pick_tags(tmp_path, 3) == ["v0.21.4+canary.20261001T120000Z"]


def test_sampling_returns_empty_when_only_tag_points_at_head(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True)

    git("init", "-b", "main")
    git("config", "user.name", "fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("commit", "--allow-empty", "-m", "head")
    git("tag", "v0.21.4+canary.20261002T120000Z")
    assert pick_tags(tmp_path, 3) == []
