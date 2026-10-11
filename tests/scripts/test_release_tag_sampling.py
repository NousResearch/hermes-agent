"""Release sampling follows tag chronology across CalVer and SemVer."""
import json
import os
import subprocess
from pathlib import Path

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


def test_sandbox_picker_skips_unresolvable_tags_instead_of_truncating(tmp_path):
    """A tag whose object is missing must skip itself, not truncate the list.

    pick-release-tags.sh resolves each tag inside a pipeline subshell under
    ``set -euo pipefail``; before the guard, the first unresolvable tag failed
    its command substitution, errexit killed the subshell, and every tag after
    it was dropped silently (exit 0) -- a run could pass with a truncated
    baseline set.
    """
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True)

    git("init", "-b", "main")
    git("config", "user.name", "fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("commit", "--allow-empty", "-m", "old")
    git("tag", "v2026.1.0")
    git("tag", "v0.21.4+canary.20260101T000000Z")
    git("commit", "--allow-empty", "-m", "middle")
    git("tag", "v2026.2.0")
    git("commit", "--allow-empty", "-m", "head")
    # The unresolvable tag sorts between the valid canary and the stable tags:
    # pre-guard, the loop died here and both stable tags were silently lost.
    broken = tmp_path / ".git" / "refs" / "tags" / "v0.21.4+canary.20261004T000000Z"
    broken.write_text("de" * 20 + "\n", encoding="utf-8")

    script = Path(__file__).resolve().parents[2] / "scripts" / "sandbox" / "pick-release-tags.sh"
    done = subprocess.run(["bash", str(script), "--repo", str(tmp_path), "--count", "10"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [
        "v0.21.4+canary.20260101T000000Z",
        "v2026.1.0",
        "v2026.2.0",
    ]
