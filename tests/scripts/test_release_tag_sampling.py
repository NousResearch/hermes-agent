"""Release sampling follows tag chronology across CalVer and SemVer, local tags or not."""
import os
import subprocess

from scripts.releases.pick_tags import pick_tags


def _git(cwd, *args, env=None):
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True)


def _seed_origin(root):
    # Origin carrying three release tags created in CalVer-then-SemVer order:
    # the names sort the other way round, so the assertions below pin
    # chronology, not lexicography. v0.27.0 is annotated to cover both tag
    # object kinds the sampling may have to date-sort.
    origin = root / "origin.git"
    _git(root, "init", "--bare", "-b", "main", str(origin))
    seed = root / "seed"
    _git(root, "init", "-b", "main", str(seed))
    _git(seed, "config", "user.name", "fixture")
    _git(seed, "config", "user.email", "fixture@example.invalid")
    for index, tag in enumerate(["v2026.7.20", "v0.27.0", "v0.28.0"]):
        date = f"2026-08-{index + 1:02d}T00:00:00Z"
        env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
        _git(seed, "commit", "--allow-empty", "-m", tag, env=env)
        if index == 1:
            _git(seed, "tag", "-a", "-m", "annotated", tag, env=env)
        else:
            _git(seed, "tag", tag, env=env)
    _git(seed, "push", str(origin), "main", "refs/tags/*:refs/tags/*")


def test_sampling_reads_origin_not_local_tags(tmp_path):
    _seed_origin(tmp_path)
    # --no-local forces the transport, so the tagless clone truly lacks the
    # tag objects, not just their refs.
    _git(tmp_path, "clone", "--no-local", str(tmp_path / "origin.git"), str(tmp_path / "with-tags"))
    _git(tmp_path, "clone", "--no-local", "--no-tags", str(tmp_path / "origin.git"), str(tmp_path / "tagless"))
    expected = ["v2026.7.20", "v0.27.0"]
    assert pick_tags(tmp_path / "with-tags", 3, "v0.28.0") == expected
    assert pick_tags(tmp_path / "tagless", 3, "v0.28.0") == expected
    assert pick_tags(tmp_path / "tagless", 1, "v0.28.0") == ["v0.27.0"]
