"""The release-tag picker reads origin, not the local checkout, and keeps CalVer order."""
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sandbox" / "pick-release-tags.sh"


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _pick(repo, *args):
    return subprocess.run(["bash", str(SCRIPT), *args, "--repo", str(repo)], check=True, capture_output=True, text=True).stdout.strip()


def _origin_with_tags(root, tags):
    origin = root / "origin.git"
    _git(root, "init", "--bare", "-b", "main", str(origin))
    seed = root / "seed"
    _git(root, "init", "-b", "main", str(seed))
    _git(seed, "config", "user.name", "fixture")
    _git(seed, "config", "user.email", "fixture@example.invalid")
    for tag in tags:
        _git(seed, "commit", "--allow-empty", "-m", tag)
        _git(seed, "tag", tag)
    _git(seed, "push", str(origin), "main", "refs/tags/*:refs/tags/*")
    return origin


def test_picking_reads_origin_not_local_tags(tmp_path):
    # CalVer names where version order is the chronology but lexicographic
    # order is not: v2026.4.13 sorts before v2026.4.3 as plain text.
    origin = _origin_with_tags(tmp_path, ["v2026.4.3", "v2026.4.13", "v2026.9.15"])
    # --no-local forces the transport, so the tagless clone truly lacks the
    # tag objects, not just their refs.
    _git(tmp_path, "clone", "--no-local", str(origin), str(tmp_path / "with-tags"))
    _git(tmp_path, "clone", "--no-local", "--no-tags", str(origin), str(tmp_path / "tagless"))
    assert _pick(tmp_path / "tagless", "--count", "2") == '["v2026.4.3","v2026.9.15"]'
    # Only the picked tags are materialized; unselected releases are not fetched.
    assert _git(tmp_path / "tagless", "tag", "--list").stdout.split() == ["v2026.4.3", "v2026.9.15"]
    expected = '["v2026.4.3","v2026.4.13","v2026.9.15"]'
    assert _pick(tmp_path / "tagless", "--count", "5") == expected
    assert _pick(tmp_path / "with-tags", "--count", "5") == expected


def test_picking_rejects_schemes_other_than_calver(tmp_path):
    origin = _origin_with_tags(tmp_path, ["v0.27.0"])
    _git(tmp_path, "clone", "--no-local", "--no-tags", str(origin), str(tmp_path / "tagless"))
    result = subprocess.run(["bash", str(SCRIPT), "--repo", str(tmp_path / "tagless")], check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "release tags found on origin" in result.stderr
