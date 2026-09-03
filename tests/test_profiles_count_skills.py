"""Regression test for ``Profile._count_skills`` tolerating OSError mid-walk.

Reproduces a Windows race: ``Path.rglob('SKILL.md')`` collects a list of
directories to walk, then ``os.scandir`` opens each one. If a directory is
removed between collection and the scandir call (manual prune, or another
process mid-update), Windows raises ``FileNotFoundError`` instead of
silently skipping. Before the fix, that exception escaped ``_count_skills``
and broke ``list_profiles()`` — every caller (including
``_pause_windows_gateways_for_update``) inherited the crash and ``hermes
update`` aborted before the dep install could run.

The fix wraps the rglob walk in ``try/except OSError`` and returns ``0``
on failure. We exercise the fix's contract directly by making the walk
raise OSError, asserting it is swallowed. Reproducing the underlying race
deterministically across platforms is impractical (the bug only fires on
Windows during a microsecond-scale TOCTOU window), so this test pins the
fix's behaviour rather than the race itself.
"""

import errno
from pathlib import Path

import pytest

from hermes_cli.profiles import _SKILL_COUNT_CACHE, _count_skills


@pytest.fixture
def skills_tree(tmp_path: Path) -> Path:
    profile = tmp_path / "profile"
    (profile / "skills" / "real-skill").mkdir(parents=True)
    (profile / "skills" / "real-skill" / "SKILL.md").write_text("# real\n")
    return profile


def test_count_skills_baseline(skills_tree: Path) -> None:
    """Sanity: with a single skill present, the count is 1."""
    _SKILL_COUNT_CACHE.clear()
    assert _count_skills(skills_tree) == 1


def test_count_skills_returns_zero_when_walk_raises_oserror(
    skills_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the underlying walk raises OSError (Windows mid-walk race for a
    removed skill dir), ``_count_skills`` must return 0 — not propagate the
    exception, which would break every caller including ``list_profiles``."""
    real_rglob = Path.rglob

    def raising_rglob(self: Path, pattern: str):
        # First call (Path.rglob in _count_skills) raises; later calls (e.g.
        # for cache invalidation elsewhere) fall through to the real impl.
        if pattern == "SKILL.md" and self.name == "skills":
            raise OSError(errno.ENOENT, "No such file or directory", str(self))
        return real_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", raising_rglob)
    _SKILL_COUNT_CACHE.clear()

    assert _count_skills(skills_tree) == 0