"""Regression tests for the _count_skills TTL cache (#53461).

_count_skills walks the whole skills/ tree with rglob. It is reached from
synchronous web-server handlers on the asyncio event-loop thread
(list_cron_jobs -> _cron_profile_dicts -> list_profiles), so a multi-second
scan on a large/slow filesystem stalls the loop, drops the WebSocket, and
restarts the gateway. The count is memoized with a short TTL so only the first
call in the window pays the scan.
"""

import time
from pathlib import Path

import hermes_cli.profiles as profiles


def _make_skill(skills_dir: Path, name: str) -> None:
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"# {name}\n")


class TestCountSkillsCache:
    def setup_method(self):
        profiles._skill_count_cache.clear()

    def test_counts_skills(self, tmp_path):
        skills = tmp_path / "skills"
        _make_skill(skills, "alpha")
        _make_skill(skills, "beta")
        assert profiles._count_skills(tmp_path) == 2

    def test_missing_dir_returns_zero(self, tmp_path):
        assert profiles._count_skills(tmp_path) == 0

    def test_repeat_call_does_not_rescan(self, tmp_path, monkeypatch):
        skills = tmp_path / "skills"
        _make_skill(skills, "alpha")

        # First call populates the cache and performs the rglob scan.
        assert profiles._count_skills(tmp_path) == 1

        # Now make rglob explode: a second call within the TTL must serve the
        # cached value WITHOUT touching the filesystem again.
        def _boom(*args, **kwargs):
            raise AssertionError("rglob called again within TTL — cache miss")

        monkeypatch.setattr(Path, "rglob", _boom)
        assert profiles._count_skills(tmp_path) == 1

    def test_cache_expires_after_ttl(self, tmp_path, monkeypatch):
        skills = tmp_path / "skills"
        _make_skill(skills, "alpha")

        fake_now = [1000.0]
        monkeypatch.setattr(profiles.time, "monotonic", lambda: fake_now[0])
        assert profiles._count_skills(tmp_path) == 1

        # Add a skill, advance past the TTL — the next call must rescan.
        _make_skill(skills, "beta")
        fake_now[0] += profiles._SKILL_COUNT_CACHE_TTL + 1
        assert profiles._count_skills(tmp_path) == 2
