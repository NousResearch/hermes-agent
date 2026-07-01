"""Profile-isolation regression tests for single-process multi-profile runtimes.

In runtimes that serve every profile from one OS process (the desktop
``tui_gateway``), the profile boundary is the context-local
``_HERMES_HOME_OVERRIDE`` ContextVar, not the process environment.  State that
escapes the request call stack — import-time-frozen path constants, direct
``os.environ`` reads, or worker threads that don't inherit the request context —
silently reverts to the launch/default profile and leaks one profile's data
into another.

These tests drive each previously-leaking site under override A then override B
with real temp HERMES_HOME directories (no mocks) and assert the *active*
profile's path is used.  They are the productionized form of the manual smoke
probes used to confirm the bug class.
"""

import threading
from pathlib import Path

import pytest

from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.fixture
def two_profiles(tmp_path):
    """Two distinct profile HERMES_HOME dirs with the dir skeleton created."""
    prof_a = tmp_path / "profA"
    prof_b = tmp_path / "profB"
    for p in (prof_a, prof_b):
        (p / "skills").mkdir(parents=True, exist_ok=True)
        (p / "state").mkdir(parents=True, exist_ok=True)
        (p / "cache").mkdir(parents=True, exist_ok=True)
    return prof_a, prof_b


def _under_override(home: Path, fn):
    """Run ``fn`` with the profile override set to ``home`` and reset after."""
    token = set_hermes_home_override(str(home))
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


# ---------------------------------------------------------------------------
# M1 — import-time path globals / direct os.environ reads
# ---------------------------------------------------------------------------

class TestSkillsHubPathResolution:
    """tools/skills_hub.py path constants must reflect the active profile."""

    def test_skills_dir_follows_override(self, two_profiles):
        prof_a, prof_b = two_profiles
        import tools.skills_hub as sh

        # Importing/touching under A must NOT pin the path for B.
        a_seen = _under_override(prof_a, lambda: Path(sh.SKILLS_DIR))
        b_seen = _under_override(prof_b, lambda: Path(sh.SKILLS_DIR))

        assert a_seen == prof_a / "skills"
        assert b_seen == prof_b / "skills"
        assert a_seen != b_seen

    def test_hub_derived_paths_follow_override(self, two_profiles):
        prof_a, prof_b = two_profiles
        import tools.skills_hub as sh

        b_lock = _under_override(prof_b, lambda: Path(sh.LOCK_FILE))
        b_audit = _under_override(prof_b, lambda: Path(sh.AUDIT_LOG))
        b_index = _under_override(prof_b, lambda: Path(sh.INDEX_CACHE_DIR))

        assert b_lock == prof_b / "skills" / ".hub" / "lock.json"
        assert b_audit == prof_b / "skills" / ".hub" / "audit.log"
        assert b_index == prof_b / "skills" / ".hub" / "index-cache"

    def test_lockfile_default_arg_resolves_active_profile(self, two_profiles):
        prof_a, prof_b = two_profiles
        from tools.skills_hub import HubLockFile, TapsManager

        lock_b = _under_override(prof_b, lambda: HubLockFile())
        taps_b = _under_override(prof_b, lambda: TapsManager())

        assert lock_b.path == prof_b / "skills" / ".hub" / "lock.json"
        assert taps_b.path == prof_b / "skills" / ".hub" / "taps.json"


class TestGatewayCacheDirResolution:
    """gateway/platforms/base.py cache getters must follow the active profile."""

    def test_image_cache_dir_follows_override(self, two_profiles):
        prof_a, prof_b = two_profiles
        import gateway.platforms.base as gb

        a_seen = _under_override(prof_a, lambda: gb.get_image_cache_dir())
        b_seen = _under_override(prof_b, lambda: gb.get_image_cache_dir())

        assert str(a_seen).startswith(str(prof_a))
        assert str(b_seen).startswith(str(prof_b))
        assert a_seen != b_seen

    def test_all_cache_getters_follow_override(self, two_profiles):
        _prof_a, prof_b = two_profiles
        import gateway.platforms.base as gb

        getters = (
            gb.get_image_cache_dir,
            gb.get_audio_cache_dir,
            gb.get_video_cache_dir,
            gb.get_document_cache_dir,
        )
        for getter in getters:
            seen = _under_override(prof_b, getter)
            assert str(seen).startswith(str(prof_b)), f"{getter.__name__} leaked: {seen}"

    def test_monkeypatched_constant_still_wins(self, two_profiles, monkeypatch, tmp_path):
        """The existing test seam (monkeypatch the module constant) is preserved."""
        _prof_a, _prof_b = two_profiles
        import gateway.platforms.base as gb

        forced = tmp_path / "forced_img"
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", forced)
        # Even with an active override, an explicit monkeypatch takes precedence.
        seen = _under_override(_prof_b, lambda: gb.get_image_cache_dir())
        assert seen == forced


class TestRichSentStorePathResolution:
    """gateway/rich_sent_store.py must honor the override, not read os.environ."""

    def test_store_path_follows_override(self, two_profiles, monkeypatch):
        prof_a, prof_b = two_profiles
        # Ensure no ambient HERMES_HOME env masks the test.
        monkeypatch.delenv("HERMES_HOME", raising=False)
        import gateway.rich_sent_store as rss

        b_seen = _under_override(prof_b, lambda: rss._store_path())
        assert b_seen.startswith(str(prof_b))
        assert b_seen.endswith("state/rich_sent_index.json")


class TestCheckpointManagerPathResolution:
    """tools/checkpoint_manager.py's checkpoint store root must honor the
    active profile — otherwise one profile's CheckpointManager instance can
    read/write code-edit checkpoints into a different profile's store under
    the multiplexed gateway."""

    def test_checkpoint_base_follows_override(self, two_profiles):
        prof_a, prof_b = two_profiles
        import tools.checkpoint_manager as cm

        a_seen = _under_override(prof_a, lambda: cm._resolve_checkpoint_base())
        b_seen = _under_override(prof_b, lambda: cm._resolve_checkpoint_base())

        assert a_seen == prof_a / "checkpoints"
        assert b_seen == prof_b / "checkpoints"
        assert a_seen != b_seen

    def test_store_path_follows_override(self, two_profiles):
        prof_a, prof_b = two_profiles
        import tools.checkpoint_manager as cm

        b_seen = _under_override(prof_b, lambda: cm._store_path())
        assert b_seen == prof_b / "checkpoints" / "store"

    def test_monkeypatched_constant_still_wins(self, two_profiles, monkeypatch, tmp_path):
        """The existing test seam (monkeypatch the module constant, see
        tests/tools/test_checkpoint_manager.py) is preserved."""
        _prof_a, prof_b = two_profiles
        import tools.checkpoint_manager as cm

        forced = tmp_path / "forced_checkpoints"
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", forced)
        seen = _under_override(prof_b, lambda: cm._resolve_checkpoint_base())
        assert seen == forced


class TestStickerCachePathResolution:
    """gateway/sticker_cache.py's cache file must honor the active profile —
    otherwise one profile's Telegram sticker-description cache leaks into a
    different profile's under the multiplexed gateway."""

    def test_cache_path_follows_override(self, two_profiles):
        prof_a, prof_b = two_profiles
        import gateway.sticker_cache as sc

        a_seen = _under_override(prof_a, lambda: sc._resolve_cache_path())
        b_seen = _under_override(prof_b, lambda: sc._resolve_cache_path())

        assert a_seen == prof_a / "sticker_cache.json"
        assert b_seen == prof_b / "sticker_cache.json"
        assert a_seen != b_seen

    def test_monkeypatched_constant_still_wins(self, two_profiles, monkeypatch, tmp_path):
        """The existing test seam (monkeypatch the module constant, see
        tests/gateway/test_sticker_cache.py) is preserved."""
        _prof_a, prof_b = two_profiles
        import gateway.sticker_cache as sc

        forced = tmp_path / "forced_sticker_cache.json"
        monkeypatch.setattr("gateway.sticker_cache.CACHE_PATH", forced)
        seen = _under_override(prof_b, lambda: sc._resolve_cache_path())
        assert seen == forced


# ---------------------------------------------------------------------------
# M2 — thread / executor context propagation
# ---------------------------------------------------------------------------

class TestThreadContextPropagation:
    """Worker threads must inherit the spawning turn's profile override."""

    def test_raw_thread_loses_override(self, two_profiles):
        """Document the underlying hazard: a bare thread does NOT inherit it."""
        _prof_a, prof_b = two_profiles
        seen = {}

        def worker():
            seen["home"] = str(get_hermes_home())

        def run():
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        _under_override(prof_b, run)
        # A bare thread falls back to the process default — this is WHY the fix
        # primitive is needed.  (Asserted as the hazard, not the desired state.)
        assert seen["home"] != str(prof_b)

    def test_propagate_primitive_preserves_override(self, two_profiles):
        _prof_a, prof_b = two_profiles
        from tools.thread_context import propagate_context_to_thread

        seen = {}

        def worker():
            seen["home"] = str(get_hermes_home())

        def run():
            t = threading.Thread(target=propagate_context_to_thread(worker))
            t.start()
            t.join()

        _under_override(prof_b, run)
        assert seen["home"] == str(prof_b)

    def test_run_async_worker_preserves_override(self, two_profiles):
        """model_tools._run_async's worker-thread branch must keep the override.

        This is the generic sync->async bridge for every async tool; if it
        leaks, every async tool that resolves get_hermes_home() leaks.
        """
        import asyncio

        _prof_a, prof_b = two_profiles
        import model_tools

        async def reads_home():
            return str(get_hermes_home())

        async def driver():
            # Inside a running loop, _run_async spawns a worker thread + loop.
            return model_tools._run_async(reads_home())

        seen = _under_override(prof_b, lambda: asyncio.run(driver()))
        assert seen == str(prof_b)
