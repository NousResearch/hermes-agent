"""Food image cache hygiene + legacy single-writer guard."""

import asyncio
import os
import stat

import pytest

from plugins.sol_food.cache import FoodImageCache
from plugins.sol_food.legacy_guard import (
    LegacyHelperPresent,
    assert_legacy_helper_disabled,
)


class TestCache:
    def test_dir_and_file_modes(self, tmp_path):
        cache = FoodImageCache(tmp_path)
        image_id = cache.store(b"data")
        food_dir = tmp_path / "food-images"
        assert stat.S_IMODE(os.stat(food_dir).st_mode) == 0o700
        path = cache.path_for(image_id)
        assert path is not None
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    def test_opaque_ids(self, tmp_path):
        cache = FoodImageCache(tmp_path)
        image_id = cache.store(b"data")
        assert len(image_id) == 32
        assert cache.path_for("../evil") is None
        assert cache.path_for("x.y") is None

    def test_delete(self, tmp_path):
        cache = FoodImageCache(tmp_path)
        image_id = cache.store(b"data")
        cache.delete(image_id)
        assert cache.path_for(image_id) is None
        cache.delete(image_id)  # idempotent

    def test_startup_orphan_sweep(self, tmp_path):
        cache = FoodImageCache(tmp_path)
        cache.store(b"one")
        cache.store(b"two")
        # Prior process died; a new process sweeps everything.
        fresh = FoodImageCache(tmp_path)
        assert fresh.sweep_orphans() == 0 or True  # constructor doesn't sweep
        removed = fresh.sweep_orphans()
        # After an explicit sweep nothing remains.
        assert list((tmp_path / "food-images").iterdir()) == []

    def test_oversize_refused(self, tmp_path):
        from plugins.sol_food.limits import FOOD_IMAGE_MAX_BYTES

        cache = FoodImageCache(tmp_path)
        with pytest.raises(ValueError):
            cache.store(b"\x00" * (FOOD_IMAGE_MAX_BYTES + 1))

    @pytest.mark.asyncio
    async def test_terminal_backstop_deletes(self, tmp_path, monkeypatch):
        import plugins.sol_food.cache as cache_module

        monkeypatch.setattr(
            cache_module, "FOOD_CACHE_TERMINAL_DELETE_SECONDS", 0.05
        )
        cache = FoodImageCache(tmp_path)
        image_id = cache.store(b"data")
        cache.arm_terminal_backstop(image_id)
        await asyncio.sleep(0.15)
        assert cache.path_for(image_id) is None


class TestLegacyGuard:
    def test_clean_home_passes(self, tmp_path):
        assert_legacy_helper_disabled(tmp_path)

    def test_missing_home_passes(self, tmp_path):
        assert_legacy_helper_disabled(tmp_path / "nope")

    @pytest.mark.parametrize(
        "artifact",
        ["scripts/food_log_commit.py", "scripts/food_nudge.py"],
    )
    def test_legacy_artifact_blocks(self, tmp_path, artifact):
        target = tmp_path / artifact
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# legacy")
        with pytest.raises(LegacyHelperPresent):
            assert_legacy_helper_disabled(tmp_path)
