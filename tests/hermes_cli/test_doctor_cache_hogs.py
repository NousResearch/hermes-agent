"""``hermes doctor`` names cache-root dirs that no pruner covers once they are big enough."""

from hermes_cli.doctor_state import unpruned_cache_hogs


def test_unpruned_cache_hogs_skips_pruned_dirs_and_small_entries(tmp_path):
    cache = tmp_path / "cache"
    for name in ("scratch", "terminal", "campaign-x", "web"):
        (cache / name).mkdir(parents=True)
        with open(cache / name / "blob", "wb") as fh:
            fh.truncate(2048)
            fh.write(b"x" * 2048)
    hogs = unpruned_cache_hogs(tmp_path, min_bytes=1024)
    assert {name for name, _ in hogs} == {"campaign-x", "web"}
    assert not unpruned_cache_hogs(tmp_path, min_bytes=1 << 20)
    assert all(size >= 2048 for _, size in hogs)


def test_unpruned_cache_hogs_skips_the_package_manager_store(tmp_path):
    """``cache/pnpm`` is pnpm's content-addressed store, not task residue.

    ``pm/packages.py::pnpm_env()`` pins store/cache/state there on purpose (Hermes-owned, shared by
    every profile); the directory grows to gigabytes partly *because* every install links from it.
    Doctor's "move task files under cache/scratch/<task>/ or delete it" advice therefore tells users
    to drop the one directory whose removal re-downloads the whole JS dependency tree, and moving it
    under ``cache/scratch/`` puts a machine-scoped cache where the 24h pruner eats it.
    """
    cache = tmp_path / "cache"
    for name in ("pnpm", "campaign-x"):
        (cache / name).mkdir(parents=True)
        with open(cache / name / "blob", "wb") as fh:
            fh.write(b"x" * 4096)
    assert [name for name, _ in unpruned_cache_hogs(tmp_path, min_bytes=1024)] == ["campaign-x"]
