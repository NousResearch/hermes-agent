"""``hermes doctor`` names cache-root dirs that no pruner covers once they are big enough."""

from hermes_cli.doctor_state import _managed_cache_dirs, unpruned_cache_hogs


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


def test_unpruned_cache_hogs_skips_hermes_owned_pm_caches(tmp_path, monkeypatch):
    import hermes_constants
    from pm import paths

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)
    partials = tmp_path / "cache" / "managed-parts"
    monkeypatch.setattr(paths, "partials_root", lambda: partials)
    assert _managed_cache_dirs() == {tmp_path / "cache" / "uv", partials}

    cache = tmp_path / "cache"
    for name in ("uv", "managed-parts", "campaign-x"):
        (cache / name).mkdir(parents=True)
        with open(cache / name / "blob", "wb") as fh:
            fh.truncate(2048)
            fh.write(b"x" * 2048)

    assert unpruned_cache_hogs(tmp_path, min_bytes=1024) == [("campaign-x", 2048)]
