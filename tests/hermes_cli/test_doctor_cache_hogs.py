"""``hermes doctor`` names cache-root dirs that no pruner covers once they are big enough."""

import pytest

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


@pytest.mark.platforms("windows")
def test_unpruned_cache_hogs_does_not_bill_junction_targets(tmp_path):
    """A junction is measured as the link it is (#156487): pnpm's ``store/v11/projects/<hash>``
    junction to the install dir must not bill the install to ``cache/pnpm/``, and a cache-root
    junction is not a cache dir of its own."""
    import _winapi

    install = tmp_path / "install"
    install.mkdir()
    (install / "big.bin").write_bytes(b"x" * (1 << 20))
    store = tmp_path / "home" / "cache" / "pnpm" / "store" / "v11"
    (store / "files").mkdir(parents=True)
    (store / "files" / "pkg").write_bytes(b"x" * 4096)
    (store / "projects").mkdir()
    _winapi.CreateJunction(str(install), str(store / "projects" / "abc123"))
    _winapi.CreateJunction(str(install), str(tmp_path / "home" / "cache" / "linked"))

    assert unpruned_cache_hogs(tmp_path / "home", min_bytes=1024) == [("pnpm", 4096)]
