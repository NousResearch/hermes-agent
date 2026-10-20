"""A failed uv-cache seed must not record completion, and links must stay links.

Regression for #141891: the seed copied with ``shutil.copytree`` (``symlinks=False``), so uv's
relative wheel links became directories that uv could not rename over ("Is a directory").

Regression: the seed copy ran inside ``except OSError: pass`` and wrote the ``.seeded`` marker
unconditionally, so a copy that died half-way was permanent — no later install retried it, and
every offline sync that needed a missing entry failed closed with "requested data wasn't found
in the cache".
"""

import os
import stat
from pathlib import Path

import pytest

from hermes_constants import get_default_hermes_root
from pm import packages
from pm import paths


def _symlink_or_skip(target: str, link: Path) -> None:
    """Create a payload link, or skip where the host cannot (Windows without Developer Mode)."""
    try:
        os.symlink(target, link)
    except OSError as exc:
        pytest.skip(f"host cannot create symlinks: {exc}")


@pytest.fixture
def payload_cache(tmp_path, monkeypatch):
    """A sealed payload shipping ``uv-cache/`` beside its store, and a machine cache that is empty."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    payload = tmp_path / "payload"
    entry = payload / "uv-cache" / "archive-v0" / "bucket"
    entry.mkdir(parents=True)
    (entry / "module.py").write_text("seeded = True\n", encoding="utf-8")
    store = payload / "tools"
    store.mkdir()
    monkeypatch.setattr(paths, "store_root", lambda: store)
    return get_default_hermes_root() / "cache" / "uv"


@pytest.fixture
def payload_cache_with_wheel_link(payload_cache):
    """The same payload with uv's wheel shape: a relative link into ``archive-v0/``.

    Creating a symlink needs Developer Mode or elevation on Windows; the link
    tests skip there instead of failing the unrelated seed regressions above.
    """
    wheel = paths.store_root().parent / "uv-cache" / "wheels-v6" / "pypi" / "pkg"
    wheel.mkdir(parents=True)
    _symlink_or_skip("../../../archive-v0/bucket", wheel / "1.0-abc123")
    (wheel / "1.0-abc123.http").write_text("x", encoding="utf-8")
    return payload_cache


def _seeded_link(machine: Path) -> Path:
    return machine / "wheels-v6" / "pypi" / "pkg" / "1.0-abc123"


def test_update_merges_the_new_payload_into_an_existing_seed(payload_cache, tmp_path, monkeypatch):
    """An update ships a payload cache for a new lock; its wheels must reach the machine cache.

    Regression: ``.seeded`` meant "done forever", so an updated install never received the new
    pins' wheels and plugin rebuilds fetched them from the index.
    """
    code = tmp_path / "code"
    code.mkdir()
    (code / "uv.lock").write_text("version = 1  # old\n", encoding="utf-8")
    monkeypatch.setattr(paths, "repo_root", lambda: code)
    packages.uv_cache_dir()

    added = paths.store_root().parent / "uv-cache" / "archive-v0" / "bucket" / "new.py"
    added.write_text("new pin\n", encoding="utf-8")
    (code / "uv.lock").write_text("version = 1  # updated\n", encoding="utf-8")
    packages.uv_cache_dir()
    assert (payload_cache / "archive-v0" / "bucket" / "new.py").read_text(encoding="utf-8-sig") == "new pin\n"


def test_partial_seed_is_retried_on_the_next_install(payload_cache, monkeypatch):
    real_seed_tree = packages._seed_tree
    attempts: list[int] = []

    def flaky(*args, **kwargs):
        """Fails the first bucket only; the seed recurses through this same module attribute."""
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("no space left on device")
        return real_seed_tree(*args, **kwargs)

    monkeypatch.setattr(packages, "_seed_tree", flaky)
    packages.uv_cache_dir()
    assert not (payload_cache / ".seeded").is_file(), "a failed seed recorded completion"

    packages.uv_cache_dir()
    assert (payload_cache / ".seeded").is_file()
    assert (payload_cache / "archive-v0" / "bucket" / "module.py").read_text(encoding="utf-8") == "seeded = True\n"


def test_retry_finishes_a_seed_that_died_inside_a_directory(payload_cache, monkeypatch):
    """A copy that already created ``archive-v0/`` left the retry nothing to do: it skipped the
    existing top-level directory and recorded completion over the hole, so uv later failed with
    "Missing .dist-info directory" for the bucket the copy never finished."""
    dist_info = paths.store_root().parent / "uv-cache" / "archive-v0" / "bucket" / "pkg-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: pkg\n", encoding="utf-8")
    real_seed_tree = packages._seed_tree
    interrupted: list[str] = []

    def dies_at_dist_info(src, dst, *args, **kwargs):
        if os.fspath(src).endswith(".dist-info") and not interrupted:
            interrupted.append(os.fspath(src))
            raise OSError("copy interrupted")
        return real_seed_tree(src, dst, *args, **kwargs)

    monkeypatch.setattr(packages, "_seed_tree", dies_at_dist_info)
    packages.uv_cache_dir()
    bucket = payload_cache / "archive-v0" / "bucket"
    # Directory order is arbitrary: the interrupted bucket may or may not have
    # copied module.py; what matters is that the hole is not recorded as done.
    assert not (bucket / "pkg-1.0.dist-info").exists()
    assert not (payload_cache / ".seeded").is_file()

    packages.uv_cache_dir()
    assert (bucket / "module.py").read_text(encoding="utf-8") == "seeded = True\n"
    assert (bucket / "pkg-1.0.dist-info" / "METADATA").read_text(encoding="utf-8") == "Name: pkg\n"
    assert (payload_cache / ".seeded").is_file()


def test_retry_recopies_a_file_the_interrupted_seed_left_short(payload_cache):
    short = payload_cache / "archive-v0" / "bucket" / "module.py"
    short.parent.mkdir(parents=True)
    short.write_text("seed", encoding="utf-8")

    packages.uv_cache_dir()
    assert short.read_text(encoding="utf-8") == "seeded = True\n"
    assert (payload_cache / ".seeded").is_file()


def test_marker_from_a_seedless_image_does_not_block_a_later_seed(payload_cache, tmp_path, monkeypatch):
    """An image without ``uv-cache/`` still writes ``.seeded`` for the current lock. The next image
    ships the seed with the same lock, and its wheels must still reach the volume."""
    code = tmp_path / "code"
    code.mkdir()
    (code / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(paths, "repo_root", lambda: code)
    seed = paths.store_root().parent / "uv-cache"
    seedless = tmp_path / "uv-cache-hidden"
    seed.rename(seedless)
    packages.uv_cache_dir()
    assert (payload_cache / ".seeded").is_file()

    seedless.rename(seed)
    packages.uv_cache_dir()
    assert (payload_cache / "archive-v0" / "bucket" / "module.py").is_file()


def test_seed_preserves_a_payload_wheel_link(payload_cache_with_wheel_link):
    """The payload's wheel entries are links into ``archive-v0/``; the merge must recreate them.

    Regression: ``shutil.copytree`` without ``symlinks=True`` follows the link, so the machine
    cache got a directory holding a second copy of the archive. uv publishes a cache wheel with
    ``rename()``, which then failed with ``Is a directory (os error 21)``.
    """
    machine = payload_cache_with_wheel_link
    packages.uv_cache_dir()

    link = _seeded_link(machine)
    assert link.is_symlink(), "the seed followed the payload link instead of recreating it"
    assert os.readlink(link) == "../../../archive-v0/bucket"
    assert stat.S_ISLNK(os.lstat(link).st_mode)
    # A relative target must stay relative: the destination mirrors the payload's depth.
    assert Path(link).resolve() == Path(machine / "archive-v0" / "bucket").resolve()
    assert (machine / "wheels-v6" / "pypi" / "pkg" / "1.0-abc123.http").is_file()


def test_seed_preserves_a_link_at_the_payload_root(payload_cache):
    """The walker checks ``is_symlink()`` first at every depth, including the payload root."""
    payload = paths.store_root().parent / "uv-cache"
    _symlink_or_skip("archive-v0/bucket", payload / "linked-bucket")

    packages.uv_cache_dir()

    link = payload_cache / "linked-bucket"
    assert link.is_symlink()
    assert os.readlink(link) == "archive-v0/bucket"
    assert (link / "module.py").read_text(encoding="utf-8") == "seeded = True\n"


def test_seed_heals_a_directory_left_by_the_old_copy(payload_cache_with_wheel_link):
    """A volume poisoned by the pre-fix seed holds a directory where the link belongs."""
    machine = payload_cache_with_wheel_link
    poisoned = _seeded_link(machine)
    poisoned.mkdir(parents=True)
    (poisoned / "module.py").write_text("stale copy\n", encoding="utf-8")

    packages.uv_cache_dir()

    assert poisoned.is_symlink()
    assert os.readlink(poisoned) == "../../../archive-v0/bucket"
    # The stale directory is gone: the path now resolves to the seeded archive.
    assert (poisoned / "module.py").read_text(encoding="utf-8") == "seeded = True\n"


def test_reseeding_over_an_existing_good_link_is_idempotent(payload_cache_with_wheel_link):
    """``symlinks=True`` alone fails here: ``os.symlink`` refuses to overwrite, so a re-seed over
    an existing link raised EEXIST — and any new lock overlaps the previous payload heavily."""
    machine = payload_cache_with_wheel_link
    packages.uv_cache_dir()
    link = _seeded_link(machine)
    (machine / ".seeded").write_text("stale", encoding="utf-8")

    packages.uv_cache_dir()

    assert link.is_symlink()
    assert os.readlink(link) == "../../../archive-v0/bucket"


def test_payload2_marker_reseeds_a_volume_marked_by_the_old_seed(payload_cache_with_wheel_link):
    """The suffix bump is what heals volumes seeded by the buggy copy: their marker reads
    ``+payload``, so the fixed seed must run again instead of trusting it."""
    machine = payload_cache_with_wheel_link
    poisoned = _seeded_link(machine)
    poisoned.mkdir(parents=True)
    (poisoned / "module.py").write_text("stale copy\n", encoding="utf-8")
    (machine / ".seeded").write_text("stale+payload", encoding="utf-8")

    packages.uv_cache_dir()

    assert poisoned.is_symlink()
    assert (machine / ".seeded").read_text(encoding="utf-8").endswith("+payload2")
