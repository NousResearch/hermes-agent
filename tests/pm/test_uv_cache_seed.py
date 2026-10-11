"""A failed uv-cache seed must not record completion.

Regression: the seed copy ran inside ``except OSError: pass`` and wrote the ``.seeded`` marker
unconditionally, so a copy that died half-way was permanent — no later install retried it, and
every offline sync that needed a missing entry failed closed with "requested data wasn't found
in the cache".
"""

import os
import shutil

import pytest

from hermes_constants import get_default_hermes_root
from pm import packages
from pm import paths


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
    real_copytree = shutil.copytree
    attempts: list[int] = []

    def flaky(*args, **kwargs):
        """Fails the first copy only; shutil recurses through this same module attribute."""
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("no space left on device")
        return real_copytree(*args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", flaky)
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
    real_copytree = shutil.copytree
    interrupted: list[str] = []

    def dies_at_dist_info(src, dst, *args, **kwargs):
        if os.fspath(src).endswith(".dist-info") and not interrupted:
            interrupted.append(os.fspath(src))
            raise OSError("copy interrupted")
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", dies_at_dist_info)
    packages.uv_cache_dir()
    bucket = payload_cache / "archive-v0" / "bucket"
    assert (bucket / "module.py").is_file() and not (bucket / "pkg-1.0.dist-info").exists()
    assert not (payload_cache / ".seeded").is_file()

    packages.uv_cache_dir()
    assert (bucket / "pkg-1.0.dist-info" / "METADATA").read_text(encoding="utf-8") == "Name: pkg\n"
    assert (payload_cache / ".seeded").is_file()


def test_retry_recopies_a_file_the_interrupted_seed_left_short(payload_cache):
    short = payload_cache / "archive-v0" / "bucket" / "module.py"
    short.parent.mkdir(parents=True)
    short.write_text("seed", encoding="utf-8")

    packages.uv_cache_dir()
    assert short.read_text(encoding="utf-8") == "seeded = True\n"
    assert (payload_cache / ".seeded").is_file()
