"""A failed uv-cache seed must not record completion.

Regression: the seed copy ran inside ``except OSError: pass`` and wrote the ``.seeded`` marker
unconditionally, so a copy that died half-way was permanent — no later install retried it, and
every offline sync that needed a missing entry failed closed with "requested data wasn't found
in the cache".
"""

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
