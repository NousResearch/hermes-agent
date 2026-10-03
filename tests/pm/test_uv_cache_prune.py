"""Pruning a uv cache to the lock never deletes a dist the lock resolves.

uv.lock records PEP 503-normalized names, while a wheel's dist-info keeps the project's own
spelling (``ruamel.yaml-0.18.16.dist-info``). A prune that compared them without collapsing
``.`` deleted the locked ruamel-yaml archive from every bundle payload, so offline venv
rebuilds had to download it.
"""

from __future__ import annotations

from pm.uv_cache_prune import prune_uv_cache_to_lock


def test_locked_dists_survive_whatever_spelling_their_wheel_uses(tmp_path):
    spellings = {"ruamel-yaml": "ruamel.yaml", "zope-interface": "zope.interface",
                 "typing-extensions": "typing_extensions", "pyyaml": "PyYAML"}
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text(
        "".join(f'[[package]]\nname = "{name}"\nversion = "1.0.0"\n\n' for name in spellings), encoding="utf-8")
    cache = tmp_path / "cache"
    for locked, spelled in spellings.items():
        (cache / "archive-v0" / locked / f"{spelled}-1.0.0.dist-info").mkdir(parents=True)
        (cache / "wheels-v5" / "pypi" / locked).mkdir(parents=True)
    stale = cache / "archive-v0" / "stale" / "left.pad-0.1.dist-info"
    stale.mkdir(parents=True)

    assert prune_uv_cache_to_lock(cache, repo) == 1
    assert not stale.exists()
    for locked in spellings:
        assert (cache / "archive-v0" / locked).is_dir()
        assert (cache / "wheels-v5" / "pypi" / locked).is_dir()
