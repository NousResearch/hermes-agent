"""pm.environments: dependency-environment selection layout.

Regression for #126816: ``payload_venv()`` must never treat a live git checkout as a
sealed payload -- see the docstring on ``payload_venv`` for the full mechanism.
"""
from __future__ import annotations

import json

from pm.environments import payload_venv


def test_git_checkout_is_never_treated_as_a_sealed_payload(tmp_path):
    """A ``.git``-bearing tree short-circuits before the manifest read.

    Without the ``.git`` check, this checkout would read ``tmp_path / "manifest.json"``
    -- a sibling file that has nothing to do with the checkout, yet exists here on purpose
    to prove the short-circuit fires before that read, not merely because the manifest
    is absent (issue #126816: on Hermes's own documented default install layout,
    ``$HERMES_HOME/hermes-agent``, that sibling path is the real home's real file).
    """
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    (repo / ".git").mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"repo": "hermes-agent", "venv": "some-payload-venv"})
    )

    assert payload_venv(repo) is None


def test_sealed_payload_without_git_still_resolves_its_venv(tmp_path):
    """Control case: the pre-existing sealed-payload contract is untouched.

    Same manifest layout as the regression test above, minus ``.git`` -- proves the new
    check is additive (skips checkouts) rather than a change to sealed-payload resolution.
    """
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    venv_dir = tmp_path / "some-payload-venv"
    venv_dir.mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"repo": "hermes-agent", "venv": "some-payload-venv"})
    )

    assert payload_venv(repo) == venv_dir


def _bare_store_python(monkeypatch, tmp_path):
    """Make this process look like PM's store Python (no venv) under a sandboxed home."""
    import sys

    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)


def test_checkout_boot_never_reads_the_sibling_manifest(tmp_path, monkeypatch):
    """``activate_dependencies`` on a checkout must not consult ``<parent>/manifest.json``.

    ``payload_venv`` short-circuits for a ``.git`` tree, but ``activate_dependencies`` then falls
    through to ``_require_own_dependencies`` -> ``store_root``, which re-derives the same sibling
    path. A manifest without a ``store`` key is valid for ``payload_venv`` (it only needs
    ``repo`` + ``venv``), so reading it here raised ``KeyError`` at import time on a bare store
    Python -- a crash the original ``payload_venv`` change introduced for checkouts only.
    """
    from pm.environments import activate_dependencies

    _bare_store_python(monkeypatch, tmp_path)
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    (repo / ".git").mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"repo": "hermes-agent", "venv": "some-payload-venv"})
    )

    activate_dependencies(repo)


def test_checkout_store_root_ignores_a_manifest_that_names_it(tmp_path, monkeypatch):
    """A checkout's store is never taken from a sibling manifest, even a complete one."""
    from pm.environments import store_root

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    (repo / ".git").mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"repo": "hermes-agent", "venv": "v", "store": "sealed-store"})
    )

    assert store_root(repo) != (tmp_path / "sealed-store").resolve()


def test_sealed_payload_store_root_still_follows_its_manifest(tmp_path, monkeypatch):
    """Control: without ``.git`` the manifest's ``store`` is still honoured."""
    from pm.environments import store_root

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"repo": "hermes-agent", "venv": "v", "store": "sealed-store"})
    )

    assert store_root(repo) == (tmp_path / "sealed-store").resolve()
