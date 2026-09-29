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
