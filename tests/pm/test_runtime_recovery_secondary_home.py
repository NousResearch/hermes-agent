"""Crash recovery validates recorded paths against every dependency home.

A journal written for a profile that resolves outside the default root (``profiles/<name>``
symlinked to another volume) used to fail recovery with "outside Hermes state", leaving the
interrupted publication unrecoverable. A path outside every dependency home is still refused.
"""
import base64
import hashlib
import json

import pytest

from hermes_cli.runtime_state import recover_publication, runtime_lock
from pm.environments import install_state_dir

pytestmark = pytest.mark.platforms("linux", "macos")

ORIGINAL = b"plugins:\n  enabled: [old]\n"
PUBLISHED = b"plugins:\n  enabled: [new]\n"


def _symlinked_profile(tmp_path, monkeypatch):
    """A live profile whose real directory is outside the default home."""
    home = tmp_path / "home"
    (home / "profiles").mkdir(parents=True)
    external = tmp_path / "external" / "worker"
    (external / "plugins").mkdir(parents=True)
    (external / "config.yaml").write_bytes(PUBLISHED)  # also the profile's identity marker
    profile = home / "profiles" / "worker"
    profile.symlink_to(external, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return profile


def _journal(tmp_path, row):
    state = install_state_dir(tmp_path / "repo")
    state.mkdir(parents=True)
    journal = state / "publication.json"
    journal.write_text(json.dumps(row))
    return journal


def test_config_publication_rolls_back_in_a_secondary_home(tmp_path, monkeypatch):
    profile = _symlinked_profile(tmp_path, monkeypatch)
    journal = _journal(tmp_path, {
        "config": str(profile / "config.yaml"),
        "previous": base64.b64encode(ORIGINAL).decode(),
        "facts_before": None,
        "config_after": hashlib.sha256(PUBLISHED).hexdigest(),
    })

    with runtime_lock(tmp_path / "repo"):
        recover_publication(tmp_path / "repo")

    assert (profile / "config.yaml").read_bytes() == ORIGINAL
    assert not journal.exists()


def test_committed_plugin_publication_recovers_in_a_secondary_home(tmp_path, monkeypatch):
    profile = _symlinked_profile(tmp_path, monkeypatch)
    target = profile / "plugins" / "probe"
    backup = target.parent / ".previous-probe"
    backup.mkdir()
    (backup / "plugin.yaml").write_text("name: probe\n", encoding="utf-8")
    journal = _journal(tmp_path, {
        "kind": "plugin", "committed": True, "facts_before": None,
        "target": str(target), "backup": str(backup),
        "metadata": str(target.parent / ".install-metadata.json"),
    })

    with runtime_lock(tmp_path / "repo"):
        recover_publication(tmp_path / "repo")

    assert not backup.exists()
    assert not journal.exists()


def test_config_outside_every_home_is_still_refused(tmp_path, monkeypatch):
    _symlinked_profile(tmp_path, monkeypatch)
    outside = tmp_path / "outside" / "config.yaml"
    outside.parent.mkdir()
    outside.write_bytes(PUBLISHED)
    journal = _journal(tmp_path, {
        "config": str(outside),
        "previous": base64.b64encode(ORIGINAL).decode(),
        "facts_before": None,
        "config_after": hashlib.sha256(PUBLISHED).hexdigest(),
    })

    with runtime_lock(tmp_path / "repo"), pytest.raises(RuntimeError, match="outside Hermes state"):
        recover_publication(tmp_path / "repo")

    assert outside.read_bytes() == PUBLISHED
    assert journal.exists()
