"""Recovery refuses corrupted metadata and never overwrites unrelated config."""
import json

import pytest

from pm.environments import install_state_dir
from hermes_cli.runtime_state import recover_publication, runtime_lock


def test_invalid_journal_cannot_write_outside_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    outside = tmp_path / "outside" / "config.yaml"
    outside.parent.mkdir()
    outside.write_text("untouched")
    state = install_state_dir(repo)
    state.mkdir(parents=True)
    journal = state / "publication.json"
    journal.write_text(json.dumps({"config": str(outside), "previous": "eA==", "facts_before": None}))
    with runtime_lock(repo), pytest.raises(RuntimeError, match="outside Hermes state"):
        recover_publication(repo)
    assert outside.read_text() == "untouched"
    assert journal.exists()


@pytest.mark.require_symlinks
def test_rollback_of_symlinked_home_config_writes_through_the_link(tmp_path, monkeypatch):
    """A home config.yaml linked into a dotfiles tree is restored in place: the link survives."""
    import base64
    import hashlib

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    tracked = tmp_path / "dotfiles" / "config.yaml"
    tracked.parent.mkdir()
    tracked.write_text("proposed")
    (home / "config.yaml").symlink_to(tracked)
    repo = tmp_path / "repo"
    state = install_state_dir(repo)
    state.mkdir(parents=True)
    (state / "publication.json").write_text(json.dumps({
        "config": str(home / "config.yaml"), "previous": base64.b64encode(b"original").decode(),
        "facts_before": None, "config_after": hashlib.sha256(b"proposed").hexdigest(),
    }))
    with runtime_lock(repo):
        recover_publication(repo)
    assert (home / "config.yaml").is_symlink()
    assert tracked.read_text() == "original"
