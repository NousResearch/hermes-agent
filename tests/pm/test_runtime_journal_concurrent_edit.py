"""Recovery does not overwrite a config edited after interrupted publication."""
import pytest


def test_recovery_refuses_to_replace_newer_config(tmp_path, monkeypatch):
    from hermes_cli.runtime_state import recover_publication, runtime_lock
    from pm.publication import PluginSelection
    import pm.paths as paths
    from pm.environments import install_state_dir

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    config = tmp_path / "config.yaml"
    config.write_bytes(b"plugins:\n  enabled: [old]\n")
    with runtime_lock(repo):
        PluginSelection({"home": str(config.parent), "enabled": ["new"], "disabled": []}).publish(repo)
        config.write_bytes(config.read_bytes() + b"model: user-choice\n")
        changed = config.read_bytes()
        with pytest.raises(RuntimeError, match="config changed"):
            recover_publication(repo)
    assert config.read_bytes() == changed
    assert (install_state_dir(repo) / "publication.json").is_file()
