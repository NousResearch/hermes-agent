"""`hermes skills untrust` must work for a project dir that no longer exists.

Trust is granted per project root and a git worktree is its own root, so the
normal lifecycle (create worktree -> trust -> finish task -> remove worktree)
leaves entries pointing at deleted directories. Those are exactly the ones a
user needs to remove, and the pre-flight `is_dir()` guard used to reject them
before the untrust branch ever ran.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.main_agent_cmds import _cmd_skills_trust


@pytest.fixture
def config_env(tmp_path, monkeypatch):
    """Isolate the config the command reads and writes.

    load_config must hand back a *copy*: the real one re-reads from disk, and
    aliasing it here would let save_config's clear() wipe the caller's own dict
    and make a broken untrust look like a successful one.
    """
    import copy

    store = {}

    def _load_config():
        return copy.deepcopy(store)

    def _save_config(data):
        store.clear()
        store.update(copy.deepcopy(data))

    monkeypatch.setattr("hermes_cli.config.load_config", _load_config)
    monkeypatch.setattr("hermes_cli.config.save_config", _save_config)
    return store


def _trusted(cfg):
    return (cfg.get("skills") or {}).get("trusted_project_dirs") or []


def _run(action, path):
    _cmd_skills_trust(SimpleNamespace(skills_action=action, path=str(path)))


class TestUntrustVanishedDir:
    def test_untrust_removes_an_entry_whose_directory_is_gone(
            self, tmp_path, config_env, capsys):
        gone = tmp_path / "retired-worktree"
        config_env["skills"] = {"trusted_project_dirs": [str(gone)]}
        assert not gone.exists()

        _run("untrust", gone)

        assert _trusted(config_env) == [], (
            "a trusted dir must be removable after the directory is deleted")
        assert "Not a directory" not in capsys.readouterr().out

    def test_untrust_keeps_other_entries(self, tmp_path, config_env):
        gone = tmp_path / "retired-worktree"
        live = tmp_path / "still-here"
        live.mkdir()
        config_env["skills"] = {"trusted_project_dirs": [str(gone), str(live)]}

        _run("untrust", gone)

        assert _trusted(config_env) == [str(live)]

    def test_untrust_still_works_for_an_existing_dir(self, tmp_path, config_env):
        live = tmp_path / "project"
        live.mkdir()
        config_env["skills"] = {"trusted_project_dirs": [str(live)]}

        _run("untrust", live)

        assert _trusted(config_env) == []

    def test_untrust_reports_when_nothing_matched(self, tmp_path, config_env, capsys):
        config_env["skills"] = {"trusted_project_dirs": [str(tmp_path / "other")]}

        _run("untrust", tmp_path / "never-trusted")

        assert "was not trusted" in capsys.readouterr().out
        assert len(_trusted(config_env)) == 1

    def test_trust_still_refuses_a_missing_directory(self, tmp_path, config_env, capsys):
        """The guard is correct for trust: don't trust what isn't there."""
        gone = tmp_path / "not-created"

        _run("trust", gone)

        assert "Not a directory" in capsys.readouterr().out
        assert _trusted(config_env) == []

    def test_untrust_matches_an_unresolvable_entry_by_raw_string(
            self, tmp_path, config_env, monkeypatch):
        """An entry under a dead mount can raise from resolve(); match literally."""
        dead = tmp_path / "dead-mount" / "project"
        config_env["skills"] = {"trusted_project_dirs": [str(dead)]}
        real_resolve = Path.resolve

        def _explode(self, *a, **kw):
            if str(self) == str(dead):
                raise OSError(5, "Input/output error")
            return real_resolve(self, *a, **kw)

        monkeypatch.setattr(Path, "resolve", _explode)

        _run("untrust", dead)

        assert _trusted(config_env) == []
