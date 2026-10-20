"""A bare ``hermes skills trust|untrust`` must resolve from the shell's cwd.

The subcommand runs in the user's shell, so "the current directory" is where
the user stands. ``find_project_root()`` without a start resolves the agent's
effective cwd instead (session cwd, then a config-pinned ``TERMINAL_CWD``,
then the process cwd) — a ``terminal.cwd`` pinned outside the repo (e.g. the
Nix module defaulting it to ``$HOME``) made the bare subcommand walk from
outside the repo and report "Not inside a git checkout" (#127025).
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.main_agent_cmds import _cmd_skills_trust


@pytest.fixture
def config_store(monkeypatch):
    """Isolate the config the command reads and writes."""
    import hermes_cli.config as config_mod

    store = {}

    def _load_config():
        return store.get("config", {"skills": {}})

    def _save_config(data):
        store["config"] = data

    monkeypatch.setattr(config_mod, "load_config", _load_config)
    monkeypatch.setattr(config_mod, "save_config", _save_config)
    return store


@pytest.fixture
def pinned_terminal_cwd_outside_repo(tmp_path, monkeypatch):
    """Point the agent-cwd ladder's TERMINAL_CWD rung outside the repo."""
    outside = tmp_path / "outside"
    outside.mkdir()

    import agent.runtime_cwd as runtime_cwd_mod
    monkeypatch.setattr(runtime_cwd_mod, "scope_terminal_cwd", lambda: str(outside))
    return outside


def _run(action, capsys):
    _cmd_skills_trust(SimpleNamespace(skills_action=action, path=None))
    return capsys.readouterr().out


def test_bare_trust_resolves_from_shell_cwd(
        tmp_path, monkeypatch, config_store, pinned_terminal_cwd_outside_repo, capsys):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.chdir(repo)

    out = _run("trust", capsys)

    assert "Not inside a git checkout" not in out
    assert f"Trusted: {repo.resolve()}" in out
    trusted = config_store["config"]["skills"]["trusted_project_dirs"]
    assert str(repo.resolve()) in trusted


def test_bare_untrust_resolves_from_shell_cwd(
        tmp_path, monkeypatch, config_store, pinned_terminal_cwd_outside_repo, capsys):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    config_store["config"] = {
        "skills": {"trusted_project_dirs": [str(repo.resolve())]}}
    monkeypatch.chdir(repo)

    out = _run("untrust", capsys)

    assert "Not inside a git checkout" not in out
    assert f"Untrusted: {repo.resolve()}" in out
    assert config_store["config"]["skills"]["trusted_project_dirs"] == []
