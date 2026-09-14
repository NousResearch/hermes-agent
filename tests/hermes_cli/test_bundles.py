"""Tests for hermes_cli/bundles.py — the `hermes bundles` CLI subcommand."""

import argparse

import pytest

from hermes_cli.bundles import (
    bundles_command,
    register_cli,
)


@pytest.fixture
def bundles_env(tmp_path, monkeypatch):
    bundles_dir = tmp_path / "skill-bundles"
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    # Reset module-level cache between tests.
    import agent.skill_bundles as mod
    mod._bundles_cache = {}
    mod._bundles_cache_mtime = None
    return bundles_dir


def _parse(argv):
    parser = argparse.ArgumentParser()
    register_cli(parser)
    return parser.parse_args(argv)


class TestBundlesCli:



    def test_create_refuses_overwrite(self, bundles_env, capsys):
        bundles_command(_parse(["create", "dup", "--skill", "s1"]))
        capsys.readouterr()
        with pytest.raises(SystemExit) as ei:
            bundles_command(_parse(["create", "dup", "--skill", "s2"]))
        assert ei.value.code == 1
        out = capsys.readouterr().out
        assert "already exists" in out.lower() or "--force" in out.lower()


    def test_create_requires_skills(self, bundles_env, capsys, monkeypatch):
        # Simulate user pressing Ctrl-D immediately at the interactive prompt.
        monkeypatch.setattr("builtins.input", lambda *_a, **_kw: (_ for _ in ()).throw(EOFError()))
        with pytest.raises(SystemExit):
            bundles_command(_parse(["create", "empty"]))


