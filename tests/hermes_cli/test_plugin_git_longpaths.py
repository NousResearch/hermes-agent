"""Plugin checkouts must support long Windows paths despite Git config isolation."""

import subprocess
from pathlib import Path

import pytest

from hermes_cli import plugins_cmd
from hermes_cli._subprocess_compat import noninteractive_git_env


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("pinned", [False, True], ids=["latest", "pinned"])
@pytest.mark.parametrize("subdir", [None, "fixture-plugin"], ids=["full", "sparse"])
def test_plugin_clone_checks_out_long_paths(tmp_path, monkeypatch, pinned, subdir):
    git = plugins_cmd._resolve_git_executable()
    if not git:
        pytest.skip("Git is required for the checkout regression")

    source = tmp_path / "source"
    source.mkdir()

    def seed_git(*args):
        return subprocess.run(
            [git, "-c", "core.longpaths=true", "-c", "user.name=Fixture",
             "-c", "user.email=fixture@example.test", "-c", "commit.gpgsign=false", *args],
            cwd=source, env=noninteractive_git_env(), capture_output=True, text=True,
            check=True, timeout=30,
        ).stdout.strip()

    seed_git("init", "-q")
    relative = Path("fixture-plugin") / ("reference-" + "a" * 90) / ("fixture-" + "b" * 90) / "marker.txt"
    tracked = source / relative
    tracked.parent.mkdir(parents=True)
    tracked.write_text("old revision\n", encoding="utf-8")
    (source / "fixture-plugin" / "plugin.yaml").write_text(
        "name: longpaths-fixture\nversion: '1.0.0'\n", encoding="utf-8",
    )
    seed_git("add", ".")
    seed_git("commit", "-qm", "old")
    old_sha = seed_git("rev-parse", "HEAD")
    tracked.write_text("new revision\n", encoding="utf-8")
    seed_git("commit", "-qam", "new")
    new_sha = seed_git("rev-parse", "HEAD")

    # The reported failure occurs even with the user's global setting enabled:
    # plugin Git intentionally ignores it, so each command must supply its own.
    global_config = tmp_path / "gitconfig"
    global_config.write_text("[core]\n\tlongpaths = true\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    target = tmp_path / "plugins" / ".install-fixture" / "plugin"
    target.parent.mkdir(parents=True)
    assert len(str(target / relative)) > 260

    revision = plugins_cmd._clone_plugin_repo(
        target, source.as_uri(), old_sha if pinned else None, subdir=subdir,
    )

    assert revision == (old_sha if pinned else new_sha)
    assert (target / relative).read_text(encoding="utf-8") == (
        "old revision\n" if pinned else "new revision\n"
    )
    status = plugins_cmd._run_plugin_git(git, target, "status", "--porcelain")
    assert status.returncode == 0, status.stderr
    assert status.stdout == ""
