"""Tagless packaging enters the real builder with an exact source identity."""
from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.bundles import desktop

from tests.ci.test_desktop_release_tag_admission import _BASH, _child_env, _git, _seed_repo


@pytest.mark.parametrize("variant", ["bundled", "store", "light"])
def test_desktop_build_reaches_the_managed_payload_with_commit_ref(tmp_path, monkeypatch, variant):
    _, repo = _seed_repo(tmp_path)
    sha = _git("rev-parse", "HEAD", cwd=repo)
    (repo / "pyproject.toml").write_text('[project]\nname="x"\nversion="9.9.9"\n', encoding='utf-8')
    monkeypatch.setenv("HERMES_PAYLOAD_TAG", "v9.9.9")
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("BUILD_NUMBER", "123")
    monkeypatch.setattr(desktop.shutil, "which", lambda name: name)
    monkeypatch.setattr(desktop, "npm_command", lambda node: [node, "npm-cli.js"])
    monkeypatch.setattr("scripts.build.windows_deps.prepare_windows_environment", lambda **kwargs: dict(kwargs["env"]))
    from pm.store import current_target
    target_arch = current_target().split("-")[1]
    def capture(argv, cwd):
        if argv[0] == "git":
            return _git(*argv[1:], cwd=cwd)
        if argv == ["uv", "--version"]:
            return "uv 0.12.0 aarch64-pc-windows-msvc"
        if "process.arch" in argv:
            return target_arch
        return "26.7.0"
    monkeypatch.setattr(desktop, "capture", capture)
    (repo / "package-lock.json").write_text("{}", encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / 'apps/desktop').mkdir(parents=True)
    (repo / 'ui-tui/dist').mkdir(parents=True)
    (repo / 'ui-tui/dist/entry.js').write_text('source fixture', encoding='utf-8')
    (repo / 'hermes_cli/web_dist').mkdir(parents=True)
    (repo / 'hermes_cli/web_dist/index.html').write_text('source fixture', encoding='utf-8')
    calls = []
    def run(argv, *, cwd, env):
        assert cwd.resolve().is_relative_to(repo.resolve())
        calls.append((argv, cwd, env.copy()))
        assert env.get("HERMES_PAYLOAD_TAG", "") == ""
        assert env["HERMES_BUILD_COMMIT"] == sha
        assert env["GITHUB_SHA"] == sha
        assert env["HERMES_PAYLOAD_VERSION"] == "0.1.2"
        if "scripts.bundles.stage" in argv:
            assert argv[argv.index("--ref") + 1] == sha
            assert "--tui" in argv and "--web" in argv
            payload = repo / 'apps/desktop/build/agent-payload'
            (payload / 'hermes-agent').mkdir(parents=True)
            (payload / 'manifest.json').write_text(json.dumps({'repo': 'hermes-agent', 'target': current_target()}), encoding='utf-8')
    monkeypatch.setattr(desktop, "run", run)
    desktop.build(repo, None, variant, ['--publish=never'], commit_build=sha)
    assert any('scripts.bundles.stage' in argv for argv, _, _ in calls) == (variant != 'light')
    argv, cwd, env = calls[-1]
    assert argv[:5] == ['node', 'npm-cli.js', 'run', 'builder', '--']
    assert '-c.extraMetadata.version=0.1.2' in argv
    assert argv[-1] == '--publish=never'
    assert cwd == repo / 'apps/desktop'
    assert env['HERMES_DESKTOP_VARIANT'] == variant
    if os.name == 'nt':
        assert 'BUILD_NUMBER' not in env
    before = len(calls)
    for tag, commit in [('v0.1.2', sha), (None, 'b' * 40), (None, 'short')]:
        with pytest.raises(ValueError):
            desktop.build(repo, tag, variant, [], commit_build=commit)
        assert len(calls) == before



def test_termux_commit_args_reach_prerequisite_checks_without_mutation(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    out = tmp_path / "must-not-be-written"
    helper = tmp_path / "bin"
    helper.mkdir()
    # Empty prerequisite commands are not build substitutes: stop at the first
    # actual prerequisite check, before any payload or output is created.
    env = _child_env(PATH=str(helper), HERMES_PAYLOAD_TAG="")
    scripts = [repo / "scripts/termux/termux_build.sh", repo / "scripts/termux/build_deb.sh"]
    for script in scripts:
        args = ["--repo", str(repo), "--commit", "a" * 40, "--out", str(out)]
        if script.name == "build_deb.sh":
            args += ["--payload", str(tmp_path / "absent")]
        result = subprocess.run([_BASH, str(script), *args], env=env, cwd=tmp_path,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "usage:" not in result.stderr
        assert not out.exists()
