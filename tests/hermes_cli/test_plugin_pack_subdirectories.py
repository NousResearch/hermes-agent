"""A pack's subdirectory selects files inside its pinned repository."""

import json
import subprocess

import pytest
import yaml

from hermes_cli.plugin_packs import parse_pack, resolve_pack_plugins
from hermes_cli.plugins_cmd import _install_plugin_core, _resolve_git_url


@pytest.mark.parametrize("repo, expected_url, prefix", [
    ("owner/repo", "https://github.com/owner/repo.git", ""),
    ("https://github.com/owner/repo", "https://github.com/owner/repo", ""),
    ("https://git.example/repo", "https://git.example/repo", ""),
    ("ssh://git@git.example/repo.git", "ssh://git@git.example/repo.git", ""),
    ("https://github.com/owner/repo/tree/main/plugins", "https://github.com/owner/repo.git", "plugins/"),
    ("owner/repo#plugins", "https://github.com/owner/repo.git", "plugins/"),
])
def test_pack_subdirectory_preserves_repository_and_existing_prefix(repo, expected_url, prefix):
    pack = parse_pack(yaml.safe_dump({
        "name": "demo", "plugins": [{"repo": repo, "subdir": "nested/demo", "ref": "a" * 40}],
    }))
    resolved, = resolve_pack_plugins(pack)
    assert _resolve_git_url(resolved.identifier) == (expected_url, prefix + "nested/demo")


def test_pack_subdirectory_installs_from_a_real_pinned_repository(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    plugin = repo / "plugins" / "demo"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: demo\nversion: 1.0.0\n", encoding="utf-8")
    (repo / "unrelated.txt").write_text("not part of the plugin", encoding="utf-8")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
            stdin=subprocess.DEVNULL,
        ).stdout.strip()

    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.com", "commit", "-qm", "init")
    revision = git("rev-parse", "HEAD")
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    pack = parse_pack(yaml.safe_dump({
        "name": "demo", "plugins": [{"repo": repo.as_uri(), "subdir": "plugins/demo", "ref": revision}],
    }))
    resolved, = resolve_pack_plugins(pack)
    target, manifest, name = _install_plugin_core(resolved.identifier, force=False, ref=revision)
    assert name == manifest["name"] == "demo"
    assert (target / "plugin.yaml").read_bytes() == (plugin / "plugin.yaml").read_bytes()
    assert not (target / "unrelated.txt").exists()
    metadata = json.loads((home / "plugins" / ".install-metadata.json").read_text(encoding="utf-8"))
    assert metadata[name]["revision"] == revision
    assert metadata[name]["source"].endswith("#plugins/demo")
