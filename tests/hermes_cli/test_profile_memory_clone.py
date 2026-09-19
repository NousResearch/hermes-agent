"""Package transfer does not introduce native memory clone semantics.

Native files retain the historical distinction: normal clones omit them, full
clones copy them unchanged. Neither behavior promises a fresh remote identity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from hermes_cli import profiles
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


def _git(root, *args):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.invalid",
               GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.invalid")
    return subprocess.run(["git", "-C", str(root), *args], env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def clone_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    caller = home / "profiles" / "caller"
    caller.mkdir(parents=True)
    (caller / "config.yaml").write_text("memory: {}\n")
    monkeypatch.setenv("HERMES_HOME", str(caller))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda name: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda name: None)
    token = set_hermes_home_override(caller)
    try:
        yield home, caller
    finally:
        reset_hermes_home_override(token)


def _source(home, name, companion):
    source = home / "profiles" / name
    source.mkdir(parents=True)
    settings = {"memory": {"provider": "clone_probe", "clone_probe": {"user_id": name}}}
    (source / "config.yaml").write_text(yaml.safe_dump(settings))
    (source / ".env").write_text(f"PACKAGE_TEST_KEY={name}\nTELEGRAM_BOT_TOKEN=strip-me\n")
    (source / "clone_probe.json").write_text(json.dumps({"peer": name}))
    package = source / "plugins" / "clone_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('# MemoryProvider\nraise AssertionError("runtime activation forbidden")\n')
    if companion:
        (package / "clone.py").write_text('raise AssertionError("clone hooks are not a host API")\n')
    (package / "payload.txt").write_text(name)
    _git(package, "init")
    _git(package, "add", ".")
    _git(package, "commit", "-m", "fixture")
    origin = "https://example.invalid/clone-probe.git"
    _git(package, "remote", "add", "origin", origin)
    from hermes_cli.plugin_inventory import capture_install_inventory
    record = {"source": origin, "revision": _git(package, "rev-parse", "HEAD"), "pinned": True}
    record["files"] = capture_install_inventory(package, package)
    (package / ".hermes-catalog.json").write_text(json.dumps({
        "catalog_name": "clone-probe", "repo": origin, "sha": record["revision"]}))
    (source / "plugins" / ".install-metadata.json").write_text(json.dumps({"clone_probe": record}))
    return source


@pytest.mark.parametrize("clone_all", [False, True])
@pytest.mark.parametrize("companion", [False, True])
def test_clone_preserves_packages_without_native_preparation(clone_env, monkeypatch, clone_all, companion):
    home, caller = clone_env
    sources = {name: _source(home, name, companion) for name in ("source-a", "source-b")}
    notified = []

    def notify(name):
        destination = profiles.get_profile_dir(name)
        assert (destination / "plugins" / "clone_probe" / "payload.txt").is_file()
        report = json.loads((destination / ".clone-report.json").read_text())
        assert report["plugins"]["copied"] == ["clone_probe"]
        assert report["plugins"]["warnings"]
        assert "needs_auth" not in report
        notified.append(name)

    monkeypatch.setattr(profiles, "_notify_multiplexer", notify)
    for index, source_name in enumerate(("source-a", "source-b", "source-a")):
        source = sources[source_name]
        destination_name = f"copy-{index}"
        destination = profiles.create_profile(destination_name, clone_from=source_name, clone_all=clone_all)
        assert get_hermes_home() == caller
        assert (destination / "plugins" / "clone_probe" / "payload.txt").read_text() == source_name
        assert not (destination / "plugins" / "clone_probe" / ".git").exists()
        assert not (destination / ".plugin-installation.lock").exists()
        assert "TELEGRAM_BOT_TOKEN" not in (destination / ".env").read_text()
        assert f"PACKAGE_TEST_KEY={source_name}" in (destination / ".env").read_text()
        config = yaml.safe_load((destination / "config.yaml").read_text())
        assert config["memory"]["clone_probe"]["user_id"] == source_name
        if clone_all:
            assert (destination / "clone_probe.json").read_bytes() == (source / "clone_probe.json").read_bytes()
        else:
            assert not (destination / "clone_probe.json").exists()
        (destination / "plugins" / "clone_probe" / "payload.txt").write_text("independent copy")
        assert (source / "plugins" / "clone_probe" / "payload.txt").read_text() == source_name
    assert notified == ["copy-0", "copy-1", "copy-2"]
