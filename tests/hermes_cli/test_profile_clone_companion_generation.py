"""Cloning after a package replacement uses the new code, not live-agent modules."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest


def _git(root, *args):
    return subprocess.run(
        ["git", "-c", "core.hooksPath=" + os.devnull, "-C", str(root), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.mark.parametrize("changed", ["clone", "helper", "both"])
def test_replaced_package_prepares_clone_from_current_generation(tmp_path, monkeypatch, changed):
    from hermes_cli import profiles
    from hermes_cli.plugins_cmd import _install_plugin_core
    from hermes_cli.profile_clone import clone_source_scope
    from plugins import memory

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", tmp_path / "no-bundled")
    # Publication remains real; only notifications to real services are disabled.
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda *_: None)
    repo = tmp_path / "generation-provider"
    repo.mkdir()
    init = '''from agent.memory_provider import MemoryProvider
from .helper import VALUE
class Provider(MemoryProvider):
    name = 'generation-provider'
    def is_available(self): return True
    def initialize(self, *args, **kwargs): pass
    def get_tool_schemas(self): return []
    def prefetch(self, *args, **kwargs): return VALUE
'''
    clone = '''from .helper import VALUE
LABEL = 'old'
def prepare_clone(*, staging_home, **kwargs):
    (staging_home / 'prepared.txt').write_text(LABEL + ':' + VALUE)
    return {'needs_auth': True}
'''
    (repo / "__init__.py").write_text(init)
    (repo / "helper.py").write_text("VALUE = 'old'\n")
    (repo / "clone.py").write_text(clone)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "old fixture")
    old_revision = _git(repo, "rev-parse", "HEAD")
    (home / "config.yaml").write_text("memory:\n  provider: generation-provider\n")
    with clone_source_scope(home):
        target, _, _ = _install_plugin_core(repo.as_uri(), force=False, ref=old_revision, python_deps=False)
        provider = memory.load_memory_provider(target.name, register_skills=False)
        assert provider is not None
        runtime_name = type(provider).__module__
        runtime_modules = {name: module for name, module in sys.modules.copy().items()
                           if name == runtime_name or name.startswith(runtime_name + ".")}
        assert provider.prefetch("probe") == "old"

    def offline(*args, **kwargs):
        raise AssertionError("clone must not access the network")

    monkeypatch.setattr(socket.socket, "connect", offline)
    monkeypatch.setattr(memory, "load_memory_provider", offline)
    first = profiles.create_profile("first", clone_from="default", clone_config=True,
                                    no_alias=True)
    assert (first / "prepared.txt").read_text() == "old:old"
    if changed in {"helper", "both"}:
        (repo / "helper.py").write_text("VALUE = 'new'\n")
    if changed in {"clone", "both"}:
        (repo / "clone.py").write_text(clone.replace("LABEL = 'old'", "LABEL = 'new'"))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "new fixture")
    new_revision = _git(repo, "rev-parse", "HEAD")
    with clone_source_scope(home):
        updated, _, _ = _install_plugin_core(repo.as_uri(), force=True, ref=new_revision,
                                             python_deps=False)
    assert updated == target
    second = profiles.create_profile("second", clone_from="default", clone_config=True,
                                     no_alias=True)
    # The published code and metadata must agree with the hook that prepared it.
    copied = second / "plugins" / target.name
    record = json.loads((second / "plugins" / ".install-metadata.json").read_text())[target.name]
    assert record["revision"] == new_revision != old_revision
    for filename in ("clone.py", "helper.py", "__init__.py"):
        assert (copied / filename).read_bytes() == (repo / filename).read_bytes()
    expected = {"helper": "old:new", "clone": "new:old", "both": "new:new"}[changed]
    assert (second / "prepared.txt").read_text() == expected
    assert (first / "prepared.txt").read_text() == "old:old"
    assert provider.prefetch("probe") == "old"
    assert all(sys.modules[name] is module for name, module in runtime_modules.items())
