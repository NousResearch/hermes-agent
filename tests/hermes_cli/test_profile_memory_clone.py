"""Clone publication is an offline, source-scoped transaction on every surface."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from agent import secret_scope
from hermes_cli import profiles
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


COMPANION = '''import json
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home

def prepare_clone(*, source_home, source_name, staging_home, destination_home, destination_name, clone_all):
    assert get_hermes_home() == source_home
    assert get_secret("CLONE_TEST_KEY") == source_name
    assert not get_secret("CALLER_ONLY_KEY")
    assert not destination_home.exists()
    assert staging_home != destination_home
    assert destination_home.name == destination_name
    raw = json.loads((source_home / "clone_probe.json").read_text())
    if raw.get("fail"):
        raise RuntimeError("secret-must-not-escape")
    raw["peer"] = destination_name
    raw["source"] = source_name
    (staging_home / "clone_probe.json").write_text(json.dumps(raw))
    return {"needs_auth": True, "secret": "must-not-be-reported"}
'''


def _git(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
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
    monkeypatch.setenv("CALLER_ONLY_KEY", "launch-secret")
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda name: None)
    # General plugin discovery is unrelated to memory companion loading; leave it disabled.
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    token = set_hermes_home_override(caller)
    secrets = secret_scope.set_secret_scope({"CLONE_TEST_KEY": "caller", "CALLER_ONLY_KEY": "caller-secret"})
    original_modules = set(sys.modules)
    try:
        yield home, caller
    finally:
        secret_scope.reset_secret_scope(secrets)
        reset_hermes_home_override(token)
        for name in set(sys.modules) - original_modules:
            if name.startswith(("_hermes_user_memory.clone_probe", "_hermes_memory_companions_")):
                sys.modules.pop(name, None)


def _source(home, name="source", *, companion=True, active=True):
    source = home / "profiles" / name
    source.mkdir(parents=True)
    (source / "config.yaml").write_text(yaml.safe_dump({"memory": {"provider": "clone_probe" if active else ""}}))
    (source / ".env").write_text(f"CLONE_TEST_KEY={name}\n")
    (source / "clone_probe.json").write_text(json.dumps({"peer": name}))
    package = source / "plugins" / "clone_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('# MemoryProvider\nraise AssertionError("provider activated during clone")\n')
    if companion:
        (package / "clone.py").write_text(COMPANION)
    (package / "payload.py").write_text('VALUE = "private-package"\n')
    _git(package, "init")
    _git(package, "add", ".")
    _git(package, "commit", "-m", "fixture")
    _git(package, "remote", "add", "origin", "https://example.invalid/clone-probe.git")
    metadata = {"clone_probe": {"source": "https://example.invalid/clone-probe.git",
                               "revision": _git(package, "rev-parse", "HEAD"), "pinned": True}}
    (source / "plugins" / ".install-metadata.json").write_text(json.dumps(metadata))
    return source


@pytest.mark.parametrize("clone_all,active,multiplex", [(False, True, False), (True, True, True), (True, False, False)])
def test_clone_prepares_private_state_before_publication(clone_env, monkeypatch, clone_all, active, multiplex):
    home, caller = clone_env
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", multiplex)
    source = _source(home, active=active)
    original = (source / "clone_probe.json").read_bytes()
    env = dict(os.environ)
    notifications = []
    def notified(name):
        destination = profiles.get_profile_dir(name)
        assert json.loads((destination / "clone_probe.json").read_text())["peer"] == name
        assert (destination / "plugins" / "clone_probe" / "payload.py").is_file()
        notifications.append(name)
    monkeypatch.setattr(profiles, "_notify_multiplexer", notified)
    for name in ("first", "second"):
        result = profiles.create_profile(name, clone_from="source", clone_all=clone_all)
        assert result == home / "profiles" / name
        assert get_hermes_home() == caller
        assert secret_scope.get_secret("CLONE_TEST_KEY") == "caller"
        assert dict(os.environ) == env
        from hermes_cli.profile_clone import clone_needs_auth
        assert clone_needs_auth(result) == ["clone_probe"]
        receipt = json.loads((result / ".clone-report.json").read_text())
        assert receipt["plugins"]["copied"] == ["clone_probe"]
        assert "must-not-be-reported" not in json.dumps(receipt)
        assert not (result / ".plugin-installation.lock").exists()
        assert not (result / "plugins" / "clone_probe" / ".git").exists()
        assert json.loads((result / "clone_probe.json").read_text())["source"] == "source"
        assert (source / "clone_probe.json").read_bytes() == original
    (source / "plugins" / "clone_probe" / "payload.py").write_text("CHANGED\n")
    assert (home / "profiles" / "first" / "plugins" / "clone_probe" / "payload.py").read_text() != "CHANGED\n"
    assert notifications == ["first", "second"]


@pytest.fixture
def installed_module(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    site.mkdir()
    module = site / "legacy_module_probe.py"
    module.write_text('raise AssertionError("provider activated during clone")\n')
    dist = site / "legacy_module_probe-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: legacy-module-probe\nVersion: 1.0\n"
    )
    (dist / "entry_points.txt").write_text(
        "[hermes_agent.memory_providers]\nlegacy_module_probe = legacy_module_probe:register\n"
    )
    monkeypatch.syspath_prepend(str(site))
    return module


@pytest.mark.parametrize("clone_all", [False, True])
@pytest.mark.parametrize("failure", ["callback", "missing", "broken-module", "legacy-native", "legacy-directory"])
def test_clone_refusal_never_publishes_source_identity(
    clone_env, monkeypatch, request, failure, clone_all,
):
    home, caller = clone_env
    source = _source(home, companion=not failure.startswith("legacy"))
    if failure == "callback":
        (source / "clone_probe.json").write_text('{"fail": true, "peer": "source"}')
    elif failure == "missing":
        (source / "config.yaml").write_text("memory:\n  provider: absent_probe\n")
    elif failure == "broken-module":
        request.getfixturevalue("installed_module").unlink()
        (source / "config.yaml").write_text("memory:\n  provider: legacy_module_probe\n")
    elif failure == "legacy-directory":
        (source / "clone_probe.json").unlink()
        (source / "clone_probe").mkdir()
        (source / "clone_probe" / "identity.json").write_text('{"peer": "source"}')
    original = (source / "config.yaml").read_bytes()
    notifications = []
    monkeypatch.setattr(profiles, "_notify_multiplexer", notifications.append)
    with pytest.raises(ValueError) as error:
        profiles.create_profile("refused", clone_from="source", clone_all=clone_all)
    assert "secret-must-not-escape" not in str(error.value)
    assert "retry" in str(error.value).lower() or "clone" in str(error.value).lower()
    assert not (home / "profiles" / "refused").exists()
    assert not list((home / "profiles").glob(".refused.staging-*"))
    assert notifications == []
    assert (source / "config.yaml").read_bytes() == original
    assert get_hermes_home() == caller
    assert secret_scope.get_secret("CLONE_TEST_KEY") == "caller"


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("failure_at", ["load_env_file", "set_secret_scope"])
def test_clone_source_scope_entry_failure_restores_caller(
    clone_env, monkeypatch, error_type, failure_at,
):
    from hermes_cli.profile_clone import clone_source_scope
    from hermes_constants import get_hermes_home_override

    home, caller = clone_env
    source = home / "profiles" / "source"
    source.mkdir()
    (source / ".env").write_text("CLONE_TEST_KEY=source\n")
    caller_home = get_hermes_home_override()
    caller_secrets = secret_scope.current_secret_scope()
    env = dict(os.environ)
    failure = error_type("scope entry failed")

    def fail_entry(*args):
        assert get_hermes_home() == source
        assert secret_scope.current_secret_scope() is caller_secrets
        raise failure

    monkeypatch.setattr(secret_scope, failure_at, fail_entry)
    with pytest.raises(error_type) as raised:
        with clone_source_scope(source):
            pytest.fail("scope body must not run after entry failure")

    assert raised.value is failure
    assert get_hermes_home_override() == caller_home
    assert get_hermes_home() == caller
    assert secret_scope.current_secret_scope() is caller_secrets
    assert dict(os.environ) == env


@pytest.mark.parametrize("clone_all", [False, True])
@pytest.mark.parametrize("configuration", ["active", "inactive", "top-level", "env-only"])
@pytest.mark.parametrize("installation", ["directory", "module"])
def test_legacy_config_clone_needs_no_companion(
    clone_env, monkeypatch, request, clone_all, configuration, installation,
):
    home, caller = clone_env
    source = _source(home, companion=False)
    (source / "clone_probe.json").unlink()
    provider = "clone_probe" if installation == "directory" else "legacy_module_probe"
    if installation == "module":
        request.getfixturevalue("installed_module")
    settings = {"user_id": "source-peer", "api_key": "${LEGACY_TEST_KEY}"}
    config: dict = {"memory": {"provider": provider if configuration == "active" else ""}}
    if configuration in {"active", "inactive"}:
        config["memory"][provider] = settings
    elif configuration == "top-level":
        config[provider] = settings
    (source / "config.yaml").write_text(yaml.safe_dump(config))
    (source / ".env").write_text("LEGACY_TEST_KEY=source-test-key\n")
    original = (source / "config.yaml").read_bytes()
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda name: None)

    for destination in ("first", "second"):
        result = profiles.create_profile(destination, clone_from="source", clone_all=clone_all)
        cloned = yaml.safe_load((result / "config.yaml").read_text())
        assert cloned["memory"]["provider"] == config["memory"]["provider"]
        if configuration in {"active", "inactive"}:
            assert cloned["memory"][provider] == settings
        elif configuration == "top-level":
            assert cloned[provider] == settings
        assert "LEGACY_TEST_KEY=source-test-key" in (result / ".env").read_text()
        assert json.loads((result / ".clone-report.json").read_text())["needs_auth"] == []
        assert get_hermes_home() == caller
        assert (source / "config.yaml").read_bytes() == original
        assert "legacy_module_probe" not in sys.modules
