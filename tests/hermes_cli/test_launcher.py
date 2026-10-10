"""Tests for the top-level `./hermes` launcher script."""

import json
import os
import runpy
import sys
import types
from pathlib import Path

import pytest


def test_launcher_delegates_to_argparse_entrypoint(monkeypatch):
    """`./hermes` should use `hermes_cli.main`, not the legacy Fire wrapper."""
    launcher_path = Path(__file__).resolve().parents[2] / "hermes"
    called = []

    fake_main_module = types.ModuleType("hermes_cli.main")

    def fake_main():
        called.append("hermes_cli.main")

    fake_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "hermes_cli.main", fake_main_module)

    fake_cli_module = types.ModuleType("cli")

    def legacy_cli_main(*args, **kwargs):
        raise AssertionError("launcher should not import cli.main")

    fake_cli_module.main = legacy_cli_main
    monkeypatch.setitem(sys.modules, "cli", fake_cli_module)

    fake_fire_module = types.ModuleType("fire")

    def legacy_fire(*args, **kwargs):
        raise AssertionError("launcher should not invoke fire.Fire")

    fake_fire_module.Fire = legacy_fire
    monkeypatch.setitem(sys.modules, "fire", fake_fire_module)

    monkeypatch.setattr(sys, "argv", [str(launcher_path), "gateway", "status"])

    runpy.run_path(str(launcher_path), run_name="__main__")

    assert called == ["hermes_cli.main"]


def test_installation_command_refuses_unpublished_launcher(monkeypatch, tmp_path):
    """PM workspace copies resolve a store Python without ever publishing
    .hermes/bin — the returned command must stay executable everywhere
    instead of embedding a missing path (issue #125043)."""
    from hermes_cli import _launchers

    import pm.environments as _pmenv
    monkeypatch.setattr(_pmenv, "installs_root", lambda: tmp_path / "installs")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: Path("/usr/bin/python3"))

    command = _launchers.installation_command(workspace, ["gateway", "run"])
    assert str(workspace / ".hermes" / "bin" / "hermes") not in command
    assert command[0] == "/usr/bin/python3" and command[1:3] == ["-I", "-c"]
    assert command[-2:] == ["gateway", "run"]

    # A published, executable launcher whose root carries committed facts keeps
    # the exact-install path.
    from pm.environments import runtime_facts_path
    runtime_facts_path(workspace).parent.mkdir(parents=True, exist_ok=True)
    runtime_facts_path(workspace).write_text(
        json.dumps({"packages": {"venv": {"environment": str(tmp_path / "venv")}}}), encoding="utf-8")
    local = workspace / ".hermes" / "bin"
    local.mkdir(parents=True)
    shim = local / "hermes"
    shim.write_text("#!/bin/sh\n", encoding="utf-8")
    shim.chmod(0o755)
    assert _launchers.installation_command(workspace, ["gateway", "run"])[0] == str(shim)
    assert _launchers.installation_command(
        workspace, module="gateway.cgroup_cleanup")[1] == "--run-module"

    # A non-executable launcher is as unusable as a missing one (facts gone too:
    # the record without a venv would divert to the owner lookup first).
    shim.chmod(0o644)
    runtime_facts_path(workspace).unlink()
    assert _launchers.installation_command(workspace, ["gateway", "run"])[0] != str(shim)


def test_installation_command_workspace_without_facts_resolves_owner_entry(monkeypatch, tmp_path):
    """A PM workspace copy publishes the launcher shim but has no facts.json of its
    own — the command must resolve the entry point from the checkout that owns the
    committed venv instead of embedding the dead shim (#125375 counter-datapoint)."""
    from hermes_cli import _launchers

    import pm.environments as _pmenv
    monkeypatch.setattr(_pmenv, "installs_root", lambda: tmp_path / "installs")

    home = tmp_path / "home"
    checkout = home / "hermes-agent"
    workspace = tmp_path / "workspace"
    venv = tmp_path / "envs" / "525b471e" / "venv"
    for d in (checkout, workspace, venv / "bin"):
        d.mkdir(parents=True)
    script = venv / "bin" / "hermes"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o755)
    (venv / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    # The checkout carries the committed facts; the workspace does not.
    from pm.environments import runtime_facts_path
    runtime_facts_path(checkout).parent.mkdir(parents=True, exist_ok=True)
    runtime_facts_path(checkout).write_text(
        json.dumps({"packages": {"venv": {"environment": str(venv)}}}), encoding="utf-8")

    # The workspace resolves the same store python and published the shim.
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: Path("/usr/bin/python3"))
    local = workspace / ".hermes" / "bin"
    local.mkdir(parents=True)
    shim = local / "hermes"
    shim.write_text("#!/bin/sh\n", encoding="utf-8")
    shim.chmod(0o755)

    monkeypatch.setattr(_launchers, "_facts_owner_root", lambda root: checkout)

    stable = checkout / ".hermes" / "bin" / "hermes"
    stable.parent.mkdir(parents=True)
    stable.write_text("#!/bin/sh\n", encoding="utf-8")
    stable.chmod(0o755)

    command = _launchers.installation_command(workspace, ["gateway", "run"])
    assert command[0] == str(stable), "the stable owner launcher, not a generation executable"
    assert command[-2:] == ["gateway", "run"]
    assert str(shim) not in command

    # Module form remains on the shim-only carrier.
    module_cmd = _launchers.installation_command(workspace, module="gateway.cgroup_cleanup")
    assert module_cmd == [str(stable), "--run-module", "gateway.cgroup_cleanup"]

    # With facts of its own, the workspace keeps its published launcher.
    runtime_facts_path(workspace).parent.mkdir(parents=True, exist_ok=True)
    runtime_facts_path(workspace).write_text(
        json.dumps({"packages": {"venv": {"environment": str(venv)}}}), encoding="utf-8")
    assert _launchers.installation_command(workspace, ["gateway", "run"])[0] == str(shim)


def test_installation_command_owner_without_record_falls_back_to_interpreter(monkeypatch, tmp_path):
    """No facts anywhere: the interpreter bootstrap form stays the answer."""
    from hermes_cli import _launchers

    import pm.environments as _pmenv
    monkeypatch.setattr(_pmenv, "installs_root", lambda: tmp_path / "installs")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: Path("/usr/bin/python3"))
    local = workspace / ".hermes" / "bin"
    local.mkdir(parents=True)
    shim = local / "hermes"
    shim.write_text("#!/bin/sh\n", encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setattr(_launchers, "_facts_owner_root", lambda root: tmp_path / "nowhere")

    command = _launchers.installation_command(workspace, ["gateway", "run"])
    assert command[0] == "/usr/bin/python3" and command[1:3] == ["-I", "-c"]


def test_installation_command_generation_workspace_last_resort_uses_sibling_venv(monkeypatch, tmp_path):
    """#125375 follow-up: with no facts record anywhere, a generation workspace
    root must still resolve the sibling venv's entry point instead of the
    interpreter form bound to the workspace (which dies at activation exactly
    like the dead shim it replaced — the workspace is not the recorded root)."""
    from hermes_cli import _launchers

    import pm.environments as _pmenv
    monkeypatch.setattr(_pmenv, "installs_root", lambda: tmp_path / "installs")
    generation = tmp_path / "installs" / "ee2f073f78aff6f9" / "environments" / "525b471e"
    workspace = generation / "workspace"
    venv = generation / "venv"
    for d in (workspace, venv / "bin"):
        d.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    script = venv / "bin" / "hermes"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o755)
    (venv / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: Path("/usr/bin/python3"))
    monkeypatch.setattr(_launchers, "_facts_owner_root", lambda root: tmp_path / "nowhere")

    stable_root = tmp_path / "checkout"
    stable_launcher = stable_root / ".hermes" / "bin" / "hermes"
    stable_launcher.parent.mkdir(parents=True)
    stable_launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    stable_launcher.chmod(0o755)
    (generation.parent.parent / "inputs").mkdir(parents=True)
    (generation.parent.parent / "inputs" / ".project-root").write_text(str(stable_root), encoding="utf-8")

    command = _launchers.installation_command(workspace, ["gateway", "run"])
    assert command[0] == str(stable_launcher), "stable launcher, not a generation executable"
    assert command[-2:] == ["gateway", "run"]

    # The previously fixed module-carrier behavior remains unchanged.
    module_cmd = _launchers.installation_command(workspace, module="gateway.cgroup_cleanup")
    assert module_cmd == [str(stable_launcher), "--run-module", "gateway.cgroup_cleanup"]

    # A script-less sibling venv does not change the stable command.
    (venv / "bin" / "hermes").unlink()
    no_script_cmd = _launchers.installation_command(workspace, module="gateway.cgroup_cleanup")
    assert no_script_cmd == [str(stable_launcher), "--run-module", "gateway.cgroup_cleanup"]

    # A plain directory named "workspace" without the generation layout (no
    # sibling venv/pyvenv.cfg) must NOT resolve a bogus entry — the
    # interpreter form stays the answer there.
    plain = tmp_path / "plain" / "workspace"
    plain.mkdir(parents=True)
    plain_command = _launchers.installation_command(plain, ["gateway", "run"])
    assert plain_command[0] == "/usr/bin/python3" and plain_command[1:3] == ["-I", "-c"]


def test_persisted_workspace_command_survives_generation_gc(monkeypatch, tmp_path):
    """A service command resolved from A must not point into A after B wins GC."""
    from hermes_cli import _launchers
    from hermes_cli.runtime_state import collect_generations
    from pm.environments import install_state_dir, runtime_facts_path

    repo = tmp_path / "checkout"
    repo.mkdir()
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: Path("/usr/bin/python3"))
    monkeypatch.setattr(_launchers, "_facts_owner_root", lambda root: tmp_path / "no-owner")
    monkeypatch.setattr("pm.environments.installs_root", lambda: tmp_path / "installs")

    stable = repo / ".hermes" / "bin" / "hermes"
    stable.parent.mkdir(parents=True)
    stable.write_text("#!/bin/sh\n", encoding="utf-8")
    stable.chmod(0o755)

    state = install_state_dir(repo)
    generations = state / "environments"
    for name in ("A", "B"):
        venv = generations / name / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("version = 3.11", encoding="utf-8")
        script = venv / "bin" / "hermes"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        script.chmod(0o755)
        (venv.parent / ".lease-managed").touch()

    workspace = generations / "A" / "workspace"
    workspace.mkdir()
    (state / "inputs").mkdir(parents=True)
    (state / "inputs" / ".project-root").write_text(str(repo), encoding="utf-8")
    runtime_facts_path(repo).parent.mkdir(parents=True, exist_ok=True)
    runtime_facts_path(repo).write_text(json.dumps({"packages": {"venv": {
        "environment": str(generations / "A" / "venv")}}}), encoding="utf-8")

    persisted = _launchers.installation_command(workspace, ["gateway", "run"])
    assert persisted[0] == str(stable)

    runtime_facts_path(repo).write_text(json.dumps({"packages": {"venv": {
        "environment": str(generations / "B" / "venv")}}}), encoding="utf-8")
    removed = collect_generations(repo, min_age_seconds=0)
    assert removed == [generations / "A"]
    assert not (generations / "A").exists()
    assert Path(persisted[0]).is_file() and os.access(persisted[0], os.X_OK)


@pytest.mark.parametrize("marker_kind", ["custom", "missing", "empty", "nonexistent", "file"])
@pytest.mark.parametrize("published", [True, False])
def test_workspace_project_root_precedes_default_checkout(
    monkeypatch, tmp_path, marker_kind, published
):
    """Two installs share a home; a valid PM marker owns persisted argv."""
    from hermes_cli import _launchers
    from pm.environments import install_state_dir, runtime_facts_path

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    default = home / "hermes-agent"
    custom = tmp_path / "custom-source"
    launchers = {}
    for checkout in (default, custom):
        checkout.mkdir(parents=True)
        state = install_state_dir(checkout)
        venv = state / "environments" / "A" / "venv"
        venv.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("version = 3.11\n", encoding="utf-8")
        runtime_facts_path(checkout).write_text(json.dumps({"packages": {"venv": {
            "environment": str(venv)}}}), encoding="utf-8")
        if published:
            shim = checkout / ".hermes" / "bin" / "hermes"
            shim.parent.mkdir(parents=True)
            shim.write_text("#!/bin/sh\n", encoding="utf-8")
            shim.chmod(0o755)
            launchers[checkout] = shim

    state = install_state_dir(custom)
    workspace = state / "environments" / "A" / "workspace"
    workspace.mkdir()
    marker = state / "inputs" / ".project-root"
    marker.parent.mkdir()
    if marker_kind != "missing":
        not_a_directory = tmp_path / "not-a-directory"
        not_a_directory.write_text("not a checkout", encoding="utf-8")
        value = {"custom": str(custom), "empty": "", "nonexistent": str(tmp_path / "absent"),
                 "file": str(not_a_directory)}[marker_kind]
        marker.write_text(value, encoding="utf-8")

    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: Path("/usr/bin/python3"))
    expected = custom if marker_kind == "custom" else default
    # Exercise real home/facts lookup rather than replacing _facts_owner_root.
    assert _launchers._facts_owner_root(workspace) == default
    for module, args in (("hermes_cli.main", ["gateway", "run"]),
                         ("gateway.cgroup_cleanup", []),
                         ("hermes_cli.stderr_timestamp", ["--error-log", "errors.log"])):
        command = _launchers.installation_command(workspace, args, module=module)
        if published:
            prefix = [] if module == "hermes_cli.main" else ["--run-module", module]
            assert command == [str(launchers[expected]), *prefix, *args]
        else:
            assert command == _launchers.runtime_command(expected, args, module=module)
        assert str(state / "environments" / "A" / "venv") not in command
