"""Isolated Desktop instance resolver, manifest, and launcher contracts.

These tests drive ``hermes_cli.desktop_instances`` against temp roots and
injected adapters. They must never touch the developer's real APPDATA,
LOCALAPPDATA, Desktop folder, SSH config, or running Hermes processes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pytest

from hermes_cli import main as cli_main
from hermes_cli.subcommands.gui import build_gui_parser


def _gui_parser(*, cmd_gui=lambda args: args):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_gui_parser(sub, cmd_gui=cmd_gui)
    return parser


def _touch(path: Path, text: str = "exe") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _store(tmp_path: Path, **overrides: Any):
    from hermes_cli.desktop_instances import DesktopInstanceStore

    hermes_root = tmp_path / "hermes-root"
    runtime_root = tmp_path / "runtime"
    canonical_exe = (
        runtime_root / "apps" / "desktop" / "release" / "win-unpacked" / "Hermes.exe"
    )
    desktop_dir = tmp_path / "Desktop"
    kwargs = dict(
        hermes_root=hermes_root,
        runtime_root=runtime_root,
        canonical_exe=canonical_exe,
        shortcut_dir=desktop_dir,
        platform="win32",
        ssh_probe=lambda host: None,
        compiler=lambda source, output: _touch(output, "compiled-launcher"),
        shortcut_writer=lambda spec: _touch(spec.path, f"shortcut->{spec.target}"),
        is_locked=lambda path: False,
        cwd=tmp_path / "cwd",
    )
    kwargs.update(overrides)
    _touch(canonical_exe)
    (runtime_root / "hermes_constants.py").write_text(
        "# runtime marker\n", encoding="utf-8"
    )
    desktop_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "cwd").mkdir(parents=True, exist_ok=True)
    return DesktopInstanceStore(**kwargs)


def _create(store, name: str = "grace", **kwargs: Any):
    defaults = dict(
        ssh_host="grace",
        remote_hermes_path="/opt/hermes/bin/hermes",
        remote_profile="default",
    )
    defaults.update(kwargs)
    return store.create(name, **defaults)


# ── name / path validation ───────────────────────────────────────────────


def test_csharp_verbatim_does_not_let_trailing_backslash_eat_the_quote():
    from hermes_cli.desktop_instances import _csharp_verbatim

    assert _csharp_verbatim("C:\\") == "C:\\\\"
    assert _csharp_verbatim('Hermes "X"') == 'Hermes ""X""'


def test_instance_name_must_be_a_safe_slug():
    from hermes_cli.desktop_instances import InstanceNameError, validate_instance_name

    assert validate_instance_name("grace") == "grace"
    assert validate_instance_name("Bear-Agent") == "bear-agent"
    with pytest.raises(InstanceNameError):
        validate_instance_name("../etc")
    with pytest.raises(InstanceNameError):
        validate_instance_name("Hermes Grace")
    with pytest.raises(InstanceNameError):
        validate_instance_name("default")
    with pytest.raises(InstanceNameError):
        validate_instance_name("desktop")


def test_remote_hermes_path_must_be_absolute():
    from hermes_cli.desktop_instances import StalePathError, validate_remote_hermes_path

    assert (
        validate_remote_hermes_path("/opt/hermes/bin/hermes")
        == "/opt/hermes/bin/hermes"
    )
    assert (
        validate_remote_hermes_path("C:\\hermes\\hermes.exe")
        == "C:\\hermes\\hermes.exe"
    )
    with pytest.raises(StalePathError):
        validate_remote_hermes_path("bin/hermes")
    with pytest.raises(StalePathError):
        validate_remote_hermes_path("")


def test_remote_profile_reuses_profile_identifier_rules():
    from hermes_cli.desktop_instances import validate_remote_profile

    assert validate_remote_profile("default") == "default"
    assert validate_remote_profile("Research") == "research"
    with pytest.raises(ValueError):
        validate_remote_profile("root")
    with pytest.raises(ValueError):
        validate_remote_profile("bad profile")


# ── layout and isolation env ─────────────────────────────────────────────


def test_instance_paths_stay_under_injected_roots_not_profiles(tmp_path):
    store = _store(tmp_path)
    instance = _create(store, display_name="Hermes Grace")

    assert (
        instance.hermes_home
        == store.hermes_root / "desktop-instances" / "grace" / "home"
    )
    assert (
        instance.user_data
        == store.hermes_root / "desktop-instances" / "grace" / "user-data"
    )
    assert (
        instance.launcher_exe.parent
        == store.hermes_root / "desktop-instances" / "grace" / "launcher"
    )
    assert "profiles" not in instance.hermes_home.parts
    assert instance.named_exe == store.canonical_exe.with_name("Hermes Grace.exe")
    assert instance.runtime_root == store.runtime_root
    assert instance.app_name == "Hermes Grace"


def test_launch_plan_uses_named_hardlink_early_user_data_and_no_shellexecute(tmp_path):
    store = _store(tmp_path)
    instance = _create(store)
    plan = store.build_launch_plan(instance)

    assert plan.executable == instance.named_exe
    assert plan.arguments == [f"--user-data-dir={instance.user_data}"]
    assert plan.use_shell_execute is False
    assert plan.env["HERMES_HOME"] == str(instance.hermes_home)
    assert plan.env["HERMES_DESKTOP_USER_DATA_DIR"] == str(instance.user_data)
    assert plan.env["HERMES_DESKTOP_HERMES_ROOT"] == str(instance.runtime_root)
    assert plan.env["HERMES_DESKTOP_APP_NAME"] == instance.app_name
    assert plan.env["HERMES_DESKTOP_CWD"] == str(store.cwd)
    assert plan.env["HERMES_DESKTOP_INSTANCE"] == instance.name
    assert (
        plan.env["HERMES_DESKTOP_AUMID"]
        == f"com.nousresearch.hermes.instance.{instance.name}"
    )
    assert plan.env["HERMES_DESKTOP_DISABLE_GLOBAL_SHORTCUTS"] == "1"
    assert plan.env["HERMES_DESKTOP_SKIP_PROTOCOL_REGISTER"] == "1"


# ── atomic non-secret manifest ───────────────────────────────────────────


def test_display_name_cannot_collide_with_canonical_exe(tmp_path):
    from hermes_cli.desktop_instances import InstanceNameError

    store = _store(tmp_path)
    with pytest.raises(InstanceNameError, match="collide"):
        _create(store, display_name="Hermes")


def test_launch_plan_can_forward_a_deep_link(tmp_path):
    store = _store(tmp_path)
    instance = _create(store)
    plan = store.build_launch_plan(instance, deep_link="hermes://blueprint/morning")
    assert plan.env["HERMES_DESKTOP_PENDING_DEEP_LINK"] == "hermes://blueprint/morning"
    assert "hermes://blueprint/morning" in plan.arguments


def test_parse_instance_deep_link_returns_none_for_reserved_or_invalid():
    from hermes_cli.desktop_instances import parse_instance_deep_link

    assert parse_instance_deep_link("hermes://instance/desktop/blueprint/x") is None
    assert parse_instance_deep_link("hermes://instance/Not Valid/blueprint/x") is None
    assert parse_instance_deep_link("hermes://instance/") is None


def test_create_writes_atomic_manifest_without_secrets(tmp_path):
    store = _store(tmp_path)
    instance = _create(store)

    raw = instance.manifest_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["version"] == 1
    assert data["name"] == "grace"
    assert data["ssh_host"] == "grace"
    assert data["remote_hermes_path"] == "/opt/hermes/bin/hermes"
    assert data["remote_profile"] == "default"
    blob = raw.lower()
    assert "token" not in blob
    assert "password" not in blob
    assert "secret" not in blob
    assert "-----begin" not in blob


def test_create_is_atomic_and_rejects_duplicates(tmp_path):
    store = _store(tmp_path)
    _create(store)
    from hermes_cli.desktop_instances import InstanceExistsError

    with pytest.raises(InstanceExistsError):
        _create(store)
    listed = store.list()
    assert [item.name for item in listed] == ["grace"]


def test_create_seeds_nonsecret_ssh_connection_json(tmp_path):
    store = _store(tmp_path)
    instance = _create(store)
    seed_path = instance.user_data / "connection.json"
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "ssh"
    assert payload["remote"]["mode"] == "ssh"
    assert payload["remote"]["host"] == "grace"
    assert payload["remote"]["remoteHermesPath"] == "/opt/hermes/bin/hermes"
    assert payload["remote"]["remoteProfile"] == "default"
    assert "token" not in payload
    assert "token" not in payload["remote"]
    assert payload.get("profiles") == {}


def test_create_does_not_clone_remote_or_ordinary_local_state(tmp_path, monkeypatch):
    ordinary = tmp_path / "ordinary-home"
    ordinary.mkdir()
    (ordinary / "auth.json").write_text('{"token": "do-not-copy"}', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(ordinary))
    monkeypatch.setenv("APPDATA", str(tmp_path / "real-appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "real-local"))
    store = _store(tmp_path)
    instance = _create(store)
    assert not (instance.hermes_home / "auth.json").exists()
    leftover = list(instance.hermes_home.rglob("*"))
    assert all("auth.json" not in str(path) for path in leftover)
    assert not (tmp_path / "real-appdata").exists() or not any(
        (tmp_path / "real-appdata").rglob("*")
    )


# ── SSH / platform / stale errors ────────────────────────────────────────


def test_create_reports_missing_ssh_alias(tmp_path):
    from hermes_cli.desktop_instances import MissingSshAliasError

    def boom(host):
        raise MissingSshAliasError(f"SSH alias {host!r} was not found")

    store = _store(tmp_path, ssh_probe=boom)
    with pytest.raises(MissingSshAliasError, match="grace"):
        _create(store)


def test_create_skip_ssh_check_bypasses_probe(tmp_path):
    def boom(host):
        raise AssertionError(f"probe should not run for {host}")

    store = _store(tmp_path, ssh_probe=boom)
    instance = _create(store, skip_ssh_check=True)
    assert instance.ssh_host == "grace"


def test_create_rejects_missing_canonical_runtime(tmp_path):
    from hermes_cli.desktop_instances import StalePathError

    store = _store(tmp_path)
    store.canonical_exe.unlink()
    with pytest.raises(StalePathError, match="canonical"):
        _create(store)


def test_unknown_platform_mutations_fail(tmp_path):
    from hermes_cli.desktop_instances import IncompatiblePlatformError

    store = _store(tmp_path, platform="aix")
    with pytest.raises(IncompatiblePlatformError):
        _create(store)


def test_linux_create_writes_desktop_entry_and_wrapper(tmp_path):
    store = _store(tmp_path, platform="linux")
    instance = _create(store)
    assert instance.shortcut_path.suffix == ".desktop"
    assert instance.shortcut_path.exists()
    desktop = instance.shortcut_path.read_text(encoding="utf-8")
    assert f"Name={instance.app_name}" in desktop
    assert str(instance.launcher_exe) in desktop
    wrapper = instance.launcher_exe.read_text(encoding="utf-8")
    assert "HERMES_HOME=" in wrapper
    assert "--user-data-dir=" in wrapper
    assert "UseShellExecute" not in wrapper
    assert instance.named_exe == store.canonical_exe


def test_macos_create_writes_command_wrapper(tmp_path):
    store = _store(tmp_path, platform="darwin")
    instance = _create(store)
    assert instance.shortcut_path.suffix == ".command"
    assert instance.shortcut_path.exists()
    script = instance.shortcut_path.read_text(encoding="utf-8")
    assert "HERMES_DESKTOP_USER_DATA_DIR=" in script
    assert "--user-data-dir=" in script


def test_instance_spec_from_ssh_connection_maps_nonsecret_fields():
    from hermes_cli.desktop_instances import isolated_instance_spec_from_ssh

    spec = isolated_instance_spec_from_ssh({
        "id": "c1",
        "kind": "ssh",
        "label": "Hermes Athena",
        "host": "bear-agent",
        "user": "bear",
        "port": 2222,
        "keyPath": "/home/bear/.ssh/id_ed25519",
        "remoteHermesPath": "/opt/hermes/bin/hermes",
        "remoteProfile": "default",
    })
    assert spec.name == "athena"
    assert spec.connection_id == "c1"
    assert spec.ssh_host == "bear-agent"
    assert spec.ssh_user == "bear"
    assert spec.ssh_port == 2222
    assert spec.ssh_key_path == "/home/bear/.ssh/id_ed25519"
    assert spec.remote_hermes_path == "/opt/hermes/bin/hermes"
    assert spec.remote_profile == "default"
    assert spec.display_name == "Hermes Athena"
    assert "token" not in spec.to_manifest()


def test_same_host_ssh_rows_are_distinct_when_user_port_or_key_differ():
    from hermes_cli.desktop_instances import isolated_instance_spec_from_ssh

    alice = isolated_instance_spec_from_ssh({
        "id": "alice-box",
        "kind": "ssh",
        "label": "Lab",
        "host": "lab.example",
        "user": "alice",
        "port": 22,
        "keyPath": "/keys/alice",
        "remoteHermesPath": "/opt/hermes/bin/hermes",
        "remoteProfile": "default",
    })
    bob = isolated_instance_spec_from_ssh({
        "id": "bob-box",
        "kind": "ssh",
        "label": "Lab",
        "host": "lab.example",
        "user": "bob",
        "port": 2200,
        "keyPath": "/keys/bob",
        "remoteHermesPath": "/opt/hermes/bin/hermes",
        "remoteProfile": "research",
    })
    assert alice.dial_identity() != bob.dial_identity()
    assert alice.connection_id != bob.connection_id


def test_open_isolated_fails_closed_when_existing_manifest_does_not_match_selected_row(
    tmp_path,
):
    from hermes_cli.desktop_instances import IsolatedInstanceSpecError

    store = _store(tmp_path)
    store.create(
        "lab",
        connection_id="alice-box",
        ssh_host="lab.example",
        ssh_user="alice",
        ssh_port=22,
        ssh_key_path="/keys/alice",
        remote_hermes_path="/opt/hermes/bin/hermes",
        remote_profile="default",
        skip_ssh_check=True,
        install_shortcut=False,
    )
    with pytest.raises(IsolatedInstanceSpecError, match="no longer matches"):
        store.open_from_connection({
            "id": "alice-box",
            "kind": "ssh",
            "label": "Lab",
            "host": "lab.example",
            "user": "alice",
            "port": 2200,
            "keyPath": "/keys/alice",
            "remoteHermesPath": "/opt/hermes/bin/hermes",
            "remoteProfile": "default",
        })


def test_open_isolated_updates_matching_connection_when_only_display_name_changes(
    tmp_path,
):
    store = _store(tmp_path)
    store.create(
        "lab",
        connection_id="alice-box",
        ssh_host="lab.example",
        ssh_user="alice",
        ssh_port=22,
        ssh_key_path="/keys/alice",
        remote_hermes_path="/opt/hermes/bin/hermes",
        remote_profile="default",
        display_name="Hermes Lab",
        skip_ssh_check=True,
        install_shortcut=False,
    )
    instance = store.open_from_connection({
        "id": "alice-box",
        "kind": "ssh",
        "label": "Hermes Lab",
        "host": "lab.example",
        "user": "alice",
        "port": 22,
        "keyPath": "/keys/alice",
        "remoteHermesPath": "/opt/hermes/bin/hermes",
        "remoteProfile": "default",
    })
    assert instance.connection_id == "alice-box"
    assert instance.ssh_user == "alice"
    assert instance.ssh_port == 22


def test_default_process_starter_scrubs_parent_provider_secrets(tmp_path, monkeypatch):
    from hermes_cli.desktop_instances import (
        LaunchPlan,
        default_process_starter,
    )

    captured: dict[str, object] = {}

    class FakePopen:
        def __init__(self, args, cwd=None, env=None):
            captured["args"] = args
            captured["cwd"] = cwd
            captured["env"] = env
            self.pid = 4242

    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-parent")
    monkeypatch.setattr(
        "hermes_cli.desktop_instances.subprocess.Popen",
        FakePopen,
    )
    plan = LaunchPlan(
        executable=tmp_path / "Hermes Grace.exe",
        arguments=["--user-data-dir=/u"],
        env={
            "HERMES_HOME": str(tmp_path / "home"),
            "HERMES_DESKTOP_INSTANCE": "grace",
        },
        cwd=str(tmp_path),
    )
    assert default_process_starter(plan) == 4242
    env = captured["env"]
    assert env["HERMES_HOME"] == str(tmp_path / "home")
    assert env["HERMES_DESKTOP_INSTANCE"] == "grace"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "sk-parent-secret" not in " ".join(str(value) for value in env.values())


def test_instance_spec_rejects_non_ssh_and_missing_remote_path():
    from hermes_cli.desktop_instances import (
        IsolatedInstanceSpecError,
        isolated_instance_spec_from_ssh,
    )

    with pytest.raises(IsolatedInstanceSpecError, match="SSH"):
        isolated_instance_spec_from_ssh({"kind": "remote", "label": "box", "host": "x"})
    with pytest.raises(IsolatedInstanceSpecError, match="absolute"):
        isolated_instance_spec_from_ssh({
            "kind": "ssh",
            "label": "box",
            "host": "lab",
            "remoteHermesPath": "rel",
        })


def test_parse_instance_deep_link_extracts_slug_and_remainder():
    from hermes_cli.desktop_instances import parse_instance_deep_link

    parsed = parse_instance_deep_link("hermes://instance/grace/blueprint/morning")
    assert parsed.instance_name == "grace"
    assert parsed.remainder == "hermes://blueprint/morning"
    assert parse_instance_deep_link("hermes://blueprint/morning") is None


def test_list_and_show_work_on_non_windows(tmp_path):
    win = _store(tmp_path)
    _create(win)
    linux = _store(tmp_path, platform="darwin")
    names = [item.name for item in linux.list()]
    assert names == ["grace"]
    shown = linux.get("grace")
    assert shown.ssh_host == "grace"


# ── native launcher source ───────────────────────────────────────────────


def test_generated_launcher_source_has_validated_windows_invariants(tmp_path):
    store = _store(tmp_path)
    instance = _create(store)
    source = store.render_launcher_source(instance)
    assert "UseShellExecute = false" in source
    assert 'Arguments = "--user-data-dir=\\"" + UserData + "\\""' in source
    assert 'EnvironmentVariables["HERMES_HOME"]' in source
    assert 'EnvironmentVariables["HERMES_DESKTOP_USER_DATA_DIR"]' in source
    assert 'EnvironmentVariables["HERMES_DESKTOP_HERMES_ROOT"]' in source
    assert 'EnvironmentVariables["HERMES_DESKTOP_APP_NAME"]' in source
    assert 'EnvironmentVariables["HERMES_DESKTOP_INSTANCE"]' in source
    assert 'EnvironmentVariables["HERMES_DESKTOP_AUMID"]' in source
    assert 'EnvironmentVariables["HERMES_DESKTOP_DISABLE_GLOBAL_SHORTCUTS"]' in source
    assert "CreateHardLink" in source
    assert "File.Delete" in source
    assert instance.app_name in source
    assert str(instance.user_data) in source
    assert str(instance.hermes_home) in source
    assert str(instance.named_exe) in source
    assert str(store.canonical_exe) in source
    assert "RunAsInvoker" not in source
    assert "--no-sandbox" not in source
    # Generated launchers may live under the current user's home; they must
    # not bake in the validated live-workaround paths from this machine.
    assert r"AppData\Local\hermes-launchers" not in source
    assert r"AppData\Local\hermes-instances" not in source


# ── hardlink refresh / lock policy ───────────────────────────────────────


def test_repair_refreshes_named_hardlink_to_canonical_bytes(tmp_path):
    store = _store(tmp_path)
    instance = _create(store)
    assert instance.named_exe.exists()
    store.canonical_exe.write_text("updated-canonical", encoding="utf-8")
    result = store.repair(instance.name)
    assert result.refreshed is True
    assert result.retained_running is False
    assert instance.named_exe.read_text(encoding="utf-8") == "updated-canonical"


def test_repair_retains_named_exe_when_running_instance_locks_it(tmp_path):
    store = _store(tmp_path)
    instance = _create(store)
    instance.named_exe.write_text("in-use", encoding="utf-8")

    def locked(path):
        return Path(path) == instance.named_exe

    store.is_locked = locked
    result = store.repair(instance.name)
    assert result.retained_running is True
    assert instance.named_exe.read_text(encoding="utf-8") == "in-use"


def test_launch_refreshes_hardlink_then_starts_named_exe(tmp_path):
    launched: list[Any] = []

    def starter(plan):
        launched.append(plan)
        return 4242

    store = _store(tmp_path, process_starter=starter)
    instance = _create(store, install_shortcut=False)
    store.canonical_exe.write_text("after-update", encoding="utf-8")
    pid = store.launch(instance.name)
    assert pid == 4242
    assert len(launched) == 1
    plan = launched[0]
    assert plan.executable == instance.named_exe
    assert plan.use_shell_execute is False
    assert instance.named_exe.read_text(encoding="utf-8") == "after-update"


# ── shortcut / remove ────────────────────────────────────────────────────


def test_create_installs_shortcut_to_native_launcher(tmp_path):
    written = []

    def writer(spec):
        written.append(spec)
        _touch(spec.path, spec.target)

    store = _store(tmp_path, shortcut_writer=writer)
    instance = _create(store)
    assert instance.shortcut_path.exists()
    assert written[0].target == str(instance.launcher_exe)
    assert written[0].icon == str(store.canonical_exe)
    assert instance.launcher_exe.exists()


def test_remove_drops_launcher_and_shortcut_but_not_remote_or_local_state(tmp_path):
    store = _store(tmp_path)
    instance = _create(store)
    (instance.hermes_home / "notes.txt").write_text(
        "keep local cache", encoding="utf-8"
    )
    store.remove(instance.name)
    assert not instance.manifest_path.exists()
    assert not instance.launcher_exe.exists()
    assert not instance.shortcut_path.exists()
    assert not instance.named_exe.exists()
    assert (instance.hermes_home / "notes.txt").read_text(
        encoding="utf-8"
    ) == "keep local cache"
    assert (instance.user_data / "connection.json").exists()


def test_remove_purge_local_still_does_not_claim_remote_deletion(tmp_path):
    store = _store(tmp_path)
    instance = _create(store)
    result = store.remove(instance.name, purge_local=True)
    assert result.remote_state_deleted is False
    assert not instance.hermes_home.exists()
    assert not instance.user_data.exists()


def test_remove_refuses_running_instance_unless_forced(tmp_path):
    from hermes_cli.desktop_instances import InstanceLockedError

    store = _store(tmp_path)
    instance = _create(store)
    store.is_locked = lambda path: Path(path) == instance.named_exe
    with pytest.raises(InstanceLockedError):
        store.remove(instance.name)
    assert instance.manifest_path.exists()
    store.remove(instance.name, force=True)
    assert not instance.manifest_path.exists()


def test_create_can_be_retried_if_launcher_materialize_fails(tmp_path):
    calls = {"n": 0}

    def flaky_compiler(source, output):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("csc unavailable")
        _touch(output, "compiled-launcher")

    store = _store(tmp_path, compiler=flaky_compiler)
    with pytest.raises(RuntimeError, match="csc unavailable"):
        _create(store)
    assert store.list() == []
    instance = _create(store)
    assert instance.manifest_path.exists()
    assert instance.launcher_exe.exists()


def test_unknown_instance_has_a_useful_error(tmp_path):
    from hermes_cli.desktop_instances import InstanceNotFoundError

    store = _store(tmp_path)
    with pytest.raises(InstanceNotFoundError, match="athena"):
        store.get("athena")


# ── CLI parser: existing desktop launch stays default ────────────────────


def test_instance_list_does_not_require_desktop_source_tree(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli.desktop_instances import DesktopInstanceStore

    root = tmp_path / "not-a-desktop-checkout"
    root.mkdir()
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    store = DesktopInstanceStore.from_defaults(
        runtime_root=root, hermes_root=tmp_path / "isolated-root"
    )
    assert store.list() == []

    ns = argparse.Namespace(
        desktop_action="instance",
        instance_action="list",
        cwd=None,
    )
    cli_main.cmd_gui(ns)
    assert "No isolated Desktop instances" in capsys.readouterr().out


def test_bare_desktop_command_still_dispatches_to_cmd_gui():
    seen = {}

    def cmd_gui(args):
        seen["args"] = args
        return "launched"

    parser = _gui_parser(cmd_gui=cmd_gui)
    ns = parser.parse_args(["desktop", "--skip-build"])
    assert ns.func is cmd_gui
    assert ns.skip_build is True
    assert getattr(ns, "desktop_action", None) in (None, "")


def test_instance_create_parser_collects_ssh_fields():
    parser = _gui_parser()
    ns = parser.parse_args([
        "desktop",
        "instance",
        "create",
        "athena",
        "--ssh-host",
        "bear-agent",
        "--remote-hermes-path",
        "/home/bear/.local/bin/hermes",
        "--remote-profile",
        "default",
        "--display-name",
        "Hermes Athena",
    ])
    assert ns.desktop_action == "instance"
    assert ns.instance_action == "create"
    assert ns.instance_name == "athena"
    assert ns.ssh_host == "bear-agent"
    assert ns.remote_hermes_path == "/home/bear/.local/bin/hermes"
    assert ns.remote_profile == "default"
    assert ns.display_name == "Hermes Athena"


def test_instance_list_launch_shortcut_repair_remove_parsers():
    parser = _gui_parser()
    listed = parser.parse_args(["desktop", "instance", "list"])
    assert listed.instance_action == "list"
    launched = parser.parse_args(["desktop", "instance", "launch", "grace"])
    assert launched.instance_action == "launch"
    assert launched.instance_name == "grace"
    shortcut = parser.parse_args(["gui", "instance", "shortcut", "grace"])
    assert shortcut.instance_action == "shortcut"
    repaired = parser.parse_args(["desktop", "instance", "repair", "--all"])
    assert repaired.instance_action == "repair"
    assert repaired.all_instances is True
    removed = parser.parse_args([
        "desktop",
        "instance",
        "remove",
        "grace",
        "--purge-local",
    ])
    assert removed.purge_local is True
