"""Service consent through real config/discovery/install; only OS peers are fake."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import pytest

from hermes_cli import config as config_api
from hermes_cli import gateway
from hermes_cli import gateway_setup_service as setup_service


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    # The installer intentionally rejects /tmp profiles. A disposable RAM-backed
    # home exercises that real guard without disabling it or touching user state.
    with tempfile.TemporaryDirectory(prefix="hermes-choice-", dir="/dev/shm") as directory:
        home = Path(directory)
        profile = home / "profile"
        profile.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("HERMES_HOME", str(profile))
        runtime = home / "run"
        runtime.mkdir()
        (runtime / "bus").touch()
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        calls = tmp_path / "calls.jsonl"
        state = tmp_path / "active"
        for name in ("systemctl", "loginctl", "sudo"):
            exe = bin_dir / name
            exe.write_text(
                f"#!{sys.executable}\n"
                "import json, os, pathlib, sys\n"
                f"calls = pathlib.Path({str(calls)!r})\n"
                "with calls.open('a') as stream: stream.write(json.dumps([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')\n"
                f"active = pathlib.Path({str(state)!r})\n"
                "args = sys.argv[1:]\n"
                "if 'enable' in args and os.getenv('CHOICE_FAIL'): sys.exit(1)\n"
                "if 'start' in args: active.touch()\n"
                "if 'is-active' in args: print('active' if active.exists() else 'inactive')\n"
                "elif 'show-user' in args: print('yes')\n"
                "elif 'show' in args: print('LoadState=not-found')\n"
                "elif 'is-system-running' in args: print('running')\n",
                encoding="utf-8",
            )
            exe.chmod(0o700)
        monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
        # Inject the owned executable at the OS command boundary, retaining the
        # live-system guard (which correctly blocks a command named systemctl).
        peer = bin_dir / "supervisor-peer"
        peer.write_bytes((bin_dir / "systemctl").read_bytes())
        peer.chmod(0o700)
        monkeypatch.setattr(gateway, "_systemctl_cmd", lambda system=False: [str(peer)])
        yield profile, calls


def _operations(calls):
    return [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []


@pytest.mark.linux_only
@pytest.mark.parametrize("choice", [None, "decline", "install"])
@pytest.mark.parametrize("installed", [False, True])
def test_import_and_ordinary_setup_never_authorize_install(supervisor, choice, installed):
    profile, calls = supervisor
    config_api.save_config({"gateway": {"service_install_choice": choice}})
    if installed:
        unit = gateway.get_systemd_unit_path()
        unit.parent.mkdir(parents=True)
        unit.write_text(gateway.generate_systemd_unit(), encoding="utf-8")
    before = (profile / "config.yaml").read_bytes()
    assert setup_service.ensure_gateway_service() is installed
    assert setup_service.ensure_gateway_service(interactive=True) is installed
    assert setup_service.ensure_gateway_service(context="import", install=True) is installed
    from hermes_cli.backup import _revive_gateway_after_import
    _revive_gateway_after_import(profile)
    assert gateway.get_systemd_unit_path().exists() is installed
    forbidden = {"enable", "enable-linger", "daemon-reload"}
    if not installed:
        forbidden.add("start")
    assert not any(set(op) & forbidden for op in _operations(calls))
    assert (profile / "config.yaml").read_bytes() == before


@pytest.mark.linux_only
@pytest.mark.parametrize("answer, fail", [(False, False), (True, False), (True, True)])
def test_explicit_setup_consent_is_durable_only_after_success(supervisor, monkeypatch, answer, fail):
    profile, calls = supervisor
    config_api.save_config({"gateway": {"service_install_choice": None}, "custom": "keep"})
    prompts = []
    monkeypatch.setattr(gateway, "prompt_yes_no", lambda *args: prompts.append(args) or answer)
    stream = io.StringIO()
    stream.isatty = lambda: True
    monkeypatch.setattr(sys, "stdin", stream)
    if fail:
        monkeypatch.setenv("CHOICE_FAIL", "1")
    config = config_api.load_config()
    result = setup_service.ensure_gateway_service(interactive=True, config=config)
    expected = None if fail else ("install" if answer else "decline")
    assert config_api.load_config()["gateway"]["service_install_choice"] == expected
    assert config["gateway"]["service_install_choice"] == expected
    assert config_api.load_config()["custom"] == "keep"
    assert result is (answer and not fail)
    assert len(prompts) == 1
    if not fail:
        setup_service.ensure_gateway_service(interactive=True)
        assert len(prompts) == 1
    if not answer:
        assert not gateway.get_systemd_unit_path().exists()
        assert not any(set(op) & {"enable", "enable-linger", "start", "daemon-reload"} for op in _operations(calls))


@pytest.mark.linux_only
@pytest.mark.parametrize("start_now", [False, True])
@pytest.mark.parametrize("consent", [False, True])
def test_wizard_service_choice_controls_installation(supervisor, monkeypatch, start_now, consent):
    profile, calls = supervisor
    config_api.save_config({"gateway": {"service_install_choice": None}})
    stream = io.StringIO()
    stream.isatty = lambda: True
    monkeypatch.setattr(sys, "stdin", stream)
    answers = iter([start_now, consent])
    monkeypatch.setattr(gateway, "prompt_yes_no", lambda *args: next(answers))
    monkeypatch.setattr(gateway, "prompt_choice", lambda *args, **kwargs: 0)
    spawned = []
    # Popen is the unmanaged process boundary; no real gateway is launched.
    original_popen = gateway.subprocess.Popen

    def popen(argv, *args, **kwargs):
        if "hermes_cli.main" in argv:
            spawned.append((argv, kwargs))
            return None
        return original_popen(argv, *args, **kwargs)

    monkeypatch.setattr(gateway.subprocess, "Popen", popen)
    setup_service._wizard_install_service("systemd")
    assert gateway.get_systemd_unit_path().exists() is consent
    assert bool(spawned) is (start_now and not consent)
    assert config_api.load_config()["gateway"]["service_install_choice"] == ("install" if consent else "decline")
    if spawned:
        assert "--replace" not in spawned[0][0]


@pytest.mark.linux_only
@pytest.mark.parametrize("fail", [False, True])
def test_explicit_install_records_only_completed_install(supervisor, monkeypatch, fail):
    from argparse import Namespace
    import subprocess

    config_api.save_config({"gateway": {"service_install_choice": "decline"}})
    if fail:
        monkeypatch.setenv("CHOICE_FAIL", "1")
    args = Namespace(start_now=False, start_on_login=True)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            gateway._cmd_install(args)
    else:
        gateway._cmd_install(args)
    assert config_api.load_config()["gateway"]["service_install_choice"] == ("decline" if fail else "install")


@pytest.mark.linux_only
def test_wizard_start_does_not_spawn_over_reserved_owner(supervisor, monkeypatch):
    from gateway.runtime_ownership import ProfileOwnership
    profile, calls = supervisor
    profile.chmod(0o700)
    config_api.save_config({"gateway": {"service_install_choice": "decline"}})
    stream = io.StringIO()
    stream.isatty = lambda: True
    monkeypatch.setattr(sys, "stdin", stream)
    monkeypatch.setattr(gateway, "prompt_yes_no", lambda *args: True)
    spawned = []
    original = gateway.subprocess.Popen
    def boundary(argv, *args, **kwargs):
        if "hermes_cli.main" in argv:
            spawned.append(argv)
            return None
        return original(argv, *args, **kwargs)
    monkeypatch.setattr(gateway.subprocess, "Popen", boundary)
    owner = ProfileOwnership()
    owner.reserve([profile])
    try:
        setup_service._wizard_install_service("systemd")
        assert not spawned
    finally:
        owner.close()
