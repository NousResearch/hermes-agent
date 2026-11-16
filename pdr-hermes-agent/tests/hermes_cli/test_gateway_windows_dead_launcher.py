"""A launcher whose interpreter is gone is re-rendered on the next ensure (#150448).

A Store (MSIX) update replaces the versioned payload folder the gateway launcher's ``python.exe``
lives in. Nothing that runs on a Store install re-rendered the launcher afterwards, so the logon
task or Startup entry ran a dead launcher, and an ensure that started the registered task waited out
its deadline with no gateway.
"""
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants
from hermes_cli import gateway, gateway_windows
from hermes_cli import gateway_runtime_service as service
from hermes_cli.gateway_runtime_service_identity import _vbs_identity, launcher_text

# Host-independent: run on every lane, including native Windows, where the launchers actually live.
pytestmark = pytest.mark.platforms("any")


def _payload(root, version):
    python = root / f"NousResearchInc.HermesAgent_{version}_x64__pub" / "agent-payload" / "python" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    return python


@pytest.fixture(params=["default", "custom"])
def store_update(request, tmp_path, monkeypatch):
    """A launcher installed from the old payload, then the Store update: old folder gone, new one current.

    Both naming branches: the platform-default home owns the bare task name (a Store user's usual
    home), any other root a hashed suffix."""
    home = hermes_constants._get_platform_default_hermes_home() if request.param == "default" else tmp_path / "custom-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    old, new = _payload(tmp_path / "WindowsApps", "2026.7363.3271.0"), _payload(tmp_path / "WindowsApps", "2026.7411.3516.0")
    suffix = service.service_suffix(home)
    name = f"Hermes_Gateway{'_' + suffix if suffix else ''}"  # what install names the task for this home
    cmd = home / "gateway-service" / f"{name}.cmd"
    cmd.parent.mkdir()
    monkeypatch.setattr(gateway, "get_python_path", lambda: str(old))
    gateway_windows._write_launcher_scripts(gateway_windows._render_launchers(gateway_windows._launcher_settings(home), cmd))
    shutil.rmtree(old.parents[2])
    monkeypatch.setattr(gateway, "get_python_path", lambda: str(new))
    return SimpleNamespace(home=home, name=name, vbs=cmd.with_suffix(".vbs"), cmd=cmd, old=old, new=new)


def _interpreter(vbs):
    return _vbs_identity(launcher_text(vbs.read_bytes()))[1][0]


def _fake_schtasks(monkeypatch, *, task=None):
    def run(argv, *_args, **_kwargs):
        if argv[0] == "whoami.exe":
            out = '"DESKTOP-1\\owner","S-1-5-21-1"\r\n'
        elif "/XML" in argv:
            out = gateway_windows._build_scheduled_task_xml(task.name, task.vbs, "DESKTOP-1\\owner")
        else:
            out = f'"\\{task.name}","N/A","Ready"\r\n' if task else ""
        return SimpleNamespace(returncode=0, stdout=out.encode(), stderr=b"")

    monkeypatch.setattr(service, "_run", run)


def test_ensure_rerenders_the_task_launcher_before_running_it(store_update, monkeypatch):
    _fake_schtasks(monkeypatch, task=store_update)

    found = service._windows(store_update.home, time.monotonic() + 5)

    assert found is not None and found.start_argv[-1] == store_update.name
    assert _interpreter(store_update.vbs) == str(store_update.new)
    assert str(store_update.old.parents[2]) not in store_update.cmd.read_text(encoding="utf-8")


def test_ensure_rerenders_the_startup_entry_launcher(store_update, monkeypatch):
    _fake_schtasks(monkeypatch)

    assert service._windows(store_update.home, time.monotonic() + 5) is None
    assert _interpreter(store_update.vbs) == str(store_update.new)


def test_an_unreadable_cmd_helper_blocks_neither_discovery_nor_the_vbs(store_update, monkeypatch):
    _fake_schtasks(monkeypatch, task=store_update)
    store_update.cmd.unlink()
    store_update.cmd.mkdir()  # exists, but reading or replacing it raises OSError

    found = service._windows(store_update.home, time.monotonic() + 5)

    assert found is not None and found.start_argv[-1] == store_update.name
    assert _interpreter(store_update.vbs) == str(store_update.new)


def test_an_unstattable_interpreter_does_not_stop_discovery(store_update, monkeypatch):
    """Python < 3.13 ``Path.exists()`` raises on access denied instead of answering False."""
    _fake_schtasks(monkeypatch, task=store_update)
    before = store_update.vbs.read_bytes()
    real_exists = Path.exists

    def exists(path, *args, **kwargs):
        if str(path) == str(store_update.old):
            raise PermissionError(13, "Access is denied", str(path))
        return real_exists(path, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)

    found = service._windows(store_update.home, time.monotonic() + 5)

    assert found is not None and found.start_argv[-1] == store_update.name
    assert store_update.vbs.read_bytes() == before


def test_a_named_profile_client_renders_the_root_launcher_for_the_root(store_update, monkeypatch):
    """A named profile's ensure discovers its multiplexer's service, so the repair runs for the root's
    home from a process whose active home is the profile: the launcher must still be the root's own."""
    installed = gateway_windows._render_launchers(gateway_windows._launcher_settings(store_update.home), store_update.cmd)
    profile = store_update.home / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    assert gateway_windows.refresh_dead_launcher(store_update.home, store_update.name)

    assert [path.read_bytes() for path, _, _ in installed] == [content.encode(enc) for _, content, enc in installed]


def test_the_repair_keeps_the_installed_home_spelling(store_update, monkeypatch):
    """Ensure passes the canonical (resolved) home, but install wrote the configured spelling, which may
    be a link to the real directory (as ``%LOCALAPPDATA%\\hermes`` can be): the repair keeps that spelling."""
    link = store_update.home.parent / "hermes-link"
    try:
        link.symlink_to(store_update.home, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem does not allow symlinks")
    monkeypatch.setenv("HERMES_HOME", str(link))
    cmd = link / "gateway-service" / store_update.cmd.name
    monkeypatch.setattr(gateway, "get_python_path", lambda: str(store_update.old))
    gateway_windows._write_launcher_scripts(gateway_windows._render_launchers(gateway_windows._launcher_settings(link), cmd))
    monkeypatch.setattr(gateway, "get_python_path", lambda: str(store_update.new))
    installed = gateway_windows._render_launchers(gateway_windows._launcher_settings(link), cmd)

    assert gateway_windows.refresh_dead_launcher(store_update.home.resolve(), store_update.name)

    assert [path.read_bytes() for path, _, _ in installed] == [content.encode(enc) for _, content, enc in installed]


def test_a_live_or_foreign_launcher_is_left_alone(store_update, monkeypatch):
    assert gateway_windows.refresh_dead_launcher(store_update.home, store_update.name)
    before = store_update.vbs.read_bytes()
    assert not gateway_windows.refresh_dead_launcher(store_update.home, store_update.name)
    assert store_update.vbs.read_bytes() == before

    edited = store_update.vbs.read_bytes().decode("utf-16").replace(str(store_update.new), str(store_update.old))
    store_update.vbs.write_bytes(edited.replace("Option Explicit", "Option Explicit\r\nMsgBox 1").encode("utf-16"))
    hand_written = store_update.vbs.read_bytes()
    assert not gateway_windows.refresh_dead_launcher(store_update.home, store_update.name)
    assert store_update.vbs.read_bytes() == hand_written

    other = store_update.home.parent / "other-home"
    other.mkdir()
    store_update.vbs.write_bytes(edited.replace(str(store_update.home), str(other)).encode("utf-16"))
    foreign = store_update.vbs.read_bytes()
    assert not gateway_windows.refresh_dead_launcher(store_update.home, store_update.name)
    assert store_update.vbs.read_bytes() == foreign
