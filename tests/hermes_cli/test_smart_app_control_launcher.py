"""Smart App Control blocks the unsigned distlib ``hermes.exe`` before it starts.

PowerShell resolves ``hermes`` to that exe and does not fall through to a
sibling ``.cmd``. Enforcement has to publish the command file and remove the
exe. Machines with Smart App Control off keep the native launcher.
"""
from pathlib import Path

import hermes_constants
from hermes_cli import _launchers
from hermes_cli._install_repair import ensure_windows_bin_launchers


def _managed(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    root = home / "hermes-agent"
    local = root / ".hermes" / "bin"
    bindir = home / "bin"
    local.mkdir(parents=True)
    bindir.mkdir()
    for name in ("hermes", "hermes-acp"):
        (local / f"{name}.exe").write_bytes(b"MZ local")
    monkeypatch.setenv("HERMES_HOME", str(home))
    hermes_constants._default_hermes_root_memo = None
    return root, bindir


def test_enforcing_app_control_publishes_cmd_and_drops_the_exe(tmp_path, monkeypatch):
    out = tmp_path / "bin"
    out.mkdir()
    exe = out / "hermes.exe"
    exe.write_bytes(b"MZ unsigned launcher")
    python = tmp_path / "store" / "python.exe"
    python.parent.mkdir()
    python.write_bytes(b"MZ python")

    def refuse_distlib():
        raise AssertionError("distlib must not mint an exe Smart App Control will block")

    monkeypatch.setattr(_launchers, "_load_script_maker", refuse_distlib)
    written = _launchers.mint_launcher(
        "hermes", tmp_path / "repo", out, python, None, windows=True, app_control=True,
    )

    assert written == out / "hermes.cmd"
    assert not exe.exists()
    body = written.read_text(encoding="utf-8")
    assert f'"{python}" -I -c ' in body


def test_enforcing_app_control_restages_an_existing_exe(tmp_path, monkeypatch):
    root, bindir = _managed(tmp_path, monkeypatch)
    store = tmp_path / "store"
    entry = store / "python-test"
    (entry / "bin").mkdir(parents=True)
    python = entry / "bin" / "python3"
    python.write_bytes(b"MZ store python")
    (entry / "python.exe").write_bytes(b"MZ store python")
    (store / "facts.json").write_text(
        '{"packages":{"python":{"entry":"python-test"}}}\n', encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    for name in ("hermes", "hermes-acp"):
        (bindir / f"{name}.exe").write_bytes(b"MZ blocked")
    monkeypatch.setattr(_launchers, "smart_app_control_enforcing", lambda: True)

    restored = ensure_windows_bin_launchers(root, windows=True, user_path_entries=[])

    assert {Path(path).name for path in restored} == {"hermes.cmd", "hermes-acp.cmd"}
    assert not (bindir / "hermes.exe").exists()
    assert not (bindir / "hermes-acp.exe").exists()
    cmd = (bindir / "hermes.cmd").read_text(encoding="utf-8")
    assert cmd.startswith("@echo off")
    assert f'"{python.resolve()}" -I -c ' in cmd


def test_app_control_off_leaves_a_healthy_exe(tmp_path, monkeypatch):
    root, bindir = _managed(tmp_path, monkeypatch)
    for name in ("hermes", "hermes-acp"):
        (bindir / f"{name}.exe").write_bytes(b"MZ healthy")
    monkeypatch.setattr(_launchers, "smart_app_control_enforcing", lambda: False)

    assert ensure_windows_bin_launchers(root, windows=True, user_path_entries=[]) == []
    assert (bindir / "hermes.exe").read_bytes() == b"MZ healthy"
