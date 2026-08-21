from types import SimpleNamespace
import subprocess

import hermes_cli.memory_setup as memory_setup
import hermes_cli.web_server as web_server


BRV_CMD = r"C:\nvm4w\nodejs\brv.CMD"


def test_dashboard_memory_provider_check_resolves_windows_cmd_shim(monkeypatch):
    """Dashboard dependency checks must resolve npm .cmd shims on Windows.

    Native Windows CreateProcess cannot launch a bare npm shim name such as
    ``brv`` from a subprocess argv list. ``shutil.which`` resolves the command
    through PATHEXT to the actual ``brv.CMD`` launcher.
    """
    monkeypatch.setattr(web_server, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        web_server.shutil,
        "which",
        lambda name: BRV_CMD if name == "brv" else None,
    )

    assert web_server._memory_provider_command_argv("brv --version") == [
        BRV_CMD,
        "--version",
    ]


def test_dashboard_memory_provider_check_preserves_unresolved_command(monkeypatch):
    monkeypatch.setattr(web_server, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(web_server.shutil, "which", lambda _name: None)

    assert web_server._memory_provider_command_argv("missing-tool --version") == [
        "missing-tool",
        "--version",
    ]


def test_cli_memory_setup_resolves_windows_cmd_shim(tmp_path, monkeypatch):
    """The CLI setup path must use the resolved .cmd launcher too."""
    plugin_dir = tmp_path / "byterover"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        """\
pip_dependencies:\n  - sys\nexternal_dependencies:\n  - name: brv\n    check: brv --version\n    install: npm install -g byterover-cli\n""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "plugins.memory.find_provider_dir",
        lambda _name: plugin_dir,
    )
    monkeypatch.setattr(memory_setup, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        memory_setup.shutil,
        "which",
        lambda name: BRV_CMD if name == "brv" else None,
    )

    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    memory_setup._install_dependencies("byterover")

    assert calls
    assert calls[-1][0] == [BRV_CMD, "--version"]
    assert calls[-1][1]["check"] is True
