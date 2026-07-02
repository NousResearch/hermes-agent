from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "tookie-osint"


def load_plugin():
    package_name = "tookie_osint_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def load_core():
    package_name = "tookie_osint_test_plugin"
    if package_name not in sys.modules:
        load_plugin()
    return sys.modules[f"{package_name}.core"]


def write_fake_root(root: Path) -> None:
    (root / "sites").mkdir(parents=True)
    (root / "config").mkdir()
    (root / "brib.py").write_text("print('fake')\n", encoding="utf-8")
    (root / "sites" / "sites.json").write_text(
        json.dumps([{"site": "https://example.com/"}]), encoding="utf-8"
    )
    (root / "sites" / "headers.txt").write_text("ua\n", encoding="utf-8")
    (root / "config" / "version").write_text("test\n", encoding="utf-8")
    (root / "requirements.txt").write_text("requests\ncolorama\n", encoding="utf-8")


def test_register_exposes_tools_and_cli_command():
    plugin = load_plugin()

    class Ctx:
        def __init__(self):
            self.tools = []
            self.commands = []
            self.cli_commands = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_command(self, *args, **kwargs):
            self.commands.append((args, kwargs))

        def register_cli_command(self, **kwargs):
            self.cli_commands.append(kwargs)

    ctx = Ctx()
    plugin.register(ctx)

    assert {tool["name"] for tool in ctx.tools} == {
        "tookie_status",
        "tookie_scan_username",
    }
    assert all(tool["toolset"] == "tookie_osint" for tool in ctx.tools)
    assert ctx.commands[0][0][0] == "tookie-osint"
    assert ctx.cli_commands[0]["name"] == "tookie-osint"


def test_status_without_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("TOOKIE_OSINT_ROOT", raising=False)
    core = load_core()

    payload = core.status_payload({})

    assert payload["available"] is False
    assert payload["root"] == ""
    assert "setup_hint" in payload


def test_save_root_and_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    root = tmp_path / "tookie-osint"
    write_fake_root(root)
    core = load_core()

    saved = core.save_root(root)
    payload = core.status_payload({})

    assert saved == root.resolve()
    assert payload["root_valid"] is True
    assert payload["site_count"] == 1


def test_scan_username_runs_in_isolated_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    root = tmp_path / "tookie-osint"
    write_fake_root(root)
    core = load_core()
    core.save_root(root)
    monkeypatch.setattr(core, "_missing_imports", lambda *, webscraper=False: [])

    def fake_run(cmd, cwd, **_kwargs):
        Path(cwd, "alice.json").write_text(
            json.dumps([{"url": "https://example.com/alice", "found": True}]),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(core.subprocess, "run", fake_run)

    payload = core.scan_username({"username": "alice", "threads": 2})

    assert payload["success"] is True
    assert payload["result"][0]["found"] is True
    assert str(tmp_path / ".hermes" / "tookie-osint" / "runs") in payload["run_dir"]


def test_scan_username_rejects_path_like_username():
    core = load_core()

    payload = core.scan_username({"username": "../alice"})

    assert payload["success"] is False
    assert "username must" in payload["error"]
