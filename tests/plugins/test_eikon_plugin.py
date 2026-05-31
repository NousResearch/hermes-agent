"""Tests for the bundled eikon install plugin."""

from __future__ import annotations

import json
import os
from pathlib import Path

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.registry import registry
from toolsets import resolve_toolset


def _ctx() -> PluginContext:
    mgr = PluginManager()
    manifest = PluginManifest(name="eikon", key="eikon", kind="backend")
    return PluginContext(manifest, mgr)


def test_register_adds_default_eikon_install_tool() -> None:
    from plugins.eikon import register

    registry.deregister("eikon_install")
    register(_ctx())

    entry = registry.get_entry("eikon_install")
    assert entry is not None
    assert entry.toolset == "eikon"
    assert entry.schema["name"] == "eikon_install"
    assert "eikon_install" in resolve_toolset("hermes-cli")

    registry.deregister("eikon_install")


def test_handler_invokes_herm_eikon_install_json(monkeypatch, tmp_path: Path) -> None:
    from plugins.eikon.tools import _handle_eikon_install, check_herm_available

    args_file = tmp_path / "args.json"
    herm = tmp_path / "herm"
    herm.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['ARGS_FILE']).write_text(json.dumps(sys.argv[1:]))\n"
        "print(json.dumps({'ok': True, 'name': 'war', 'n': 0, 'bytes': 0, 'sources': {}, 'active': None}))\n",
        encoding="utf-8",
    )
    herm.chmod(0o755)
    monkeypatch.setenv("HERM_EIKON_HERM_BIN", str(herm))
    monkeypatch.setenv("ARGS_FILE", str(args_file))

    assert check_herm_available() is True
    result = json.loads(_handle_eikon_install({
        "source": "https://eikon.liftaris.dev/eikons/ares/",
        "name": "war",
        "media": False,
        "set_active": False,
    }))

    assert result == {"ok": True, "name": "war", "n": 0, "bytes": 0, "sources": {}, "active": None}
    assert json.loads(args_file.read_text(encoding="utf-8")) == [
        "eikon", "install", "https://eikon.liftaris.dev/eikons/ares/",
        "--json", "--name", "war", "--no-source", "--no-use",
    ]


def test_handler_returns_json_error_on_failure(monkeypatch, tmp_path: Path) -> None:
    from plugins.eikon.tools import _handle_eikon_install

    herm = tmp_path / "herm"
    herm.write_text("#!/usr/bin/env sh\necho nope >&2\nexit 7\n", encoding="utf-8")
    herm.chmod(0o755)
    monkeypatch.setenv("HERM_EIKON_HERM_BIN", str(herm))

    result = json.loads(_handle_eikon_install({"source": "ares"}))

    assert result["ok"] is False
    assert "herm eikon install failed" in result["error"]
    assert "nope" in result["error"]


def test_handler_rejects_missing_source() -> None:
    from plugins.eikon.tools import _handle_eikon_install

    assert json.loads(_handle_eikon_install({"source": ""})) == {
        "ok": False,
        "error": "source is required",
    }
