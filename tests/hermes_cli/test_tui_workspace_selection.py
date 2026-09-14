"""Install selection and freshness must describe the same dependency closure."""
import json
from types import SimpleNamespace

import pytest

from hermes_cli import main_tui_launch as launch


@pytest.mark.parametrize("termux", [False, True])
def test_install_and_freshness_share_nested_workspace_roots(tmp_path, monkeypatch, termux):
    tui = tmp_path / "ui-tui"
    child = tui / "packages" / "hermes-ink"
    child.mkdir(parents=True)
    (tui / "package.json").write_text("{}", encoding="utf-8")
    (child / "package.json").write_text("{}", encoding="utf-8")
    keys = launch._tui_selected_workspace_keys(tui, tmp_path)
    assert "ui-tui/packages/hermes-ink" in keys
    wanted = {key: {} for key in keys}
    wanted["ui-tui/packages/hermes-ink"] = {"devDependencies": {"child-dev": "1"}}
    wanted["node_modules/child-dev"] = {"version": "1"}
    wanted["apps/desktop"] = {"dependencies": {"electron": "1"}}
    wanted["node_modules/electron"] = {"version": "1"}
    (tmp_path / "package-lock.json").write_text(json.dumps({"packages": wanted}), encoding="utf-8")
    ink = tmp_path / "node_modules" / "@hermes" / "ink"
    ink.mkdir(parents=True)
    (ink / "package.json").write_text("{}", encoding="utf-8")
    marker = tmp_path / "node_modules" / ".package-lock.json"
    marker.write_text('{"packages": {}}', encoding="utf-8")
    assert launch._tui_need_npm_install(tui)
    marker.write_text(json.dumps({"packages": {"node_modules/child-dev": {"version": "1"}}}), encoding="utf-8")
    assert not launch._tui_need_npm_install(tui)
    # A future selected workspace must reach argv without a second hardcoded list.
    keys.add("ui-tui/packages/future")
    monkeypatch.setattr(launch, "_tui_selected_workspace_keys", lambda *_: keys)
    monkeypatch.setattr(launch, "_tui_node_bin", lambda _: "npm")
    calls = []
    monkeypatch.setattr(launch.subprocess, "run", lambda argv, **kw: (calls.append((argv, kw)) or SimpleNamespace(returncode=0)))
    launch._install_tui_dependencies(tui, termux_startup=termux)
    argv, kw = calls[0]
    assert {argv[i + 1] for i, arg in enumerate(argv) if arg == "--workspace"} == keys
    assert "apps/desktop" not in argv
    assert "--include=dev" in argv
    assert ("--include-workspace-root=false" in argv) == termux
    assert kw["cwd"] == str(tmp_path)
    (tui / "package-lock.json").write_text("{}", encoding="utf-8")
    calls.clear()
    launch._install_tui_dependencies(tui, termux_startup=termux)
    assert "--workspace" not in calls[0][0]
    assert calls[0][1]["cwd"] == str(tui)


def test_updater_installs_ink_without_desktop(tmp_path, monkeypatch):
    from hermes_cli import main, update_cmd_deps as deps
    from tools import browser_tool_install
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "_resolve_node_runtime_npm", lambda: "npm")
    monkeypatch.setattr(main, "_npm_lockfile_changed", lambda _: True)
    monkeypatch.setattr(main, "_nixos_build_env", lambda: {})
    monkeypatch.setattr(browser_tool_install, "warm_agent_browser_npx_cache", lambda: None)
    monkeypatch.setattr(deps, "_record_npm_lockfile_hash", lambda _: None)
    calls = []
    monkeypatch.setattr(main, "_run_npm_install_deterministic", lambda *a, **kw: (calls.append((a, kw)) or SimpleNamespace(returncode=0)))
    assert deps._update_node_dependencies() == []
    args = calls[0][1]["extra_args"]
    assert "ui-tui/packages/hermes-ink" in args
    assert "ui-tui" in args and "web" in args
    assert "apps/desktop" not in args
    assert "--include-workspace-root" in args
    # A nested manifest edit must also invalidate the updater's skip cache.
    (tmp_path / "package.json").write_text(
        json.dumps({"workspaces": ["ui-tui", "ui-tui/packages/*"]}), encoding="utf-8"
    )
    child = tmp_path / "ui-tui" / "packages" / "hermes-ink"
    child.mkdir(parents=True)
    manifest = child / "package.json"
    manifest.write_text("{}", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    before = deps._npm_manifests_digest()
    manifest.write_text('{"dependencies":{"new-dep":"1"}}', encoding="utf-8")
    assert deps._npm_manifests_digest() != before
