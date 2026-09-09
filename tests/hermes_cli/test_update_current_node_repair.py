"""The commit_count == 0 path must repair Node deps, not just Python (#77211).

A previous ``hermes update`` whose npm install failed printed "Fix npm and
re-run `hermes update`" — but re-running hit the "Already up to date!" early
return before the Node refresh, so the advice could never work. The repair
now runs through ``_repair_node_deps_on_current_checkout``, which delegates
to ``_update_node_dependencies`` (self-gating on the lockfile hash, recorded
only after a successful install, so healthy installs stay a cheap no-op).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli import update_cmd, update_cmd_deps


def test_current_checkout_repairs_failed_node_deps(capsys):
    """A recorded failure surfaces the fix-npm hint, not 'Already up to date!'."""
    completion = MagicMock()
    with patch.object(
        update_cmd, "_update_node_dependencies", return_value=["ui-tui, web workspaces"]
    ), patch.object(update_cmd, "_m") as m:
        update_cmd._repair_node_deps_on_current_checkout(completion)

    m.return_value._build_web_ui.assert_not_called()
    completion.assert_called_once()
    assert "could not be repaired" in completion.call_args[0][0]
    out = capsys.readouterr().out
    assert "Node.js refresh failed for: ui-tui, web workspaces" in out
    assert "Fix npm and re-run `hermes update`." in out


def test_current_checkout_healthy_node_deps_reports_up_to_date():
    """A clean refresh (or lockfile-hash no-op) still says 'Already up to date!'."""
    completion = MagicMock()
    with patch.object(
        update_cmd, "_update_node_dependencies", return_value=[]
    ), patch.object(update_cmd, "_m") as m, patch.object(
        update_cmd, "_rebuild_desktop_after_update", return_value=True
    ):
        update_cmd._repair_node_deps_on_current_checkout(completion)

    # The refresh pairs with the web build like every other call site.
    m.return_value._build_web_ui.assert_called_once()
    completion.assert_called_once_with("✓ Already up to date!")


def test_force_bypasses_the_lockfile_unchanged_early_return(monkeypatch, tmp_path):
    """A Node *runtime* upgrade changes no npm manifest, so the lockfile-hash gate would normally
    skip the reinstall — force=True must run it anyway (issue #106456: native modules need
    rebuilding against the new Node ABI even though the lockfile itself didn't change)."""
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    ran_install = {"called": False}

    def fake_run_npm_install_deterministic(*args, **kwargs):
        ran_install["called"] = True
        return SimpleNamespace(returncode=0, stderr="")

    fake_m = SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        _resolve_node_runtime_npm=lambda: "/usr/bin/npm",
        _is_windows_npm_path=lambda _: False,
        _npm_lockfile_changed=lambda _: False,
        _nixos_build_env=lambda: {},
        _run_npm_install_deterministic=fake_run_npm_install_deterministic,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_m)
    monkeypatch.setattr(update_cmd_deps, "_record_npm_lockfile_hash", lambda root: None)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda env=None: env or {})
    monkeypatch.setattr(
        "tools.browser_tool_install.warm_agent_browser_npx_cache", lambda: True, raising=False)

    failures = update_cmd_deps._update_node_dependencies(force=True)

    assert ran_install["called"] is True
    assert failures == []


def test_force_does_not_bypass_the_no_package_json_early_return(monkeypatch, tmp_path):
    """`force` only bypasses the lockfile-freshness gate. A checkout with no package.json at all
    (no Node project here) is a different, legitimate early return — force=True must not run npm
    install anyway just because the caller wants a rebuild."""
    fake_m = SimpleNamespace(PROJECT_ROOT=tmp_path)
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_m)

    failures = update_cmd_deps._update_node_dependencies(force=True)

    assert failures == []


def test_force_does_not_bypass_unresolvable_npm(monkeypatch, tmp_path):
    """No usable npm at all (not even a Windows one from WSL) is a hard stop regardless of
    `force` — forcing a rebuild can't work without an npm to run it with."""
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    fake_m = SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        _resolve_node_runtime_npm=lambda: None,
        _is_windows_npm_path=lambda _: False,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_m)
    monkeypatch.setattr("hermes_constants.is_wsl", lambda: False, raising=False)

    failures = update_cmd_deps._update_node_dependencies(force=True)

    assert failures == []


def test_force_still_reports_failure_when_npm_install_fails(monkeypatch, tmp_path, capsys):
    """force=True bypassing the lockfile gate must not swallow a genuine npm install failure —
    the caller (doctor --upgrade-node) needs the failure label to report a broken upgrade rather
    than a false 'success'."""
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")

    def failing_run_npm_install_deterministic(*args, **kwargs):
        return SimpleNamespace(returncode=1, stderr="npm ERR! something broke\n")

    fake_m = SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        _resolve_node_runtime_npm=lambda: "/usr/bin/npm",
        _is_windows_npm_path=lambda _: False,
        _npm_lockfile_changed=lambda _: False,
        _nixos_build_env=lambda: {},
        _run_npm_install_deterministic=failing_run_npm_install_deterministic,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_m)
    hash_recorded = {"called": False}
    monkeypatch.setattr(
        update_cmd_deps, "_record_npm_lockfile_hash",
        lambda root: hash_recorded.__setitem__("called", True))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda env=None: env or {})
    monkeypatch.setattr(
        "tools.browser_tool_install.warm_agent_browser_npx_cache", lambda: True, raising=False)

    failures = update_cmd_deps._update_node_dependencies(force=True)

    assert failures == ["ui-tui, web workspaces"]
    assert hash_recorded["called"] is False
    assert "npm install failed" in capsys.readouterr().out


def test_force_records_lockfile_hash_on_success_so_the_next_plain_update_is_a_no_op(monkeypatch, tmp_path):
    """After a forced rebuild succeeds, the lockfile hash must still be recorded — otherwise every
    subsequent plain `hermes update` (force=False) would think the lockfile is still unrecorded
    and keep reinstalling, defeating the whole point of the freshness gate."""
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")

    def fake_run_npm_install_deterministic(*args, **kwargs):
        return SimpleNamespace(returncode=0, stderr="")

    fake_m = SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        _resolve_node_runtime_npm=lambda: "/usr/bin/npm",
        _is_windows_npm_path=lambda _: False,
        _npm_lockfile_changed=lambda _: False,
        _nixos_build_env=lambda: {},
        _run_npm_install_deterministic=fake_run_npm_install_deterministic,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_m)
    recorded_roots = []
    monkeypatch.setattr(
        update_cmd_deps, "_record_npm_lockfile_hash", lambda root: recorded_roots.append(root))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda env=None: env or {})
    monkeypatch.setattr(
        "tools.browser_tool_install.warm_agent_browser_npx_cache", lambda: True, raising=False)

    failures = update_cmd_deps._update_node_dependencies(force=True)

    assert failures == []
    assert recorded_roots == [tmp_path]
