"""Profile-required plugin admission before an updated source tree restarts the gateway."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import venv
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import (
    main, update_cmd, update_cmd_zip, update_completion, update_pause_record, update_plugin_guards)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


@pytest.fixture
def guarded_profile(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "sample"
    plugin = profile / "plugins" / "context-check"
    plugin.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    profile.joinpath("config.yaml").write_text(
        "plugins:\n  enabled: [context-check]\n  update_guards: [context-check]\n",
        encoding="utf-8")
    plugin.joinpath("plugin.yaml").write_text(
        "name: context-check\nversion: 1.0.0\npost_update_check: check.py\n",
        encoding="utf-8")
    plugin.joinpath("check.py").write_text(
        """import argparse
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--hermes-root', required=True)
p.add_argument('--profile', required=True)
a = p.parse_args()
assert a.profile == 'sample'
assert 'HERMES_TEST_SECRET' not in os.environ
assert Path(a.hermes_root, 'compatible.txt').read_text() == 'ready'
""", encoding="utf-8")
    return home, profile, plugin


def test_guard_checks_exact_pulled_tree_without_ambient_credentials(
        guarded_profile, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_TEST_SECRET", "never pass to checker")
    source = tmp_path / "source"
    source.mkdir()
    (source / "compatible.txt").write_text("ready", encoding="utf-8")
    checks = update_plugin_guards.prepare_update_guards()
    assert len(checks) == 1 and checks[0].profile == "sample"
    assert update_plugin_guards.check_pulled_plugin_compatibility(source, checks) is None
    (source / "compatible.txt").write_text("broken", encoding="utf-8")
    assert "exited" in update_plugin_guards.check_pulled_plugin_compatibility(source, checks)


def test_guard_checks_selected_python_and_holds_paused_gateway_until_pass(
        guarded_profile, tmp_path):
    _, _, plugin = guarded_profile
    source = tmp_path / "source"
    source.mkdir()
    (source / "compatible.txt").write_text("ready", encoding="utf-8")
    selected = tmp_path / "selected-python"
    venv.EnvBuilder(with_pip=False).create(selected)
    interpreter = selected / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    (source / "expected-prefix.txt").write_text(str(selected.resolve()), encoding="utf-8")
    script = plugin / "check.py"
    script.write_text(script.read_text(encoding="utf-8") + "\n"
                      "import sys\n"
                      "assert Path(sys.prefix).resolve() == Path(Path(a.hermes_root, 'expected-prefix.txt').read_text())\n",
                      encoding="utf-8")
    checks = update_plugin_guards.prepare_update_guards()
    update_plugin_guards.arm_pending(source, checks)
    assert update_pause_record.tree_is_whole({}, source) == (
        False, "a required plugin has not passed the selected runtime check")
    assert "exited" in update_plugin_guards.check_pulled_plugin_compatibility(source, checks)
    assert update_plugin_guards.check_pulled_plugin_compatibility(
        source, checks, python_executable=interpreter) is None
    update_plugin_guards.clear_pending(source)
    assert update_pause_record.tree_is_whole({}, source) == (True, "")


def test_guarded_completion_transport_preserves_frozen_declarations(guarded_profile):
    checks = update_plugin_guards.prepare_update_guards()
    assert update_plugin_guards.guards_from_records(update_plugin_guards.guard_records(checks)) == checks


@pytest.mark.parametrize("compatible", [True, False])
def test_completion_runs_guard_under_selected_runtime_before_gateway_phase(
        guarded_profile, tmp_path, monkeypatch, compatible):
    import pm
    import pm.client
    import pm.environments
    import pm.receipt
    from hermes_cli import gitlock, venv_sync

    source = tmp_path / "checkout"
    source.mkdir()
    (source / "compatible.txt").write_text("ready" if compatible else "broken", encoding="utf-8")
    checks = update_plugin_guards.prepare_update_guards()
    update_plugin_guards.arm_pending(source, checks)
    events = []
    monkeypatch.setattr(venv_sync, "arm_completion", lambda root: None)
    monkeypatch.setattr(venv_sync, "collect_superseded_generations", lambda root: events.append("collect"))
    monkeypatch.setattr(gitlock, "convert_treeless_checkout_first", lambda root: None)
    monkeypatch.setattr(pm.client, "ensure_tools_for_sync", lambda: None)
    monkeypatch.setattr(pm, "sync_venv", lambda **kwargs: events.append("sync"))
    monkeypatch.setattr(pm.receipt, "worker_context", lambda update_id: nullcontext())
    monkeypatch.setattr(pm.receipt, "last_for_update", lambda update_id: None)
    monkeypatch.setattr(pm.environments, "project_python", lambda root: Path(sys.executable))
    monkeypatch.setattr(pm.environments, "activation_environment", lambda root: {})
    monkeypatch.setattr(update_completion.subprocess, "call", lambda *args, **kwargs: events.append("child") or 0)
    request = {"source": str(source), "receipt": {"update_id": "guarded"},
               "bytecode_cache": str(tmp_path / "bc"),
               "plugin_update_guards": update_plugin_guards.guard_records(checks)}
    request_path, result_path = tmp_path / "request.json", tmp_path / "result.json"
    if compatible:
        update_completion._prepare(request, request_path, result_path)
        assert events == ["sync", "collect", "child"]
        # The marker remains until the freshly selected interpreter enters _finish.
        assert update_plugin_guards.pending_path(source).exists()
    else:
        with pytest.raises(RuntimeError, match="required plugin failed under the selected runtime"):
            update_completion._prepare(request, request_path, result_path)
        assert events == ["sync"]
        assert update_plugin_guards.pending_path(source).exists()


def test_guard_declaration_is_profile_owned_and_fails_closed(guarded_profile, tmp_path):
    _, profile, plugin = guarded_profile
    plugin.joinpath("check.py").unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        update_plugin_guards.prepare_update_guards()
    plugin.joinpath("check.py").write_text("pass\n", encoding="utf-8")
    checks = update_plugin_guards.prepare_update_guards()
    plugin.joinpath("check.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    assert "changed during update" in update_plugin_guards.check_pulled_plugin_compatibility(tmp_path, checks)
    profile.joinpath("config.yaml").write_text("plugins:\n  enabled: []\n  update_guards: [context-check]\n")
    with pytest.raises(ValueError, match="not enabled"):
        update_plugin_guards.prepare_update_guards()


def test_enabled_plugin_declaration_runs_and_checker_path_cannot_escape(guarded_profile):
    _, profile, plugin = guarded_profile
    profile.joinpath("config.yaml").write_text("plugins:\n  enabled: [context-check]\n")
    assert len(update_plugin_guards.prepare_update_guards()) == 1
    profile.joinpath("config.yaml").write_text("plugins:\n  enabled: []\n")
    assert update_plugin_guards.prepare_update_guards() == ()
    profile.joinpath("config.yaml").write_text(
        "plugins:\n  enabled: [context-check]\n  update_guards: [context-check]\n")
    plugin.joinpath("plugin.yaml").write_text(
        "name: context-check\npost_update_check: ../other.py\n")
    with pytest.raises(ValueError, match="unsafe post_update_check path"):
        update_plugin_guards.prepare_update_guards()


def test_required_plugin_disabled_veto_refuses_before_checkout_move(guarded_profile):
    _, profile, _ = guarded_profile
    profile.joinpath("config.yaml").write_text(
        "plugins:\n  enabled: [context-check]\n  disabled: [context-check]\n"
        "  update_guards: [context-check]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="but it is disabled"):
        update_plugin_guards.prepare_update_guards()


def test_same_head_update_checks_before_dependency_completion(guarded_profile, tmp_path, monkeypatch):
    source = tmp_path / "checkout"
    source.mkdir()
    (source / "compatible.txt").write_text("broken", encoding="utf-8")
    monkeypatch.setattr(main, "PROJECT_ROOT", source)
    guards = update_plugin_guards.prepare_update_guards()

    def no_completion(_request):
        raise AssertionError("dependency completion must not start")

    monkeypatch.setattr(update_cmd, "_complete_source_update", no_completion)
    plan = SimpleNamespace(auto_stash_ref=None, parked_branch_switched=False)
    with pytest.raises(SystemExit) as stopped:
        update_cmd._finish_already_up_to_date(
            ["git"], "main", "main", plan, gw_input_fn=None,
            completion_request={}, plugin_update_guards=guards)
    assert stopped.value.code == 1


def test_failed_guard_rolls_git_source_back_and_fixed_retry_succeeds(
        guarded_profile, tmp_path, monkeypatch, capsys):
    home, _, _ = guarded_profile
    default_config = home / "config.yaml"
    lab_config = home / "profiles" / "local-lab" / "config.yaml"
    lab_config.parent.mkdir(parents=True)
    default_config.write_text("model:\n  default: existing-default\n", encoding="utf-8")
    lab_config.write_text("model:\n  default: existing-lab\n", encoding="utf-8")
    profile_configs_before = (default_config.read_bytes(), lab_config.read_bytes())
    source = tmp_path / "checkout"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.email", "test@example.invalid")
    _git(source, "config", "user.name", "Test")
    marker = source / "compatible.txt"
    marker.write_text("ready", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "working")
    previous = _git(source, "rev-parse", "HEAD")
    monkeypatch.setattr(main, "PROJECT_ROOT", source)
    guards = update_plugin_guards.prepare_update_guards()

    marker.write_text("broken", encoding="utf-8")
    _git(source, "commit", "-am", "incompatible upstream")
    _git(source, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(source, "reset", "--hard", previous)

    def pull():
        return update_cmd._pull_updates(
            ["git"], "main", None, prompt_for_restore=False, gw_input_fn=None,
            discard_local_changes=False, keep_stash=False, plugin_update_guards=guards)

    original_move = update_cmd._move_checkout_to

    def checked_move(*args):
        assert update_plugin_guards.pending_reason(source)
        return original_move(*args)

    def failed_move(*args):
        assert update_plugin_guards.pending_reason(source)
        raise OSError("test checkout move failure")

    monkeypatch.setattr(update_cmd, "_move_checkout_to", failed_move)
    with pytest.raises(OSError, match="checkout move failure"):
        pull()
    assert update_plugin_guards.pending_reason(source) is None
    assert _git(source, "rev-parse", "HEAD") == previous

    monkeypatch.setattr(update_cmd, "_move_checkout_to", checked_move)
    with pytest.raises(SystemExit) as stopped:
        pull()
    assert stopped.value.code == 1
    assert "required plugin update check" in capsys.readouterr().out
    assert _git(source, "rev-parse", "HEAD") == previous
    assert marker.read_text() == "ready"
    assert update_plugin_guards.pending_reason(source) is None
    assert (default_config.read_bytes(), lab_config.read_bytes()) == profile_configs_before

    _git(source, "reset", "--hard", "origin/main")
    marker.write_text("ready", encoding="utf-8")
    _git(source, "commit", "-am", "compatible upstream")
    corrected = _git(source, "rev-parse", "HEAD")
    _git(source, "update-ref", "refs/remotes/origin/main", corrected)
    _git(source, "reset", "--hard", previous)
    assert pull() == previous
    assert _git(source, "rev-parse", "HEAD") == corrected
    assert update_plugin_guards.pending_reason(source)
    assert (default_config.read_bytes(), lab_config.read_bytes()) == profile_configs_before

    # A first update could die after moving Z -> A but before runtime admission.
    # A second candidate B must not erase A's hold when it rolls back to A.
    marker.write_text("broken", encoding="utf-8")
    _git(source, "commit", "-am", "second incompatible upstream")
    _git(source, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(source, "reset", "--hard", corrected)
    with pytest.raises(SystemExit) as second:
        pull()
    assert second.value.code == 1
    assert _git(source, "rev-parse", "HEAD") == corrected
    assert update_pause_record.tree_is_whole({}, source)[0] is False
    from gateway import run_bootstrap
    from gateway.config import GatewayConfig

    pending = update_plugin_guards.pending_path(source)
    with monkeypatch.context() as scoped:
        scoped.setattr(update_plugin_guards, "pending_path", lambda _source: pending)
        assert asyncio.run(run_bootstrap.start_gateway(GatewayConfig(), replace=True)) is False

    # Returning all the way to the originally admitted revision discharges it.
    _git(source, "reset", "--hard", previous)
    assert _git(source, "status", "--porcelain") == ""
    from pm.environments import runtime_facts_path

    facts = runtime_facts_path(source)
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text('{"selected":"changed-after-partial-update"}', encoding="utf-8")
    update_plugin_guards.clear_after_verified_rollback(source, previous)
    assert update_plugin_guards.pending_reason(source)  # source alone is not an app+deps rollback
    facts.unlink()
    update_plugin_guards.clear_after_verified_rollback(source, previous)
    assert update_plugin_guards.pending_reason(source) is None


def test_guarded_zip_update_refuses_before_download(guarded_profile, monkeypatch, capsys):
    def no_download(*args, **kwargs):
        raise AssertionError("ZIP download must not start")
    monkeypatch.setattr(update_cmd_zip, "_download_and_swap_zip", no_download)
    with pytest.raises(SystemExit) as stopped:
        update_cmd_zip._update_via_zip(SimpleNamespace(), completion_request={})
    assert stopped.value.code == 1
    assert "cannot verify required plugin update checks" in capsys.readouterr().out
