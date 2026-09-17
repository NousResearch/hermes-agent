"""Stale-module regressions now covered at the post-swap interpreter boundary.

The former purge missed new root modules (the utils/base_url_origin field failure)
and could recreate hermes_logging's listener or hermes_constants' ContextVar.
The fresh-interpreter handoff supersedes that mechanism: new code imports in the
child while parent module identities stay intact. Keep this test at its old path
during the sync so fork coverage survives the upstream modify/delete conflict.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants
import hermes_logging
from hermes_cli import update_cmd, update_cmd_config, update_handoff, update_receipt
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan


@pytest.fixture
def handoff_env(tmp_path, monkeypatch):
    """All files stay temporary; process discovery and code-identity probes are stubbed."""
    home = tmp_path / "home"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", checkout)
    monkeypatch.setattr(update_receipt, "_current", None)
    monkeypatch.setattr(update_receipt, "_code_identity", lambda refresh=False: {})
    monkeypatch.setattr(update_receipt, "_receipt_dir", lambda: home / "logs" / "update_receipts")
    monkeypatch.setattr(update_cmd_config, "_LAST_SIBLING_SNAPSHOTS", {"work": "snapshot-work"})
    monkeypatch.setattr(update_handoff, "_running_from_windows_shim", lambda: False)
    monkeypatch.setattr(update_handoff, "post_swap_python", lambda: Path(sys.executable))
    # The real handoff passes stdin to Popen; pytest's capture stream has no fd.
    with (tmp_path / "stdin").open("w+", encoding="utf-8") as child_stdin:
        monkeypatch.setattr(sys, "stdin", child_stdin)
        yield home, checkout


def _payload_kwargs(token):
    return dict(
        swap="git", branch="main", pre_pull_sha="a" * 40, is_fork=False,
        opts=SimpleNamespace(
            pre_update_version="old", active_lazy_features=["voice"],
            active_tool_dependencies={"tool": ["dependency"]}),
        gateway_mode=False, had_desktop_app_before_update=True,
        pre_update_snapshot_id="snapshot-default",
        _pre_update_plan=UpdatePlan(
            expected_sha="a" * 40,
            runtimes=[RuntimeRecord(kind="gateway", profile="work", pid=41)]),
        _windows_gateway_resume=token,
    )


@pytest.mark.real_post_swap_handoff
def test_handoff_imports_fresh_root_and_package_without_mutating_parent(
    handoff_env, monkeypatch, capfd,
):
    """Exercise the real spawn with a synthetic checkout, never the installed CLI."""
    home, checkout = handoff_env
    package = checkout / "hermes_cli"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli_output.py").write_text(
        "def new_package_symbol(): return 'fresh package'\n", encoding="utf-8")
    (checkout / "utils.py").write_text(
        "def new_root_symbol(): return 'fresh root'\n", encoding="utf-8")

    import utils as parent_utils
    from hermes_cli import cli_output as parent_cli_output

    # Model an old module generation, not an empty module: unrelated lazy
    # imports still need existing exports such as utils.atomic_json_write.
    stale = {}
    for original in (parent_utils, parent_cli_output):
        module = types.ModuleType(original.__name__)
        module.__dict__.update(vars(original))
        stale[original.__name__] = module
    assert not hasattr(stale["utils"], "new_root_symbol")
    assert not hasattr(stale["hermes_cli.cli_output"], "new_package_symbol")
    for name, module in stale.items():
        monkeypatch.setitem(sys.modules, name, module)
    listener = hermes_logging._queue_listener
    context_token = hermes_constants.set_hermes_home_override(home)
    token = {"resume_needed": True, "profiles": {"work": 41}}
    update_receipt.begin_update_receipt()
    update_receipt.record_step("git_pull", True, "checkout replaced")

    # Only the command entry point is substituted. The production handoff writes
    # the payload, starts a real interpreter, waits for it and relays its status.
    # -I/-S excludes host/site imports; the child imports only stdlib + these fakes.
    script = (
        "import json, sys; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); "
        "from utils import new_root_symbol; "
        "from hermes_cli.cli_output import new_package_symbol; "
        "payload = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')); "
        "print(json.dumps({'symbols': [new_root_symbol(), new_package_symbol()], "
        "'payload': payload}))"
    )
    monkeypatch.setattr(
        update_handoff, "post_swap_command",
        lambda handoff_path, argv_tail: [
            sys.executable, "-I", "-S", "-B", "-c", script, str(checkout), str(handoff_path)],
    )
    try:
        with pytest.raises(SystemExit) as exit_info:
            update_cmd._hand_off_post_swap(SimpleNamespace(yes=True), **_payload_kwargs(token))

        assert exit_info.value.code == 0
        child = json.loads(capfd.readouterr().out.strip())
        assert child["symbols"] == ["fresh root", "fresh package"]
        assert child["payload"]["windows_gateway_resume"]["resume_needed"] is True
        assert token["resume_needed"] is False
        assert child["payload"]["receipt"]["steps"][0]["name"] == "git_pull"
        assert child["payload"]["sibling_snapshots"] == {"work": "snapshot-work"}
        restored_plan = UpdatePlan.from_dict(child["payload"]["plan"])
        assert isinstance(restored_plan.runtimes[0], RuntimeRecord)
        assert restored_plan.runtimes[0].profile == "work"
        assert update_receipt.finalize_pending_update_receipt(0) is None
        for name, module in stale.items():
            assert sys.modules[name] is module
        assert stale["utils"].atomic_json_write is parent_utils.atomic_json_write
        assert not hasattr(stale["utils"], "new_root_symbol")
        assert not hasattr(stale["hermes_cli.cli_output"], "new_package_symbol")
        assert sys.modules["hermes_logging"] is hermes_logging
        assert hermes_logging._queue_listener is listener
        assert sys.modules["hermes_constants"] is hermes_constants
        assert hermes_constants.get_hermes_home_override() == str(home)
    finally:
        # Reloading the constants module would replace its ContextVar and make
        # this reset fail even if the same module object survived.
        hermes_constants.reset_hermes_home_override(context_token)


@pytest.mark.windows_only
@pytest.mark.real_post_swap_handoff
@pytest.mark.parametrize("spawn_ok", [True, False], ids=["detached", "spawn-refused"])
def test_windows_handoff_transfers_or_retains_recovery_ownership(
    handoff_env, monkeypatch, spawn_ok,
):
    """Never wait on a shim child; a refused spawn leaves recovery with the parent."""
    home, _ = handoff_env
    monkeypatch.setattr(update_handoff, "_running_from_windows_shim", lambda: True)
    token = {"resume_needed": True, "profiles": {"work": 41}}
    update_receipt.begin_update_receipt()
    update_receipt.record_step("git_pull", True, "checkout replaced")
    spawned = {}
    markers = []
    monkeypatch.setattr(
        update_cmd._m(), "_write_update_incomplete_marker", lambda: markers.append("incomplete"))
    recovered = []

    def recover(token):
        recovered.append(token)
        token["resume_needed"] = False

    monkeypatch.setattr(update_cmd._m(), "_resume_windows_gateways_after_update", recover)

    def fake_popen(cmd, **kwargs):
        spawned["kwargs"] = kwargs
        spawned["payload"] = update_handoff.read_handoff(cmd[cmd.index("--post-swap") + 1])
        if not spawn_ok:
            raise OSError("synthetic spawn refusal")
        return SimpleNamespace(wait=lambda *a, **k: pytest.fail("shim parent must not wait"))

    monkeypatch.setattr(update_handoff.subprocess, "Popen", fake_popen)
    with pytest.raises(SystemExit) as exit_info:
        update_cmd._hand_off_post_swap(SimpleNamespace(yes=True), **_payload_kwargs(token))

    assert exit_info.value.code == (0 if spawn_ok else 1)
    assert spawned["kwargs"]["stdin"] == subprocess.DEVNULL
    assert spawned["payload"]["windows_gateway_resume"]["resume_needed"] is True
    assert token["resume_needed"] is False
    assert recovered == ([] if spawn_ok else [token])
    assert markers == ([] if spawn_ok else ["incomplete"])

    if spawn_ok:
        assert update_receipt.finalize_pending_update_receipt(0) is None
        # Model the detached child's receipt ownership after the parent exits.
        update_receipt.resume_update_receipt(spawned["payload"]["receipt"])
    else:
        assert update_receipt._current is not None

    receipt_path = update_receipt.finalize_pending_update_receipt(exit_info.value.code)
    assert receipt_path is not None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_steps = ["git_pull"] if spawn_ok else ["git_pull", "post_swap_handoff"]
    assert [step["name"] for step in receipt["steps"]] == expected_steps
    assert receipt["outcome"] == ("success" if spawn_ok else "failed")
    assert update_receipt.finalize_pending_update_receipt(exit_info.value.code) is None
    assert list((home / "logs" / "update_receipts").glob("update_*.json")) == [receipt_path]


@pytest.mark.parametrize("failure_stage", ["options", "tail"])
def test_post_swap_refusal_resumes_gateways_before_hard_exit(
    handoff_env, monkeypatch, failure_stage,
):
    """Config and install refusals must not leave recovery to the child's skipped atexit."""
    import atexit

    resume_token = {"resume_needed": True, "profiles": {"work": 41}}
    payload = update_cmd._post_swap_payload(**_payload_kwargs(resume_token))
    handoff_path = update_handoff.write_handoff(payload)
    resumed = []
    monkeypatch.setattr(atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_ensure_non_trampoline_git", lambda cmd: cmd)

    def resolve_options(args, gateway_mode):
        if failure_stage == "options":
            raise SystemExit(2)
        return update_cmd._UpdateOptions(
            active_lazy_features=None, active_tool_dependencies=None,
            pre_update_version="new", gw_input_fn=None, assume_yes=True,
            keep_stash=False, switch_branch=False, discard_local_changes=False)

    def refuse_tail(*args, **kwargs):
        raise SystemExit(2)

    def resume(token):
        resumed.append(token)
        token["resume_needed"] = False

    monkeypatch.setattr(update_cmd, "_resolve_update_options", resolve_options)
    monkeypatch.setattr(update_cmd, "_finish_pulled_update", refuse_tail)
    monkeypatch.setattr(update_cmd._m(), "_resume_windows_gateways_after_update", resume)
    with pytest.raises(SystemExit) as exit_info:
        update_cmd._cmd_update_impl(SimpleNamespace(post_swap=str(handoff_path)), gateway_mode=False)

    assert exit_info.value.code == 2
    assert len(resumed) == 1
    assert resumed[0]["profiles"] == {"work": 41}
    assert resumed[0]["resume_needed"] is False
    assert not handoff_path.exists()
