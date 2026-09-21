"""``hermes update`` finishes in an interpreter born on the pulled code (hermes_cli/update_handoff.py).

Class: every post-swap phase used to run in the pre-pull process and lazily import NEW source
into an OLD ``sys.modules`` graph, so any rename between the two commits crashed the updater
after the code swap (#87134, #112465, #112558, #112604). The boundary under test: the parent
stops at the swap and re-executes ``hermes update --post-swap <file>``; the child resumes the
receipt and owns the tail.
"""

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from hermes_cli import update_cmd, update_handoff, update_receipt
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan
from hermes_cli.update_lock import HANDOFF_PID_ENV


@pytest.fixture(autouse=True)
def _isolated_handoff_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", tmp_path / "checkout")
    monkeypatch.setattr(update_receipt, "_current", None)
    monkeypatch.setattr(update_receipt, "_code_identity", lambda refresh=False: {})
    monkeypatch.setattr(update_handoff, "post_swap_python", lambda: Path(sys.executable))


def _opts():
    return SimpleNamespace(
        pre_update_version="0.21.3", active_lazy_features=["voice"], active_tool_dependencies={"x": 1})


@pytest.mark.real_post_swap_handoff
def test_parent_uses_handoff_module_loaded_before_checkout_changes(monkeypatch):
    """The pre-swap updater must not import a newly pulled module into its stale module graph."""
    called = {}

    def pre_pull_handoff(payload, *, argv_tail):
        called["argv_tail"] = argv_tail
        return 0

    pulled_handoff = ModuleType("hermes_cli.update_handoff")
    pulled_handoff.continue_update_in_fresh_interpreter = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("post-pull handoff module was imported into the pre-pull interpreter"))
    monkeypatch.setattr(update_handoff, "continue_update_in_fresh_interpreter", pre_pull_handoff)
    monkeypatch.setitem(sys.modules, "hermes_cli.update_handoff", pulled_handoff)

    with pytest.raises(SystemExit) as exit_info:
        update_cmd._hand_off_post_swap(
            SimpleNamespace(yes=True, no_gateway_restart=False, branch=None, gateway=False),
            swap="git", branch="main", opts=_opts(), gateway_mode=False,
            had_desktop_app_before_update=False)

    assert exit_info.value.code == 0
    assert called["argv_tail"] == ["--yes"]


@pytest.mark.real_post_swap_handoff
def test_parent_reexecs_tail_on_pulled_tree_and_relays_exit_code(monkeypatch, tmp_path):
    """The pre-swap interpreter runs NOTHING after the swap: it spawns ``python -m hermes_cli.main
    update <flags> --post-swap <file>`` with the detached receipt + plan in the file and exits
    with the child's code, leaving no open receipt for its own boundary to finalize."""
    import hermes_cli.update_cmd_config as cfg
    monkeypatch.setattr(update_handoff, "_running_from_windows_shim", lambda: False)
    monkeypatch.delenv(HANDOFF_PID_ENV, raising=False)
    monkeypatch.setattr(cfg, "_LAST_SIBLING_SNAPSHOTS", {"work": "snap-w"})
    spawned = {}

    def fake_popen(cmd, env=None, **kwargs):
        spawned["cmd"], spawned["env"] = cmd, env
        spawned["payload"] = json.loads(open(cmd[cmd.index("--post-swap") + 1], encoding="utf-8").read())
        return SimpleNamespace(wait=lambda timeout=None: 7)

    monkeypatch.setattr(update_handoff.subprocess, "Popen", fake_popen)
    update_receipt._current = None
    update_receipt.begin_update_receipt()
    update_receipt.record_step("pre_update_backup", True, "snapshot=s1")
    plan = UpdatePlan(expected_sha="a" * 40, runtimes=[RuntimeRecord(kind="gateway", profile="default", pid=4, code_sha="a" * 40)])
    args = SimpleNamespace(yes=True, no_gateway_restart=True, branch="main", gateway=False)
    token = {"resume_needed": True, "profiles": {"default": 4}}

    with pytest.raises(SystemExit) as exit_info:
        update_cmd._hand_off_post_swap(
            args, swap="git", branch="main", pre_pull_sha="a" * 40, is_fork=False, opts=_opts(),
            gateway_mode=False, had_desktop_app_before_update=True, pre_update_snapshot_id="s1",
            _pre_update_plan=plan, _windows_gateway_resume=token)

    assert exit_info.value.code == 7
    assert spawned["cmd"][:4] == [sys.executable, "-m", "hermes_cli.main", "update"]
    assert spawned["cmd"][4:] == ["--yes", "--no-gateway-restart", "--branch", "main", "--post-swap", spawned["cmd"][-1]]
    env = spawned["env"]
    assert env[update_handoff.POST_SWAP_ENV] == "1" and env["HERMES_UPDATE_REEXEC"] == "1"
    assert env[HANDOFF_PID_ENV] == str(os.getpid())
    payload = spawned["payload"]
    assert payload["receipt"]["steps"][0]["name"] == "pre_update_backup"
    assert payload["plan"]["runtimes"][0]["profile"] == "default"
    assert payload["pre_update_version"] == "0.21.3" and payload["pre_pull_sha"] == "a" * 40
    assert payload["sibling_snapshots"] == {"work": "snap-w"}
    # The child owns the Windows resume: its copy still says resume_needed, the parent's does not.
    assert payload["windows_gateway_resume"]["resume_needed"] is True and token["resume_needed"] is False
    # Detached: the parent's command-boundary finalize is a no-op — the child writes the receipt.
    assert update_receipt._current is None
    assert update_receipt.finalize_pending_update_receipt(7) is None


@pytest.mark.real_post_swap_handoff
@pytest.mark.parametrize("upstream_moves", [False, True])
def test_upstream_sync_finishes_before_final_tree_handoff(monkeypatch, upstream_moves):
    events = []
    origin_sha, upstream_sha = "b" * 40, "c" * 40
    head = [origin_sha]
    opts = SimpleNamespace(assume_yes=True, gw_input_fn=None)
    monkeypatch.setattr(update_cmd, "_invalidate_update_cache", lambda: None)
    monkeypatch.setattr(update_cmd, "_verify_head_after_pull", lambda *a, **k: head[0])
    monkeypatch.setattr(update_cmd, "_capture_head_sha", lambda *a: head[0])

    def sync(*args, **kwargs):
        events.append("upstream")
        if upstream_moves:
            head[0] = upstream_sha

    def handoff(args, **kwargs):
        events.append(("handoff", head[0]))
        assert kwargs["pre_pull_sha"] == "a" * 40
        assert kwargs["opts"] is opts
        raise SystemExit(0)

    monkeypatch.setattr(update_cmd._m(), "_sync_with_upstream_if_needed", sync)
    monkeypatch.setattr(update_cmd, "_write_fleet_restart_pending_marker",
                        lambda **kw: events.append(("marker", kw["expected_sha"])))
    monkeypatch.setattr(update_cmd, "_sweep_bytecode_after_update", lambda branch: events.append("sweep"))
    monkeypatch.setattr(update_cmd, "_hand_off_post_swap", handoff)
    with pytest.raises(SystemExit) as exit_info:
        update_cmd._apply_pulled_update(
            ["git"], "main", "a" * 40, SimpleNamespace(in_place_update=False), opts,
            gateway_mode=False, is_fork=True, desktop_dir=Path("unused"),
            had_desktop_app_before_update=False, pre_update_snapshot_id=None,
            _pre_update_plan=None, _windows_gateway_resume=None, args=SimpleNamespace())
    final_sha = upstream_sha if upstream_moves else origin_sha
    assert exit_info.value.code == 0
    assert events == ["upstream", ("marker", final_sha), "sweep", ("handoff", final_sha)]


def test_post_swap_tail_never_changes_checkout_again(monkeypatch):
    events = []
    opts = update_cmd._UpdateOptions(
        active_lazy_features=None, active_tool_dependencies=None, pre_update_version="old",
        gw_input_fn=None, assume_yes=True, keep_stash=False, switch_branch=False,
        discard_local_changes=False)
    monkeypatch.setattr(update_cmd._m(), "_sync_with_upstream_if_needed",
                        lambda *a, **k: pytest.fail("tree swap after interpreter handoff"))
    monkeypatch.setattr(update_cmd, "_sync_python_dependencies_after_pull",
                        lambda *a, **k: events.append("dependencies"))
    monkeypatch.setattr(update_cmd, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(update_cmd._m(), "_build_web_ui", lambda *a: None)
    monkeypatch.setattr(update_cmd, "_rebuild_desktop_after_update", lambda *a, **k: True)
    monkeypatch.setattr(update_cmd, "_branch_head_suffix", lambda *a: "")
    monkeypatch.setattr(update_cmd, "_run_post_update_maintenance", lambda **k: True)
    monkeypatch.setattr(update_cmd, "_restart_gateway_fleet_after_update", lambda *a: None)
    monkeypatch.setattr(update_cmd, "_resume_windows_gateways_and_merge_outcome", lambda *a: None)
    monkeypatch.setattr(update_cmd, "_verify_fleet_after_update", lambda *a, **k: events.append("verify"))
    update_cmd._finish_pulled_update(
        ["git"], "main", "a" * 40, opts, gateway_mode=False, is_fork=True,
        desktop_dir=Path("unused"), had_desktop_app_before_update=False,
        pre_update_snapshot_id=None, _pre_update_plan=None, _windows_gateway_resume=None)
    assert events == ["dependencies", "verify"]


@pytest.mark.real_post_swap_handoff
def test_parent_records_failure_when_child_cannot_start(monkeypatch, tmp_path):
    """No child → the parent takes the receipt back (a pulled update must leave a record), drops
    the install breadcrumb for the next launch and exits 1."""
    monkeypatch.setattr(update_handoff, "_running_from_windows_shim", lambda: False)
    monkeypatch.setattr(update_handoff.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("no python")))
    marker = {}
    monkeypatch.setattr(update_cmd._m(), "_write_update_incomplete_marker", lambda: marker.setdefault("written", True))
    update_receipt._current = None
    update_receipt.begin_update_receipt()

    with pytest.raises(SystemExit) as exit_info:
        update_cmd._hand_off_post_swap(
            SimpleNamespace(), swap="git", branch="main", opts=_opts(), gateway_mode=False,
            had_desktop_app_before_update=False)

    assert exit_info.value.code == 1 and marker == {"written": True}
    assert [s["name"] for s in update_receipt._current.data["steps"]] == ["post_swap_handoff"]
    update_receipt._current = None


def test_child_resumes_receipt_and_runs_git_tail_from_payload(monkeypatch, tmp_path):
    started = "2026-09-16T15:57:48+00:00"
    handoff = tmp_path / "post_swap.json"
    handoff.write_text(json.dumps({
        "swap": "git", "branch": "main", "pre_pull_sha": "a" * 40, "is_fork": True,
        "gateway_mode": False, "had_desktop_app_before_update": True, "pre_update_snapshot_id": "s1",
        "pre_update_version": "0.21.3", "active_lazy_features": ["voice"], "active_tool_dependencies": {},
        "plan": UpdatePlan(expected_sha="a" * 40, runtimes=[RuntimeRecord(kind="gateway", profile="work", pid=9)]).to_dict(),
        "windows_gateway_resume": None, "sibling_snapshots": {"work": "snap-w"},
        "receipt": {"schema": 1, "started_at": started, "outcome": "running", "steps": [{"name": "pre_update_backup", "ok": True}],
                    "skips": [], "gateway_restart": {}, "fleet": [], "pre_update": {"sha": "a" * 40}, "argv": ["hermes", "update"]},
    }), encoding="utf-8")
    import hermes_cli.update_cmd_config as cfg
    seen = {}
    monkeypatch.setattr(cfg, "_LAST_SIBLING_SNAPSHOTS", {})
    monkeypatch.setattr(update_cmd, "_base_git_cmd", lambda: ["git"])
    monkeypatch.setattr(update_cmd, "_ensure_non_trampoline_git", lambda cmd: cmd)
    monkeypatch.setattr(update_cmd, "_finish_pulled_update", lambda *a, **kw: seen.update(args=a, kw=kw))
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda args, gateway_mode: update_cmd._UpdateOptions(
        active_lazy_features=None, active_tool_dependencies=None, pre_update_version="NEW", gw_input_fn=None,
        assume_yes=True, keep_stash=False, switch_branch=False, discard_local_changes=False))
    update_receipt._current = None

    update_cmd._run_post_swap_phase(SimpleNamespace(post_swap=str(handoff), yes=True), gateway_mode=False)

    assert update_receipt._current.data["started_at"] == started
    assert update_receipt._current.data["post_swap_pid"] == os.getpid()
    assert [s["name"] for s in update_receipt._current.data["steps"]] == ["pre_update_backup"]
    git_cmd, branch, pre_pull_sha, opts = seen["args"]
    assert (git_cmd, branch, pre_pull_sha) == (["git"], "main", "a" * 40)
    # Pre-update snapshots come from the payload, never re-read from the new tree.
    assert opts.pre_update_version == "0.21.3" and opts.active_lazy_features == ["voice"]
    plan = seen["kw"]["_pre_update_plan"]
    assert isinstance(plan, UpdatePlan) and isinstance(plan.runtimes[0], RuntimeRecord)
    assert plan.runtimes[0].profile == "work" and seen["kw"]["is_fork"] is True
    assert cfg._LAST_SIBLING_SNAPSHOTS == {"work": "snap-w"}
    assert not handoff.exists()  # consumed: nothing left to leak
    update_receipt._current = None
