"""Every `hermes update` exit before the apply step names its closed reason (hermes.update.run
``failure_class``), in-process and through the parked copy alike; exits that fire before the
receipt opens still count once; a run that committed and owes work never reads ``failed``."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import update_receipt
from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_update as update_metrics


@pytest.fixture
def rows(tmp_path, monkeypatch):
    captured: list[tuple[str, dict]] = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("hermes_cli.config.read_raw_config_readonly",
                        lambda: {"telemetry": {"shared_metrics": {"enabled": True}}})
    monkeypatch.setattr(relay_shared_metrics, "enabled", lambda: True)
    monkeypatch.setattr(relay_shared_metrics, "record_process_mark", lambda mark, data: captured.append((mark, data)))
    monkeypatch.setattr(update_receipt, "_code_identity", lambda refresh=False: {"sha": "a" * 40, "commit_date": None})
    with update_receipt.update_receipt_scope():
        yield SimpleNamespace(runs=lambda: [d for m, d in captured if m == contract.UPDATE_RUN_MARK],
                              home=tmp_path / "home")


def _run(stop: str | None, *, applied: bool = False, code: int = 1, user_action: bool = False) -> dict:
    update_receipt.begin_update_receipt()
    update_receipt.record_stage("plan", "success")
    if user_action:
        update_receipt.record_user_action("local_changes", "stash ref /home/alice/private")
    if stop:
        update_receipt.record_stop_reason(stop)
    if applied:
        update_receipt.record_stage("apply", "success", mode="git")
    path = update_receipt.finalize_pending_update_receipt(code, f"sys.exit({code})")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("stop", "kwargs", "expected"), [
    *[(name, {}, (name, "failed")) for name in sorted(contract.UPDATE_STOP_CLASSES - {"lock_held", "managed_install"})],
    ("not_a_class: /home/alice", {}, ("aborted_before_apply", "failed")),  # only closed tokens pass
    (None, {}, ("aborted_before_apply", "failed")),  # the fallback stays for an exit with no reason
    ("fetch_failed", {"applied": True}, ("deps_failed", "failed")),  # a reason never outlives the apply
    (None, {"applied": True, "code": 1, "user_action": True}, ("local_changes_parked", "partial")),
])
def test_a_pre_apply_exit_reads_its_own_class_in_process_and_parked(rows, stop, kwargs, expected):
    """Invariant: the class is the exit that fired (recorded at it), the parked copy a pre-pull
    interpreter writes classifies the same run identically and keeps no free text, and a committed
    run that only owes the user's parked changes is ``partial``, never ``failed``."""
    receipt = _run(stop, **kwargs)
    (run,) = rows.runs()
    assert (run["failure_class"], run["outcome"]) == expected
    assert contract.counter_dimensions_are_valid(contract.UPDATE_RUN_METRIC, run)
    parked = update_receipt._metric_receipt(receipt)
    assert "/home/alice" not in json.dumps(parked)
    assert update_metrics.update_receipt_fields(json.loads(json.dumps(parked)))[0] == run
    for text in ("alice secret repo: x", "Windows gateway recovery failed: C:\\Users\\alice"):
        kept = update_receipt._metric_receipt({**receipt, "stop_reason": text})["stop_reason"]
        assert "alice" not in kept and kept in {"-", "Windows gateway recovery failed"}


@pytest.mark.parametrize(("stderr", "argv", "code", "expected"), [
    ("fatal: Unable to create '/x/.git/index.lock': File exists.", ["git", "merge"], 128, "git_index_locked"),
    ("git fetch timed out after 300s (a stalled remote)", ["git", "fetch"], 124, "git_timeout"),
    ("error: No space left on device", ["git", "checkout"], 1, "disk_full"),
    ("error: could not write index", ["git", "stash", "push"], 1, "local_changes_blocked"),
    ("fatal: bad object", ["git", "reset", "--hard"], 128, "checkout_move_failed"),
    ("whatever", ["uv", "pip", "install"], 1, None),
])
def test_a_failed_git_call_names_its_class_from_the_argv_and_fixed_phrases(stderr, argv, code, expected):
    import subprocess

    from hermes_cli.update_cmd_zip import _zip_stop_class

    assert update_receipt.git_error_stop_class(subprocess.CalledProcessError(code, argv, stderr=stderr)) == expected
    assert _zip_stop_class(OSError(28, "No space left on device"), downloaded=True) == "disk_full"
    assert _zip_stop_class(ValueError("x"), downloaded=False) == "download_failed"


def test_exits_before_the_receipt_opens_count_once_and_leave_the_holders_receipt_alone(rows, monkeypatch):
    """Invariant: the update-lock refusal and the Git-operation refusal fire before this run's
    receipt opens; each still yields exactly one row with its class, and neither writes a receipt
    (``latest.json`` belongs to the update holding the lock)."""
    from hermes_cli import main, update_cmd, update_lock, update_owning_install

    update_receipt.begin_update_receipt()  # the holder's open run
    holders = (rows.home / "logs/update_receipts/latest.json").read_bytes()
    update_receipt._current.set(None)

    class Held:
        holder = None

        def __init__(self, **_kw):
            pass

        def acquire(self):
            return False

    monkeypatch.setattr(update_owning_install, "retarget_to_owning_install", lambda root: None)
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kw: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda state: None)
    monkeypatch.setattr(update_lock, "UpdateLock", Held)
    monkeypatch.setattr(update_lock, "describe_holder", lambda holder: "another update is running")
    with pytest.raises(SystemExit) as refused:
        main.cmd_update(SimpleNamespace(gateway=False))
    monkeypatch.setattr(update_cmd, "git_operation_in_progress", lambda root: "rebase")
    with pytest.raises(SystemExit) as in_progress:
        update_cmd._cmd_update_impl(SimpleNamespace(), gateway_mode=False)

    assert (refused.value.code, in_progress.value.code) == (2, 1)
    assert [(r["failure_class"], r["outcome"], r["failed_stage"]) for r in rows.runs()] == [
        ("lock_held", "refused", "none"), ("git_in_progress", "failed", "other")]
    assert all(contract.counter_dimensions_are_valid(contract.UPDATE_RUN_METRIC, r) for r in rows.runs())
    assert (rows.home / "logs/update_receipts/latest.json").read_bytes() == holders


def _marker(home, *lines: str) -> None:
    (home / ".hermes-update-in-progress").parent.mkdir(parents=True, exist_ok=True)
    (home / ".hermes-update-in-progress").write_text("".join(f"{line}\n" for line in lines))


def test_an_exit_before_the_receipt_reads_desktop_only_under_its_own_hand_off(rows, monkeypatch):
    """Invariant: a pre-receipt exit carries the initiator its receipt would: ``desktop`` when the
    marker names this process as the hand-off's delegate or names its hand-off partner, ``cli`` when
    the marker belongs to some other update (a terminal `hermes update` refused by a Desktop run)."""
    import os

    from hermes_cli.update_cmd_common import _record_stop

    monkeypatch.delenv("HERMES_UPDATE_HANDOFF_PID", raising=False)
    pid, other = os.getpid(), 999_999
    _marker(rows.home, other, 1, "ct:1.000", f"delegate:{pid} ct:2.000")  # the posix/Windows hand-off
    _record_stop("lock_held", without_receipt="refused")
    monkeypatch.setenv("HERMES_UPDATE_HANDOFF_PID", str(other))  # the Tauri updater's claim
    _marker(rows.home, other, 1, "ct:1.000")
    _record_stop("git_in_progress", without_receipt="failed")
    monkeypatch.delenv("HERMES_UPDATE_HANDOFF_PID")  # a CLI run refused by someone else's update
    _record_stop("lock_held", without_receipt="refused")
    (rows.home / ".hermes-update-in-progress").unlink()
    _record_stop("managed_install", without_receipt="refused")

    assert [(r["failure_class"], r["kind"]) for r in rows.runs()] == [
        ("lock_held", "desktop"), ("git_in_progress", "desktop"), ("lock_held", "cli"), ("managed_install", "cli")]


def test_a_lock_refusal_of_the_hand_offs_own_update_child_is_a_failed_run(rows, monkeypatch):
    """Invariant: when the lock refuses the claim whose delegate line names this very process, the
    Desktop quit to run this update and reports the refusal as a failed update, so the row reads
    ``failed`` (also for its child: Windows relaunches the updater under the named process); a refusal under someone else's claim (Tauri partner, a CLI run) stays ``refused``."""
    import os

    from hermes_cli.update_cmd_common import _record_stop

    monkeypatch.delenv("HERMES_UPDATE_HANDOFF_PID", raising=False)
    pid, other = os.getpid(), 999_999
    _marker(rows.home, other, 1, "ct:1.000", f"delegate:{pid} ct:2.000")  # our own hand-off's claim
    _record_stop("lock_held", without_receipt="refused")
    _marker(rows.home, other, 1, "ct:1.000", f"delegate:{os.getppid()} ct:2.000")  # Windows: relaunched child
    _record_stop("lock_held", without_receipt="refused")
    monkeypatch.setenv("HERMES_UPDATE_HANDOFF_PID", str(other))
    _marker(rows.home, other, 1, "ct:1.000")
    _record_stop("lock_held", without_receipt="refused")
    monkeypatch.delenv("HERMES_UPDATE_HANDOFF_PID")
    _record_stop("lock_held", without_receipt="refused")

    assert [(r["outcome"], r["kind"], r["failure_class"]) for r in rows.runs()] == [
        ("failed", "desktop", "lock_held"), ("failed", "desktop", "lock_held"), ("refused", "desktop", "lock_held"),
        ("refused", "cli", "lock_held")]
    assert all(contract.counter_dimensions_are_valid(contract.UPDATE_RUN_METRIC, r) for r in rows.runs())


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="POSIX permissions; root ignores them")
def test_a_checkout_move_names_a_held_index_lock_apart_from_a_permission_error(rows, tmp_path, monkeypatch):
    """Invariant: the fast-forward's git error goes through the shared classifier: an index.lock that
    exists is ``git_index_locked``; git refused permission to create one is ``permission_denied``."""
    import subprocess

    from hermes_cli import update_cmd, update_cmd_git

    monkeypatch.setattr(update_cmd, "_git_run", update_cmd_git._git_run)  # the real git runner, no main import
    repo = tmp_path / "repo"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    cmd = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    git = lambda *a: subprocess.run([*cmd, *a], check=True, capture_output=True, text=True).stdout.strip()
    repo.mkdir()
    git("init", "-q")
    (repo / "a").write_text("a")
    git("add", "a")
    git("commit", "-qm", "one")
    old = git("rev-parse", "HEAD")
    (repo / "a").write_text("b")
    git("commit", "-qam", "two")
    new = git("rev-parse", "HEAD")
    git("reset", "-q", "--hard", old)

    def move() -> str:
        update_receipt.begin_update_receipt()
        with pytest.raises(SystemExit):
            update_cmd._move_checkout_to(cmd, "main", "origin/main", new, old)
        path = update_receipt.finalize_pending_update_receipt(1, "sys.exit(1)")
        return json.loads(path.read_text(encoding="utf-8"))["stop_class"]

    (repo / ".git/index.lock").write_text("")
    held = move()
    (repo / ".git/index.lock").unlink()
    index_dir = tmp_path / "readonly-index"
    index_dir.mkdir()
    (index_dir / "index").write_bytes((repo / ".git/index").read_bytes())
    monkeypatch.setenv("GIT_INDEX_FILE", str(index_dir / "index"))
    index_dir.chmod(0o555)
    try:
        denied = move()
    finally:
        index_dir.chmod(0o755)
    assert (held, denied) == ("git_index_locked", "permission_denied")
    assert git("rev-parse", "HEAD") == old


@pytest.mark.parametrize(("site", "raised", "expected"), [
    ("channel", ValueError("Invalid channel name"), "channel_unresolved"),
    ("pause", RuntimeError("Could not stop Windows gateway service x"), "gateway_pause_failed"),
])
def test_an_exception_before_the_apply_names_its_exit_through_the_command_boundary(
        rows, monkeypatch, site, raised, expected):
    """Invariant: an exception that leaves ``_cmd_update_impl`` before the apply mark is recorded by
    the exit that raised it; the command boundary (main.cmd_update) never has to read its type name,
    so it never collapses to ``exception``."""
    from hermes_cli import main, update_cmd, update_lock, update_owning_install

    def boom(*_a, **_k):
        raise raised

    class Free:
        holder = None

        def __init__(self, **_kw):
            pass

        def acquire(self):
            return True

        def release(self):
            pass

    monkeypatch.setattr(update_owning_install, "retarget_to_owning_install", lambda root: None)
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kw: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda state: None)
    monkeypatch.setattr(update_lock, "UpdateLock", Free)
    monkeypatch.setattr(update_cmd, "git_operation_in_progress", lambda root: None)
    monkeypatch.setattr(update_cmd._check, "clear_git_debris", lambda root: None)
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: update_cmd._UpdateOptions(
        pre_update_version=None, gw_input_fn=None, assume_yes=True, keep_stash=False,
        switch_branch=False, discard_local_changes=False, no_gateway_restart=True))
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan",
                        lambda args: update_receipt.begin_update_receipt())
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda args: None)
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", boom if site == "pause" else lambda: None)
    monkeypatch.setattr(main, "_desktop_packaged_executable", lambda d: None)
    monkeypatch.setattr(main, "_desktop_dist_exists", lambda d: False)
    monkeypatch.setattr(main, "_installed_desktop_apps", list)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda **_: (False, ["git"], False))
    monkeypatch.setattr(update_cmd, "_source_completion_request", lambda *a: {"home": str(rows.home)})
    monkeypatch.setattr(main, "_resolve_update_branch", lambda args: "main")
    monkeypatch.setattr(update_cmd, "_source_update_channel", boom if site == "channel" else lambda args: "main")
    with pytest.raises(type(raised)):
        main.cmd_update(SimpleNamespace(gateway=False, branch=None, channel=None))

    (run,) = rows.runs()
    assert (run["failure_class"], run["outcome"], run["apply_mode"]) == (expected, "failed", "unknown")
    assert contract.counter_dimensions_are_valid(contract.UPDATE_RUN_METRIC, run)


def test_a_timed_out_target_read_skips_the_preflight_instead_of_failing_the_update(monkeypatch, capsys):
    """Invariant: the startup-syntax preflight reads the target from git's object store; a read that
    times out (a blobless install lazily fetches each blob) is an unread file, never an exception
    that ends the update as ``subprocess_failed``. The post-move syntax check stays the backstop."""
    import subprocess

    from pathlib import Path

    from hermes_cli import update_cmd_commit

    def slow(git_cmd, args, **kwargs):
        raise subprocess.TimeoutExpired([*git_cmd, *args], 120)

    monkeypatch.setattr(update_cmd_commit, "run_git", slow)
    assert update_cmd_commit.target_syntax_error(["git"], Path("."), "a" * 40, ["cli.py", "run_agent.py"]) is None
    assert "Syntax preflight skipped (slow object fetch)" in capsys.readouterr().out


@pytest.mark.parametrize("consent", ["on", "off"])
def test_the_bare_bootstrap_interpreter_parks_and_the_next_start_applies_the_gate(rows, consent, monkeypatch):
    """Invariant: a run finalized by the ``-I -S`` bootstrap interpreter (dependency preparation
    failed; no ruamel, so it cannot read consent) never emits and never drops the run: it parks only
    the bounded fields, and the next normal start applies the collection gate: counted once while
    collection is on, purged unrecorded while it is off."""
    import subprocess
    from pathlib import Path

    from hermes_cli.observability import shared_metrics_process as process_metrics

    saved: list[tuple[str, dict]] = []
    monkeypatch.setattr(relay_shared_metrics, "record_process_marks_saved",
                        lambda marks: saved.extend(marks) or len(marks), raising=False)
    monkeypatch.setattr("hermes_cli.config.read_raw_config_readonly",
                        lambda: {"telemetry": {"shared_metrics": {"enabled": consent == "on"}}})
    monkeypatch.setattr(process_metrics, "_STATE", {})
    monkeypatch.setattr(process_metrics, "_report_dead_markers", lambda home, own: update_metrics.report_pending_updates())

    home = rows.home
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(f"telemetry:\n  shared_metrics:\n    enabled: {str(consent == 'on').lower()}\n")
    pending = update_metrics.pending_updates_dir(home)
    pending.mkdir(parents=True)
    (pending / "earlier.json").write_text("{}")
    receipt = {
        "schema": 1, "pid": 0, "update_id": "0123456789abcdef0123456789abcdef", "outcome": "success",
        "started_at": "2026-10-07T10:00:00+00:00", "finished_at": "2026-10-07T10:01:00+00:00", "exit_code": 0,
        "argv": ["hermes", "update", "/home/alice/private"], "pre_update": {"sha": "a" * 40},
        "stages": [{"name": "apply", "outcome": "success", "mode": "git"}, {"name": "deps", "outcome": "failed"}],
    }
    tree = Path(update_receipt.__file__).resolve().parents[1]
    script = ("import json, sys; sys.path.insert(0, sys.argv[1]); from hermes_cli import update_receipt as r; "
              "assert 'ruamel' not in sys.modules; r._publish_shared_metrics(json.loads(sys.argv[2]))")
    subprocess.run([sys.executable, "-I", "-S", "-c", script, str(tree), json.dumps(receipt)], check=True,
                   env={"HERMES_HOME": str(home), "HOME": str(home.parent), "PATH": os.environ.get("PATH", "")})
    parked = sorted(path.name for path in pending.glob("*.json"))
    assert parked == ["0123456789abcdef0123456789abcdef.json", "earlier.json"]  # nothing emitted, nothing lost
    assert "alice" not in (pending / parked[0]).read_text()
    assert rows.runs() == [] and not list(home.rglob("metrics.sqlite3"))

    monkeypatch.setattr(process_metrics.threading, "Thread", lambda target, args, **kw: SimpleNamespace(
        start=lambda: target(*args)))
    monkeypatch.setattr(process_metrics.atexit, "register", lambda *a: None)
    monkeypatch.setattr(process_metrics.sys, "excepthook", process_metrics.sys.excepthook)
    process_metrics.begin_process("cli")  # the next normal start
    runs = [d for m, d in saved if m == contract.UPDATE_RUN_MARK]
    if consent == "off":
        assert runs == [] and not pending.exists()
    else:
        assert [(r["outcome"], r["failed_stage"], r["apply_mode"]) for r in runs] == [("success", "none", "git")]
        assert list(pending.glob("*.json")) == []
