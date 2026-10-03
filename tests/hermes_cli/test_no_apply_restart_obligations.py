"""No-apply update runs must not found manual-serve restart obligations.

A failed channel-resolution abort ("...No update was applied.") and an
already-up-to-date run finalize a receipt whose pre and post SHAs are
identical: the checkout never moved. The receipt's ``plan.runtimes`` is a
pre-run inventory of processes the run never touched, so retaining it as
restart debt materializes durable ``serve_restart_pending`` rows and prints
"manual restart still pending" on every CLI start for processes that already
serve the current checkout — the recurring false-alarm class behind the
2026-09-22 and 2026-09-28 incidents. Receipts that DID move the checkout
keep founding obligations exactly as before, and ``pending_manual_serves``
inherited from earlier apply receipts stay owed regardless of this
receipt's own outcome.
"""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import process_identity
from hermes_cli import source_releases
from hermes_cli import update_cmd
from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import update_receipt
from hermes_cli.release_channels import ChannelNotFound
from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.update_channel import set_install_channel
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan
from hermes_cli.update_serve_obligations import (
    defer_manual_serve,
    retain_receipt_manual_serves,
)
from hermes_constants import get_hermes_home

SHA_BEFORE = "1" * 40
SHA_AFTER = "2" * 40


def manual_runtime(pid=900, create=1000.0, profile="work"):
    return asdict(
        RuntimeRecord(
            kind="serve", profile=profile, pid=pid,
            supervisor="manual-serve", restart_via="respawn-argv",
            detail={"create_time": create},
        )
    )


def pending_dir():
    return get_hermes_home() / "serve_restart_pending"


def pending_rows():
    return list(pending_dir().glob("*.json")) if pending_dir().is_dir() else []


def write_latest(receipt):
    root = get_hermes_home() / "logs" / "update_receipts"
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps(receipt), encoding="utf-8")


def update_receipt_payload(runtimes, *, pre=SHA_BEFORE, post=SHA_BEFORE):
    """Shape of the 2026-09-28 abort receipt: failed, exit 1, unchanged checkout."""
    return {
        "schema": 1, "outcome": "failed", "exit_code": 1, "stop_reason": "sys.exit(1)",
        "finished_at": "2026-09-28T10:25:15.311012+00:00",
        "pre_update": {"sha": pre}, "post_update": {"sha": post},
        "gateway_restart": {}, "fleet": [], "plan": {"runtimes": runtimes},
    }


def startup_warning(monkeypatch, capsys, *, alive=True):
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a, **k: alive)
    fleet._warn_pending_fleet_restart_on_startup()
    return capsys.readouterr().err


class _FakeLock:
    holder = None

    def acquire(self):
        return True

    def release(self):
        pass


@pytest.fixture
def update_source(tmp_path, monkeypatch):
    """A git checkout + sandboxed home wired for the real update command boundary."""
    home, origin, checkout = (tmp_path / name for name in ("home", "origin", "checkout"))
    home.mkdir()
    origin.mkdir()

    def git(root, *args):
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                              text=True, stdin=subprocess.DEVNULL).stdout.strip()

    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "Fixture")
    git(origin, "config", "user.email", "fixture@example.invalid")
    git(origin, "config", "commit.gpgsign", "false")
    commits = []
    for label in ("installed", "published", "unpublished"):
        (origin / "content.txt").write_text(label)
        git(origin, "add", "content.txt")
        git(origin, "commit", "-m", label)
        commits.append(git(origin, "rev-parse", "HEAD"))
    git(tmp_path, "clone", str(origin), str(checkout))
    git(checkout, "checkout", "--detach", commits[0])
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(checkout))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", checkout)
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: checkout)
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=hermes_main.cmd_update)
    opts = update_cmd._UpdateOptions(pre_update_version=None, gw_input_fn=None,
        assume_yes=True, keep_stash=False, switch_branch=False, discard_local_changes=False)
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: opts)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda *_: None)
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (False, ["git"], False))
    monkeypatch.setattr(hermes_main, "detect_install_method", lambda *a, **k: "git", raising=False)
    monkeypatch.setattr(hermes_main, "_install_hangup_protection", lambda gateway_mode: None, raising=False)
    monkeypatch.setattr(hermes_main, "_finalize_update_output", lambda state: None, raising=False)
    # The per-file test interpreter is the production install's venv; without this stub the
    # boundary would re-exec the update against the owning install instead of the fixture.
    monkeypatch.setattr("hermes_cli.update_owning_install.retarget_to_owning_install", lambda root: None)
    import hermes_cli.update_lock as update_lock_mod

    monkeypatch.setattr(update_lock_mod, "UpdateLock", _FakeLock)
    return SimpleNamespace(home=home, root=checkout, commits=commits, parser=parser)


def test_no_apply_abort_receipt_materializes_no_rows_and_no_startup_warning(monkeypatch, capsys):
    """The 2026-09-28 defect: a failed, nothing-applied receipt carried live manual-serve
    inventory; the next CLI start filed durable reminders and warned on every startup."""
    runtime = manual_runtime()
    write_latest(update_receipt_payload([runtime]))
    warning = startup_warning(monkeypatch, capsys)
    assert "pid 900" not in warning
    assert "manual restart still pending" not in warning
    assert pending_rows() == []


def test_apply_receipt_still_materializes_rows_and_startup_warning(monkeypatch, capsys):
    """Counter-probe: a receipt whose update moved the checkout keeps founding the debt."""
    write_latest(update_receipt_payload([manual_runtime()], pre=SHA_BEFORE, post=SHA_AFTER))
    warning = startup_warning(monkeypatch, capsys)
    assert "serve [work] pid 900" in warning
    assert "manual restart still pending" in warning
    assert len(pending_rows()) == 1


def test_inherited_pending_serves_survive_a_no_apply_receipt(monkeypatch, capsys):
    """A no-apply receipt must not drop debts inherited from earlier APPLY receipts:
    only its own plan inventory is unfounded, carried-in rows stay owed."""
    carried = dict(manual_runtime(), detail={"create_time": 2000.0})
    write_latest({**update_receipt_payload([]), "pending_manual_serves": [carried]})
    warning = startup_warning(monkeypatch, capsys)
    assert "serve [work] pid 900" in warning
    assert len(pending_rows()) == 1


def test_retain_splits_plan_inventory_from_inherited_debts(monkeypatch):
    from hermes_cli.update_serve_obligations import receipt_applied_no_code

    runtime = manual_runtime()
    no_apply = update_receipt_payload([runtime])
    assert receipt_applied_no_code(no_apply) is True
    assert retain_receipt_manual_serves(no_apply) == []
    apply = update_receipt_payload([runtime], pre=SHA_BEFORE, post=SHA_AFTER)
    assert receipt_applied_no_code(apply) is False
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a, **k: True)
    assert retain_receipt_manual_serves(apply) == []  # transferred, not pending
    assert len(pending_rows()) == 1
    # An inherited row that cannot be transferred stays on the no-apply receipt's books:
    # without a readable create_time the durable reminder cannot be filed (#116507), so the
    # debt rides along as pending instead.
    inherited = {**update_receipt_payload([]), "pending_manual_serves": [dict(runtime, pid=903, detail={})]}
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a, **k: None)
    pending = retain_receipt_manual_serves(inherited)
    assert pending == [inherited["pending_manual_serves"][0]]


def test_finalize_stamps_no_apply_and_rotation_drops_unfounded_carry(monkeypatch):
    update_receipt.begin_update_receipt()
    path = update_receipt.finalize_update_receipt("failed")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # The test checkout does not move during the run: pre == post.
    assert data["pre_update"]["sha"] == data["post_update"]["sha"]
    assert data["no_apply"] is True

    # A previous receipt that moved the checkout keeps its debts across rotation... (the
    # inherited row here cannot be filed durably — no create_time, unknowable pid — so it
    # must ride along as pending_manual_serves.)
    apply_receipt = update_receipt_payload(
        [dict(manual_runtime(pid=901), detail={"create_time": None})], pre=SHA_BEFORE, post=SHA_AFTER)
    write_latest(apply_receipt)
    monkeypatch.setattr(process_identity, "_pid_alive_matches", lambda *a, **k: None)
    update_receipt.begin_update_receipt()
    path = update_receipt.finalize_update_receipt("failed")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["pending_manual_serves"] == apply_receipt["plan"]["runtimes"]

    # ...while a no-apply previous receipt carries nothing forward.
    write_latest(update_receipt_payload([manual_runtime(pid=902)]))
    update_receipt.begin_update_receipt()
    path = update_receipt.finalize_update_receipt("failed")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert "pending_manual_serves" not in data

    # A receipt that provably moved the checkout is never stamped no_apply, even when
    # begin inherited foreign identity fields.
    write_latest(update_receipt_payload([], pre=SHA_BEFORE, post=SHA_AFTER))
    update_receipt.begin_update_receipt(
        previous={"pre_update": {"sha": SHA_BEFORE}, "post_update": {"sha": SHA_AFTER}})
    path = update_receipt.finalize_update_receipt("success")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["no_apply"] is False


def test_durable_row_from_no_apply_history_discharges_while_serving_current_checkout(monkeypatch, capsys):
    """Reconciliation: an already-materialized reminder is unfounded when its process is live,
    runs out of this updater's own checkout, and every receipt newer than the process is a
    no-apply run — no update it could have missed exists."""
    pending_dir().mkdir(parents=True, exist_ok=True)
    row = {"kind": "serve", "profile": "work", "pid": 900, "create_time": 1000.0}
    (pending_dir() / f"900-{float(1000.0).hex()}.json").write_text(json.dumps(row), encoding="utf-8")
    write_latest(update_receipt_payload([manual_runtime()]))  # finished after the process started

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            repo_root = Path(update_receipt.__file__).resolve().parent.parent
            return [str(repo_root / "hermes_cli" / "main.py"), "serve"]

        def create_time(self):
            return 1000.0

        def is_running(self):
            return True

        def status(self):
            return "running"

    import psutil

    monkeypatch.setattr(psutil, "Process", FakeProc)
    warning = startup_warning(monkeypatch, capsys)
    assert "pid 900" not in warning
    assert pending_rows() == []


def test_durable_row_survives_when_an_apply_landed_after_the_process_started(monkeypatch, capsys):
    """Counter-probe: an update that moved the checkout after the process started keeps the
    reminder legitimate — the process may hold pre-apply modules in memory."""
    pending_dir().mkdir(parents=True, exist_ok=True)
    row = {"kind": "serve", "profile": "work", "pid": 900, "create_time": 1000.0}
    (pending_dir() / f"900-{float(1000.0).hex()}.json").write_text(json.dumps(row), encoding="utf-8")
    write_latest(update_receipt_payload([manual_runtime()], pre=SHA_BEFORE, post=SHA_AFTER))

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            repo_root = Path(update_receipt.__file__).resolve().parent.parent
            return [str(repo_root / "hermes_cli" / "main.py"), "serve"]

        def create_time(self):
            return 1000.0

        def is_running(self):
            return True

        def status(self):
            return "running"

    import psutil

    monkeypatch.setattr(psutil, "Process", FakeProc)
    warning = startup_warning(monkeypatch, capsys)
    assert "serve [work] pid 900" in warning
    assert len(pending_rows()) == 1


def test_durable_row_survives_without_receipt_coverage_since_process_start(monkeypatch, capsys):
    """Conservatism: when no update receipt finished after the process started, the receipts
    cannot prove absence of an apply — the reminder stays."""
    pending_dir().mkdir(parents=True, exist_ok=True)
    row = {"kind": "serve", "profile": "work", "pid": 900, "create_time": 1000.0}
    (pending_dir() / f"900-{float(1000.0).hex()}.json").write_text(json.dumps(row), encoding="utf-8")
    # A receipt older than the process start is outside the coverage window.
    write_latest({**update_receipt_payload([manual_runtime()]),
                  "finished_at": "1970-01-01T00:16:40+00:00"})

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            repo_root = Path(update_receipt.__file__).resolve().parent.parent
            return [str(repo_root / "hermes_cli" / "main.py"), "serve"]

        def create_time(self):
            return 1000.0

        def is_running(self):
            return True

        def status(self):
            return "running"

    import psutil

    monkeypatch.setattr(psutil, "Process", FakeProc)
    warning = startup_warning(monkeypatch, capsys)
    assert "serve [work] pid 900" in warning
    assert len(pending_rows()) == 1


def test_channel_abort_through_cli_boundary_leaves_no_restart_obligations(update_source, monkeypatch, capsys):
    """End to end: `hermes update` aborting on a missing channel record (the 2026-09-28
    stable.json 404) must stamp its receipt no_apply, write no fleet marker, and leave
    nothing for the next CLI start to materialize or warn about."""
    runtime = RuntimeRecord(
        kind="serve", profile="default", pid=900,
        supervisor="manual-serve", restart_via="respawn-argv",
        detail={"create_time": 1000.0},
    )
    monkeypatch.setattr(
        "hermes_cli.update_inventory.collect_runtime_inventory",
        lambda: UpdatePlan(runtimes=[runtime]),
    )
    set_install_channel("stable", update_source.root)

    def missing(name, repository):
        raise ChannelNotFound("Channel object not found: releases/channels/stable.json")

    monkeypatch.setattr(source_releases, "_resolve_channel", missing)
    args = update_source.parser.parse_args(["update", "--yes"])
    with pytest.raises(SystemExit) as exc:
        hermes_main.cmd_update(args)
    assert exc.value.code == 1
    assert "No update was applied" in capsys.readouterr().out

    receipt = json.loads(
        (update_source.home / "logs" / "update_receipts" / "latest.json").read_text(encoding="utf-8"))
    assert receipt["outcome"] == "failed"
    assert receipt["pre_update"]["sha"] == receipt["post_update"]["sha"]
    assert receipt.get("no_apply") is True
    assert receipt["plan"]["runtimes"][0]["pid"] == 900  # inventory recorded, not acted on
    assert not fleet._fleet_restart_obligation_armed()

    # The next CLI start must not materialize the unfounded inventory.
    warning = startup_warning(monkeypatch, capsys)
    assert "pid 900" not in warning
    assert pending_rows() == []
