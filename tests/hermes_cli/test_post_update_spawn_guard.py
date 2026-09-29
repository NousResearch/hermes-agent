"""Spawn guard step (t_7b0df4cf): registry, skips, verdicts, mutation.

All hermetic: legs never spawn here (monkeypatched); the real subprocess
proof is the acceptance run of the wired step against an installed tree.
"""
from pathlib import Path

from hermes_cli import post_update
from hermes_cli.post_update import step_spawn_guard


def _root_env(root: Path, *extra: str) -> dict:
    import os

    return {"PYTHONPATH": os.pathsep.join([str(root), *extra])}


# ── registry ─────────────────────────────────────────────────────────


def test_spawn_guard_in_home_and_boot_steps():
    home = dict(post_update.HOME_STEPS)
    boot = dict(post_update.BOOT_HOME_STEPS)
    assert home["spawn_guard"] is step_spawn_guard
    # Boot runs it too (BOOT_HOME_STEPS excludes only sync_skills); boot's
    # per-revision record still gates the run, the step gates the generation.
    assert boot["spawn_guard"] is step_spawn_guard


# ── recorded skips ───────────────────────────────────────────────────


def test_skips_on_wheel_install(tmp_path, monkeypatch):
    import cron.scheduler_worker_env as worker_env_mod

    monkeypatch.setattr(
        worker_env_mod, "pin_hermes_tree_on_pythonpath", lambda env, _root: env)
    assert step_spawn_guard(tmp_path) == {"ok": True, "skipped": "wheel-install"}


def test_skips_without_committed_site(tmp_path, monkeypatch):
    import cron.scheduler_worker_env as worker_env_mod

    monkeypatch.setattr(
        worker_env_mod, "pin_hermes_tree_on_pythonpath",
        lambda env, root: _root_env(Path(root)))
    assert step_spawn_guard(tmp_path) == {"ok": True, "skipped": "no-committed-site"}


def test_skips_when_generation_unchanged(tmp_path, monkeypatch):
    import cron.scheduler_worker_env as worker_env_mod

    monkeypatch.setattr(
        worker_env_mod, "pin_hermes_tree_on_pythonpath",
        lambda env, root: _root_env(Path(root), str(tmp_path / "site")))
    monkeypatch.setattr(
        post_update, "_spawn_guard_generation_fingerprint", lambda _root: "abc")
    state = tmp_path / "spawn-guard.json"
    state.write_text('{"generation": "abc"}\n', encoding="utf-8")
    monkeypatch.setattr(post_update, "_spawn_guard_state_path", lambda _root: state)
    assert step_spawn_guard(tmp_path) == {"ok": True, "skipped": "generation-unchanged"}


# ── verdicts ─────────────────────────────────────────────────────────


def _passing_step(monkeypatch, tmp_path):
    import cron.scheduler_worker_env as worker_env_mod

    monkeypatch.setattr(
        worker_env_mod, "pin_hermes_tree_on_pythonpath",
        lambda env, root: _root_env(Path(root), str(tmp_path / "site")))
    monkeypatch.setattr(
        post_update, "_spawn_guard_generation_fingerprint", lambda _root: "gen-1")
    monkeypatch.setattr(
        post_update, "_spawn_guard_state_path",
        lambda _root: tmp_path / "state.json")


def test_passes_and_records_generation(tmp_path, monkeypatch):
    _passing_step(monkeypatch, tmp_path)
    monkeypatch.setattr(post_update, "_spawn_guard_leg_kanban", lambda _r: (True, "ok"))
    monkeypatch.setattr(post_update, "_spawn_guard_leg_dm", lambda _r: (True, "ok"))
    monkeypatch.setattr(post_update, "_spawn_guard_leg_cron", lambda _r: (True, "ok"))

    assert step_spawn_guard(tmp_path) == {"ok": True, "legs": "3/3"}
    import json

    assert json.loads((tmp_path / "state.json").read_text())["generation"] == "gen-1"


def test_failure_names_the_leg(tmp_path, monkeypatch):
    _passing_step(monkeypatch, tmp_path)
    monkeypatch.setattr(post_update, "_spawn_guard_leg_kanban", lambda _r: (True, "ok"))
    monkeypatch.setattr(post_update, "_spawn_guard_leg_dm", lambda _r: (True, "ok"))
    monkeypatch.setattr(
        post_update, "_spawn_guard_leg_cron",
        lambda _r: (False, "IMPORT-FAIL stderr: No module named 'ruamel'"))

    result = step_spawn_guard(tmp_path)
    assert result["ok"] is False
    assert "L3-cron-externo" in result["error"]
    assert "ruamel" in result["error"]
    assert not (tmp_path / "state.json").exists()


# ── mutation ─────────────────────────────────────────────────────────


def test_root_only_mutation_rebuilds_prefix_pin(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("GUARDA_3_SPAWNS_MUTATION", "root-only")
    env = {"PYTHONPATH": os.pathsep.join(["kept", str(tmp_path / "repo"), str(tmp_path / "site")])}
    out = post_update._spawn_guard_pin(env, tmp_path / "repo")
    assert out["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path / "repo"), "kept", str(tmp_path / "site")]


# ── update-tail reporting ────────────────────────────────────────────


def test_report_prints_pass_and_fail(capsys):
    from hermes_cli.update_cmd_maint import _report_spawn_guard_after_update

    _report_spawn_guard_after_update({"ok": True, "legs": "3/3"})
    assert "Spawn-path guard: 3/3" in capsys.readouterr().out

    _report_spawn_guard_after_update({"ok": True, "skipped": "generation-unchanged"})
    assert "generation-unchanged" in capsys.readouterr().out

    _report_spawn_guard_after_update({"ok": False, "error": "L3-cron-externo: boom"})
    out = capsys.readouterr().out
    assert "Spawn-path guard FAILED" in out
    assert "L3-cron-externo" in out
