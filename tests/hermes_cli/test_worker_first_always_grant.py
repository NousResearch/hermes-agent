"""An "Always" approval inside a frozen managed worker reaches config.yaml.

A fresh profile has no ``command_allowlist`` in the live file. An ordinary worker's
snapshot does not have it either. A safe-mode or ``--ignore-user-config`` worker
freezes a copy of the schema, so the snapshot does contain ``command_allowlist: []``.
Neither is an operator deletion: the list exists only because the schema default
was merged in, and the grant must be written. An explicit empty list the operator
put in an ordinary snapshot and then removed stays removed.

Ordinary path: ``build_policy`` on the raw file, then ``_worker_config``, bound with
``_bind_worker_config``. Bypass path: ``_bypass_policy`` then ``_worker_config``, bound
with ``_bind_safe_worker_policy`` the way the worker bootstrap binds it. Always goes
through ``tools.approval._persist_choice``.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
KEY = "recursive delete"


def _snapshot(home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    from gateway.run import _load_gateway_config
    from gateway.session_managed_worker import _worker_config
    from gateway.session_policy import build_policy
    policy = build_policy({"source": "cli", "cwd": str(home)}, _load_gateway_config(home / "config.yaml"))
    return _worker_config(SimpleNamespace(profile_id=str(home)), policy)


def _bypass_snapshot(home, monkeypatch, *, safe_mode):
    monkeypatch.setenv("HERMES_HOME", str(home))
    from gateway.session_local import _bypass_policy
    from gateway.session_managed_worker import _worker_config
    params = {"source": "cli", "cwd": str(home), "model": "m"}
    if safe_mode:
        params["safe_mode"] = True
    else:
        params["ignore_user_config"] = True
    policy = _bypass_policy(params, private_secrets={})
    return _worker_config(SimpleNamespace(profile_id=str(home)), policy)


def _always_in_worker(tmp_path, monkeypatch, live_yaml, bind=True, *, safe_mode=False, action="always"):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(live_yaml, encoding="utf-8")
    if bind == "bypass":
        snapshot = _bypass_snapshot(home, monkeypatch, safe_mode=safe_mode)
        bind_mode = "bypass"
    elif bind:
        snapshot = _snapshot(home, monkeypatch)
        bind_mode = "worker"
    else:
        snapshot = None
        bind_mode = "off"
    script = tmp_path / "grant.py"
    script.write_text(
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n" + textwrap.dedent(
            """
            import json, os
            snap = json.loads(os.environ["GRANT_SNAPSHOT"])
            mode = os.environ["GRANT_BIND"]
            if mode == "bypass":
                from agent.safe_worker_policy import _bind_safe_worker_policy
                _bind_safe_worker_policy(
                    safe_mode=os.environ["GRANT_SAFE"] == "1",
                    ignore_user_config=True,
                    config=snap,
                )
            elif mode == "worker":
                from agent.safe_worker_policy import _bind_worker_config
                _bind_worker_config(snap)
            if os.environ["GRANT_ACTION"] == "approvals":
                from hermes_cli.config import load_config, save_config
                cfg = load_config()
                cfg["approvals"]["mode"] = "off"
                save_config(cfg)
                print(json.dumps({"ok": True}))
            else:
                from tools import approval
                approval._persist_choice("route-1", "always", [os.environ["GRANT_KEY"]])
                print(json.dumps({"in_memory": approval._is_permanently_approved(os.environ["GRANT_KEY"])}))
            """
        ),
        encoding="utf-8",
    )
    from tests.gateway.fixtures.local_recovery_probe import child_env
    env = child_env()
    env.update(
        HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(home), GRANT_KEY=KEY,
        GRANT_SNAPSHOT=json.dumps(snapshot), GRANT_BIND=bind_mode,
        GRANT_SAFE="1" if safe_mode else "0", GRANT_ACTION=action,
        HERMES_DISABLE_LAZY_INSTALLS="1",
    )
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=90, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    import hermes_yaml
    on_disk = hermes_yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
    return json.loads(proc.stdout.strip().splitlines()[-1]), on_disk, snapshot


def _assert_saved(on_disk, allowlist):
    assert on_disk.get("model", {}).get("default") == "m"
    assert on_disk.get("command_allowlist") == allowlist, on_disk
    assert set(on_disk) <= {"model", "command_allowlist", "_config_version"}, on_disk


def test_first_always_in_frozen_worker_persists(tmp_path, monkeypatch):
    receipt, on_disk, snapshot = _always_in_worker(tmp_path, monkeypatch, "model:\n  default: m\n")
    assert "command_allowlist" not in snapshot, snapshot
    assert receipt["in_memory"] is True
    _assert_saved(on_disk, [KEY])


@pytest.mark.parametrize(
    "live_yaml, bind, expected",
    [
        ("model:\n  default: m\ncommand_allowlist: []\n", True, [KEY]),
        ("model:\n  default: m\ncommand_allowlist:\n- other\n", True, ["other", KEY]),
        ("model:\n  default: m\n", False, [KEY]),
    ],
    ids=["explicit-empty", "existing-list", "in-process"],
)
def test_controls_persist(tmp_path, monkeypatch, live_yaml, bind, expected):
    receipt, on_disk, _ = _always_in_worker(tmp_path, monkeypatch, live_yaml, bind=bind)
    assert receipt["in_memory"] is True
    _assert_saved(on_disk, expected)


def _bind_snapshot(monkeypatch, snapshot):
    import copy
    from hermes_cli import config as cfg
    monkeypatch.setattr(cfg, "_worker_config_snapshot", lambda: copy.deepcopy(snapshot))
    return cfg


def test_rebase_applies_an_edit_to_a_default_only_list(monkeypatch):
    from hermes_cli.config_worker_save import rebase_for_save
    snapshot = {"model": {"default": "m"}}
    cfg = _bind_snapshot(monkeypatch, snapshot)
    desired = cfg.load_config()
    desired["command_allowlist"] = ["recursive delete"]
    saved = rebase_for_save(desired, {"model": {"default": "m"}}, False)
    assert saved["command_allowlist"] == ["recursive delete"]


def test_rebase_keeps_a_list_the_snapshot_carried_and_the_operator_deleted(monkeypatch):
    from hermes_cli.config_worker_save import rebase_for_save
    snapshot = {"model": {"default": "m"}, "command_allowlist": ["a *"]}
    cfg = _bind_snapshot(monkeypatch, snapshot)
    desired = cfg.load_config()
    desired["command_allowlist"] = ["a *", "b *"]
    saved = rebase_for_save(desired, {"model": {"default": "m"}}, False)
    assert "command_allowlist" not in saved


def test_rebase_applies_an_edit_to_a_default_only_mapping(monkeypatch):
    import copy
    from hermes_cli.config_worker_save import rebase_for_save
    snapshot = {"model": {"default": "m"}}
    cfg = _bind_snapshot(monkeypatch, snapshot)
    base = cfg.load_config()
    desired = copy.deepcopy(base)
    desired["approvals"]["mode"] = "off"
    saved = rebase_for_save(desired, {"model": {"default": "m"}}, False)
    expected = copy.deepcopy(base["approvals"])
    expected["mode"] = "off"
    assert saved["approvals"] == expected
    assert saved["model"] == {"default": "m"}
    assert set(saved) == {"model", "approvals"}


def test_rebase_keeps_a_mapping_the_snapshot_carried_and_the_operator_deleted(monkeypatch):
    from hermes_cli.config_worker_save import rebase_for_save
    snapshot = {"model": {"default": "m"}, "approvals": {"mode": "manual"}}
    cfg = _bind_snapshot(monkeypatch, snapshot)
    desired = cfg.load_config()
    desired["approvals"]["mode"] = "off"
    saved = rebase_for_save(desired, {"model": {"default": "m"}}, False)
    assert "approvals" not in saved


def test_rebase_applies_an_edit_to_a_nested_default_only_list(monkeypatch):
    from hermes_cli.config_worker_save import rebase_for_save
    # The parent is in the snapshot and the live file. ``deny`` is not: it comes
    # only from the schema defaults under that parent.
    snapshot = {"model": {"default": "m"}, "approvals": {"mode": "manual"}}
    live = {"model": {"default": "m"}, "approvals": {"mode": "manual"}}
    cfg = _bind_snapshot(monkeypatch, snapshot)
    desired = cfg.load_config()
    desired["approvals"]["deny"] = ["rm -rf"]
    saved = rebase_for_save(desired, live, False)
    assert saved["approvals"] == {"mode": "manual", "deny": ["rm -rf"]}


_BYPASS_FRESH = "model:\n  default: m\ndisplay:\n  skin: mono\n"


def _assert_kept(on_disk, **extra):
    assert on_disk.get("model") == {"default": "m"}, on_disk
    assert on_disk.get("display") == {"skin": "mono"}, on_disk
    for key, value in extra.items():
        assert on_disk.get(key) == value, on_disk
    assert set(on_disk) <= {"model", "display", "_config_version", *extra}, on_disk


@pytest.mark.parametrize("safe_mode", [True, False], ids=["safe-mode", "ignore-user-config"])
def test_first_always_in_bypass_worker_persists(tmp_path, monkeypatch, safe_mode):
    receipt, on_disk, snapshot = _always_in_worker(
        tmp_path, monkeypatch, _BYPASS_FRESH, bind="bypass", safe_mode=safe_mode,
    )
    assert snapshot.get("command_allowlist") == []
    assert receipt["in_memory"] is True
    _assert_kept(on_disk, command_allowlist=[KEY])


@pytest.mark.parametrize("safe_mode", [True, False], ids=["safe-mode", "ignore-user-config"])
def test_bypass_worker_persists_a_default_only_mapping_edit(tmp_path, monkeypatch, safe_mode):
    receipt, on_disk, snapshot = _always_in_worker(
        tmp_path, monkeypatch, _BYPASS_FRESH, bind="bypass", safe_mode=safe_mode, action="approvals",
    )
    assert snapshot.get("approvals", {}).get("mode") == "smart"
    assert receipt == {"ok": True}
    _assert_kept(on_disk, approvals={"mode": "off"})


def test_bypass_worker_keeps_an_existing_allowlist_and_unrelated_keys(tmp_path, monkeypatch):
    receipt, on_disk, snapshot = _always_in_worker(
        tmp_path, monkeypatch, _BYPASS_FRESH + "command_allowlist:\n- other\n",
        bind="bypass", safe_mode=True,
    )
    assert snapshot.get("command_allowlist") == []
    assert receipt["in_memory"] is True
    _assert_kept(on_disk, command_allowlist=["other", KEY])


def test_rebase_keeps_an_explicit_empty_list_the_operator_deleted(monkeypatch):
    from hermes_cli.config_worker_save import rebase_for_save
    snapshot = {"model": {"default": "m"}, "command_allowlist": []}
    cfg = _bind_snapshot(monkeypatch, snapshot)
    desired = cfg.load_config()
    desired["command_allowlist"] = ["recursive delete"]
    saved = rebase_for_save(desired, {"model": {"default": "m"}}, False)
    assert "command_allowlist" not in saved


def test_rebase_keeps_a_nested_list_the_snapshot_carried_and_the_operator_deleted(monkeypatch):
    from hermes_cli.config_worker_save import rebase_for_save
    snapshot = {"model": {"default": "m"}, "approvals": {"mode": "manual", "deny": ["old"]}}
    live = {"model": {"default": "m"}, "approvals": {"mode": "manual"}}
    cfg = _bind_snapshot(monkeypatch, snapshot)
    desired = cfg.load_config()
    desired["approvals"]["deny"] = ["old", "new"]
    saved = rebase_for_save(desired, live, False)
    assert saved["approvals"] == {"mode": "manual"}
