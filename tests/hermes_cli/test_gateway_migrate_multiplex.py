"""``hermes gateway migrate``: preflight verdicts, apply/rollback bookkeeping, and the update hook.

Service layer is faked through the module's ``_installed_services`` / ``_service_op`` seams (the same
shape ``hermes gateway install`` tests use); the default gateway boot is faked by writing the
``served_profiles`` record the real multiplexer writes. Blockers reuse the gateway's own credential
fingerprint and port-binding predicates, so the tests assert verdict → effect, not internal lists.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants
from hermes_cli import gateway_migrate as gm


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """default + coder + ops; both secondaries run a 'live' standalone gateway with a systemd unit."""
    root = tmp_path / "hermes"
    for sub in ("profiles/coder", "profiles/ops"):
        (root / sub).mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    (root / ".env").write_text("TELEGRAM_BOT_TOKEN=111111:default-token\n", encoding="utf-8")
    (root / "profiles/coder/.env").write_text("TELEGRAM_BOT_TOKEN=222222:coder-token\n", encoding="utf-8")
    (root / "profiles/ops/.env").write_text("DISCORD_BOT_TOKEN=ops-discord-333333\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    for name in ("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "API_SERVER_KEY", "WEBHOOK_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)

    state = SimpleNamespace(
        # profile -> installed unit(s); a tuple is one unit, a list is every installed unit.
        services={"coder": ("systemd", False), "ops": ("systemd", False)},
        pids={"coder": 4101, "ops": 4102},
        ops=[],
        refused_at_start={},
    )

    def _service_op(kind, system, verb, home, *, run_as_user=None):
        name = _name(home)
        state.ops.append((name, verb))
        if verb == "start" and name != "default":
            # What the real `hermes -p <name> gateway run` checks first: is a live multiplexer
            # still recorded as serving me? (exit 78 if so — the unit is then parked for good).
            from hermes_cli.gateway import named_profile_served_by_running_multiplexer
            state.refused_at_start[name] = named_profile_served_by_running_multiplexer(name)
        if verb == "uninstall":
            remaining = [u for u in _units(state.services.get(name)) if u != (kind, system)]
            if remaining:
                state.services[name] = remaining
            else:
                state.services.pop(name, None)
        elif verb == "install":
            state.services[name] = (kind, system)
        elif verb in ("start", "restart") and name == "default":
            (root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(root)}))
            runtime_path = root / "gateway_state.json"
            runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
            runtime.update({"pid": os.getpid(), "hermes_home": str(root), "gateway_state": "running"})
            # Multiplex startup records ownership; standalone startup historically preserved the
            # old key, which is the stale-state half of #109473's rollback failure.
            if _config_flag(root):
                runtime["served_profiles"] = ["default", "coder", "ops"]
            runtime_path.write_text(json.dumps(runtime))

    import gateway.status as status
    # The default gateway the fixture "starts" is this process; the served probe verifies identity.
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "hermes gateway run")
    monkeypatch.setattr(gm, "_installed_services", lambda home: _units(state.services.get(_name(home))))
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda home: state.pids.get(_name(home)))
    monkeypatch.setattr(gm, "_service_op", _service_op)
    monkeypatch.setattr(gm, "_stop_gateway_process", lambda home: state.pids.pop(_name(home), None))
    monkeypatch.setattr(gm, "_spawn_detached_gateway", lambda home: _fake_spawn(state, home))
    # The compensator's liveness proof reads what a started gateway actually wrote, so a start that
    # returns without producing a gateway (the ExecMainStatus=75 respawn loop) is NOT "restored".
    monkeypatch.setattr(gm, "_wait_for_live_gateway",
                        lambda home, timeout: _pid_of_started_gateway(home))
    monkeypatch.setattr(gm, "_host_supports_migration", lambda: None)
    # This fleet uses core Telegram/Discord config only. Do not discover the
    # entire plugin catalog for each temporary profile; plugin configuration
    # has separate integration coverage. Keep the real identity resolver for
    # the system-unit identity tests below.
    from hermes_cli import plugins
    monkeypatch.setattr(plugins, "discover_plugins", lambda force=False: None)
    # Part of the faked service layer: the real check asks systemd's questions (root, NSS user).
    monkeypatch.setattr(gm, "_preflight_apply", lambda plan, target, run_as_user: None)
    # Identity inputs are pinned too, or the verdict depends on the HOST: the fake pids 4101/4102 may
    # name a real root-owned process in /proc on a CI runner (a spurious "UNIX privilege boundary"
    # blocker), and the user-scope unit path lives under the real $HOME. The whole fleet runs as this
    # user with no unit files on disk unless a test writes some (it repoints _SYSTEM_UNIT_DIR itself).
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_migrate_guards as guards
    monkeypatch.setattr(guards, "_pid_uid", lambda pid: root.stat().st_uid if pid in state.pids.values() else None)
    _unit_path = gw.get_systemd_unit_path
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: _unit_path(system=True) if system
                        else tmp_path / "user-units" / f"{gw.get_service_name()}.service")
    state.root = root
    return state


def _fake_spawn(state, home: Path) -> bool:
    root = state.root
    (root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(root)}))
    return True


def _pid_of_started_gateway(home: Path):
    marker = home / "gateway.pid"
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text()).get("pid")
    except ValueError:
        return None


def _name(home: Path) -> str:
    return hermes_constants.profile_name_for_home(home) or "default"


def _units(recorded) -> list:
    if recorded is None:
        return []
    return list(recorded) if isinstance(recorded, list) else [recorded]


_real_preflight = gm._preflight_apply


def _config_flag(root: Path):
    import yaml
    raw = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    return (raw.get("gateway") or {}).get("multiplex_profiles")


def test_dry_run_and_blocked_preflight_change_nothing(fleet, capsys):
    plan = gm.build_migration_plan()
    assert not plan.blocked and plan.eligible_for_migration()
    assert [p.name for p in plan.standalone_secondaries] == ["coder", "ops"]

    gm.cmd_migrate(SimpleNamespace(multiplex=True, dry_run=True, yes=True))  # returns; no exit
    assert "dry run" in capsys.readouterr().out
    # Blocked: coder reuses the default's Telegram token -> the gateway's own fingerprint says duplicate.
    (fleet.root / "profiles/coder/.env").write_text("TELEGRAM_BOT_TOKEN=111111:default-token\n", encoding="utf-8")
    blocked = gm.build_migration_plan()
    assert blocked.blocked and "profile_routes" in blocked.blockers[0] and "'coder'" in blocked.blockers[0]
    with pytest.raises(SystemExit) as exc:
        gm.cmd_migrate(SimpleNamespace(multiplex=True, dry_run=False, yes=True))
    assert exc.value.code == 1
    assert fleet.ops == [] and fleet.services == {"coder": ("systemd", False), "ops": ("systemd", False)}
    assert fleet.pids == {"coder": 4101, "ops": 4102} and _config_flag(fleet.root) is None
    assert not (fleet.root / gm.MANIFEST_NAME).exists()
    assert "nothing will be changed" in capsys.readouterr().out


def test_parked_profile_remains_in_migration_inventory(fleet):
    home = fleet.root / "profiles/coder"
    (home / "gateway.parked").touch()

    plan = gm.build_migration_plan()
    inventory = {p.name: p for p in plan.standalone_secondaries}
    assert "coder" in inventory
    assert inventory["coder"].home == home
    assert inventory["coder"].pid == fleet.pids["coder"]
    assert inventory["coder"].services == [("systemd", False)]

    # Parking must not hide migration preflight conflicts either.
    (home / ".env").write_text("TELEGRAM_BOT_TOKEN=111111:default-token\n", encoding="utf-8")
    blocked = gm.build_migration_plan()
    assert any("'coder'" in reason and "credential" in reason for reason in blocked.blockers)


def test_migration_removes_parked_footprint_without_waiting_for_it_to_serve(fleet, monkeypatch):
    home = fleet.root / "profiles/coder"
    (home / "gateway.parked").touch()
    service_op = gm._service_op

    def boot_unparked_profiles(kind, system, verb, home, **kwargs):
        service_op(kind, system, verb, home, **kwargs)
        if _name(home) == "default" and verb in ("start", "restart"):
            from hermes_cli.profiles import profiles_to_serve
            path = fleet.root / "gateway_state.json"
            runtime = json.loads(path.read_text())
            runtime["served_profiles"] = [name for name, _ in profiles_to_serve(True)]
            path.write_text(json.dumps(runtime))

    monkeypatch.setattr(gm, "_service_op", boot_unparked_profiles)
    plan = gm.build_migration_plan()
    assert any("verify it serves 2 profiles" in line for line in gm.format_plan(plan, dry_run=True))
    ok, manifest = _apply_capturing_manifest(plan, served_wait=0.1)
    assert ok
    assert "coder" not in fleet.pids and "coder" not in fleet.services
    assert (home / "gateway.parked").exists()
    assert {p["profile"] for p in manifest["secondaries"]} == {"coder", "ops"}
    gm._write_manifest(fleet.root, manifest)
    assert not gm.build_migration_plan().interrupted


def _apply_capturing_manifest(plan, *, served_wait=5.0, restore=False):
    """Apply, returning (ok, manifest-as-written). A CONFIRMED convergence deletes the manifest
    (manifest present == unfinished), so the only way to inspect it is to read it mid-flight."""
    seen = {}
    real_wait = gm._wait_for_served

    def _wait(default_home, expected, timeout):
        seen["manifest"] = json.loads((default_home / gm.MANIFEST_NAME).read_text(encoding="utf-8"))
        return real_wait(default_home, expected, timeout)

    gm._wait_for_served = _wait
    try:
        ok = gm.apply_migration(plan, served_wait=served_wait)
    finally:
        gm._wait_for_served = real_wait
    if restore:  # put it back to drive the compensator, which only ever runs on an UNfinished apply
        gm._write_manifest(plan.default_home, seen["manifest"])
    return ok, seen["manifest"]


def test_apply_clears_the_manifest_on_success_and_the_compensator_restores(fleet, capsys):
    plan = gm.build_migration_plan()
    ok, manifest = _apply_capturing_manifest(plan)
    assert ok is True
    assert not (fleet.root / gm.MANIFEST_NAME).exists(), "a confirmed migration is not 'interrupted'"
    assert {s["profile"] for s in manifest["secondaries"]} == {"coder", "ops"}
    assert all(s["service"] == {"kind": "systemd", "system": False} for s in manifest["secondaries"])
    assert _config_flag(fleet.root) is True
    assert "coder" not in fleet.services and "ops" not in fleet.services and fleet.pids == {}
    # The default is brought up on the SAME service manager the secondaries used.
    assert fleet.services["default"] == ("systemd", False)
    assert ("default", "install") in fleet.ops and ("default", "start") in fleet.ops
    assert "serves 3 profiles" in capsys.readouterr().out
    # Idempotent: a second run sees the live multiplexer and refuses cleanly.
    again = gm.build_migration_plan()
    assert again.already_multiplexed and gm.apply_migration(again) is True

    runtime_path = fleet.root / "gateway_state.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["platforms"] = {
        "telegram": {"state": "connected"},
        "coder:telegram": {"state": "connected"},
        "ops:discord": {"state": "connected"},
    }
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    fleet.ops.clear()
    gm._write_manifest(fleet.root, manifest)  # the compensator only ever runs on a FAILED apply
    assert gm.rollback_migration(fleet.root) is True
    assert _config_flag(fleet.root) is False
    # ONE gateway, not a fleet: the flipped host lock refuses N per-profile gateways, and the
    # compensator clears the served record first, so every secondary it used to start would lose
    # the flock race and respawn at exit 75 forever.
    assert fleet.services == {"default": ("systemd", False)}
    assert [op for op in fleet.ops if op[0] != "default"] == []
    assert fleet.ops[-1] == ("default", "restart")
    assert "coder, ops" in capsys.readouterr().out
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["served_profiles"] == []
    assert runtime["platforms"] == {"telegram": {"state": "connected"}}
    from hermes_cli.gateway import named_profile_served_by_running_multiplexer
    assert named_profile_served_by_running_multiplexer("coder") is False
    assert not (fleet.root / gm.MANIFEST_NAME).exists()


def test_migration_preserves_root_system_service_user_for_default_install(fleet, monkeypatch):
    """A root-owned secondary system unit must be replaced with an explicit root unit."""
    fleet.services["coder"] = ("systemd", True)
    monkeypatch.setattr(
        gm,
        "_systemd_service_user",
        lambda home, services: "root" if _name(home) == "coder" and ("systemd", True) in services else None,
    )
    plan = gm.build_migration_plan()
    root_secondary = next(p for p in plan.standalone_secondaries if p.name == "coder")
    assert root_secondary.run_as_user == "root"
    installs = []

    def _service_op(kind, system, verb, home, *, run_as_user=None):
        if _name(home) == "default" and verb == "install":
            installs.append((kind, system, run_as_user))
        fleet.services.pop(_name(home), None) if verb == "uninstall" else None
        if verb == "install":
            fleet.services[_name(home)] = (kind, system)
        if verb in ("start", "restart") and _name(home) == "default":
            (fleet.root / "gateway_state.json").write_text(json.dumps({
                "served_profiles": ["default", "coder", "ops"]}))

    monkeypatch.setattr(gm, "_service_op", _service_op)
    monkeypatch.setattr(gm, "_wait_for_served", lambda *args: ["default", "coder", "ops"])

    assert gm.apply_migration(plan, served_wait=0.1) is True
    assert installs == [("systemd", True, "root")]


def test_rollback_that_cannot_bring_the_gateway_back_reports_failure_and_keeps_the_manifest(
        fleet, monkeypatch, capsys):
    """A service-manager ``start`` that returns is not proof of a live gateway. Returning True on
    it printed '✓ Restored' over a unit respawning every 5s at ExecMainStatus=75."""
    ok, _manifest = _apply_capturing_manifest(gm.build_migration_plan(), restore=True)
    assert ok is True
    fleet.ops.clear()
    real_op = gm._service_op

    def _silent_start(kind, system, verb, home, *, run_as_user=None):
        if verb in ("start", "restart") and _name(home) == "default":
            (fleet.root / "gateway.pid").unlink(missing_ok=True)
            return  # the unit "starts" and produces nothing (ExecMainStatus=75, respawning)
        real_op(kind, system, verb, home, run_as_user=run_as_user)

    monkeypatch.setattr(gm, "_service_op", _silent_start)
    monkeypatch.setattr(gm, "_COMPENSATOR_WAIT_SECONDS", 0.5)
    assert gm.rollback_migration(fleet.root) is False
    out = capsys.readouterr().out
    assert "no gateway confirmed serving this host" in out and "Compensation incomplete" in out
    assert _config_flag(fleet.root) is False
    assert [op for op in fleet.ops if op[0] != "default"] == [], "no secondary may be rebuilt"
    assert (fleet.root / gm.MANIFEST_NAME).exists()


def test_rollback_never_respawns_a_detached_secondary(fleet, monkeypatch):
    """Detached gateways have no supervisor to loop, but they lose the same flock race and simply
    die — leaving the operator a process that vanished. The compensator spawns the default only."""
    gm._write_manifest(fleet.root, {"version": 1, "flag_was": False, "default": {"service": None}, "secondaries": [
        {"profile": n, "home": str(fleet.root / "profiles" / n), "pid": 4100, "service": None} for n in ("coder", "ops")]})
    fleet.services.clear()
    fleet.pids = {}
    spawned = []

    def _spawn(home):
        spawned.append(_name(home))
        fleet.pids[_name(home)] = 5000 + len(spawned)
        return _fake_spawn(fleet, home)

    monkeypatch.setattr(gm, "_spawn_detached_gateway", _spawn)
    assert gm.rollback_migration(fleet.root) is True
    assert spawned == ["default"]


def test_malformed_manifest_is_refused_before_any_mutation(fleet, capsys):
    (fleet.root / gm.MANIFEST_NAME).write_text(
        json.dumps({"version": 1, "flag_was": False, "default": [], "secondaries": [{"home": "x"}]}), encoding="utf-8")
    assert gm.rollback_migration(fleet.root) is False
    assert _config_flag(fleet.root) is None and fleet.ops == []
    assert "fix or delete the manifest" in capsys.readouterr().out


def test_secondary_port_binder_is_notice_with_ingress_and_blocker_without(fleet, monkeypatch):
    """The verdict follows the adapter's ``serves_profile_prefix`` declaration, not a hardcoded list."""
    (fleet.root / "profiles/ops/.env").write_text(
        "DISCORD_BOT_TOKEN=ops-discord-333333\nAPI_SERVER_KEY=ops-api-key-abcdef\n", encoding="utf-8")
    (fleet.root / "profiles/ops/config.yaml").write_text(
        "platforms:\n  api_server:\n    enabled: true\n    extra:\n      port: 9999\n", encoding="utf-8")
    plan = gm.build_migration_plan()
    assert not plan.blocked, plan.blockers
    assert any("api_server" in n and "/p/ops/" in n for n in plan.notices), plan.notices

    monkeypatch.setattr(gm, "platform_serves_profile_prefix", lambda value: False)
    plan = gm.build_migration_plan()
    assert plan.blocked and "api_server" in plan.blockers[0] and "/p/ops/" in plan.blockers[0]


def test_serves_profile_prefix_is_read_from_adapter_classes():
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.webhook import WebhookAdapter
    assert APIServerAdapter.serves_profile_prefix and WebhookAdapter.serves_profile_prefix
    assert gm.platform_serves_profile_prefix("api_server") is True
    assert gm.platform_serves_profile_prefix("webhook") is True
    # Every adapter that binds through shared_ingress.bind_listener is served at /p/<profile>/
    # for a secondary, so migrate must report it as a notice, never a blocker.
    for platform in ("sms", "line", "teams", "bluebubbles", "whatsapp_cloud", "msgraph_webhook"):
        assert gm.platform_serves_profile_prefix(platform) is True, platform
    # An outbound-only adapter never declares it (and never needs to).
    assert gm.platform_serves_profile_prefix("telegram") is False


def test_update_hook_migrates_when_unblocked_and_only_warns_when_blocked(fleet, capsys):
    fleet.services["default"] = ("systemd", False)  # same service domain as the secondaries
    gm.maybe_auto_migrate_after_update()
    out = capsys.readouterr().out
    assert "Migrating per-profile gateways" in out and "serves 3 profiles" in out
    assert _config_flag(fleet.root) is True and not (fleet.root / gm.MANIFEST_NAME).exists()

    # Blocked fleet: warning block with the fix + one-liner; nothing changes.
    for f in ("gateway.pid", "gateway_state.json", gm.MANIFEST_NAME):
        (fleet.root / f).unlink(missing_ok=True)
    (fleet.root / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    fleet.services.update({"coder": ("systemd", False), "default": ("systemd", False)})
    fleet.pids.update({"coder": 4101}); fleet.ops.clear()
    (fleet.root / "profiles/coder/.env").write_text("TELEGRAM_BOT_TOKEN=111111:default-token\n", encoding="utf-8")
    gm.maybe_auto_migrate_after_update()
    out = capsys.readouterr().out
    assert gm.MIGRATE_COMMAND in out and "profile_routes" in out
    assert fleet.ops == [] and _config_flag(fleet.root) is None


def test_update_hook_never_touches_single_profile_or_already_multiplexed(fleet, capsys):
    fleet.services.clear(); fleet.pids.clear()  # secondaries exist but run no gateway of their own
    gm.maybe_auto_migrate_after_update()
    assert capsys.readouterr().out == "" and _config_flag(fleet.root) is None


@pytest.mark.parametrize(
    ("secondary_service", "secondary_uid", "secondary_home", "expected"),
    [
        (("systemd", True), 1000, "profiles/coder", "different service domain"),
        (("launchd", False), 1000, "profiles/coder", "different service domain"),
        (("systemd", False), 2000, "profiles/coder", "UNIX privilege boundary"),
        (("systemd", False), 1000, "external", "outside"),
    ],
)
def test_update_hook_refuses_to_cross_service_user_or_home_boundary(
    fleet, capsys, monkeypatch, secondary_service, secondary_uid, secondary_home, expected,
):
    """#109954: the unattended hook must not fold a secondary that sits behind a kernel-enforced
    boundary (other service domain, other UNIX user, HERMES_HOME outside profiles/). It prints the
    boundary + the explicit command and touches nothing; the dry-run plan shows the same finding as a
    notice and the explicit command stays available."""
    fleet.services["default"] = ("systemd", False)
    fleet.services["ops"] = secondary_service
    ops_home = fleet.root.parent / "external-ops" if secondary_home == "external" else fleet.root / secondary_home
    monkeypatch.setattr(
        gm, "_gateway_identity",
        lambda home, pid, service: (secondary_uid if _name(home) == "ops" else 1000, ops_home if _name(home) == "ops" else home),
        raising=False,  # absent on the pre-fix module: the test must then fail on behaviour, not on the seam
    )

    gm.maybe_auto_migrate_after_update()

    out = capsys.readouterr().out
    assert expected in out and gm.MIGRATE_COMMAND in out and "'ops'" in out
    assert fleet.ops == [] and fleet.pids == {"coder": 4101, "ops": 4102}
    assert fleet.services == {"default": ("systemd", False), "coder": ("systemd", False), "ops": secondary_service}
    assert _config_flag(fleet.root) is None and not (fleet.root / gm.MANIFEST_NAME).exists()

    plan = gm.build_migration_plan()
    assert not plan.blocked and any(expected in n for n in plan.notices)
    gm.cmd_migrate(SimpleNamespace(multiplex=True, dry_run=True, yes=True))
    assert expected in capsys.readouterr().out


def test_update_hook_still_migrates_same_user_same_scope_profiles_under_the_default_tree(fleet, capsys, monkeypatch):
    """The guard is a boundary check, not a kill switch: one user, one service domain, everything under
    profiles/ (the shape `hermes profile create` produces) still auto-migrates."""
    fleet.services["default"] = ("systemd", False)
    monkeypatch.setattr(gm, "_gateway_identity", lambda home, pid, service: (1000, home), raising=False)

    gm.maybe_auto_migrate_after_update()

    out = capsys.readouterr().out
    assert "Migrating per-profile gateways" in out and "serves 3 profiles" in out
    assert _config_flag(fleet.root) is True and ("ops", "uninstall") in fleet.ops


def test_update_hook_folds_a_unit_less_default_when_every_secondary_shares_one_manager(fleet, capsys, monkeypatch):
    """The #118097 fleet: the default profile never had a gateway unit, every secondary runs under the
    SAME manager (all launchd, one macOS user). That is one service domain, and the manager the plan
    elects as the target (``target_service_kind``) is the reference the guard must agree with — not the
    default's empty unit list, which turned every such fleet into "blockers" instead of a fold.
    Controls: a default unit under another manager, and two managers among the secondaries, still refuse."""
    from hermes_cli.gateway_migrate_guards import auto_migration_blockers
    monkeypatch.setattr(gm, "_gateway_identity", lambda home, pid, service: (1000, home), raising=False)
    fleet.services.update({"coder": ("launchd", False), "ops": ("launchd", False)})
    assert "default" not in fleet.services
    plan = gm.build_migration_plan()
    assert plan.target_service_kind() == ("launchd", False)
    assert auto_migration_blockers(plan) == []

    gm.maybe_auto_migrate_after_update()
    out = capsys.readouterr().out
    assert "serves 3 profiles" in out and _config_flag(fleet.root) is True
    assert ("coder", "uninstall") in fleet.ops and ("ops", "uninstall") in fleet.ops
    assert ("default", "install") in fleet.ops and fleet.services == {"default": ("launchd", False)}

    # Control 1: the default HAS a unit under another manager -> still a different service domain.
    fleet.services.update({"default": ("systemd", False), "coder": ("launchd", False), "ops": ("launchd", False)})
    fleet.pids.update({"coder": 4101, "ops": 4102})
    blockers = auto_migration_blockers(gm.build_migration_plan())
    assert blockers and all("different service domain" in b for b in blockers)
    # Control 2: unit-less default, secondaries under two managers -> the non-elected one refuses.
    fleet.services.pop("default")
    fleet.services.update({"coder": ("launchd", False), "ops": ("systemd", False)})
    blockers = auto_migration_blockers(gm.build_migration_plan())
    assert len(blockers) == 1 and "'ops'" in blockers[0] and "different service domain" in blockers[0]


def test_auto_multiplex_migration_false_opts_out_of_the_update_hook_but_not_the_explicit_command(fleet, capsys):
    """``gateway.auto_multiplex_migration: false`` is a durable opt-out: an otherwise-eligible fleet is
    left alone by ``hermes update`` (no output, no ops, no flag flip), while the operator typing
    ``migrate --multiplex`` still migrates. Only the nested key counts."""
    assert gm.build_migration_plan().eligible_for_migration()  # would migrate but for the flag
    (fleet.root / "config.yaml").write_text(
        "model:\n  default: x\nauto_multiplex_migration: false\ngateway:\n  auto_multiplex_migration: false\n",
        encoding="utf-8")

    gm.maybe_auto_migrate_after_update()
    assert capsys.readouterr().out == ""
    assert fleet.ops == [] and _config_flag(fleet.root) is None
    assert fleet.services == {"coder": ("systemd", False), "ops": ("systemd", False)}
    assert fleet.pids == {"coder": 4101, "ops": 4102}
    assert not (fleet.root / gm.MANIFEST_NAME).exists()

    # A top-level alias is NOT honoured; absent and an explicit true keep the automatic behaviour.
    from hermes_cli.gateway_migrate_guards import auto_migration_opted_out
    (fleet.root / "config.yaml").write_text(
        "model:\n  default: x\nauto_multiplex_migration: false\n", encoding="utf-8")
    assert auto_migration_opted_out(fleet.root) is False
    (fleet.root / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    assert auto_migration_opted_out(fleet.root) is False
    (fleet.root / "config.yaml").write_text(
        "model:\n  default: x\ngateway:\n  auto_multiplex_migration: true\n", encoding="utf-8")
    assert auto_migration_opted_out(fleet.root) is False

    # The opt-out governs the AUTOMATIC path only: an explicit --multiplex is an explicit request.
    (fleet.root / "config.yaml").write_text(
        "model:\n  default: x\ngateway:\n  auto_multiplex_migration: false\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        gm.cmd_migrate(SimpleNamespace(multiplex=True, dry_run=False, yes=True))
    assert exc.value.code == 0
    assert _config_flag(fleet.root) is True and not (fleet.root / gm.MANIFEST_NAME).exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"auto_multiplex_migration": False},
        {"auto_migrate": False},
        {"auto_multiplex_migration": True, "auto_migrate": False},
        {"auto_multiplex_migration": False, "auto_migrate": True},
    ],
    ids=["canonical-false", "legacy-false", "legacy-false-conflict", "canonical-false-conflict"],
)
def test_automatic_migration_canonical_and_legacy_false_each_veto(fleet, capsys, settings):
    """The canonical setting and the briefly shipped legacy name both fail safe on false.

    Keep the fleet otherwise eligible so a no-op cannot be explained by a service-domain blocker.
    """
    fleet.services["default"] = ("systemd", False)
    gateway_yaml = "".join(f"  {key}: {str(value).lower()}\n" for key, value in settings.items())
    (fleet.root / "config.yaml").write_text(
        f"model:\n  default: x\ngateway:\n{gateway_yaml}", encoding="utf-8")

    gm.maybe_auto_migrate_after_update()

    assert capsys.readouterr().out == ""
    assert fleet.ops == [] and _config_flag(fleet.root) is None
    assert fleet.pids == {"coder": 4101, "ops": 4102}
    assert not (fleet.root / gm.MANIFEST_NAME).exists()


@pytest.mark.parametrize("source", ["expanded-user", "managed-overlay"])
def test_automatic_migration_opt_out_uses_effective_bool_config(fleet, monkeypatch, source):
    """The presence-sensitive update guard sees the same expanded/managed config and bool strings
    as gateway runtime, while retaining the legacy-key false veto."""
    managed = fleet.root / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    if source == "expanded-user":
        monkeypatch.setenv("AUTO_MIGRATION_ENABLED", "false")
        (fleet.root / "config.yaml").write_text(
            "gateway:\n  auto_migrate: ${AUTO_MIGRATION_ENABLED}\n", encoding="utf-8")
    else:
        (managed / "config.yaml").write_text(
            'gateway:\n  auto_multiplex_migration: "false"\n', encoding="utf-8")

    from hermes_cli.gateway_migrate_guards import auto_migration_opted_out

    assert auto_migration_opted_out(fleet.root) is True


def test_automatic_migration_config_key_is_canonical_schema_key():
    from hermes_cli.config import _validate_config_key

    assert _validate_config_key("gateway.auto_multiplex_migration") == (True, None)


def test_explicit_migrate_with_no_standalone_secondaries_still_flips_flag_and_restarts_default(fleet, capsys, monkeypatch):
    """The user typed --multiplex: 'nothing to migrate' + flag left off was a no-op the user did not ask
    for. The update hook keeps its no-op (previous test); the explicit command proceeds."""
    fleet.services.clear(); fleet.pids.clear()

    def _detached(home):  # no service manager anywhere -> detached start writes the served record
        fleet.services["default-detached"] = True
        (fleet.root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(fleet.root)}))
        (fleet.root / "gateway_state.json").write_text(json.dumps({
            "pid": os.getpid(), "hermes_home": str(fleet.root), "gateway_state": "running",
            "served_profiles": ["default", "coder", "ops"]}))
        return True
    monkeypatch.setattr(gm, "_spawn_detached_gateway", _detached)
    assert not gm.build_migration_plan().standalone_secondaries
    seen = {}
    real_wait = gm._wait_for_served
    monkeypatch.setattr(gm, "_wait_for_served", lambda home, exp, t: (
        seen.setdefault("m", json.loads((home / gm.MANIFEST_NAME).read_text(encoding="utf-8"))),
        real_wait(home, exp, t))[1])
    with pytest.raises(SystemExit) as exc:
        gm.cmd_migrate(SimpleNamespace(multiplex=True, dry_run=False, yes=True))
    assert exc.value.code == 0
    manifest = seen["m"]
    assert fleet.services.pop("default-detached") is True
    assert _config_flag(fleet.root) is True
    assert not (fleet.root / gm.MANIFEST_NAME).exists()
    assert manifest["secondaries"] == [] and manifest["flag_was"] is False
    assert ("default", "install") not in fleet.ops
    out = capsys.readouterr().out
    assert "serves 3 profiles" in out
    # That same manifest is what the compensator would have used: flag restored, nothing to reinstall.
    gm._write_manifest(fleet.root, manifest)
    assert gm.rollback_migration(fleet.root) is True and _config_flag(fleet.root) is False


def test_failed_default_bringup_compensates_to_one_detached_gateway(fleet, monkeypatch, capsys):
    """#110850: the last step (install/start the default) is the one that can fail after the
    destructive ones. It must not leave the flag on with no gateway anywhere. The compensator can
    no longer rebuild the fleet (the flipped host lock refuses it), and the recorded service
    manager is the very thing that just failed — so it falls back to ONE detached gateway."""
    real_op = gm._service_op

    def _refusing(kind, system, verb, home, *, run_as_user=None):
        if verb == "install" and _name(home) == "default":
            raise ValueError("Refusing to install the gateway system service as root; pass --run-as-user root")
        real_op(kind, system, verb, home, run_as_user=run_as_user)

    monkeypatch.setattr(gm, "_service_op", _refusing)
    assert gm.apply_migration(gm.build_migration_plan(), served_wait=0.1) is False
    out = capsys.readouterr().out
    assert "Restoring one host gateway" in out and "Compensated: one host gateway is running" in out
    assert "falling back to a detached gateway" in out
    assert _config_flag(fleet.root) is False
    # Compensation is a single gateway; the removed secondaries are named, not rebuilt.
    assert "coder, ops" in out and "NOT reinstalled" in out
    assert "coder" not in fleet.services and "ops" not in fleet.services
    assert not (fleet.root / gm.MANIFEST_NAME).exists()
    # One gateway, no fleet left: a re-run is a clean convergence, never a refusal or a rollback.
    after = gm.build_migration_plan()
    assert not after.blocked and after.standalone_secondaries == []


def test_interrupted_apply_is_resumed_from_the_manifest_not_short_circuited(fleet, monkeypatch, capsys):
    """Flag flipped, secondaries gone, default never came up (the process died mid-apply): the re-run
    must finish the migration from the manifest — with the recorded User= — instead of reporting
    'already multiplexed' over a fleet with no gateway at all."""
    fleet.services["coder"] = ("systemd", True)
    monkeypatch.setattr(gm, "_systemd_service_user", lambda home, services: "root" if _name(home) == "coder" else None)
    with pytest.MonkeyPatch.context() as dying:
        dying.setattr(gm, "_restart_default", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            gm.apply_migration(gm.build_migration_plan(), served_wait=0.1)
    assert _config_flag(fleet.root) is True and "coder" not in fleet.services and "default" not in fleet.services
    installs = []
    real_op = gm._service_op

    def _recording(kind, system, verb, home, *, run_as_user=None):
        if verb == "install":
            installs.append((_name(home), kind, system, run_as_user))
        real_op(kind, system, verb, home, run_as_user=run_as_user)

    monkeypatch.setattr(gm, "_service_op", _recording)
    plan = gm.build_migration_plan()
    assert plan.interrupted and not plan.already_multiplexed
    assert gm.apply_migration(plan, served_wait=5.0) is True
    assert installs == [("default", "systemd", True, "root")]
    assert "serves 3 profiles" in capsys.readouterr().out


def test_every_installed_unit_of_a_secondary_is_removed_and_recorded(fleet, capsys):
    """A profile carrying a user AND a system unit: both are stopped/uninstalled (recording only the
    first found left the other live beside the multiplexer) and BOTH are recorded, so a resume knows
    the full topology. The compensator no longer reinstalls them — one gateway per host."""
    fleet.services["coder"] = [("systemd", False), ("systemd", True)]
    plan = gm.build_migration_plan()
    coder = next(p for p in plan.standalone_secondaries if p.name == "coder")
    assert coder.services == [("systemd", False), ("systemd", True)]
    ok, manifest = _apply_capturing_manifest(plan, restore=True)
    assert ok is True
    assert "coder" not in fleet.services
    coder_rec = next(r for r in manifest["secondaries"] if r["profile"] == "coder")
    assert [(s["kind"], s["system"]) for s in coder_rec["services"]] == [("systemd", False), ("systemd", True)]
    capsys.readouterr()
    assert gm.rollback_migration(fleet.root) is True
    assert [op for op in fleet.ops if op[0] == "coder" and op[1] in ("install", "start")] == [], \
        "the compensator restores one gateway, not a two-unit per-profile fleet"

    # The unattended hook does not resolve an ambiguous two-unit topology on its own.
    for f in (gm.MANIFEST_NAME, "gateway.pid", "gateway_state.json"):
        (fleet.root / f).unlink(missing_ok=True)
    (fleet.root / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    fleet.services.update({"default": ("systemd", False), "coder": [("systemd", False), ("systemd", True)]})
    fleet.pids.update({"coder": 4101, "ops": 4102}); fleet.ops.clear()
    gm.maybe_auto_migrate_after_update()
    assert "more than one installed service" in capsys.readouterr().out and fleet.ops == []


@pytest.mark.linux_only
def test_unresolvable_system_unit_user_is_unknown_principal_not_directory_owner(fleet, tmp_path, monkeypatch, capsys):
    """A system unit pinned to a User= this host cannot resolve: the principal is unknown, never the
    profile directory's owner, and unknown blocks the unattended path."""
    from hermes_cli import gateway as gw
    from hermes_cli.gateway_migrate_guards import gateway_identity
    unit_dir = tmp_path / "system"; unit_dir.mkdir()
    monkeypatch.setattr(gw, "_SYSTEM_UNIT_DIR", unit_dir)
    coder_home = fleet.root / "profiles/coder"
    with gm._home_env(coder_home):
        gw.get_systemd_unit_path(system=True).write_text("[Service]\nUser=nobody-such-user-xyz\n", encoding="utf-8")
    uid, _home = gateway_identity(coder_home, None, [("systemd", True)])
    assert uid is None  # NOT coder_home.stat().st_uid
    # Same unit shape without a system unit keeps the directory-owner answer for user-scope gateways.
    assert gateway_identity(coder_home, None, [("systemd", False)])[0] == coder_home.stat().st_uid

    fleet.services.update({"default": ("systemd", True), "coder": ("systemd", True)})
    monkeypatch.setattr(gm, "_gateway_identity",
                        lambda home, pid, services: (None if _name(home) == "coder" else 1000, home))
    gm.maybe_auto_migrate_after_update()
    out = capsys.readouterr().out
    assert "cannot be resolved" in out and gm.MIGRATE_COMMAND in out
    assert fleet.ops == [] and _config_flag(fleet.root) is None


def test_opt_out_reads_effective_config_managed_false_wins_and_string_false_is_false(fleet, tmp_path, monkeypatch):
    """The opt-out authorizes an unattended destructive action, so it reads the same effective config
    the CLI does: a managed ``false`` overrides the user's ``true``; a hand-written ``"false"`` string is
    an opt-out, not a truthy value; the declared default keeps absent == opted in."""
    from hermes_cli import config as cfg
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.gateway_migrate_guards import auto_migration_opted_out
    from hermes_cli import managed_scope
    assert DEFAULT_CONFIG["gateway"]["auto_multiplex_migration"] is True
    assert auto_migration_opted_out(fleet.root) is False  # absent -> DEFAULT_CONFIG value

    (fleet.root / "config.yaml").write_text(
        "model:\n  default: x\ngateway:\n  auto_multiplex_migration: 'false'\n", encoding="utf-8")
    assert auto_migration_opted_out(fleet.root) is True

    (fleet.root / "config.yaml").write_text(
        "model:\n  default: x\ngateway:\n  auto_multiplex_migration: true\n", encoding="utf-8")
    assert auto_migration_opted_out(fleet.root) is False
    managed = tmp_path / "managed"; managed.mkdir()
    (managed / "config.yaml").write_text("gateway:\n  auto_multiplex_migration: false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    with gm._home_env(fleet.root):
        assert cfg.load_config()["gateway"]["auto_multiplex_migration"] is False
    assert auto_migration_opted_out(fleet.root) is True
    gm.maybe_auto_migrate_after_update()
    assert fleet.ops == [] and _config_flag(fleet.root) is None


@pytest.mark.parametrize("default_unit_preinstalled", [False, True])
def test_interruption_after_the_default_unit_exists_is_still_interrupted_not_already_multiplexed(
    fleet, monkeypatch, capsys, default_unit_preinstalled,
):
    """``systemd_install`` writes the unit before the start that can be killed; an existing stopped
    default unit can be interrupted mid-restart. Either way the flag is on and a default unit exists
    but nothing serves anyone: that is an interrupted migration to resume, not a completed one. Only
    a LIVE multiplexer that recorded ``served_profiles`` counts as already multiplexed."""
    if default_unit_preinstalled:
        fleet.services["default"] = ("systemd", False)
    real_op = gm._service_op

    def _killed_at_start(kind, system, verb, home, *, run_as_user=None):
        if verb in ("start", "restart") and _name(home) == "default":
            raise KeyboardInterrupt()
        real_op(kind, system, verb, home, run_as_user=run_as_user)

    with pytest.MonkeyPatch.context() as dying:
        dying.setattr(gm, "_service_op", _killed_at_start)
        with pytest.raises(KeyboardInterrupt):
            gm.apply_migration(gm.build_migration_plan(), served_wait=0.1)
    assert _config_flag(fleet.root) is True and fleet.services == {"default": ("systemd", False)}
    assert (fleet.root / gm.MANIFEST_NAME).exists()

    plan = gm.build_migration_plan()
    assert plan.interrupted and not plan.already_multiplexed
    fleet.ops.clear()
    assert gm.apply_migration(plan, served_wait=5.0) is True
    assert fleet.ops[-1] == ("default", "restart") and "serves 3 profiles" in capsys.readouterr().out
    # Postcondition met: the next plan sees the live multiplexer and stops.
    assert gm.build_migration_plan().already_multiplexed


@pytest.mark.parametrize("failing_op", [("ops", "stop"), ("coder", "uninstall"), ("default", "flag")])
def test_failure_anywhere_in_the_destructive_phase_restores_the_removed_secondaries(fleet, monkeypatch, capsys, failing_op):
    """The compensation boundary covers the whole destructive phase, not only the default bring-up:
    a later secondary's stop, a unit unlink/daemon-reload, or the flag write failing after an earlier
    secondary was removed must put that secondary back (the manifest alone is not a restored fleet)."""
    name, verb = failing_op
    real_op = gm._service_op

    def _failing(kind, system, verb_, home, *, run_as_user=None):
        if (_name(home), verb_) == (name, verb):
            raise RuntimeError(f"{name} {verb} failed")
        real_op(kind, system, verb_, home, run_as_user=run_as_user)

    monkeypatch.setattr(gm, "_service_op", _failing)
    if verb == "flag":
        real_flag = gm._write_multiplex_flag
        monkeypatch.setattr(gm, "_write_multiplex_flag",
                            lambda home, value: (_ for _ in ()).throw(OSError("read-only config")) if value else real_flag(home, value))
    assert gm.apply_migration(gm.build_migration_plan(), served_wait=0.1) is False
    out = capsys.readouterr().out
    assert "Restoring one host gateway" in out and "Compensated: one host gateway is running" in out
    assert _config_flag(fleet.root) is not True
    # Whatever the destructive phase had already removed stays removed: the compensator restores
    # ONE gateway and never re-installs or re-starts a per-profile one.
    assert [op for op in fleet.ops if op[0] != "default" and op[1] in ("install", "start")] == []
    assert not (fleet.root / gm.MANIFEST_NAME).exists()
    # Secondaries the phase had not reached yet are untouched, so a corrected re-run is a normal
    # migration; the ones it had already removed are simply served by the restored host gateway.
    after = gm.build_migration_plan()
    assert not after.blocked and after.eligible_for_migration()


@pytest.mark.linux_only
def test_known_bringup_refusal_is_rejected_before_any_secondary_is_touched(fleet, monkeypatch, capsys):
    """A system-unit fleet with no recorded User= run by root is the #110850 refusal: known from the plan,
    so it is refused before a working gateway is stopped rather than discovered and rolled back."""
    from hermes_cli import gateway as gw
    fleet.services.update({"coder": ("systemd", True), "ops": ("systemd", True)})
    monkeypatch.setattr(gm, "_systemd_service_user", lambda home, services: None)
    monkeypatch.setattr(gm, "_preflight_apply", _real_preflight)
    monkeypatch.setattr(gw, "_require_root_for_system_service", lambda action: None)  # we are "root"
    for var in ("SUDO_USER", "USER", "LOGNAME"):
        monkeypatch.setenv(var, "root")
    assert gm.apply_migration(gm.build_migration_plan(), served_wait=0.1) is False
    out = capsys.readouterr().out
    assert "before changing anything" in out and "--run-as-user root" in out
    assert fleet.ops == [] and _config_flag(fleet.root) is None and not (fleet.root / gm.MANIFEST_NAME).exists()


@pytest.mark.linux_only
def test_unknown_default_system_principal_blocks_the_update_hook(fleet, tmp_path, monkeypatch, capsys):
    """Mirror of the unknown-secondary case: the default's system unit names a User= this host cannot
    resolve while both secondaries are known root system units. Folding INTO an unidentifiable
    principal is the same boundary; known-same uid still folds, known-different still refuses."""
    from hermes_cli import gateway as gw
    from hermes_cli.gateway_migrate_guards import auto_migration_blockers, gateway_identity
    unit_dir = tmp_path / "system"; unit_dir.mkdir()
    monkeypatch.setattr(gw, "_SYSTEM_UNIT_DIR", unit_dir)
    with gm._home_env(fleet.root):
        gw.get_systemd_unit_path(system=True).write_text("[Service]\nUser=no-such-pr111062-user\n", encoding="utf-8")
    for name in ("coder", "ops"):
        with gm._home_env(fleet.root / "profiles" / name):
            gw.get_systemd_unit_path(system=True).write_text("[Service]\nUser=root\n", encoding="utf-8")
    fleet.services.update({"default": ("systemd", True), "coder": ("systemd", True), "ops": ("systemd", True)})
    fleet.pids.clear()  # stopped units everywhere: identity comes from User=, resolved for real
    plan = gm.build_migration_plan()
    assert plan.default.uid is None and {p.uid for p in plan.standalone_secondaries} == {0}
    assert gateway_identity(fleet.root, None, [("systemd", True)])[0] is None
    blockers = auto_migration_blockers(plan)
    assert len(blockers) == 1 and "default gateway" in blockers[0] and "cannot be resolved" in blockers[0]
    gm.maybe_auto_migrate_after_update()
    out = capsys.readouterr().out
    assert "cannot be resolved" in out and gm.MIGRATE_COMMAND in out
    assert fleet.ops == [] and _config_flag(fleet.root) is None

    # Controls: same known uid folds; a different known uid refuses.
    monkeypatch.setattr(gm, "_gateway_identity", lambda home, pid, services: (0, home))
    assert auto_migration_blockers(gm.build_migration_plan()) == []
    monkeypatch.setattr(gm, "_gateway_identity", lambda home, pid, services: (0 if _name(home) == "default" else 1000, home))
    assert any("UNIX privilege boundary" in b for b in auto_migration_blockers(gm.build_migration_plan()))


# --------------------------------------------------------------------------- multiplex-only convergence


def test_migrate_leaves_exactly_one_host_unit_serving_every_profile_and_is_idempotent(fleet, capsys):
    """(a) Two per-profile units in, ONE host unit out, and a second run changes nothing.

    The unit count is the contract: multiplex-only means one supervisor unit per host, not one per
    profile, and the host gateway must report every profile as served.
    """
    fleet.services["default"] = ("systemd", False)
    assert len(fleet.services) == 3  # default + the two per-profile units
    with pytest.raises(SystemExit) as exc:
        gm.cmd_migrate(SimpleNamespace(multiplex=True, dry_run=False, yes=True))
    assert exc.value.code == 0
    assert fleet.services == {"default": ("systemd", False)}
    assert fleet.pids == {}
    served = json.loads((fleet.root / "gateway_state.json").read_text(encoding="utf-8"))["served_profiles"]
    assert sorted(served) == ["coder", "default", "ops"]
    capsys.readouterr()

    fleet.ops.clear()
    plan = gm.build_migration_plan()
    assert plan.already_multiplexed and not plan.standalone_secondaries
    gm.cmd_migrate(SimpleNamespace(multiplex=True, dry_run=False, yes=True))  # returns, no exit
    assert fleet.ops == [] and fleet.services == {"default": ("systemd", False)}
    assert "already multiplexing" in capsys.readouterr().out


def test_half_migrated_host_converges_on_a_re_run(fleet, capsys):
    """(b) Flag already on, a per-profile unit + live gateway still there: the re-run must converge.

    This is the upgrade shape — a migration that died after the flag write, or a unit an operator
    reinstalled by hand. Reading the FLAG as "already multiplexed" made the re-run a no-op on
    exactly the host that still had two gateways.
    """
    (fleet.root / "config.yaml").write_text(
        "model:\n  default: x\ngateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    fleet.services = {"default": ("systemd", False), "coder": ("systemd", False)}
    fleet.pids = {"coder": 4101}

    plan = gm.build_migration_plan()
    assert plan.multiplex_flag_on is True
    assert not plan.already_multiplexed, "a host with a live per-profile gateway is not converged"
    assert [p.name for p in plan.standalone_secondaries] == ["coder"]

    with pytest.raises(SystemExit) as exc:
        gm.cmd_migrate(SimpleNamespace(multiplex=True, dry_run=False, yes=True))
    assert exc.value.code == 0
    assert fleet.services == {"default": ("systemd", False)} and fleet.pids == {}
    out = capsys.readouterr().out
    assert "Half-migrated host" in out and "serves 3 profiles" in out
    assert gm.build_migration_plan().already_multiplexed


def test_plan_names_every_process_it_will_sigterm_before_it_signals_anything(fleet, capsys):
    """A running gateway is never stopped silently: the dry run names the pids and changes nothing."""
    gm.cmd_migrate(SimpleNamespace(multiplex=True, dry_run=True, yes=True))
    out = capsys.readouterr().out
    assert "SIGTERM" in out and "coder (pid 4101)" in out and "ops (pid 4102)" in out
    assert fleet.ops == [] and fleet.pids == {"coder": 4101, "ops": 4102}






def test_windows_task_detection_reads_both_the_task_and_the_startup_fallback(monkeypatch):
    """`hermes gateway install` falls back to a Startup-folder entry when it cannot register a
    task; a migration that removed only the task would leave that entry launching a second
    gateway at the next logon."""
    from hermes_cli import gateway_windows as gww
    for task, startup, expected in ((True, False, True), (False, True, True), (False, False, False)):
        monkeypatch.setattr(gww, "is_task_registered", lambda t=task: t)
        monkeypatch.setattr(gww, "is_startup_entry_installed", lambda s=startup: s)
        assert gm._windows_task_installed() is expected
    # A schtasks failure is not evidence of a second gateway: fail closed to "not installed".
    monkeypatch.setattr(gww, "is_task_registered", lambda: (_ for _ in ()).throw(OSError("schtasks")))
    assert gm._windows_task_installed() is False


def test_windows_is_migratable_and_only_s6_is_refused(monkeypatch):
    """The host predicate: Windows used to be a flat refusal with hand-migration instructions.
    s6 stays refused -- its per-profile gateways are slots the container's own boot registers."""
    from hermes_cli import gateway as gw
    monkeypatch.setattr(gw, "_running_under_s6", lambda: False)
    assert gm._host_supports_migration() is None

    monkeypatch.setattr(gw, "_running_under_s6", lambda: True)
    reason = gm._host_supports_migration()
    assert reason is not None


def test_standalone_profile_is_listed_left_alone_and_not_a_fold_target(fleet):
    """A profile that opts itself out via `gateway.standalone: true` keeps its own gateway: it is
    neither a blocker nor a fold target — the plan names it under standalone_by_config instead."""
    root = fleet.root
    (root / "profiles" / "ops" / "config.yaml").write_text("gateway:\n  standalone: true\n", encoding="utf-8")
    plan = gm.build_migration_plan()
    assert "ops" not in [p.name for p in plan.profiles]
    assert "ops" not in [p.name for p in plan.standalone_secondaries]
    assert "coder" in [p.name for p in plan.standalone_secondaries]
    assert plan.standalone_by_config == ("ops",)
    payload = json.loads(json.dumps(plan.to_dict()))
    assert payload["standalone_by_config"] == list(plan.standalone_by_config)
    assert "ops" not in [p["profile"] for p in payload["profiles"]]
    assert any("ops" in line for line in gm.format_plan(plan, dry_run=True))
