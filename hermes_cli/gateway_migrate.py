"""``hermes gateway migrate --multiplex`` / ``--standalone``: move a per-profile-gateway install onto one
multiplexed default gateway (and back), with a table-driven preflight.

Standalone per-profile gateways stay supported; this is a migration path, not a removal. The
preflight reuses the gateway's own conflict logic (``GatewayRunner._adapter_credential_fingerprint``,
``platform_binds_port``, the adapters' ``serves_profile_prefix`` declaration) so its verdict matches
what the multiplexer would do at startup. ``hermes update`` calls :func:`maybe_auto_migrate_after_update`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

MANIFEST_NAME = "gateway_migration.json"
MIGRATE_COMMAND = "hermes gateway migrate --multiplex"
_SERVED_WAIT_SECONDS = 90.0


# --------------------------------------------------------------------------- data


@dataclass
class ProfileGateway:
    """One profile's standalone gateway footprint: live PID and/or installed service."""
    name: str
    home: Path
    pid: Optional[int] = None
    service: Optional[tuple[str, bool]] = None  # ("systemd", system) | ("launchd", False)

    @property
    def is_default(self) -> bool:
        return self.name == "default"

    @property
    def has_gateway(self) -> bool:
        return self.pid is not None or self.service is not None

    def service_label(self) -> str:
        if self.service is None:
            return "none"
        kind, system = self.service
        return f"{kind} ({'system' if system else 'user'})" if kind == "systemd" else kind

    def to_dict(self) -> dict:
        return {
            "profile": self.name, "home": str(self.home), "pid": self.pid,
            "service": None if self.service is None else {"kind": self.service[0], "system": self.service[1]},
        }


@dataclass
class MigrationPlan:
    default_home: Path
    profiles: list[ProfileGateway]
    multiplex_flag_on: bool
    live_served: Optional[list[str]]  # served_profiles the live default gateway recorded, if any
    blockers: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)

    @property
    def secondaries(self) -> list[ProfileGateway]:
        return [p for p in self.profiles if not p.is_default]

    @property
    def default(self) -> ProfileGateway:
        return next(p for p in self.profiles if p.is_default)

    @property
    def already_multiplexed(self) -> bool:
        return self.multiplex_flag_on or bool(self.live_served and len(self.live_served) > 1)

    @property
    def standalone_secondaries(self) -> list[ProfileGateway]:
        return [p for p in self.secondaries if p.has_gateway]

    @property
    def blocked(self) -> bool:
        return bool(self.blockers)

    def target_service_kind(self) -> Optional[tuple[str, bool]]:
        """Service manager the default gateway should end up on: its own, else the one the
        secondaries used (so a systemd-managed fleet stays systemd-managed)."""
        if self.default.service is not None:
            return self.default.service
        return next((p.service for p in self.secondaries if p.service is not None), None)

    def to_dict(self) -> dict:
        return {
            "default_home": str(self.default_home),
            "profiles": [p.to_dict() for p in self.profiles],
            "multiplex_flag_on": self.multiplex_flag_on,
            "live_served": self.live_served,
            "already_multiplexed": self.already_multiplexed,
            "blockers": list(self.blockers),
            "notices": list(self.notices),
            "eligible": self.eligible_for_migration(),
            "command": MIGRATE_COMMAND,
        }

    def eligible_for_migration(self) -> bool:
        """>= 2 profiles, at least one secondary with its own gateway, multiplex off, no blockers.
        This is the AUTO-migration (``hermes update``) bar; the explicit command also proceeds with
        zero standalone secondaries (see :func:`cmd_migrate`)."""
        return (
            len(self.profiles) >= 2 and bool(self.standalone_secondaries)
            and not self.already_multiplexed and not self.blocked
        )


# --------------------------------------------------------------------------- home / env plumbing


@contextlib.contextmanager
def _home_env(home: Path) -> Iterator[None]:
    """Run service-manager helpers as if ``home`` were the active HERMES_HOME. Both the contextvar
    override (``get_hermes_home``) and ``os.environ`` (``gateway.status`` identity files, unit
    generation) are switched, then restored."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    import hermes_constants
    previous = os.environ.get("HERMES_HOME")
    token = set_hermes_home_override(str(home))
    os.environ["HERMES_HOME"] = str(home)
    hermes_constants._default_hermes_root_memo = None
    try:
        yield
    finally:
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = previous
        hermes_constants._default_hermes_root_memo = None


def _default_home() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root()


def _profile_homes() -> list[tuple[str, Path]]:
    from hermes_cli.profiles import profiles_to_serve
    return list(profiles_to_serve(multiplex=True))


def _live_gateway_pid(home: Path) -> Optional[int]:
    """PID of a standalone gateway owned by ``home`` (pid file, then runtime status), else None."""
    from gateway.status import get_running_pid, get_runtime_status_running_pid, read_runtime_status
    with contextlib.suppress(Exception):
        pid = get_running_pid(home / "gateway.pid", cleanup_stale=False)
        if pid is not None:
            return pid
    with contextlib.suppress(Exception):
        return get_runtime_status_running_pid(read_runtime_status(home / "gateway_state.json"), expected_home=home)
    return None


def _installed_service(home: Path) -> Optional[tuple[str, bool]]:
    """Installed service kind for ``home``'s gateway (unit / plist on disk), else None."""
    from hermes_cli import gateway as gw
    with _home_env(home):
        if gw.supports_systemd_services():
            for system in (False, True):
                if gw.get_systemd_unit_path(system=system).exists():
                    return ("systemd", system)
        if gw.is_macos() and gw.get_launchd_plist_path().exists():
            return ("launchd", False)
    return None


def _service_op(kind: str, system: bool, verb: str, home: Path) -> None:
    """``stop`` / ``uninstall`` / ``start`` / ``restart`` / ``install`` on ``home``'s service."""
    from hermes_cli import gateway as gw
    with _home_env(home):
        if verb == "install":
            if kind == "launchd":
                gw.launchd_install()
            else:
                gw.systemd_install(system=system, non_interactive=True)
            return
        gw._service_call(kind, verb, system)


def _stop_gateway_process(home: Path) -> None:
    from hermes_cli.profiles import _stop_gateway_process
    _stop_gateway_process(home)


def _spawn_detached_gateway(home: Path) -> bool:
    from hermes_cli import gateway as gw
    with _home_env(home):
        return gw._spawn_detached_gateway()


def _read_multiplex_flag(default_home: Path) -> bool:
    from gateway.config import _env_multiplex_profiles_override
    env = _env_multiplex_profiles_override()
    if env is not None:
        return env
    cfg_path = default_home / "config.yaml"
    if not cfg_path.exists():
        return False
    from hermes_cli.config import read_user_config_raw
    cfg = read_user_config_raw(cfg_path) or {}
    gateway_section = cfg.get("gateway") if isinstance(cfg.get("gateway"), dict) else {}
    return bool(cfg.get("multiplex_profiles") or gateway_section.get("multiplex_profiles"))


def _write_multiplex_flag(default_home: Path, value: bool) -> None:
    """Set ``gateway.multiplex_profiles`` in the DEFAULT profile's config.yaml through the config API
    (same read-guard + nested-set + atomic write ``hermes config set`` uses; no raw YAML edits)."""
    from hermes_cli.config import _set_nested, _write_user_config, require_readable_config_before_write
    cfg_path = default_home / "config.yaml"
    user_config = require_readable_config_before_write(cfg_path)
    # A stale top-level alias would shadow the nested key the docs describe.
    user_config.pop("multiplex_profiles", None)
    _set_nested(user_config, "gateway.multiplex_profiles", value)
    _write_user_config(cfg_path, user_config)


# --------------------------------------------------------------------------- preflight checks


def _profile_gateway_config(home: Path):
    """This profile's ``GatewayConfig`` read exactly the way the multiplexer reads it: under the
    profile's own secret scope with multiplexing active, so a missing token stays missing instead of
    borrowing the CLI process's ``os.environ`` (which holds the launch profile's ``.env``)."""
    from gateway.config import load_gateway_config
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(home):
        return load_gateway_config()


@contextlib.contextmanager
def _multiplex_read_mode() -> Iterator[None]:
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        yield
    finally:
        set_multiplex_active(previous)


def _credential_probe(platform_config) -> SimpleNamespace:
    """Config-shaped stand-in for ``GatewayRunner._adapter_credential_fingerprint`` (which probes
    adapter attributes): token/api_key plus the id-style credentials adapters expose from ``extra``."""
    extra = getattr(platform_config, "extra", None) or {}
    return SimpleNamespace(
        token=getattr(platform_config, "token", None) or getattr(platform_config, "api_key", None),
        _app_id=extra.get("app_id"), _client_id=extra.get("client_id"), _bot_id=extra.get("bot_id"),
        _project_secret=extra.get("project_secret"), config=platform_config,
    )


def _credential_claims(config) -> dict[tuple, str]:
    """``(platform, fingerprint)`` for every enabled platform with a discoverable credential."""
    from gateway.run import GatewayRunner
    claims: dict[tuple, str] = {}
    for platform, platform_config in config.platforms.items():
        if not platform_config.enabled:
            continue
        fp = GatewayRunner._adapter_credential_fingerprint(_credential_probe(platform_config))
        if fp is not None:
            claims[(platform.value, fp)] = platform.value
    return claims


def _check_duplicate_credentials(plan: MigrationPlan, configs: dict[str, object]) -> None:
    """BLOCKER: the same bot credential configured on two profiles — the multiplexer would park
    the duplicate adapter, so one profile's bot would go silent after migration."""
    owners: dict[tuple, str] = {}
    for profile in plan.profiles:  # default first: it wins the claim, like at multiplexer startup
        cfg = configs.get(profile.name)
        if cfg is None:
            continue
        for claim, platform_value in _credential_claims(cfg).items():
            owner = owners.setdefault(claim, profile.name)
            if owner == profile.name:
                continue
            plan.blockers.append(
                f"Profiles '{owner}' and '{profile.name}' both configure {platform_value} with the same "
                f"credential: the bot can only belong to one profile; remove the token from "
                f"'{profile.name}' or keep it in {owner} and route {profile.name}'s chats with "
                f"profile_routes (gateway.profile_routes in {owner}'s config.yaml)."
            )


def platform_serves_profile_prefix(platform_value: str) -> bool:
    """True when the adapter for ``platform_value`` declares ``serves_profile_prefix`` (it answers
    ``/p/<profile>/...`` on the default listener). Read from the adapter CLASS — builtin table or the
    plugin registry entry — never from a hand-kept list, so new ingress adapters count automatically."""
    from gateway.platforms.base import BasePlatformAdapter

    def _declares(cls) -> bool:
        return isinstance(cls, type) and issubclass(cls, BasePlatformAdapter) and bool(
            getattr(cls, "serves_profile_prefix", False))

    with contextlib.suppress(Exception):
        from gateway.config import Platform
        from gateway.run import _BUILTIN_ADAPTERS, _builtin_adapter_import
        spec = _BUILTIN_ADAPTERS.get(Platform(platform_value))
        if spec is not None:
            adapter_cls, _ok = _builtin_adapter_import(spec[0], spec[1], spec[2])
            return _declares(adapter_cls)
    with contextlib.suppress(Exception):
        # Plugin-shipped adapters (sms, line, teams, feishu, wecom, ...) only exist in the registry
        # after discovery; a bare CLI process has not run it yet.
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform_value)
        if entry is not None:
            factory = entry.adapter_factory
            if _declares(factory):
                return True
            # Lambda factories: the adapter class lives in the factory's module.
            import importlib
            module = importlib.import_module(factory.__module__)
            return any(_declares(getattr(module, name)) for name in dir(module))
    return False


def _listener_url(default_cfg, platform_value: str, profile: str) -> str:
    from gateway.config import Platform
    extra = {}
    with contextlib.suppress(Exception):
        extra = (default_cfg.platforms.get(Platform(platform_value)) or SimpleNamespace(extra={})).extra or {}
    defaults = {"api_server": ("127.0.0.1", 8642), "webhook": ("0.0.0.0", 8644)}
    host, port = defaults.get(platform_value, ("<host>", "<port>"))
    host = extra.get("host") or host
    port = extra.get("port") or port
    tail = {"api_server": "/v1/...", "webhook": "/webhooks/<route>"}.get(platform_value, "/...")
    return f"http://{host}:{port}/p/{profile}{tail}"


def _check_secondary_port_binders(plan: MigrationPlan, configs: dict[str, object]) -> None:
    """BLOCKER when a secondary enables a port-binding platform with no ``/p/<profile>/`` ingress
    (the multiplexer skips the whole profile); NOTICE (URL changes) when the ingress exists."""
    from gateway.config import platform_binds_port
    default_cfg = configs.get("default")
    for profile in plan.secondaries:
        cfg = configs.get(profile.name)
        if cfg is None:
            continue
        for platform, platform_config in cfg.platforms.items():
            if not platform_config.enabled or not platform_binds_port(platform.value, platform_config.extra):
                continue
            if platform_serves_profile_prefix(platform.value):
                plan.notices.append(
                    f"Profile '{profile.name}': {platform.value} moves onto the default listener at "
                    f"{_listener_url(default_cfg, platform.value, profile.name)} (its key/secret is "
                    f"unchanged; update clients that call the old per-profile port)."
                )
            else:
                plan.blockers.append(
                    f"Profile '{profile.name}' enables {platform.value}, which binds its own port and has no "
                    f"/p/{profile.name}/ ingress on the default listener yet; the multiplexer would skip "
                    f"the whole profile. Disable it there (platforms.{platform.value}.enabled: false) or "
                    f"keep '{profile.name}' on a standalone gateway (hermes -p {profile.name} gateway start --force)."
                )


_PREFLIGHT_CHECKS: tuple[Callable[[MigrationPlan, dict[str, object]], None], ...] = (
    _check_duplicate_credentials,
    _check_secondary_port_binders,
)


def _load_profile_configs(plan: MigrationPlan) -> dict[str, object]:
    configs: dict[str, object] = {}
    with _multiplex_read_mode():
        for profile in plan.profiles:
            try:
                configs[profile.name] = _profile_gateway_config(profile.home)
            except Exception as exc:  # unreadable config is itself a blocker, not a crash
                plan.blockers.append(f"Profile '{profile.name}': could not load its gateway config ({exc}).")
    return configs


def build_migration_plan() -> MigrationPlan:
    """Enumerate profiles + their gateway footprint, then run every preflight check."""
    from hermes_cli.gateway_multiplex_served import recorded_served_profiles
    default_home = _default_home()
    profiles = [
        ProfileGateway(name=name, home=home, pid=_live_gateway_pid(home), service=_installed_service(home))
        for name, home in _profile_homes()
    ]
    plan = MigrationPlan(
        default_home=default_home, profiles=profiles,
        multiplex_flag_on=_read_multiplex_flag(default_home),
        live_served=recorded_served_profiles(default_home),
    )
    if len(plan.profiles) < 2:
        plan.notices.append("Only one profile exists: nothing to multiplex.")
        return plan
    configs = _load_profile_configs(plan)
    for check in _PREFLIGHT_CHECKS:
        check(plan, configs)
    plan.notices.append(
        "Profiles created after the migration are served by the running multiplexer as soon as "
        "they exist (it rescans profiles/ on create/delete and every 30s)."
    )
    return plan


# --------------------------------------------------------------------------- printing


def _print(lines: list[str]) -> None:
    for line in lines:
        print(line)


def format_plan(plan: MigrationPlan, *, dry_run: bool) -> list[str]:
    head = "Migration plan (dry run — nothing changed)" if dry_run else "Migration plan"
    lines = [head, f"  default home: {plan.default_home}", "", "  profile      gateway pid   service"]
    for p in plan.profiles:
        lines.append(f"  {p.name:<12} {str(p.pid or '-'):<13} {p.service_label()}")
    lines.append("")
    if plan.already_multiplexed:
        lines.append("  ✓ The default gateway is already multiplexing"
                     + (f" (serving {', '.join(plan.live_served)})" if plan.live_served else " (flag on)") + ".")
        return lines
    steps = []
    for p in plan.standalone_secondaries:
        what = " + ".join(x for x in (f"stop pid {p.pid}" if p.pid else "", f"uninstall {p.service_label()}" if p.service else "") if x)
        steps.append(f"  - {p.name}: {what}")
    if len(plan.profiles) < 2:  # the notice already says "only one profile exists"
        return lines + _plan_tail(plan)
    if not steps:
        lines.append("  No secondary profile runs its own gateway; the only step is turning the flag on:")
    else:
        lines += ["  Steps:", *steps]
    lines.append(f"  - default: set gateway.multiplex_profiles: true in {plan.default_home / 'config.yaml'}")
    target = plan.target_service_kind()
    lines.append(f"  - default: {'restart' if plan.default.has_gateway else 'start'} the gateway"
                 + (f" via {target[0]}" if target else " (detached)") + f", verify it serves {len(plan.profiles)} profiles")
    lines.append(f"  - record the previous state in {plan.default_home / MANIFEST_NAME} (rollback: hermes gateway migrate --standalone)")
    return lines + _plan_tail(plan)


def _plan_tail(plan: MigrationPlan) -> list[str]:
    lines: list[str] = []
    if plan.blockers:
        lines += ["", "  ✗ Blockers (fix these first, nothing will be changed):"]
        lines += [f"    • {b}" for b in plan.blockers]
    if plan.notices:
        lines += ["", "  Notices:"]
        lines += [f"    • {n}" for n in plan.notices]
    return lines


def format_update_warning(plan: MigrationPlan) -> list[str]:
    return [
        "⚠ Your profiles each run their own gateway. A single multiplexed gateway is the recommended",
        "  setup, but this install cannot be migrated automatically yet:",
        *[f"    • {b}" for b in plan.blockers],
        f"  After fixing the above, run:  {MIGRATE_COMMAND}",
        "  (`hermes update` will migrate automatically once nothing blocks it.)",
    ]


# --------------------------------------------------------------------------- apply / rollback


def _manifest_path(default_home: Path) -> Path:
    return default_home / MANIFEST_NAME


def _read_manifest(default_home: Path) -> Optional[dict]:
    path = _manifest_path(default_home)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_manifest(default_home: Path, data: dict) -> None:
    _manifest_path(default_home).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _wait_for_served(default_home: Path, expected: set[str], timeout: float) -> Optional[list[str]]:
    """Poll the default's ``gateway_state.json`` until ``served_profiles`` covers ``expected``."""
    from hermes_cli.gateway_multiplex_served import recorded_served_profiles
    deadline = time.monotonic() + timeout
    served: Optional[list[str]] = None
    while time.monotonic() < deadline:
        with _home_env(default_home):
            served = recorded_served_profiles(default_home)
        if served is not None and expected <= set(served):
            return served
        time.sleep(0.5)
    return served


def _restart_default(plan_default: ProfileGateway, target: Optional[tuple[str, bool]], default_home: Path) -> str:
    """Bring the default gateway up on the new flag value; returns a one-line description."""
    if plan_default.service is not None:
        kind, system = plan_default.service
        _service_op(kind, system, "restart", default_home)
        return f"restarted the default gateway via {kind}"
    if target is not None:
        kind, system = target
        _service_op(kind, system, "install", default_home)
        _service_op(kind, system, "start", default_home)
        return f"installed and started the default gateway via {kind}"
    verb = "restarted" if plan_default.pid is not None else "started"
    if plan_default.pid is not None:
        _stop_gateway_process(default_home)
    if not _spawn_detached_gateway(default_home):
        raise RuntimeError("could not spawn the default gateway (detached)")
    return f"{verb} the default gateway (detached; no service manager was in use)"


def apply_migration(plan: MigrationPlan, *, served_wait: float = _SERVED_WAIT_SECONDS) -> bool:
    """Stop/uninstall every secondary gateway, flip the flag, bring up the multiplexer, verify.
    Returns True when the multiplexer verifiably serves every profile."""
    if plan.blocked:
        _print(["✗ Migration refused:", *[f"  • {b}" for b in plan.blockers]])
        return False
    if plan.already_multiplexed:
        print("✓ Already multiplexed — nothing to do.")
        return True
    manifest = {
        "version": 1, "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "flag_was": plan.multiplex_flag_on,
        "default": plan.default.to_dict(),
        "secondaries": [p.to_dict() for p in plan.standalone_secondaries],
    }
    for p in plan.standalone_secondaries:
        if p.service is not None:
            kind, system = p.service
            _service_op(kind, system, "stop", p.home)
            _service_op(kind, system, "uninstall", p.home)
            print(f"  ✓ {p.name}: stopped and removed its {p.service_label()} service")
        if p.pid is not None:
            _stop_gateway_process(p.home)
            print(f"  ✓ {p.name}: stopped standalone gateway (pid {p.pid})")
        # Record progressively so a crash mid-way still leaves a usable rollback manifest.
        _write_manifest(plan.default_home, manifest)
    _write_multiplex_flag(plan.default_home, True)
    _write_manifest(plan.default_home, manifest)
    print(f"  ✓ default: gateway.multiplex_profiles: true ({plan.default_home / 'config.yaml'})")
    print(f"  ✓ {_restart_default(plan.default, plan.target_service_kind(), plan.default_home)}")

    expected = {p.name for p in plan.profiles}
    served = _wait_for_served(plan.default_home, expected, served_wait)
    if served is not None and expected <= set(served):
        _print(["", f"✓ Migrated: the default gateway now serves {len(served)} profiles: {', '.join(served)}",
                f"  Rollback any time with: hermes gateway migrate --standalone",
                *[f"  • {n}" for n in plan.notices]])
        return True
    missing = sorted(expected - set(served or []))
    _print(["", f"⚠ Migration applied, but the default gateway has not confirmed serving: {', '.join(missing)}",
            "  Check `hermes gateway status` and the gateway log; the flag and manifest are in place.",
            "  Rollback: hermes gateway migrate --standalone"])
    return False


def rollback_migration(default_home: Optional[Path] = None) -> bool:
    """``--standalone``: flag off, reinstall/start the recorded per-profile gateways, restart default."""
    default_home = default_home or _default_home()
    manifest = _read_manifest(default_home)
    if manifest is None:
        print(f"✗ No migration manifest at {_manifest_path(default_home)}; nothing to roll back.")
        print("  To leave multiplex mode by hand: hermes config set gateway.multiplex_profiles false && hermes gateway restart")
        return False
    _write_multiplex_flag(default_home, bool(manifest.get("flag_was", False)))
    print("  ✓ default: gateway.multiplex_profiles restored")
    default_rec = manifest.get("default") or {}
    default_service = default_rec.get("service")
    default_gw = ProfileGateway(
        "default", default_home, pid=_live_gateway_pid(default_home),
        service=(default_service["kind"], bool(default_service.get("system"))) if default_service else _installed_service(default_home),
    )
    if default_gw.has_gateway:
        print(f"  ✓ {_restart_default(default_gw, None, default_home)}")
    ok = True
    for rec in manifest.get("secondaries", []):
        home = Path(rec["home"])
        name = rec["profile"]
        try:
            service = rec.get("service")
            if service:
                kind, system = service["kind"], bool(service.get("system"))
                _service_op(kind, system, "install", home)
                _service_op(kind, system, "start", home)
                print(f"  ✓ {name}: reinstalled and started its {kind} service")
            elif rec.get("pid"):
                if _spawn_detached_gateway(home):
                    print(f"  ✓ {name}: started its standalone gateway (detached)")
                else:
                    ok = False
                    print(f"  ✗ {name}: could not start its standalone gateway")
        except Exception as exc:
            ok = False
            print(f"  ✗ {name}: {exc}")
    if ok:
        _manifest_path(default_home).unlink(missing_ok=True)
        print("✓ Rolled back to per-profile gateways.")
    else:
        print(f"⚠ Rollback incomplete; manifest kept at {_manifest_path(default_home)}.")
    return ok


# --------------------------------------------------------------------------- CLI + update hook


def _host_supports_migration() -> Optional[str]:
    """Reason the host cannot be migrated by this command (s6 slots / Windows tasks), else None."""
    from hermes_cli import gateway as gw
    if gw._running_under_s6():
        return "s6-supervised container: per-profile gateways are s6 slots; set gateway.multiplex_profiles on the default profile and restart the container instead."
    if gw.is_windows():
        return "Windows Scheduled Tasks are not migrated automatically; set gateway.multiplex_profiles true, stop the per-profile tasks, and `hermes gateway restart`."
    return None


def cmd_migrate(args) -> None:
    """``hermes gateway migrate [--multiplex|--standalone] [--dry-run] [--yes]``."""
    if getattr(args, "standalone", False):
        sys.exit(0 if rollback_migration() else 1)
    reason = _host_supports_migration()
    if reason:
        print(f"✗ {reason}")
        sys.exit(1)
    plan = build_migration_plan()
    dry_run = getattr(args, "dry_run", False)
    _print(format_plan(plan, dry_run=dry_run))
    if dry_run:
        return
    if plan.already_multiplexed:
        return
    if plan.blocked or len(plan.profiles) < 2:
        sys.exit(1 if plan.blocked else 0)
    # Zero standalone secondaries is still a migration when the user asks for it explicitly: the flag
    # goes on and the default gateway restarts (the update hook keeps treating that case as a no-op).
    if not getattr(args, "yes", False) and sys.stdin.isatty():
        from hermes_cli.setup import prompt_yes_no
        if not prompt_yes_no("Apply this migration now?", True):
            print("Aborted; nothing changed.")
            return
    print()
    sys.exit(0 if apply_migration(plan) else 1)


def maybe_auto_migrate_after_update() -> None:
    """``hermes update`` hook: with >= 2 profiles, per-profile gateways present and multiplex off,
    migrate automatically when unblocked (deterministic, never prompts) or print the blocker block."""
    if _host_supports_migration() is not None:
        return
    plan = build_migration_plan()
    if plan.already_multiplexed or len(plan.profiles) < 2 or not plan.standalone_secondaries:
        return
    print()
    if plan.blocked:
        _print(format_update_warning(plan))
        return
    print("→ Migrating per-profile gateways onto one multiplexed default gateway...")
    _print(format_plan(plan, dry_run=False))
    apply_migration(plan)
