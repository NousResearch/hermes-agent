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
    services: list[tuple[str, bool]] = field(default_factory=list)

    @property
    def is_default(self) -> bool:
        return self.name == "default"

    @property
    def has_gateway(self) -> bool:
        return self.pid is not None or bool(self.services)

    def service_label(self) -> str:
        if not self.services:
            return "none"
        return ", ".join(f"{kind} ({'system' if system else 'user'})"
                         if kind == "systemd" else kind for kind, system in self.services)

    def to_dict(self) -> dict:
        return {
            "profile": self.name, "home": str(self.home), "pid": self.pid,
            "services": [{"kind": kind, "system": system} for kind, system in self.services],
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
        # The flag can be left behind by a partially applied migration. Standalone secondary
        # gateways still have to be stopped/uninstalled before this fleet is complete.
        if self.standalone_secondaries:
            return False
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
        return next((service for p in self.profiles for service in p.services), None)

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


def _profile_homes(profile_allowlist: Optional[list[str]] = None) -> list[tuple[str, Path]]:
    from hermes_cli.profiles import profiles_to_serve
    return list(profiles_to_serve(multiplex=True, profile_allowlist=profile_allowlist))


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


def _installed_services(home: Path) -> list[tuple[str, bool]]:
    """Every installed service scope for ``home``'s gateway (units / plist on disk)."""
    from hermes_cli import gateway as gw
    services = []
    with _home_env(home):
        if gw.supports_systemd_services():
            for system in (False, True):
                if gw.get_systemd_unit_path(system=system).exists():
                    services.append(("systemd", system))
        if gw.is_macos() and gw.get_launchd_plist_path().exists():
            services.append(("launchd", False))
    return services


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
    from agent.secret_scope import load_env_file
    scoped_env = dict(os.environ)
    scoped_env.update(load_env_file(default_home / ".env"))
    env = _env_multiplex_profiles_override(scoped_env)
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
                    f"Profile '{profile.name}' enables {platform.value}; use the shared listener at "
                    f"{_listener_url(default_cfg, platform.value, profile.name)}."
                )
            else:
                plan.blockers.append(
                    f"Profile '{profile.name}' enables {platform.value}, which still binds its own port; "
                    f"the multiplexer has no /p/{profile.name}/ ingress for this adapter. Disable it there "
                    f"(platforms.{platform.value}.enabled: false) or keep '{profile.name}' on a standalone "
                    f"gateway (hermes -p {profile.name} gateway start --force)."
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
    allowlist = None
    with contextlib.suppress(Exception):
        allowlist = getattr(_profile_gateway_config(default_home), "multiplex_profile_allowlist", None)
    profiles = [
        ProfileGateway(name=name, home=home, pid=_live_gateway_pid(home), services=_installed_services(home))
        for name, home in _profile_homes(allowlist)
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
        what = " + ".join(x for x in (f"stop pid {p.pid}" if p.pid else "", f"uninstall {p.service_label()}" if p.services else "") if x)
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
    if plan_default.services:
        kind, system = plan_default.services[0]
        for extra_kind, extra_system in plan_default.services[1:]:
            _service_op(extra_kind, extra_system, "stop", default_home)
            _service_op(extra_kind, extra_system, "uninstall", default_home)
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


def _restore_default_gateway(default_home: Path, default_rec: dict) -> None:
    """Restore exactly the default gateway footprint recorded before migration."""
    desired_services = _recorded_services(default_rec)
    current_services = _installed_services(default_home)
    for service in current_services:
        _service_op(*service, "stop", default_home)
        if service not in desired_services:
            _service_op(*service, "uninstall", default_home)
    if _live_gateway_pid(default_home) is not None:
        _stop_gateway_process(default_home)
    for service in desired_services:
        if service not in current_services:
            _service_op(*service, "install", default_home)
        _service_op(*service, "start", default_home)
    if not desired_services and default_rec.get("pid"):
        if not _spawn_detached_gateway(default_home):
            raise RuntimeError("could not restore the default gateway (detached)")


def _recorded_services(rec: dict) -> list[tuple[str, bool]]:
    # Version-one manifests recorded only one service; retain rollback for those on disk.
    rows = rec.get("services")
    if rows is None:
        rows = [rec["service"]] if rec.get("service") else []
    return [(row["kind"], bool(row.get("system"))) for row in rows]


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
        "version": 2, "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "flag_was": plan.multiplex_flag_on,
        "default": plan.default.to_dict(),
        "secondaries": [p.to_dict() for p in plan.standalone_secondaries],
    }
    # The complete rollback record must exist before the first stop/uninstall/kill/config mutation.
    _write_manifest(plan.default_home, manifest)
    for p in plan.standalone_secondaries:
        for kind, system in p.services:
            _service_op(kind, system, "stop", p.home)
            _service_op(kind, system, "uninstall", p.home)
            print(f"  ✓ {p.name}: stopped and removed its {p.service_label()} service")
        if p.pid is not None:
            _stop_gateway_process(p.home)
            print(f"  ✓ {p.name}: stopped standalone gateway (pid {p.pid})")
    _write_multiplex_flag(plan.default_home, True)
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
    ok = True
    try:
        _write_multiplex_flag(default_home, bool(manifest.get("flag_was", False)))
        _restore_default_gateway(default_home, manifest.get("default") or {})
        print("  ✓ default: restored its pre-migration gateway state")
    except Exception as exc:
        ok = False
        print(f"  ✗ default: {exc}")
    for rec in manifest.get("secondaries", []):
        home = Path(rec["home"])
        name = rec["profile"]
        from hermes_constants import named_profile_is_deleted
        if not home.is_dir() or named_profile_is_deleted(home):
            print(f"  - {name}: deleted profile skipped")
            continue
        try:
            services = _recorded_services(rec)
            for kind, system in services:
                _service_op(kind, system, "install", home)
                _service_op(kind, system, "start", home)
                print(f"  ✓ {name}: reinstalled and started its {kind} service")
            if not services and rec.get("pid"):
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
    try:
        migrated = apply_migration(plan)
    except Exception:
        rollback_migration(plan.default_home)
        raise
    if not migrated:
        restored = rollback_migration(plan.default_home)
        raise RuntimeError("Automatic gateway migration failed" + ("; standalone gateways restored" if restored else "; rollback incomplete"))
