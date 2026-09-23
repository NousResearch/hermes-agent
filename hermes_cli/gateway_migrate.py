"""``hermes gateway migrate --multiplex``: converge a per-profile-gateway install onto the ONE
host gateway, with a table-driven preflight.

Multiplex-only (Teknium ruling): exactly one ``hermes gateway run`` per host, serving every
profile. This command is the supported convergence path, and it is defined by TOPOLOGY, not by a
config flag — a host is converged when no secondary profile owns a gateway process or a supervisor
unit any more. That makes it re-runnable: a half-migrated host (flag flipped, a unit left behind, a
crash between the two) converges on the next run instead of being reported "already multiplexed".

There is no ``--standalone`` rollback command: reinstalling per-profile services is no longer a
supported topology. The rollback machinery survives as the COMPENSATOR inside a single failed
apply (:func:`rollback_migration`) — which restores exactly ONE gateway, the default's, because
the host gateway lock now refuses the fleet it used to rebuild — and the recorded manifest is what
the next re-run resumes from. A manifest on disk outranks every preflight gate: a resume
compensates an already-destructive state instead of initiating one.

The preflight reuses the gateway's own conflict logic (``GatewayRunner._adapter_credential_fingerprint``,
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
    # EVERY installed unit: [("systemd", system), ("launchd", False)]. A profile can carry a user AND a
    # system unit at once; recording only the first found left the other live beside the multiplexer.
    services: list[tuple[str, bool]] = field(default_factory=list)
    run_as_user: Optional[str] = None  # User= recorded by a system-scope systemd unit
    uid: Optional[int] = None  # owner of the gateway process/unit; None = unknown (never "different")
    runtime_home: Optional[Path] = None  # HERMES_HOME the installed unit pins, when it differs from ``home``

    @property
    def is_default(self) -> bool:
        return self.name == "default"

    @property
    def service(self) -> Optional[tuple[str, bool]]:
        """The unit the default gateway is (re)started through; None when nothing is installed."""
        return self.services[0] if self.services else None

    @property
    def has_gateway(self) -> bool:
        return self.pid is not None or bool(self.services)

    @property
    def has_system_unit(self) -> bool:
        return ("systemd", True) in self.services

    def service_label(self) -> str:
        return " + ".join(_service_label(s) for s in self.services) if self.services else "none"

    def to_dict(self) -> dict:
        return {
            "profile": self.name, "home": str(self.home), "pid": self.pid,
            "service": None if self.service is None else _service_dict(self.service),
            "services": [_service_dict(s) for s in self.services],
            "run_as_user": self.run_as_user,
            "uid": self.uid, "runtime_home": None if self.runtime_home is None else str(self.runtime_home),
        }


def _service_label(service: tuple[str, bool]) -> str:
    kind, system = service
    if kind == "systemd":
        return f"systemd ({'system' if system else 'user'})"
    return "Windows scheduled task" if kind == "windows" else kind


def _service_dict(service: tuple[str, bool]) -> dict:
    return {"kind": service[0], "system": service[1]}


@dataclass
class MigrationPlan:
    default_home: Path
    profiles: list[ProfileGateway]
    multiplex_flag_on: bool
    live_served: Optional[list[str]]  # served_profiles the live default gateway recorded, if any
    # The migration manifest on disk, when one exists. Its PRESENCE is the "this host is already
    # mid-migration, with units removed" signal that turns every later gate from a refusal into a
    # finding (see :func:`apply_migration`): a resume compensates a destructive state, it never
    # initiates one.
    manifest: Optional[dict] = None
    # A manifest with the flag on and no LIVE default gateway: an earlier apply died between flipping
    # the flag and the multiplexer confirming it is up (#110850). An installed unit is not proof of
    # anything — `systemd_install` writes the unit before the start that can still fail or be killed.
    # Not "already multiplexed" — resumable.
    interrupted: bool = False
    blockers: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    # Profiles that authored `gateway.standalone: true`: they keep their own gateway and are neither
    # a blocker nor a fold target — the plan names them so the operator knows they were left alone.
    standalone_by_config: tuple[str, ...] = ()

    @property
    def secondaries(self) -> list[ProfileGateway]:
        return [p for p in self.profiles if not p.is_default]

    @property
    def default(self) -> ProfileGateway:
        return next(p for p in self.profiles if p.is_default)

    @property
    def already_multiplexed(self) -> bool:
        """Converged: nothing is left for this command to do.

        TOPOLOGY, not the flag. A host whose flag is on while a secondary still owns a gateway
        process or a supervisor unit is HALF-migrated; reporting that as "already multiplexed"
        made the re-run a no-op on exactly the host that needed it most (#100896).
        """
        if self.interrupted or self.standalone_secondaries:
            return False
        return self.multiplex_flag_on or bool(self.live_served and len(self.live_served) > 1)

    @property
    def standalone_secondaries(self) -> list[ProfileGateway]:
        return [p for p in self.secondaries if p.has_gateway]

    @property
    def expected_served_names(self) -> set[str]:
        from hermes_cli.profiles import profile_is_parked
        return {p.name for p in self.profiles if p.is_default or not profile_is_parked(p.home)}

    @property
    def blocked(self) -> bool:
        return bool(self.blockers)

    def target_service_kind(self) -> Optional[tuple[str, bool]]:
        """Service manager the default gateway should end up on: its own, else the one the
        secondaries used (so a systemd-managed fleet stays systemd-managed)."""
        if self.default.service is not None:
            return self.default.service
        return next((p.service for p in self.secondaries if p.service is not None), None)

    def target_run_as_user(self) -> Optional[str]:
        """Preserve the system unit identity that the default unit replaces."""
        if self.default.run_as_user:
            return self.default.run_as_user
        return next((p.run_as_user for p in self.secondaries if p.run_as_user), None)

    def to_dict(self) -> dict:
        return {
            "default_home": str(self.default_home),
            "profiles": [p.to_dict() for p in self.profiles],
            "standalone_by_config": list(self.standalone_by_config),
            "multiplex_flag_on": self.multiplex_flag_on,
            "live_served": self.live_served,
            "already_multiplexed": self.already_multiplexed,
            "interrupted": self.interrupted,
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
    return list(profiles_to_serve(multiplex=True, include_parked=True))


def _live_gateway_pid(home: Path) -> Optional[int]:
    """Verified PID of a gateway SERVING ``home``, else None (never raises: a probe failure must
    not abort a migration plan).

    Topology REPORTING, not ownership: ``live_gateway_pid_for_home`` deliberately answers with the
    host multiplexer's PID for every home it serves (``gateway.status._host_gateway_serves_home``),
    so "this profile is being served" reads as a live PID. Use :func:`_own_gateway_pid` for the
    question this command acts on.
    """
    from gateway.status import live_gateway_pid_for_home
    with contextlib.suppress(Exception):
        return live_gateway_pid_for_home(home)
    return None


def _host_gateway_owner():
    """The ONE live host gateway (``gateway.host_attach.HostGateway``) or None. Never raises."""
    with contextlib.suppress(Exception):
        from gateway.host_attach import host_gateway
        return host_gateway()
    return None


def _own_gateway_pid(home: Path, owner) -> Optional[int]:
    """PID of a gateway ``home`` OWNS, else None.

    Ownership is what this command may stop, and the host multiplexer is not owned by the profiles
    it serves: it is one process, launched from one home. Deriving ownership from
    :func:`_live_gateway_pid` made every secondary on a CONVERGED host report the host gateway's
    PID, so the plan called the host half-migrated and each `hermes update` SIGTERMed the only
    gateway it had. The host process counts only for the home it was actually launched from; every
    other profile it serves owns nothing.
    """
    pid = _live_gateway_pid(home)
    if pid is None or owner is None or pid != owner.pid:
        return pid
    with contextlib.suppress(Exception):
        from gateway.status import _same_hermes_home
        return pid if _same_hermes_home(Path(owner.home), Path(home)) else None
    return None


def _gateway_identity(home: Path, pid: Optional[int], services: list[tuple[str, bool]]) -> tuple[Optional[int], Path]:
    from hermes_cli.gateway_migrate_guards import gateway_identity
    return gateway_identity(home, pid, services)


def _installed_services(home: Path) -> list[tuple[str, bool]]:
    """Every installed service for ``home``'s gateway (units / plist / scheduled task), user scope first."""
    from hermes_cli import gateway as gw
    found: list[tuple[str, bool]] = []
    with _home_env(home):
        if gw.supports_systemd_services():
            found.extend(("systemd", system) for system in (False, True) if gw.get_systemd_unit_path(system=system).exists())
        if gw.is_macos() and gw.get_launchd_plist_path().exists():
            found.append(("launchd", False))
        if gw.is_windows() and _windows_task_installed():
            found.append(("windows", False))
    return found


def _windows_task_installed() -> bool:
    """Is a per-profile Windows gateway installed for the ACTIVE ``HERMES_HOME``?

    ``get_task_name()`` is home-suffixed, so this answers per profile exactly the way the systemd
    unit path does. Either half counts: ``hermes gateway install`` falls back to a Startup-folder
    entry when it cannot register a scheduled task, and a migration that removed only the task
    would leave the fallback launching a second gateway at the next logon.
    """
    from hermes_cli import gateway_windows as gww
    with contextlib.suppress(Exception):
        return bool(gww.is_task_registered() or gww.is_startup_entry_installed())
    return False


def _systemd_service_user(home: Path, services: list[tuple[str, bool]]) -> Optional[str]:
    """Read ``User=`` before migration removes a system-scope unit."""
    if ("systemd", True) not in services:
        return None
    from hermes_cli import gateway as gw
    with _home_env(home):
        return gw._read_systemd_user_from_unit(gw.get_systemd_unit_path(system=True))


def _service_op(kind: str, system: bool, verb: str, home: Path, *, run_as_user: Optional[str] = None) -> None:
    """``stop`` / ``uninstall`` / ``start`` / ``restart`` / ``install`` on ``home``'s service."""
    from hermes_cli import gateway as gw
    with _home_env(home):
        if verb == "install":
            if kind == "launchd":
                gw.launchd_install()
            elif kind == "windows":
                # Non-interactive: the migration already asked; prompting here would hang a
                # supervised/`--yes` run on a console that has no operator.
                from hermes_cli import gateway_windows as gww
                gww.install(start_now=True, start_on_login=True)
            else:
                gw.systemd_install(system=system, run_as_user=run_as_user, non_interactive=True)
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
    """The operator's EXPLICIT opt-in only. The unset default (on) is settled by the default gateway at
    boot and refused while a secondary runs its own gateway — exactly the fleet this command folds —
    so the plan reads it as "not yet multiplexed" and the migration proceeds."""
    from hermes_cli.gateway_multiplex_mode import explicit_multiplex_flag
    return explicit_multiplex_flag(default_home) is True


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


def _credential_key_names(platform_value: str) -> str:
    """Env key NAMES (never values) that make ``platform_value`` connect as a bot, e.g.
    ``TELEGRAM_BOT_TOKEN``; the platform id when no key is registered (config.yaml-only token)."""
    from hermes_cli.profile_channels import credential_env_keys
    names = sorted(key for key, pid in credential_env_keys().items() if pid == platform_value)
    return "/".join(names) or f"the {platform_value} token"


def duplicate_credential_lines(configs: list[tuple[str, object]]) -> list[str]:
    """One finding per platform credential two profiles both hold, with the remedy. The SINGLE
    source for the migrate preflight, ``hermes doctor`` and ``hermes gateway status``, so all three
    name the same duplicates the same way: profile names + key names only, never a value or hash."""
    owners: dict[tuple, str] = {}
    lines: list[str] = []
    for name, cfg in configs:  # default first: it wins the claim, like at multiplexer startup
        for claim, platform_value in _credential_claims(cfg).items():
            owner = owners.setdefault(claim, name)
            if owner == name:
                continue
            key = _credential_key_names(platform_value)
            lines.append(
                f"Profiles '{owner}' and '{name}' both hold the same {platform_value} credential ({key}): "
                f"one platform token can serve only one gateway, so the bot answers from whichever profile "
                f"claims it first and the other's adapter is parked. Give '{name}' its own bot token, or remove "
                f"{key} from the profile that should not own it (or keep it in {owner} and route {name}'s "
                f"chats with profile_routes — gateway.profile_routes in {owner}'s config.yaml), then run "
                f"{MIGRATE_COMMAND}."
            )
    return lines


def duplicate_credential_findings() -> list[str]:
    """The preflight's duplicate-credential check read straight from the local profile homes, for
    diagnostics that have no migration plan (doctor, gateway status). A profile whose gateway config
    does not load is skipped here — ``build_migration_plan`` reports that one as its own blocker."""
    configs: list[tuple[str, object]] = []
    with _multiplex_read_mode():
        for name, home in _profile_homes():
            with contextlib.suppress(Exception):
                configs.append((name, _profile_gateway_config(home)))
    return duplicate_credential_lines(configs)


def _check_duplicate_credentials(plan: MigrationPlan, configs: dict[str, object]) -> None:
    """BLOCKER: the same bot credential configured on two profiles — the multiplexer would park
    the duplicate adapter, so one profile's bot would go silent after migration."""
    plan.blockers.extend(duplicate_credential_lines(
        [(p.name, configs[p.name]) for p in plan.profiles if p.name in configs]))


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
                    f"add a /p/<profile>/ ingress for it first."
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
    """Enumerate profiles + their OWN gateway footprint, then run every preflight check."""
    from hermes_cli.gateway_multiplex_served import recorded_served_profiles
    default_home = _default_home()
    # Probed ONCE: every profile's ownership verdict is relative to the same host process.
    owner = _host_gateway_owner()
    profiles = []
    for name, home in _profile_homes():
        pid, services = _own_gateway_pid(home, owner), _installed_services(home)
        uid, runtime_home = _gateway_identity(home, pid, services)
        profiles.append(ProfileGateway(name=name, home=home, pid=pid, services=services,
                                       run_as_user=_systemd_service_user(home, services), uid=uid,
                                       runtime_home=None if runtime_home == home else runtime_home))
    plan = MigrationPlan(
        default_home=default_home, profiles=profiles,
        multiplex_flag_on=_read_multiplex_flag(default_home),
        live_served=recorded_served_profiles(default_home),
        manifest=_read_manifest(default_home),
    )
    from hermes_cli.profiles import profiles_to_serve
    foldable = {name for name, _home in _profile_homes()}
    plan.standalone_by_config = tuple(
        name for name, _home in profiles_to_serve(True, include_standalone=True)
        if name != "default" and name not in foldable)
    plan.interrupted = plan.multiplex_flag_on and _manifest_not_yet_served(plan.manifest, plan.live_served)
    if len(plan.profiles) < 2:
        plan.notices.append("Only one profile exists: nothing to multiplex.")
        return plan
    configs = _load_profile_configs(plan)
    for check in _PREFLIGHT_CHECKS:
        check(plan, configs)
    from hermes_cli.gateway_migrate_guards import auto_migration_blockers
    # Notices, not blockers: the explicit command is the operator's decision; only the update hook
    # refuses to cross these boundaries on its own.
    plan.notices.extend(f"Not migrated automatically by `hermes update`: {b}" for b in auto_migration_blockers(plan))
    plan.notices.append(
        "Profiles created after the migration are served by the running multiplexer as soon as "
        "they exist (it rescans profiles/ on create/delete and every 30s)."
    )
    return plan


# --------------------------------------------------------------------------- printing


def _print(lines: list[str]) -> None:
    for line in lines:
        print(line)


def _signalled_gateways(plan: MigrationPlan) -> list[str]:
    """Every RUNNING gateway process ``apply_migration`` will signal, in apply order.

    Derived from the steps the apply actually takes, not from "secondaries that happen to expose a
    PID". Two classes were silently missing and both killed live processes: a secondary with an
    installed unit and no readable PID is stopped through ``_service_op`` (which SIGTERMs whatever
    the supervisor is running), and the DEFAULT's own live gateway is replaced by
    :func:`_restart_default`. A dry run that promises nothing will be stopped and then stops the
    host's only gateway breaks the contract this block exists to keep.
    """
    def _who(p: ProfileGateway, suffix: str = "") -> str:
        where = f"pid {p.pid}" if p.pid else f"supervised by {p.service_label()}"
        return f"{p.name} ({where}{suffix})"

    signalled = [_who(p) for p in plan.standalone_secondaries if p.pid or p.services]
    default = plan.default
    if default.services or default.pid:
        signalled.append(_who(default, ", restarted onto the new flag"))
    return signalled


def format_plan(plan: MigrationPlan, *, dry_run: bool) -> list[str]:
    head = "Migration plan (dry run — nothing changed)" if dry_run else "Migration plan"
    lines = [head, f"  default home: {plan.default_home}", "", "  profile      gateway pid   service"]
    for p in plan.profiles:
        lines.append(f"  {p.name:<12} {str(p.pid or '-'):<13} {p.service_label()}")
    if plan.standalone_by_config:
        lines.append(f"  Standalone by config (gateway.standalone: true), left alone: "
                     f"{', '.join(plan.standalone_by_config)}")
        lines.append("    (temporary compatibility shim; remove the key and re-run once the gaps it "
                     "covers for you are fixed)")
    lines.append("")
    if plan.already_multiplexed:
        lines.append("  ✓ The default gateway is already multiplexing"
                     + (f" (serving {', '.join(plan.live_served)})" if plan.live_served else " (flag on)") + ".")
        return lines
    if plan.interrupted:
        lines.append(f"  ↻ An earlier migration was interrupted before the default gateway came up "
                     f"(flag on, no live multiplexer; manifest {plan.default_home / MANIFEST_NAME}); this run resumes it.")
    elif plan.multiplex_flag_on and plan.standalone_secondaries:
        lines.append("  ↻ Half-migrated host: the flag is on, but the profile(s) below still own a "
                     "gateway. This run converges them.")
    steps = []
    signalled = _signalled_gateways(plan)
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
    # A resume installs through the units the MANIFEST recorded (the live ones are already gone),
    # which is the mechanism ``apply_migration`` picks — printing this plan's guess contradicted it.
    target = _resume_target(plan)[0] if plan.manifest is not None else plan.target_service_kind()
    lines.append(f"  - default: {'restart' if plan.default.has_gateway else 'start'} the gateway"
                 + (f" via {target[0]}" if target else " (detached)") + f", verify it serves {len(plan.expected_served_names)} profiles")
    lines.append(f"  - record the previous state in {plan.default_home / MANIFEST_NAME} "
                 f"(used to undo a FAILED apply, and to resume this command after a crash)")
    if signalled:
        # Never stop a running gateway without saying so first, and say it in the imperative
        # tense the operator can still act on: this block prints BEFORE anything is signalled.
        lines += ["",
                  "  ⚠ This SIGTERMs running gateway process(es): " + ", ".join(signalled) + ".",
                  "    They drain in-flight turns and exit; their profiles are served by the host "
                  "gateway afterwards."]
    return lines + _plan_tail(plan)


def _plan_tail(plan: MigrationPlan) -> list[str]:
    lines: list[str] = []
    if plan.blockers:
        if plan.manifest is None:
            lines += ["", "  ✗ Blockers (fix these first, nothing will be changed):"]
        else:
            # This host is already mid-migration with units removed, so a blocker cannot be a
            # refusal any more: refusing would leave it with nobody serving and no way forward.
            lines += ["",
                      f"  ⚠ Blockers found, but a migration is already in progress on this host "
                      f"(manifest {_manifest_path(plan.default_home)}).",
                      "    This run RESUMES it and converges anyway — refusing would strand the "
                      "profiles whose",
                      "    gateways an earlier attempt already removed. Fix these afterwards:"]
        lines += [f"    • {b}" for b in plan.blockers]
    if plan.notices:
        lines += ["", "  Notices:"]
        lines += [f"    • {n}" for n in plan.notices]
    return lines


def _manifest_secondaries(manifest: dict) -> Optional[list[dict]]:
    """The manifest's secondary records, or None when the (hand-edited) manifest is malformed."""
    recs = manifest.get("secondaries", [])
    if not isinstance(recs, list) or not all(isinstance(r, dict) and r.get("profile") and r.get("home") for r in recs):
        return None
    return recs


def _service_from_dict(service) -> Optional[tuple[str, bool]]:
    if isinstance(service, dict) and service.get("kind"):
        return str(service["kind"]), bool(service.get("system"))
    return None


def _recorded_services(rec: dict) -> list[tuple[str, bool]]:
    """Every service a manifest record names: ``services`` (all installed units) or, in a manifest
    written before that key existed, the single ``service``."""
    recorded = rec.get("services")
    if isinstance(recorded, list):
        return [s for s in (_service_from_dict(r) for r in recorded) if s is not None]
    service = _service_from_dict(rec.get("service"))
    return [service] if service is not None else []


def _recorded_run_as_user(rec: dict) -> Optional[str]:
    user = rec.get("run_as_user")
    return user if isinstance(user, str) and user else None


def _target_from_manifest(manifest: dict) -> tuple[Optional[tuple[str, bool]], Optional[str]]:
    """Service manager + ``User=`` the resumed default install should use, read from the recorded
    footprint (the units themselves are gone by the time an interrupted apply is re-run)."""
    recs = [r for r in (manifest.get("default"), *(_manifest_secondaries(manifest) or [])) if isinstance(r, dict)]
    target = next((s for r in recs for s in _recorded_services(r)), None)
    user = next((u for r in recs if (u := _recorded_run_as_user(r)) is not None), None)
    return target, user


_NOTHING_RECORDED = "no gateway was recorded; nothing to restore"


def format_update_warning(plan: MigrationPlan, auto_blockers: list[str]) -> list[str]:
    return [
        "⚠ Your profiles each run their own gateway. A single multiplexed gateway is the recommended",
        "  setup, but this install cannot be migrated automatically yet:",
        *[f"    • {b}" for b in (*plan.blockers, *auto_blockers)],
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


def _manifest_not_yet_served(manifest: Optional[dict], live_served: Optional[list[str]]) -> bool:
    """The postcondition ``apply_migration`` waits for, re-derived from live state: a LIVE default that
    recorded serving every unparked profile the manifest migrated. Anything less — no live gateway, an
    installed-but-dead unit (``systemd_install`` writes the unit before the start that can still fail),
    a standalone default never restarted — is a half-applied migration, not "already multiplexed".
    Profiles created after the migration are not in the manifest, so they cannot flag it as interrupted."""
    if manifest is None:
        return False
    from hermes_cli.profiles import profile_is_parked
    recs = [r for r in (manifest.get("default"), *(_manifest_secondaries(manifest) or [])) if isinstance(r, dict)]
    migrated = {str(r.get("profile") or "default") for r in recs
                if not r.get("home") or not profile_is_parked(Path(r["home"]))} | {"default"}
    return not migrated <= set(live_served or [])


def _write_manifest(default_home: Path, data: dict) -> None:
    from utils import atomic_json_write

    atomic_json_write(_manifest_path(default_home), data)


def _reconcile_standalone_runtime(default_home: Path, secondary_names: set[str]) -> None:
    """Remove only status owned by multiplexing, without re-stamping gateway identity."""
    from gateway.status import read_runtime_status
    from utils import atomic_json_write

    path = default_home / "gateway_state.json"
    runtime = read_runtime_status(path)
    if runtime is None:
        return
    runtime["served_profiles"] = []
    platforms = runtime.get("platforms")
    if isinstance(platforms, dict):
        prefixes = tuple(f"{name}:" for name in secondary_names if name)
        runtime["platforms"] = {
            key: value for key, value in platforms.items()
            if not (isinstance(key, str) and key.startswith(prefixes))
        }
    atomic_json_write(path, runtime, indent=None, separators=(",", ":"))


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


def _restart_default(
    plan_default: ProfileGateway,
    target: Optional[tuple[str, bool]],
    default_home: Path,
    *,
    run_as_user: Optional[str] = None,
) -> str:
    """Bring the default gateway up on the new flag value; returns a one-line description."""
    if plan_default.service is not None:
        kind, system = plan_default.service
        _service_op(kind, system, "restart", default_home)
        return f"restarted the default gateway via {kind}"
    if target is not None:
        kind, system = target
        _service_op(kind, system, "install", default_home, run_as_user=run_as_user)
        _service_op(kind, system, "start", default_home)
        return f"installed and started the default gateway via {kind}"
    verb = "restarted" if plan_default.pid is not None else "started"
    if plan_default.pid is not None:
        _stop_gateway_process(default_home)
    if not _spawn_detached_gateway(default_home):
        raise RuntimeError("could not spawn the default gateway (detached)")
    return f"{verb} the default gateway (detached; no service manager was in use)"


def _remove_secondary_gateways(plan: MigrationPlan) -> None:
    for p in plan.standalone_secondaries:
        for kind, system in p.services:
            _service_op(kind, system, "stop", p.home)
            _service_op(kind, system, "uninstall", p.home)
            print(f"  ✓ {p.name}: stopped and removed its {_service_label((kind, system))} service")
        if p.pid is not None:
            _stop_gateway_process(p.home)
            print(f"  ✓ {p.name}: stopped standalone gateway (pid {p.pid})")


def _preflight_apply(plan: MigrationPlan, target: Optional[tuple[str, bool]], run_as_user: Optional[str]) -> Optional[str]:
    """A failure of the destructive phase that is knowable from the plan alone, refused BEFORE any
    working per-profile gateway is stopped: rollback is the fallback for surprises, not the plan.
    Mirrors the checks ``systemd_install``/``_service_call`` make on a system unit (root, resolvable
    ``User=``) and the config write's read-guard."""
    from hermes_cli import gateway as gw
    from hermes_cli.config import require_readable_config_before_write
    try:
        require_readable_config_before_write(plan.default_home / "config.yaml")
    except Exception as exc:
        return f"default: config.yaml cannot be updated ({exc})"
    touches_system_unit = target == ("systemd", True) or any(p.has_system_unit for p in plan.standalone_secondaries)
    if touches_system_unit:
        try:
            gw._require_root_for_system_service("migration")
        except Exception as exc:
            return str(exc)
    if plan.default.service is None and target == ("systemd", True):
        if run_as_user is None:
            try:
                gw._system_service_identity()  # the #110850 refusal (implicit root), before anything is removed
            except ValueError as exc:
                return f"default: {exc}"
        else:
            import pwd
            try:
                pwd.getpwnam(run_as_user)
            except KeyError:
                return f"default: the recorded service user '{run_as_user}' does not exist on this host"
    return None


def _resume_target(plan: MigrationPlan) -> tuple[Optional[tuple[str, bool]], Optional[str]]:
    """Service manager + ``User=`` an apply will use: the manifest's record wins on a resume (the
    live units it describes are already gone), else what the plan can still see."""
    target, run_as_user = plan.target_service_kind(), plan.target_run_as_user()
    if plan.manifest is None:
        return target, run_as_user
    m_target, m_user = _target_from_manifest(plan.manifest)
    return m_target or target, m_user or run_as_user


def _resume_findings(plan: MigrationPlan, target, run_as_user) -> list[str]:
    """Safety checks re-run on a RESUME and reported as findings instead of refusals.

    Both halves matter. Skipping them (the resume branch used to return before ``_preflight_apply``
    ran at all) let a duplicate bot credential, a ``/p/`` ingress gap or an unresolvable service
    user reach the multiplexer unannounced. Enforcing them as blockers is worse: a resume runs on
    a host whose secondary units are ALREADY removed, so refusing does not prevent damage — it
    makes the damage permanent, which is exactly how a stranded host lost its only documented
    recovery. So: run every check, name every finding, converge anyway.
    """
    findings = [*plan.blockers]
    apply_blocker = _preflight_apply(plan, target, run_as_user)
    if apply_blocker is not None:
        findings.append(apply_blocker)
    if not findings:
        return []
    return ["  ⚠ Preflight findings — resuming anyway (this host is mid-migration; refusing would",
            "    leave its profiles with no gateway at all). Fix these once it is converged:",
            *[f"    • {f}" for f in findings]]


def apply_migration(plan: MigrationPlan, *, served_wait: float = _SERVED_WAIT_SECONDS) -> bool:
    """Flip the flag, stop/uninstall every secondary gateway, bring up the multiplexer, verify.
    Returns True when the multiplexer verifiably serves every profile.

    Every step after the manifest write is fallible (a config write, a secondary's stop or its
    unit's daemon-reload, a system unit that needs ``--run-as-user``, an unreachable user bus) and
    runs inside ONE compensating boundary: on failure the manifest written before the first
    destructive step brings ONE gateway back (:func:`rollback_migration`), so the host never ends
    with nobody serving. The flag goes on first so an apply killed anywhere after it is resumable
    from the manifest (flag on + manifest + no live multiplexer = interrupted).

    **Resume beats preflight.** The manifest is read BEFORE the blocker gate: a manifest means an
    earlier apply already removed units, so this run is a compensation, not an initiation, and a
    gate that refuses it traps the host forever (no ``--standalone``, no rollback command). The
    checks still run — as findings, see :func:`_resume_findings`. With no manifest nothing has been
    destroyed yet and both gates refuse exactly as before.
    """
    if plan.already_multiplexed:
        print("✓ Already multiplexed — nothing to do.")
        return True
    target, run_as_user = _resume_target(plan)
    manifest = plan.manifest
    if manifest is not None:
        # A manifest on disk means an earlier apply got past its first destructive step: it either
        # died mid-way (``interrupted``) or its compensator did not finish. Either way the manifest
        # is the ONLY record of the units that existed, so resume from it and never overwrite it.
        # Re-running this command IS the recovery.
        print(f"  ↻ resuming the migration recorded in {_manifest_path(plan.default_home)}")
        _print(_resume_findings(plan, target, run_as_user))
    else:
        if plan.blocked:
            _print(["✗ Migration refused:", *[f"  • {b}" for b in plan.blockers]])
            return False
        blocker = _preflight_apply(plan, target, run_as_user)
        if blocker is not None:
            _print(["✗ Migration refused before changing anything:", f"  • {blocker}"])
            return False
        manifest = {
            "version": 1, "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "flag_was": plan.multiplex_flag_on,
            "default": plan.default.to_dict(),
            "secondaries": [p.to_dict() for p in plan.standalone_secondaries],
        }
        # Recovery metadata must exist before the first destructive operation; the manifest never
        # changes afterwards, so this is the only write it needs.
        _write_manifest(plan.default_home, manifest)
    try:
        _write_multiplex_flag(plan.default_home, True)
        print(f"  ✓ default: gateway.multiplex_profiles: true ({plan.default_home / 'config.yaml'})")
        _remove_secondary_gateways(plan)  # on resume: whatever an apply killed mid-removal left installed
        print(f"  ✓ {_restart_default(plan.default, target, plan.default_home, run_as_user=run_as_user)}")
    except Exception as exc:
        _print([f"  ✗ migration failed ({exc})",
                "  ↩ Restoring one host gateway so this host is not left without one..."])
        rolled_back = rollback_migration(plan.default_home)
        if not rolled_back:
            print(f"  Re-run {MIGRATE_COMMAND} to resume from the manifest.")
        return False

    expected = plan.expected_served_names
    served = _wait_for_served(plan.default_home, expected, served_wait)
    if served is not None and expected <= set(served):
        # Manifest present == migration UNFINISHED. That is the whole resume/half-migrated signal
        # (`MigrationPlan.interrupted`), so a CONFIRMED convergence must clear it -- otherwise the
        # host reports itself interrupted forever and never says "already multiplexing".
        _manifest_path(plan.default_home).unlink(missing_ok=True)
        _print(["", f"✓ Migrated: the default gateway now serves {len(served)} profiles: {', '.join(served)}",
                *[f"  • {n}" for n in plan.notices]])
        return True
    missing = sorted(expected - set(served or []))
    _print(["", f"⚠ Migration applied, but the default gateway has not confirmed serving: {', '.join(missing)}",
            "  Check `hermes gateway status` and the gateway log; the flag and manifest are in place.",
            f"  Re-run {MIGRATE_COMMAND} once it is healthy — it resumes from the manifest."])
    return False


_COMPENSATOR_WAIT_SECONDS = 30.0


def _wait_for_live_gateway(home: Path, timeout: float) -> Optional[int]:
    """Bounded wait for a gateway that is actually UP for ``home``; None on timeout.

    A service-manager ``start`` that returns is not proof: systemd returns after its own start
    timeout on a unit that keeps failing, and taking that for success is how the compensator
    printed "✓ Restored" over a unit respawning every 5 s at ``ExecMainStatus=75``.
    """
    deadline = time.monotonic() + timeout
    while True:
        pid = _live_gateway_pid(home)
        if pid is not None:
            return pid
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.5)


def rollback_migration(default_home: Optional[Path] = None) -> bool:
    """COMPENSATOR for a failed apply: put ONE gateway back on this host.

    Not a user-facing rollback and deliberately NOT a fleet restore. It used to reinstall and
    start every recorded per-profile gateway, which the host gateway lock now forbids by
    construction: this function clears the multiplex-owned served record first, so
    ``gateway.host_attach`` tells each secondary to "start normally" and the flock loser exits 75
    into a supervisor respawn loop (observed live: a secondary won the race while the default's
    unit sat at ``NRestarts=38, ExecMainStatus=75``, respawning every 5 s — ``StartLimitIntervalSec=0``
    defeats the limiter). Under multiplex-only a restored fleet is not a state we want to be able
    to reach, so the compensation is the one gateway the topology allows: the DEFAULT's, on the
    recorded flag. The secondaries whose units this attempt already removed are named in the
    output and are served by that gateway as soon as it multiplexes (``flag_was`` is either unset
    or the retired ``false``; both resolve to multiplex at boot once nothing blocks it).

    Returns True only when a gateway is VERIFIABLY live again; the manifest is kept otherwise so
    the next ``hermes gateway migrate --multiplex`` resumes.
    """
    default_home = default_home or _default_home()
    manifest = _read_manifest(default_home)
    if manifest is None:
        print(f"✗ No migration manifest at {_manifest_path(default_home)}; nothing to restore.")
        return False
    incomplete = (f"⚠ Compensation incomplete; manifest kept at {_manifest_path(default_home)}.\n"
                  f"  Re-run {MIGRATE_COMMAND} — it resumes from the manifest.")
    secondaries = _manifest_secondaries(manifest)
    if secondaries is None:
        # Refuse before touching anything: a hand-edited manifest is not a rollback authority.
        print(f"✗ Malformed secondary records in {_manifest_path(default_home)}; fix or delete the manifest.")
        return False
    try:
        _write_multiplex_flag(default_home, bool(manifest.get("flag_was", False)))
        print("  ✓ default: gateway.multiplex_profiles restored")
    except Exception as exc:
        print(f"  ✗ default: could not restore gateway.multiplex_profiles ({exc})")
        print(incomplete)
        return False

    default_rec = manifest.get("default")
    default_gw = ProfileGateway(
        "default", default_home, pid=_own_gateway_pid(default_home, _host_gateway_owner()),
        services=(_recorded_services(default_rec) if isinstance(default_rec, dict) else []) or _installed_services(default_home),
    )
    names = [str(rec["profile"]) for rec in secondaries]
    # The live multiplexer's record still claims every secondary, and every lifecycle verb reads it;
    # clear it before the default comes back up on the restored flag.
    try:
        _reconcile_standalone_runtime(default_home, set(names))
        print("  ✓ default: cleared multiplex-owned runtime status")
    except Exception as exc:
        print(f"  ✗ default: could not clear multiplex-owned runtime status ({exc})")
        print(incomplete)
        return False

    if not default_gw.has_gateway and not any(rec.get("pid") or _recorded_services(rec) for rec in secondaries):
        _manifest_path(default_home).unlink(missing_ok=True)
        print(f"✓ Nothing was running before this attempt ({_NOTHING_RECORDED}); the flag is restored.")
        return True

    target, run_as_user = _target_from_manifest(manifest)
    try:
        print(f"  ✓ {_restart_default(default_gw, target, default_home, run_as_user=run_as_user)}")
    except Exception as exc:
        # The recorded service manager is exactly what the failed apply could not drive (a refused
        # system-unit install, a read-only unit dir). Falling back to a detached gateway still
        # honours one-gateway-per-host, and a host with a running gateway beats a correct unit.
        print(f"  ✗ default: could not bring the host gateway back up via the recorded service "
              f"manager ({exc}); falling back to a detached gateway")
        try:
            spawned = _spawn_detached_gateway(default_home)
        except Exception as spawn_exc:
            print(f"  ✗ default: the detached fallback also failed ({spawn_exc})")
            spawned = False
        if not spawned:
            print(incomplete)
            return False
        print("  ✓ default: started the host gateway detached (no service manager)")
    live = _wait_for_live_gateway(default_home, _COMPENSATOR_WAIT_SECONDS)
    if live is None:
        print(f"  ✗ default: no gateway confirmed serving this host within {_COMPENSATOR_WAIT_SECONDS:.0f}s "
              f"(check `hermes gateway status` and the gateway log)")
        print(incomplete)
        return False
    _manifest_path(default_home).unlink(missing_ok=True)
    print(f"✓ Compensated: one host gateway is running again (pid {live}) on the recorded flag.")
    if names:
        print(f"  Per-profile gateways this attempt had already removed: {', '.join(names)}. They are "
              f"NOT reinstalled — one gateway per host is the only supported topology — and the host "
              f"gateway serves them as soon as it multiplexes.\n"
              f"  Run {MIGRATE_COMMAND} to finish converging.")
    return True


# --------------------------------------------------------------------------- CLI + update hook


def _host_supports_migration() -> Optional[str]:
    """Reason the host cannot be converged by this command (s6 slots), else None.

    Windows IS handled now: per-profile Scheduled Tasks (and the Startup-folder fallback) are
    detected and removed like any other unit. s6 is not, and cannot be from here: the per-profile
    gateways are slots the container image registers at boot
    (``hermes_cli/container_boot.py::reconcile_profile_gateways``), so the convergence belongs to
    the container's own boot, not to a process inside it.
    """
    from hermes_cli import gateway as gw
    if gw._running_under_s6():
        return ("s6-supervised container: per-profile gateways are s6 slots registered by the "
                "container's boot, not by this process. Restart the container so its boot "
                "reconciles them; nothing on this host was changed.")
    return None


def cmd_migrate(args) -> None:
    """``hermes gateway migrate [--multiplex] [--dry-run] [--yes]``."""
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
    # A manifest means this host is already mid-migration with units removed: the blocked/too-few-
    # profiles gates describe a migration that has not started yet, and applying them here is what
    # left a stranded host refusing its own documented recovery forever.
    if plan.manifest is None and (plan.blocked or len(plan.profiles) < 2):
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
    migrate automatically when unblocked (deterministic, never prompts) or print the blocker block.
    ``gateway.auto_multiplex_migration: false`` on the default profile opts out; a secondary behind a
    service-domain / UNIX-user / HERMES_HOME boundary blocks this path only (the explicit command decides)."""
    from hermes_cli.gateway_migrate_guards import auto_migration_blockers, auto_migration_opted_out
    if _host_supports_migration() is not None or auto_migration_opted_out(_default_home()):
        return
    plan = build_migration_plan()
    if plan.already_multiplexed or len(plan.profiles) < 2:
        return
    # A stranded host (manifest on disk, units already removed, nobody serving) has no standalone
    # secondaries left to find, so the "nothing to fold" exit skipped the one host that needs this
    # most. A manifest makes the update hook a RESUME, and a resume is not gated on blockers.
    resuming = plan.manifest is not None
    if not plan.standalone_secondaries and not resuming:
        return
    print()
    auto_blockers = auto_migration_blockers(plan)
    if not resuming and (plan.blocked or auto_blockers):
        _print(format_update_warning(plan, auto_blockers))
        return
    print("→ Migrating per-profile gateways onto one multiplexed default gateway...")
    _print(format_plan(plan, dry_run=False))
    apply_migration(plan)
