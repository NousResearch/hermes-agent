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
what the multiplexer would do at startup. CLI/update orchestration lives in ``nous_cli.gateway_migrate``.
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


@dataclass(frozen=True)
class MigrationResult:
    """Structured migration outcome; CLI surfaces decide how to render its lines."""

    succeeded: bool
    lines: tuple[str, ...] = ()



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
    if kind == "s6":
        return "s6 slot"
    return "Windows scheduled task" if kind == "windows" else kind


def _remove_verb(service: tuple[str, bool]) -> str:
    """An s6 slot is parked down (it stays registered as the `hermes -p X gateway start` target), every
    other unit is uninstalled."""
    return "park" if service[0] == "s6" else "uninstall"


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
        from gateway.profile_serving import profile_is_parked
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
        zero standalone secondaries when the explicit CLI asks for migration."""
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


def default_home() -> Path:
    return _default_home()


def _profile_homes() -> list[tuple[str, Path]]:
    from gateway.profile_serving import profiles_to_serve
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
    from gateway.migration_guards import gateway_identity
    return gateway_identity(home, pid, services)


def _installed_services(home: Path) -> list[tuple[str, bool]]:
    """Every installed service for ``home``'s gateway (units / plist / scheduled task), user scope first.

    Under s6 the footprint is the SLOT: the root slot always (it is what a restart goes through),
    a named profile's slot only while it is UP — a registered-down slot is what the container's
    boot leaves behind for every named profile and is not a gateway."""
    from gateway.service_manager import detect_service_manager
    from gateway.systemd_identity import unit_path
    from gateway.systemd_runtime import supports_services

    found: list[tuple[str, bool]] = []
    if detect_service_manager() == "s6":
        from gateway.s6_service import named_slot_name, slot_is_up
        from gateway.s6_manager import S6ServiceManager
        from hermes_constants import profile_name_for_home
        name = profile_name_for_home(home) or "default"
        slot_dir = S6ServiceManager().scandir / named_slot_name(name)
        if slot_dir.is_dir() and (name == "default" or slot_is_up(name)):
            found.append(("s6", False))
        return found
    with _home_env(home):
        if supports_services():
            found.extend(("systemd", system) for system in (False, True) if unit_path(system=system).exists())
        if sys.platform == "darwin":
            from gateway.launchd_service import get_launchd_plist_path
            if get_launchd_plist_path().exists():
                found.append(("launchd", False))
        if sys.platform == "win32" and _windows_task_installed():
            found.append(("windows", False))
    return found


def _windows_task_installed() -> bool:
    """Is a per-profile Windows gateway installed for the ACTIVE ``HERMES_HOME``?

    ``get_task_name()`` is home-suffixed, so this answers per profile exactly the way the systemd
    unit path does. Either half counts: ``hermes gateway install`` falls back to a Startup-folder
    entry when it cannot register a scheduled task, and a migration that removed only the task
    would leave the fallback launching a second gateway at the next logon.
    """
    from gateway import windows_service as gww
    with contextlib.suppress(Exception):
        return bool(gww.is_task_registered() or gww.is_startup_entry_installed())
    return False


def _systemd_service_user(home: Path, services: list[tuple[str, bool]]) -> Optional[str]:
    """Read ``User=`` before migration removes a system-scope unit."""
    if ("systemd", True) not in services:
        return None
    from gateway.systemd_identity import read_unit_user, unit_path
    with _home_env(home):
        return read_unit_user(unit_path(system=True))


def _service_op(kind: str, system: bool, verb: str, home: Path, *, run_as_user: Optional[str] = None) -> None:
    """Run one service lifecycle verb through the Gateway-owned platform backend."""
    if kind == "s6":
        return _s6_slot_op(verb, home)
    with _home_env(home):
        if kind == "systemd":
            from gateway import systemd_lifecycle
            if verb == "install":
                systemd_lifecycle.install(
                    system=system,
                    run_as_user=run_as_user,
                    remove_legacy=True,
                )
            else:
                systemd_lifecycle.service_call(verb, system)
            return
        if kind == "launchd":
            from gateway import launchd_service
            if verb == "install":
                launchd_service.launchd_install()
            else:
                getattr(launchd_service, f"launchd_{verb}")()
            return
        if kind == "windows":
            from gateway import windows_service
            if verb == "install":
                # Non-interactive: migration already owns the decision.
                windows_service.install(start_now=True, start_on_login=True)
            else:
                getattr(windows_service, verb)()
            return
        raise ValueError(f"unsupported gateway service backend: {kind}")


def _stop_gateway_process(home: Path) -> None:
    from gateway.process_lifecycle import stop_gateway_process
    stop_gateway_process(home)

def _s6_slot_op(verb: str, home: Path) -> None:
    """The s6 leg of :func:`_service_op`. A named profile's slot is never uninstalled: it stays
    registered DOWN (``down`` file) as the target of a later ``hermes -p X gateway start``, exactly
    the shape the container's boot produces. The root slot is (re)started so it re-reads its config."""
    from gateway.s6_service import bring_root_slot_up, park_named_slot
    from hermes_constants import profile_name_for_home
    name = profile_name_for_home(home) or "default"
    if name == "default":
        if verb in ("start", "restart"):
            bring_root_slot_up()
        return  # install/uninstall/stop of the root slot are not migration steps
    if verb == "stop":
        park_named_slot(name)


def _spawn_detached_gateway(home: Path) -> bool:
    from gateway.launchd_service import _spawn_detached_gateway as spawn_detached_gateway
    with _home_env(home):
        return spawn_detached_gateway()

def _read_multiplex_flag(default_home: Path) -> bool:
    """The operator's EXPLICIT opt-in only. The unset default (on) is settled by the default gateway at
    boot and refused while a secondary runs its own gateway — exactly the fleet this command folds —
    so the plan reads it as "not yet multiplexed" and the migration proceeds."""
    from gateway.multiplex_mode import explicit_multiplex_flag
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
    from gateway.served_profiles import recorded_served_profiles
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
    from gateway.profile_serving import profiles_to_serve
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
    from gateway.migration_guards import auto_migration_blockers
    # Notices, not blockers: the explicit command is the operator's decision; only the update hook
    # refuses to cross these boundaries on its own.
    plan.notices.extend(f"Not migrated automatically by `hermes update`: {b}" for b in auto_migration_blockers(plan))
    plan.notices.append(
        "Profiles created after the migration are served by the running multiplexer as soon as "
        "they exist (it rescans profiles/ on create/delete and every 30s)."
    )
    return plan


# --------------------------------------------------------------------------- manifest helpers


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
    from gateway.profile_serving import profile_is_parked
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
    from gateway.served_profiles import recorded_served_profiles
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


def _remove_secondary_gateways(plan: MigrationPlan) -> list[str]:
    lines: list[str] = []
    for p in plan.standalone_secondaries:
        for kind, system in p.services:
            _service_op(kind, system, "stop", p.home)
            _service_op(kind, system, "uninstall", p.home)
            done = "parked (down file)" if kind == "s6" else "removed"
            lines.append(
                f"  ✓ {p.name}: stopped and {done} its {_service_label((kind, system))}"
            )
        if p.pid is not None:
            _stop_gateway_process(p.home)
            lines.append(f"  ✓ {p.name}: stopped standalone gateway (pid {p.pid})")
    return lines

def _preflight_apply(plan: MigrationPlan, target: Optional[tuple[str, bool]], run_as_user: Optional[str]) -> Optional[str]:
    """A failure of the destructive phase that is knowable from the plan alone, refused BEFORE any
    working per-profile gateway is stopped: rollback is the fallback for surprises, not the plan.
    Mirrors the checks ``systemd_install``/``_service_call`` make on a system unit (root, resolvable
    ``User=``) and the config write's read-guard."""
    from gateway.systemd_identity import require_root, system_service_identity
    from hermes_cli.config import require_readable_config_before_write
    try:
        require_readable_config_before_write(plan.default_home / "config.yaml")
    except Exception as exc:
        return f"default: config.yaml cannot be updated ({exc})"
    touches_system_unit = target == ("systemd", True) or any(p.has_system_unit for p in plan.standalone_secondaries)
    if touches_system_unit:
        try:
            require_root("migration")
        except Exception as exc:
            return str(exc)
    if plan.default.service is None and target == ("systemd", True):
        if run_as_user is None:
            try:
                system_service_identity()  # the #110850 refusal (implicit root), before anything is removed
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


def planned_target_service(plan: MigrationPlan) -> Optional[tuple[str, bool]]:
    """Service manager the current migration plan will use for the default gateway."""
    return _resume_target(plan)[0]


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


def apply_migration_result(
    plan: MigrationPlan,
    *,
    served_wait: float = _SERVED_WAIT_SECONDS,
) -> MigrationResult:
    """Apply a migration and return structured output instead of printing."""
    lines: list[str] = []
    if plan.already_multiplexed:
        return MigrationResult(True, ("✓ Already multiplexed — nothing to do.",))

    target, run_as_user = _resume_target(plan)
    manifest = plan.manifest
    if manifest is not None:
        lines.append(f"  ↻ resuming the migration recorded in {_manifest_path(plan.default_home)}")
        lines.extend(_resume_findings(plan, target, run_as_user))
    else:
        if plan.blocked:
            return MigrationResult(
                False,
                tuple(["✗ Migration refused:", *[f"  • {b}" for b in plan.blockers]]),
            )
        blocker = _preflight_apply(plan, target, run_as_user)
        if blocker is not None:
            return MigrationResult(
                False,
                ("✗ Migration refused before changing anything:", f"  • {blocker}"),
            )
        manifest = {
            "version": 1,
            "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "flag_was": plan.multiplex_flag_on,
            "default": plan.default.to_dict(),
            "secondaries": [p.to_dict() for p in plan.standalone_secondaries],
        }
        _write_manifest(plan.default_home, manifest)

    try:
        _write_multiplex_flag(plan.default_home, True)
        lines.append(
            f"  ✓ default: gateway.multiplex_profiles: true "
            f"({plan.default_home / 'config.yaml'})"
        )
        lines.extend(_remove_secondary_gateways(plan))
        lines.append(
            f"  ✓ {_restart_default(plan.default, target, plan.default_home, run_as_user=run_as_user)}"
        )
    except Exception as exc:
        lines.extend([
            f"  ✗ migration failed ({exc})",
            "  ↩ Restoring one host gateway so this host is not left without one...",
        ])
        rollback = rollback_migration_result(plan.default_home)
        lines.extend(rollback.lines)
        if not rollback.succeeded:
            lines.append(f"  Re-run {MIGRATE_COMMAND} to resume from the manifest.")
        return MigrationResult(False, tuple(lines))

    expected = plan.expected_served_names
    served = _wait_for_served(plan.default_home, expected, served_wait)
    if served is not None and expected <= set(served):
        _manifest_path(plan.default_home).unlink(missing_ok=True)
        lines.extend([
            "",
            f"✓ Migrated: the default gateway now serves {len(served)} profiles: {', '.join(served)}",
            *[f"  • {n}" for n in plan.notices],
        ])
        return MigrationResult(True, tuple(lines))

    missing = sorted(expected - set(served or []))
    lines.extend([
        "",
        "⚠ Migration applied, but the default gateway has not confirmed serving: "
        + ", ".join(missing),
        "  Check `hermes gateway status` and the gateway log; the flag and manifest are in place.",
        f"  Re-run {MIGRATE_COMMAND} once it is healthy — it resumes from the manifest.",
    ])
    return MigrationResult(False, tuple(lines))


def apply_migration(plan: MigrationPlan, *, served_wait: float = _SERVED_WAIT_SECONDS) -> bool:
    """Compatibility boolean result for non-CLI callers."""
    return apply_migration_result(plan, served_wait=served_wait).succeeded


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


def rollback_migration_result(default_home: Optional[Path] = None) -> MigrationResult:
    """Compensate a failed apply and return structured output without printing."""
    lines: list[str] = []
    default_home = default_home or _default_home()
    manifest = _read_manifest(default_home)
    if manifest is None:
        return MigrationResult(
            False,
            (f"✗ No migration manifest at {_manifest_path(default_home)}; nothing to restore.",),
        )

    incomplete = (
        f"⚠ Compensation incomplete; manifest kept at {_manifest_path(default_home)}.\n"
        f"  Re-run {MIGRATE_COMMAND} — it resumes from the manifest."
    )
    secondaries = _manifest_secondaries(manifest)
    if secondaries is None:
        return MigrationResult(
            False,
            (
                f"✗ Malformed secondary records in {_manifest_path(default_home)}; "
                "fix or delete the manifest.",
            ),
        )

    try:
        _write_multiplex_flag(default_home, bool(manifest.get("flag_was", False)))
        lines.append("  ✓ default: gateway.multiplex_profiles restored")
    except Exception as exc:
        lines.extend([
            f"  ✗ default: could not restore gateway.multiplex_profiles ({exc})",
            incomplete,
        ])
        return MigrationResult(False, tuple(lines))

    default_rec = manifest.get("default")
    default_gw = ProfileGateway(
        "default",
        default_home,
        pid=_own_gateway_pid(default_home, _host_gateway_owner()),
        services=(
            _recorded_services(default_rec) if isinstance(default_rec, dict) else []
        ) or _installed_services(default_home),
    )
    names = [str(rec["profile"]) for rec in secondaries]
    try:
        _reconcile_standalone_runtime(default_home, set(names))
        lines.append("  ✓ default: cleared multiplex-owned runtime status")
    except Exception as exc:
        lines.extend([
            f"  ✗ default: could not clear multiplex-owned runtime status ({exc})",
            incomplete,
        ])
        return MigrationResult(False, tuple(lines))

    if not default_gw.has_gateway and not any(
        rec.get("pid") or _recorded_services(rec) for rec in secondaries
    ):
        _manifest_path(default_home).unlink(missing_ok=True)
        lines.append(
            f"✓ Nothing was running before this attempt ({_NOTHING_RECORDED}); the flag is restored."
        )
        return MigrationResult(True, tuple(lines))

    target, run_as_user = _target_from_manifest(manifest)
    try:
        lines.append(
            f"  ✓ {_restart_default(default_gw, target, default_home, run_as_user=run_as_user)}"
        )
    except Exception as exc:
        lines.append(
            "  ✗ default: could not bring the host gateway back up via the recorded service "
            f"manager ({exc}); falling back to a detached gateway"
        )
        try:
            spawned = _spawn_detached_gateway(default_home)
        except Exception as spawn_exc:
            lines.append(f"  ✗ default: the detached fallback also failed ({spawn_exc})")
            spawned = False
        if not spawned:
            lines.append(incomplete)
            return MigrationResult(False, tuple(lines))
        lines.append("  ✓ default: started the host gateway detached (no service manager)")

    live = _wait_for_live_gateway(default_home, _COMPENSATOR_WAIT_SECONDS)
    if live is None:
        lines.extend([
            f"  ✗ default: no gateway confirmed serving this host within {_COMPENSATOR_WAIT_SECONDS:.0f}s "
            "(check `hermes gateway status` and the gateway log)",
            incomplete,
        ])
        return MigrationResult(False, tuple(lines))

    _manifest_path(default_home).unlink(missing_ok=True)
    lines.append(
        f"✓ Compensated: one host gateway is running again (pid {live}) on the recorded flag."
    )
    if names:
        lines.append(
            f"  Per-profile gateways this attempt had already removed: {', '.join(names)}. They are "
            "NOT reinstalled — one gateway per host is the only supported topology — and the host "
            "gateway serves them as soon as it multiplexes.\n"
            f"  Run {MIGRATE_COMMAND} to finish converging."
        )
    return MigrationResult(True, tuple(lines))


def rollback_migration(default_home: Optional[Path] = None) -> bool:
    """Compatibility boolean result for non-CLI callers."""
    return rollback_migration_result(default_home).succeeded


# --------------------------------------------------------------------------- host capability


def _host_supports_migration() -> Optional[str]:
    """Reason the host cannot be converged by this command, else None.

    Windows IS handled (per-profile Scheduled Tasks and the Startup-folder fallback are removed
    like any other unit). s6 IS handled too, in-process: a named profile's slot that is UP is
    parked (``s6-svc -d`` + ``down`` file) and its autostart intent folded into the root slot the
    same way the container's boot does it (``gateway.s6_service``). What cannot be
    done from here is register a slot the boot never created — that is the one refusal left.
    """
    from gateway.service_manager import detect_service_manager
    if detect_service_manager() != "s6":
        return None
    from gateway.s6_service import named_slot_name
    from gateway.s6_manager import S6ServiceManager
    scandir = S6ServiceManager().scandir
    if not (scandir / named_slot_name("default")).is_dir():
        return (f"s6-supervised container without a root gateway slot ({scandir / named_slot_name('default')}); "
                "the container's boot registers it — restart the container.")
    return None

# Public name for CLI orchestration; private seam remains patchable for domain consumers.
def host_migration_refusal() -> Optional[str]:
    return _host_supports_migration()
