"""Validated, credential-free gateway endpoint discovery for local clients."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from gateway.runtime_contract import RuntimeState

# Total startup deadline a client waits for a cold daemon. A cold boot imports the whole
# runtime, seeds the skill library and warms the turn machinery: ~10-20 s on a loaded
# 32-core CI runner with dozens of sibling boots, where 30 s tipped every attaching client
# of a parallel e2e shard into `starting: deadline`. A minute is still a verdict, not a hang.
DEFAULT_ENSURE_TIMEOUT = 60.0


@dataclass(frozen=True)
class GatewayEndpoint:
    profile_id: str
    instance_id: str
    authority_epoch: int
    runtime_protocol: int
    api_origin: str
    supervisor: Literal["none", "systemd", "launchd", "windows", "external"]
    capabilities: frozenset[str]
    # Home whose control socket answers for this endpoint: the profile's own home, or the
    # default multiplexer's root when the profile is served by it.
    control_home: str | None = None
    # Commit the owner BOOTED from (its ``identify`` ``code_sha``); None when it cannot name one.
    # A gateway that outlived ``hermes update`` keeps serving old code, so an attaching client
    # compares this with its own checkout and restarts the owner instead of re-attaching it.
    code_sha: str | None = None


@dataclass(frozen=True)
class GatewayDiscovery:
    state: RuntimeState
    endpoint: GatewayEndpoint | None = None
    reason_code: str | None = None
    # Human-readable cause behind a bounded ``reason_code`` when the owner published one (the
    # multiplexer's park reason), or the redacted output tail of a start this call made that died
    # (``runtime_exited``), or the unsafe local control node a discovery refused (path, owner uid or
    # mode, and the fix); never raw peer data.
    detail: str | None = None


def attachable(result: GatewayDiscovery) -> bool:
    """Whether a client may attach to this discovery/ensure verdict.

    ``ready`` owners, plus a restart's after-turn wait (``draining``/``restart_wait``): that owner
    still owns its turns and must stay reachable to answer a parked approval/clarify (#139041).
    Shared by every client bootstrap (``connect_gateway`` and Ink's ``gateway_bootstrap.py``)."""
    if result.endpoint is None:
        return False
    from gateway.drain_control import RESTART_WAIT_REASON
    return result.state == "ready" or (result.state == "draining" and result.reason_code == RESTART_WAIT_REASON)


def _canonical_home(home: str | Path) -> str:
    return os.path.normcase(str(Path(home).expanduser().resolve()))


def gateway_profile_home(home: str | Path) -> Path:
    """The profile home whose gateway serves *home*: the home itself, or for a private overlay home
    (a task dir mirroring a profile through symlinks, its ``gateway.lock`` among them) the profile its
    lock names. A client compares the owner's ``profile_id`` with this, never the overlay path.
    Not normcased: ``profile_id`` keeps the owner's own spelling (Windows)."""
    return _overlay_owner_home(Path(home).expanduser().resolve())


def _overlay_owner_home(home: Path) -> Path:
    """Return the owner named by a private overlay's gateway lock, if it has one."""
    if os.name == "nt":
        return home
    from hermes_cli.gateway_runtime_discovery import DiscoveryError, _private_lock, _private_node
    lock = home / "gateway.lock"
    try:
        # An overlay home intentionally shares only the owner's lock.  Do not turn an arbitrary
        # link into a profile alias: both homes and the target lock must meet ordinary discovery's
        # private-control checks.
        _private_node(home, kind="directory", home=True)
        target = lock.resolve(strict=True)
        if not lock.is_symlink() or target.name != "gateway.lock":
            return home
        _private_node(target.parent, kind="directory", home=True)
        _private_lock(target)
    except (FileNotFoundError, OSError, DiscoveryError):
        return home
    return target.parent


def _endpoint(payload: dict, home: Path, control_home: Path | None = None) -> GatewayDiscovery:
    if type(payload.get("runtime_protocol")) is not int or payload["runtime_protocol"] != 1:
        return GatewayDiscovery("incompatible", reason_code="runtime_protocol")
    profiles = payload.get("served_profiles")
    if not isinstance(profiles, list):
        return GatewayDiscovery("inaccessible", reason_code="profile_mismatch")
    # Canonicalize BOTH sides: on Windows normcase lower-cases the served spelling, so a
    # caller's mixed-case Path (WorkerRPC) never matched and every worker saw owner_unavailable.
    wanted = _canonical_home(home)
    matches = [p for p in profiles if isinstance(p, dict)
               and isinstance(p.get("home"), str) and _canonical_home(p["home"]) == wanted]
    if len(matches) != 1 or not isinstance(matches[0].get("profile_id"), str) or not matches[0]["profile_id"]:
        return GatewayDiscovery("inaccessible", reason_code="profile_mismatch")
    state = payload.get("state")
    # A graceful restart's after-turn wait still owns its in-flight turns (#139041): it stays
    # ``draining`` (exit 6, never a fresh-start target) but carries its endpoint, so an interactive
    # client can attach to answer a turn parked on approval/clarify. New input queues for the next owner.
    from gateway.drain_control import RESTART_WAIT_REASON
    restart_wait = state == "draining" and payload.get("drain_reason") == RESTART_WAIT_REASON
    if state in {"starting", "draining", "conflict"} and not restart_wait:
        # An auto-started gateway ending through its idle exit names that, so its clients wait
        # for it to go and start a fresh one instead of reporting a refusal.
        idle = state == "draining" and payload.get("drain_reason") == "idle_exit"
        return GatewayDiscovery(state, reason_code="idle_exit" if idle else None)
    if state != "ready" and not restart_wait:
        return GatewayDiscovery("inaccessible", reason_code="invalid_runtime_state")
    capabilities = payload.get("capabilities")
    if (not isinstance(capabilities, list) or not all(isinstance(c, str) for c in capabilities)
            or "session-authority-v1" not in capabilities):
        return GatewayDiscovery("incompatible", reason_code="session_authority_unavailable")
    epoch, instance = payload.get("authority_epoch"), payload.get("instance_id")
    if type(epoch) is not int or epoch <= 0 or not isinstance(instance, str) or not instance:
        return GatewayDiscovery("inaccessible", reason_code="invalid_runtime_identity")
    origin = payload.get("api_origin")
    if not isinstance(origin, str):
        return GatewayDiscovery("inaccessible", reason_code="invalid_api_origin")
    try:
        parsed = urlsplit(origin)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment or not parsed.port
                or not ipaddress.ip_address(parsed.hostname).is_loopback):
            return GatewayDiscovery("inaccessible", reason_code="invalid_api_origin")
    except ValueError:
        return GatewayDiscovery("inaccessible", reason_code="invalid_api_origin")
    supervisor = payload.get("supervisor")
    if supervisor not in {"none", "systemd", "launchd", "windows", "external"}:
        return GatewayDiscovery("inaccessible", reason_code="invalid_supervisor")
    return GatewayDiscovery("draining" if restart_wait else "ready", GatewayEndpoint(
        profile_id=matches[0]["profile_id"], instance_id=instance,
        authority_epoch=epoch, runtime_protocol=1, api_origin=origin,
        supervisor=supervisor, capabilities=frozenset(capabilities),
        control_home=str(control_home) if control_home is not None else None,
        code_sha=code_sha if isinstance(code_sha := payload.get("code_sha"), str) and code_sha else None,
    ), reason_code=RESTART_WAIT_REASON if restart_wait else None)


def control_home_for(home: Path, endpoint: GatewayEndpoint | None) -> Path:
    """Home whose control socket mints tickets for *endpoint* (the multiplexer's for a served profile)."""
    if endpoint is not None and endpoint.control_home:
        return Path(endpoint.control_home)
    return home


def _starting_multiplexer_root(home: Path) -> Path | None:
    """The root whose reservation is pending AND whose boot policy would serve *home*, else None.
    A ``gateway.standalone: true`` secondary is never served by it, so a reserved (or ready) root
    must not make its client wait out the deadline instead of starting the profile's own owner.
    One uncached config parse plus a lock probe: call it once per ensure iteration."""
    from hermes_cli.gateway_runtime_discovery import missing_owner_state
    from hermes_cli.gateway_runtime_multiplex import implied_host_root
    root = implied_host_root(home)
    return root if root is not None and missing_owner_state(root) == "starting" else None


def _standalone_multiplexer_reason(root: Path) -> str | None:
    """A live host's reason for not serving its named profiles, if it published one.

    A held root lock ordinarily means that a named profile should wait for its
    multiplexer.  A host which published a guard refusal is different: it is a
    standalone gateway and can never add the named profile during that boot. Read
    with the same liveness rules as ``hermes gateway status``'s standalone box.
    """
    from hermes_cli.gateway_multiplex_mode import recorded_standalone_reason
    reason = recorded_standalone_reason(root / "gateway_state.json")
    return reason[:500] if reason else None


def _parked_by_multiplexer(payload: dict, home: Path) -> GatewayDiscovery | None:
    """A secondary the multiplexer could not serve (unusable store, home owned elsewhere) is
    published under ``parked_profiles`` (name -> reason). That is a terminal verdict for its
    clients: waiting for a ``served_profiles`` entry that will never appear is the wrong answer."""
    parked = payload.get("parked_profiles")
    if not isinstance(parked, dict):
        return None
    from hermes_cli.profiles import normalize_profile_name
    name = normalize_profile_name(home.name)
    reason = next((r for n, r in parked.items() if normalize_profile_name(str(n)) == name), None)
    if reason is None:
        return None
    return GatewayDiscovery("inaccessible", reason_code="profile_parked", detail=str(reason)[:500] or None)


def _served_by_multiplexer(home: Path, *, timeout: float) -> GatewayDiscovery | None:
    """A served secondary has no socket of its own: the default multiplexer's control socket
    answers for it, and its ``identify`` lists the home under ``served_profiles``. Only the
    multiplexer's live descriptor proves service; a held ``gateway.lock`` alone does not."""
    from hermes_cli.gateway_runtime_discovery import DiscoveryError, query_identify
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
    root = multiplexer_root_for(home)
    if root is None:
        return None
    try:
        payload = query_identify(root, timeout=timeout)
    except (FileNotFoundError, ConnectionRefusedError, TimeoutError, DiscoveryError, PermissionError,
            OSError, ValueError, TypeError):
        return None
    result = _endpoint(payload, home, control_home=root)
    if result.state == "inaccessible" and result.reason_code == "profile_mismatch":
        return _parked_by_multiplexer(payload, home)  # else: the multiplexer does not serve this profile
    return result


_NATIVE_WINDOWS = os.name == "nt"


def _owner_pid(home: Path) -> int | None:
    """The PID a refusal may name: one of *home*'s gateway records whose process is provably that
    gateway, else None (the caller keeps the generic ``authorization``).

    Proof is the shared identity chain: alive, its recorded start time known and equal to the live
    one (a recycled PID differs), its recorded home not another profile's, and its live command line
    (or, when that is unreadable across the elevation boundary, its record) a gateway's. A stale
    ``gateway.pid`` falls through to ``gateway.lock`` instead of masking it."""
    from gateway.status import (
        _get_process_start_time,
        _live_pid_from_record,
        _read_json_file,
        _record_matches_live_gateway_pid,
        recorded_gateway_home_conflicts,
    )
    for name in ("gateway.pid", "gateway.lock"):
        record = _read_json_file(home / name, bare_pid_ok=True)
        pid = _live_pid_from_record(record)
        if pid is None or pid == os.getpid():
            continue
        recorded = record.get("start_time")
        if recorded is None or recorded != _get_process_start_time(pid):
            continue  # unverifiable or recycled: never name it
        if recorded_gateway_home_conflicts(record, expected_home=home):
            continue
        if not _record_matches_live_gateway_pid(record, pid):
            continue
        return pid
    return None


def _refused(home: Path, *, windows: bool = _NATIVE_WINDOWS) -> GatewayDiscovery:
    """The owner's control channel refused this same-user client. On Windows (*windows*, the host
    fact) name the boundary (an elevated owner, or one in another logon session) with its remedy
    instead of the generic ``authorization`` that reads as a credential failure (#139845, #140252)."""
    if windows and (pid := _owner_pid(home)) is not None:
        from gateway.runtime_bootstrap_windows import probe_refused_owner
        try:
            named = probe_refused_owner(pid)
        except OSError:
            named = None
        if named is not None:
            return GatewayDiscovery("inaccessible", reason_code=named[0], detail=named[1])
    return GatewayDiscovery("inaccessible", reason_code="authorization")


def discover_gateway_endpoint(profile_home: str | Path, *, timeout: float = 2.0) -> GatewayDiscovery:
    """Read live control identity, preserving uncertainty instead of spawning.

    A named profile served by the default multiplexer resolves to the multiplexer's endpoint
    with ``profile_id`` = the profile's own home, so clients attach to it with that identity.
    """
    from hermes_cli.gateway_runtime_discovery import DiscoveryError, missing_owner_state, query_identify

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    home = _overlay_owner_home(Path(_canonical_home(profile_home)))
    try:
        return _endpoint(query_identify(home, timeout=timeout), home)
    except (FileNotFoundError, ConnectionRefusedError):
        served = _served_by_multiplexer(home, timeout=timeout)
        if served is not None:
            return served
        state = missing_owner_state(home)
        if state == "inaccessible" and os.name != "nt":
            from gateway.runtime_ownership import unreadable_lock_recovery
            if detail := unreadable_lock_recovery(home / "gateway.lock"):
                return GatewayDiscovery(state, reason_code="foreign_stale_lock", detail=detail)
        return GatewayDiscovery(state)
    except TimeoutError:
        return GatewayDiscovery("inaccessible", reason_code="control_timeout")
    except DiscoveryError as exc:
        # The code stays bounded; the detail names the local node, its owner or mode, and the fix.
        return GatewayDiscovery("inaccessible", reason_code=exc.reason, detail=exc.bounded_detail(500))
    except PermissionError as exc:
        from gateway.runtime_bootstrap_windows import RefusedOwner
        if isinstance(exc, RefusedOwner):  # the client proved the boundary from the control pipe (#140706)
            return GatewayDiscovery("inaccessible", reason_code=exc.reason_code, detail=str(exc))
        return _refused(home)
    except (OSError, ValueError, TypeError):
        # Raw peer data, paths and URLs may contain secrets; report bounded codes.
        return GatewayDiscovery("inaccessible", reason_code="invalid_control_peer")


def _update_fenced(homes) -> bool:
    """True while an update owns the install, judged the way main's launchers and the Desktop
    judge ``.hermes-update-in-progress``: a live (or still being written) marker, or a dead or
    malformed one while the checkout lock is held (a killed update's tree still runs). A dead
    marker over a free lock is a killed update: it never blocks a launch.

    Read-only: a client never clears the updater's fence; the launch-time reclaim
    (``update_lock.read_live_update``) and the next update remove a dead marker. A marker that
    exists but cannot be read is unjudgeable and fences.
    """
    import time
    from hermes_cli import update_lock

    for fence_home in homes:
        path = fence_home / update_lock.MARKER_NAME
        try:
            raw = path.read_bytes()
            if not raw:
                # An exclusive first publish in flight (contract A3), not yet a dead claim.
                if time.time() - path.stat().st_mtime < update_lock.EMPTY_MARKER_GRACE_SECONDS:
                    return True
        except FileNotFoundError:
            continue
        except OSError:
            return True
        if update_lock.judge_marker(raw)[0] in {"live", "ours"} or update_lock.checkout_lock_held():
            return True
    return False


def _fenced_verdict(home: Path, timeout: float) -> GatewayDiscovery:
    """What ensure answers while an update owns the install. The update's own drain
    (``pause-for-update``) is a restart's after-turn wait whose owner may hold a turn parked on
    approval/clarify, so it is handed out for clients to answer it (#139041). Discovery only:
    nothing is started or replaced while fenced, and any other owner state stays ``update_paused``."""
    observed = discover_gateway_endpoint(home, timeout=timeout)
    if observed.state == "draining" and attachable(observed):
        return observed
    return GatewayDiscovery("draining", reason_code="update_paused")


_REPLACE_FAILED_DETAIL = "an older Hermes gateway still owns this profile after a stop; run: hermes gateway stop"
# ``draining`` reasons after which the owner is going away by itself: its own idle exit, or the
# older-Hermes owner this call just stopped. Ensure waits it out and starts the current owner.
_RETIRING = frozenset({"idle_exit", "replaced"})


def _observe_replacing_stale(home: Path, deadline: float, replaced: set[Path]) -> GatewayDiscovery:
    """Discovery, except that a gateway an older Hermes left on this home is stopped once (like
    ``hermes gateway stop``) and reported as retiring; a refusal keeps ``incompatible`` + its remedy."""
    from hermes_cli.gateway_runtime_service import remaining
    observed = discover_gateway_endpoint(home, timeout=remaining(deadline))
    if observed.state != "incompatible" or observed.reason_code != "runtime_protocol":
        return observed
    if home in replaced:
        return GatewayDiscovery("incompatible", reason_code="runtime_protocol", detail=_REPLACE_FAILED_DETAIL)
    from hermes_cli import gateway_runtime_replace as replace
    replaced.add(home)
    refusal = replace.replace_incompatible_owner(home, deadline=deadline)
    if refusal is not None:
        return GatewayDiscovery("incompatible", reason_code="runtime_protocol",
                                detail=replace.REFUSAL_DETAIL.get(refusal, _REPLACE_FAILED_DETAIL))
    return GatewayDiscovery("draining", reason_code="replaced")


def ensure_gateway_runtime(profile_home: str | Path, *, timeout: float = DEFAULT_ENSURE_TIMEOUT,
                           idle_exit: bool = False) -> GatewayDiscovery:
    """Ensure once; pending remains pending at deadline.

    Never installs anything. The one owner it replaces is a gateway an older Hermes left running on
    this install's own home (``incompatible``/``runtime_protocol``, the upgrade case): it is stopped
    the way ``hermes gateway stop`` stops it, once, and the current one started like any absent
    owner. Another install's or a newer Hermes's gateway stays refused, with the reason in ``detail``
    (``hermes_cli.gateway_runtime_replace``).

    A successful service command or Popen is not session readiness. After an
    owner/start request is observed this invocation never launches another.
    ``idle_exit``: an unmanaged daemon this call starts ends itself once idle (client entrypoints:
    chat, TUI, cron/kanban/ACP clients). A gateway that is ending that way reports ``draining``
    (``idle_exit``); this call waits for it to go and starts the replacement transparently.
    """
    import time
    from hermes_cli.gateway_runtime_service import (
        RuntimeStartError, discover_existing_gateway_service, remaining,
        start_existing_gateway_service,
    )
    from hermes_constants import get_default_hermes_root, get_process_hermes_home

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    home = _overlay_owner_home(Path(_canonical_home(profile_home)))
    requested = False
    child = None
    child_home, child_offset = home, 0
    # A gateway ending through its own idle exit is not a refusal: wait for it to go (its lock is
    # still held for a moment after its control socket closes, which reads as "starting"), then
    # start the replacement this call owes the client.
    retiring = False
    replaced: set[Path] = set()
    delay = 0.025
    try:
        while True:
            # The install-root marker also covers profiles.
            if _update_fenced({home, get_process_hermes_home(), get_default_hermes_root()}):
                return _fenced_verdict(home, remaining(deadline))
            # Poll BEFORE discovering: an exit observed first, then "absent", cannot be a child that
            # published its owner a moment later. A winning racer shows up as starting/ready instead.
            status = child.poll() if child is not None else None
            observed = _observe_replacing_stale(home, deadline, replaced)
            if observed.reason_code == "control_timeout":
                return GatewayDiscovery("starting", reason_code="deadline")
            if observed.state == "draining" and observed.reason_code in _RETIRING:
                retiring, requested, child = True, False, None
                time.sleep(min(0.1, remaining(deadline)))
                continue
            if observed.state not in {"absent", "starting"}:
                return observed
            if status is not None and observed.state == "absent":
                # The one start this invocation requested died without leaving an owner: waiting
                # out the deadline cannot change that, and this call never launches a second one.
                # Its own (redacted) output is the answer the user needs, not a pointer to a file.
                from hermes_cli.gateway_runtime_start import startup_failure_report
                return GatewayDiscovery("inaccessible", reason_code="runtime_exited",
                                        detail=startup_failure_report(child_home, child_offset, status, child.pid))
            if observed.state == "starting" and not retiring:
                requested = True
            if observed.state == "absent":
                retiring = False
            reserved = _starting_multiplexer_root(home) if observed.state == "absent" and not requested else None
            if reserved is not None and (reason := _standalone_multiplexer_reason(reserved)):
                return GatewayDiscovery("inaccessible", reason_code="profile_parked", detail=reason)
            # A default multiplexer that is still starting will serve this named profile; never
            # spawn a competing per-profile daemon while its reservation is pending.
            if reserved is not None:
                requested = True
            if observed.state == "absent" and not requested:
                # A named profile the default multiplexer serves has no daemon of its own: start
                # (or await) the MULTIPLEXER. A per-profile spawn here would become a second owner
                # that blocks the multiplexer's next all-or-nothing reserve.
                from hermes_cli.gateway_runtime_multiplex import implied_host_root, multiplexer_serves_home
                target = multiplexer_serves_home(home) or home
                service = discover_existing_gateway_service(target, deadline=deadline)
                if service is None and target == home and (host := implied_host_root(home)) is not None:
                    # Main's guard refuses a named profile a gateway of its own unless it is standalone
                    # or already installed; with no multiplexer evidence yet, the host is still its owner.
                    target = host
                    service = discover_existing_gateway_service(target, deadline=deadline)
                # Runtime locks settle races remaining after this second probe.
                observed = discover_gateway_endpoint(home, timeout=remaining(deadline))
                if observed.state != "absent":
                    continue
                if service is not None:
                    start_existing_gateway_service(service, deadline=deadline)
                else:
                    from hermes_cli.gateway_runtime_start import spawn_unmanaged_gateway, stdio_log_size
                    child_home, child_offset = target, stdio_log_size(target)
                    child = spawn_unmanaged_gateway(target, deadline=deadline, idle_exit=idle_exit)
                requested = True
            time.sleep(min(delay, remaining(deadline)))
            # Capped low: each probe is a lock-file read or one local identify, and READY is noticed
            # within ~50 ms instead of up to 250 ms later on every cold start.
            delay = min(delay * 1.5, 0.05)
    except TimeoutError:
        return GatewayDiscovery("starting", reason_code="deadline")
    except RuntimeStartError as exc:
        return GatewayDiscovery(exc.state, reason_code=exc.reason)
    except PermissionError:
        return _refused(home)
    except OSError:
        return GatewayDiscovery("inaccessible", reason_code="runtime_start_failed")
