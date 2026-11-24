"""Dashboard routes for ``hermes gateway migrate --backends``: move this home from the old backend
(``hermes dashboard`` / ``hermes serve``) to the gateway's Webapp on the same address.

* ``GET  /api/gateway/migrate/backends/plan``    the ``--apply --dry-run --json`` plan, the read-only
  ``--backends`` report and the last run's status. Changes nothing.
* ``POST /api/gateway/migrate/backends/apply``   starts ``--apply --yes --detach --json`` (body may carry
  ``address``, ``keep_webapp_address``, ``force_interrupt``). The request may come from the old dashboard,
  the very process the run stops (possibly inside a systemd unit whose cgroup dies on stop): ``--detach``
  starts the run as its own transient unit / session, so it survives. The request returns once that run
  says whether it holds this home's run lock: 409 ``already_running`` when another run holds it.
* ``POST /api/gateway/migrate/backends/restore`` starts ``--restore --yes --detach --json`` (works from the
  gateway's own listener too: the detached run outlives the gateway it stops); 409 the same way.
* ``GET  /api/gateway/migrate/backends/status``  the ``--status --json`` shape, plus the run log's tail and
  ``alive`` (whether a ``running`` record's process still lives).

Every route takes ``?profile=``: it acts on that profile's Hermes home, resolved per request as the rest of
the dashboard resolves it (``_config_profile_scope``), and the detached run is started with ``-p <profile>``.
A profile this server does not have is a 404 ``unknown_profile``; there is no fallback to another home.
Without ``?profile=`` the routes act on this process's own home.

After a swap the gateway answers the same address with this same app, so ``status`` keeps working there
(after one re-sign-in: the gateway's session cookies are not the old dashboard's).

Authorization is the app-wide gate every ``/api/*`` route gets (session token on loopback, sign-in cookie
and the own-origin write rule on a gated bind, a native grant on the gateway's listener) plus the
``_require_token`` check of the other sensitive routes; none of these paths is public. Responses carry key NAMES
only: argv, cwd and environment of the old backend are never returned, and every outgoing string is
scrubbed of this home's ``HERMES_DASHBOARD_*`` values.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from hermes_cli.web_deps import LateState, late

_log = logging.getLogger("hermes_cli.web_server")
_require_token = late("_require_token")


def _authorized(request: Request) -> None:
    """The check of the other sensitive dashboard routes, on top of the app-wide gate."""
    _require_token(request)


router = APIRouter(dependencies=[Depends(_authorized)])

ACTION_NAME = "gateway-migrate-backends"
_LOG_TAIL_LINES = 40
#: How long one scan for old backends answers the plan route. The System page asks for the plan on every
#: load, and the scan (the process table, ``systemctl list-unit-files``) costs the same whether or not an
#: old backend exists. A started apply or restore forgets it, and the apply route always scans again, so
#: a cached list never starts a move; at worst the card shows a change on this machine minutes late.
_SCAN_TTL_SECONDS = 300.0

get_hermes_home = late("get_hermes_home", "hermes_cli.config")
_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")

# Fields of the engine's old-backend record a UI may see. ``argv`` / ``cwd`` are left out: an ExecStart
# may carry ``env NAME=value`` or a token argument.
_BACKEND_FIELDS = ("supervisor", "unit", "scope", "pid", "unit_file_state", "was_active")
_REPORT_FIELDS = ("supervisor", "running", "pid", "unit", "scope", "state", "this_home", "not_applied", "notes")
_SECRET_PREFIX = "HERMES_DASHBOARD_"


class ApplyBody(BaseModel):
    """The contract's apply flags; anything else is refused (a typo must not run a different plan).

    ``expect``: the plan's ``expect_token`` the user confirmed. The route plans again before it starts the
    run; when the fresh plan's token differs it answers 409 ``plan_changed`` (with the fresh, unfiltered
    plan) and starts nothing, and the background run gets ``--expect <token>`` to check again under the
    run lock, so a confirmation never moves a backend the user was not shown."""

    model_config = ConfigDict(extra="forbid")
    address: Optional[str] = None
    keep_webapp_address: bool = False
    force_interrupt: bool = False
    expect: Optional[str] = None


def _host(home: Path):
    """The engine's host for *home* (tests substitute the engine's fake host)."""
    from hermes_cli.gateway_migrate_backends_host import SystemHost
    return SystemHost(home)


# --------------------------------------------------------------------------- profile


class UnknownProfile(Exception):
    """``?profile=`` names no profile this server has."""


@dataclass(frozen=True)
class Target:
    """The Hermes home a request acts on. ``profile`` is the name a ``hermes -p`` command takes (None when
    this home is no profile home, e.g. a custom ``HERMES_HOME``); ``requested`` is the query's own value."""

    home: Path
    profile: Optional[str]
    requested: Optional[str]

    def run(self, fn: Callable[[], Any]) -> Any:
        """``fn()`` in this profile's home and secret scope (the process's own without ``?profile=``)."""
        with _config_profile_scope(self.requested):
            return fn()

    def command(self, flags: str) -> str:
        """The engine's own spelling for this home (``hermes -p NAME``, with ``HERMES_HOME=ROOT`` pinned for
        a profile outside the standard root; PowerShell quoting on Windows, as the engine prints it)."""
        from hermes_cli.gateway_migrate_backends_plan import hermes_command
        return f"{hermes_command(str(self.home), powershell=os.name == 'nt')} gateway migrate --backends {flags}"

    def commands(self) -> dict:
        """The commands the card shows (it never spells them itself: only the engine knows the root pin)."""
        return {"status": self.command("--status"), "restore": self.command("--restore")}


def _resolve(profile: Optional[str]) -> Target:
    """The home ``?profile=`` names, resolved as every profile-scoped dashboard route resolves it."""
    from hermes_constants import profile_name_for_home
    requested = (profile or "").strip() or None
    if requested is None or requested.lower() == "current":
        home = Path(get_hermes_home())
        return Target(home, profile_name_for_home(home), None)
    try:
        with _config_profile_scope(requested):
            home = Path(get_hermes_home())
    except HTTPException as exc:  # 400 (not a profile name) or 404 (no such profile): never another home
        raise UnknownProfile(requested) from exc
    return Target(home, profile_name_for_home(home) or requested, requested)


async def _target(profile: Optional[str]) -> Target:
    return await asyncio.to_thread(_resolve, profile)


def _unknown_profile(name: str) -> JSONResponse:
    return JSONResponse(status_code=404, content={
        "ok": False, "error": "unknown_profile", "message": f"This server has no profile named {name!r}."})


async def _off_loop(target: Target, fn: Callable[[], Any]) -> Any:
    return await asyncio.to_thread(target.run, fn)


# --------------------------------------------------------------------------- secret scrubbing


def _secret_values(home: Path) -> list[str]:
    """Values of ``HERMES_DASHBOARD_*`` in this process and this home's ``.env`` (longest first)."""
    values = {v for k, v in os.environ.items() if k.startswith(_SECRET_PREFIX)}
    from dotenv import dotenv_values
    try:
        values |= {v for k, v in dotenv_values(home / ".env").items() if k.startswith(_SECRET_PREFIX) and v}
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        _log.debug("backend migration: .env not readable for scrubbing: %s", type(exc).__name__)
    return sorted((v for v in values if v and len(v) >= 4), key=len, reverse=True)


# A URL's userinfo anywhere in a string (a configured ``dashboard.public_url`` may carry one, and it is
# not in the ``HERMES_DASHBOARD_*`` values above): everything from ``scheme://`` to the authority's last
# ``@`` (the authority ends at the next ``/``, ``?``, ``#`` or whitespace) becomes ``***@``. Userinfo may
# hold ``'``, ``"``, ``;``, ``:`` or ``%40``, so no other character ends it; an ``@`` in a path, query or
# fragment is left alone.
_URL_USERINFO = re.compile(r"(\b[A-Za-z][A-Za-z0-9+.-]*://)[^/?#\s]*@")
#: Opaque, credential-free: never scrubbed (a secret that happens to spell part of it must not change it).
TOKEN_KEY = "expect_token"


def _scrub(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for s in secrets:
            value = value.replace(s, "***")
        return _URL_USERINFO.sub(r"\1***@", value)
    if isinstance(value, dict):
        return {k: v if k == TOKEN_KEY else _scrub(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v, secrets) for v in value]
    return value


def _strip_userinfo(url: Optional[str]) -> Optional[str]:
    if not url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


# --------------------------------------------------------------------------- views


# ``hermes <verb>`` at the start of a command (not ``-p``/``--profile`` already, not a path or unit name).
_BARE_HERMES = re.compile(r"(?<![\w./-])hermes (?!-p |--profile)")


def qualify(text: Any, profile: Optional[str]) -> Any:
    """*text* (a string or a list of them) with every ``hermes`` command naming ``-p <profile>``: a step
    the engine prints for this home must not act on another one (a sticky ``hermes profile use`` choice,
    or the default home for a named profile)."""
    if not profile:
        return text
    if isinstance(text, list):
        return [qualify(t, profile) for t in text]
    return _BARE_HERMES.sub(f"hermes -p {profile} ", text) if isinstance(text, str) else text


def _plan_view(plan, profile: Optional[str]) -> dict:
    from hermes_cli.gateway_migrate_backends_apply import plan_token
    from hermes_cli.gateway_migrate_backends_plan import format_plan
    shown = copy.copy(plan)  # the text is formatted from the sanitized URL, never the configured one
    shown.public_url = _strip_userinfo(plan.public_url)
    raw = shown.to_dict()
    old = raw.get("old_backend") or None
    raw["old_backend"] = {k: old.get(k) for k in _BACKEND_FIELDS} if old else None
    raw["unsupported"] = [{**{k: u.get(k) for k in ("supervisor", "unit", "pid")},
                           "instructions": qualify(u.get("instructions"), profile),
                           "steps": qualify(list(u.get("steps") or []), profile)}
                          for u in raw.get("unsupported") or []]
    raw["text"] = qualify(format_plan(shown, dry_run=True), profile)
    raw[TOKEN_KEY] = plan_token(plan)
    # Evidence of an old backend (found, or found but not movable here). A blocker alone (config.yaml
    # unreadable, say) is not one: such a plan is not "nothing to do", yet nothing was found.
    raw["found"] = bool(plan.candidates or plan.unsupported)
    return raw


def _report_view(items) -> list[dict]:
    from dataclasses import asdict
    return [{k: asdict(i).get(k) for k in _REPORT_FIELDS} for i in items]


def _log_path(home: Path) -> Path:
    return Path(home) / "logs" / "backend-migration.log"


def _log_tail(home: Path) -> list[str]:
    from hermes_cli.web_routers.actions import _tail_lines
    return _tail_lines(_log_path(home), _LOG_TAIL_LINES)


# The ``--detach`` launcher this process last started for each home. Kept per home: a launch for one
# profile never makes another profile's button answer "already starting".
_LAUNCHERS: dict[str, Any] = {}
_LAUNCHERS_LOCK = threading.Lock()


def _launcher(home: Path) -> Optional[dict]:
    """The short-lived ``--detach`` launcher this process spawned for *home*, if any (gone after a swap)."""
    proc = _LAUNCHERS.get(str(home))
    if proc is None:
        return None
    code = proc.poll()
    return {"pid": proc.pid, "running": code is None, "exit_code": code}


_ROLLBACK_SUFFIXES = ("; rolled back", "; rollback incomplete")


class _Answered:
    """A host whose liveness answer was already taken, so the status and ``alive`` come from one check."""

    def __init__(self, alive: bool):
        self.alive = alive

    def same_process(self, pid, start_time) -> bool:
        return self.alive


def _run_view(host, profile: Optional[str]) -> dict:
    """``--status --json`` plus ``alive`` (a ``running`` record: does its process still live, the engine's own
    pid + start-time check) and ``reason`` (a failed or rolled-back run: why, without the rollback suffix),
    all from one read of the manifest and the refusal record and one liveness check. Manual steps name
    the profile."""
    from hermes_cli.gateway_migrate_backends_apply import (
        _in_progress, read_manifest, status_payload, status_payload_from, unreadable_manifest)
    from hermes_cli.gateway_migrate_backends_detach import read_refusal, refusal_status
    if unreadable_manifest(host.home) is not None:
        return {**status_payload(host), "reason": None}  # the engine's own "cannot be trusted" status
    m, refusal = read_manifest(host.home), read_refusal(host.home)
    payload = refusal_status(refusal, m) if refusal is not None else None
    alive = None
    if payload is None:
        alive = _in_progress(host, m) if m is not None and m.get("state") == "running" else None
        payload = status_payload_from(_Answered(bool(alive)), m or {})
    payload["manual_steps"] = qualify(list(payload.get("manual_steps") or []), profile)
    if payload["state"] == "running":
        return {**payload, "alive": bool(alive), "reason": None}
    reason = None
    if payload["state"] in ("rolled_back", "failed") and not payload.get("refused"):
        reason = payload["message"]
        for suffix in _ROLLBACK_SUFFIXES:
            reason = reason.split(suffix, 1)[0]
    return {**payload, "reason": (reason or "").strip() or None}


def _status_view(host, target: Target, request: Request) -> dict:
    from hermes_cli.web_server_surface import surface
    return {**_run_view(host, target.profile), "log_tail": _log_tail(host.home), "launcher": _launcher(host.home),
            "served_by": surface(request.app.state), "now": time.time(), "profile": target.profile,
            "commands": target.commands()}


def _respond(payload: dict, host, status_code: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=_scrub(payload, _secret_values(Path(host.home))))


def _refuse(host, status_code: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return _respond({"ok": False, "error": code, "message": message, **extra}, host, status_code)


def _unfinished(host, target: Target) -> Optional[JSONResponse]:
    """409 while a run is live or an earlier one died mid-way (its rollback belongs in a terminal:
    the CLI runs it in the foreground, and it may restart the process answering this request). First of
    all, 409 for a profile a plugin runs elsewhere: no move or undo is started for it, and no command
    that would roll one back is named."""
    from hermes_cli.gateway_migrate_backends_apply import _in_progress, read_manifest, unreadable_manifest
    from hermes_cli.profiles_run_elsewhere import run_elsewhere_refusal
    if (elsewhere := run_elsewhere_refusal(host.home)) is not None:
        return _refuse(host, 409, "blocked", "Nothing was started.", blockers=[f"{elsewhere} Nothing changed."])
    if (problem := unreadable_manifest(host.home)) is not None:
        return _refuse(host, 409, "manifest_unreadable", f"{problem}. Nothing changed; see "
                       f"{target.command('--status')} on this machine.", command=target.command("--status"))
    m = read_manifest(host.home)
    if m is None or m.get("state") not in ("running", "failed"):
        return None
    if _in_progress(host, m):
        return _refuse(host, 409, "already_running", "A move is already running.", step=m.get("step"),
                       status_command=target.command("--status"))
    command = target.command("--apply")
    return _refuse(host, 409, "unfinished_run",
                   f"An earlier move did not finish (state {m.get('state')}, step {m.get('step')}). "
                   f"Open a terminal on this machine and run: {command} (it undoes the earlier run first).",
                   command=command)


def _parse_address(value: Optional[str]):
    if value is None:
        return None
    from hermes_cli.gateway_migrate_backends_plan import parse_address
    parsed = parse_address(value)
    if parsed is None:
        raise HTTPException(status_code=400, detail="address must be HOST:PORT")
    return parsed


class _Detached:
    """*host* as the button's ``--detach`` run sees it: that run is started in its own unit, so "this
    process runs inside the gateway's unit" (true of a gateway answering the request) is not about it."""

    def __init__(self, host):
        self._host = host

    def __getattr__(self, name):
        return getattr(self._host, name)

    def own_unit(self) -> None:
        return None


def _manual_gateway_blockers(host, plan) -> None:
    """The engine's gateway service refusals for a manual-only plan: its steps are withheld while one
    stands, as the automatic path refuses."""
    from hermes_cli.gateway_migrate_backends_apply import _gateway_service_blockers
    _gateway_service_blockers(host, plan)
    if plan.blockers:
        plan.unsupported = [(item, why, []) for item, why, _steps in plan.unsupported]


def _build_plan(host, body: ApplyBody):
    """The plan the button's run would follow, as ``--apply --dry-run`` reports it (with the apply path's
    own refusals), for a run started with ``--detach``."""
    from hermes_cli.gateway_migrate_backends_apply import _owner_blockers, read_manifest
    from hermes_cli.gateway_migrate_backends_plan import BackendPlan, _own_cgroup_inside, build_plan
    m = read_manifest(host.home)
    if m is not None and m.get("state") == "done":
        return BackendPlan(home=str(host.home))  # already migrated: a retired unit is never offered again
    plan = build_plan(host, address=_parse_address(body.address), keep_webapp=body.keep_webapp_address,
                      force_interrupt=body.force_interrupt)
    if plan.rec and plan.rec.get("supervisor") == "systemd" and _own_cgroup_inside(plan):
        # The button's run is always --detach: "this command runs inside the old unit" is true of THIS
        # process, not of the run it starts outside that unit. Strip it before _owner_blockers so it does
        # not mask the unknown-running-turns check.
        own = f"This command runs inside {plan.rec['unit']} itself"
        plan.blockers = [b for b in plan.blockers if not b.startswith(own)]
    if plan.rec and not plan.unsupported:
        _owner_blockers(_Detached(host), plan, force_interrupt=body.force_interrupt)
    elif plan.unsupported:
        _manual_gateway_blockers(host, plan)
    return plan


# --------------------------------------------------------------------------- the plan route's scan


_SCANS: dict[str, tuple[float, list]] = {}
_SCANS_LOCK = threading.Lock()


def _scan(host) -> list:
    """*host*'s old backends, scanned at most once per :data:`_SCAN_TTL_SECONDS` for the plan route."""
    key, now = str(host.home), time.monotonic()
    with _SCANS_LOCK:
        cached = _SCANS.get(key)
        if cached is not None and now - cached[0] < _SCAN_TTL_SECONDS:
            return cached[1]
    items = host.backends()
    with _SCANS_LOCK:
        _SCANS[key] = (time.monotonic(), items)
    return items


def _forget_scan(home: Path) -> None:
    """A started apply or restore changes which backends run: the next plan scans again."""
    with _SCANS_LOCK:
        _SCANS.pop(str(home), None)


def _plan_read(host, body: ApplyBody, profile: Optional[str]) -> tuple[Any, list, dict]:
    """The plan, the ``--backends`` report and the last run, from ONE scan (``build_plan`` reads the same
    list through the host)."""
    items = _scan(host)
    host.backends = lambda: items
    return _build_plan(host, body), items, _run_view(host, profile)


# --------------------------------------------------------------------------- the detached run


def _apply_argv(body: ApplyBody) -> list[str]:
    argv = ["gateway", "migrate", "--backends", "--apply", "--yes", "--detach", "--json"]
    if body.address:
        argv += ["--address", body.address.strip()]
    if body.expect:
        argv += ["--expect", body.expect]
    if body.keep_webapp_address:
        argv.append("--keep-webapp-address")
    if body.force_interrupt:
        argv.append("--force-interrupt")
    return argv


def _launcher_busy(home: Path) -> bool:
    """This process is still waiting on a launcher it started for *home* (one per home; two requests that
    both get past this check are still decided by the engine's per-home run lock)."""
    proc = _LAUNCHERS.get(str(home))
    return proc is not None and proc.poll() is None


def _spawn(argv: list[str], target: Target, output):
    """``hermes [-p <profile>] <argv>`` for the target's home: ``-p`` pins the child's home even where a sticky
    ``hermes profile use`` choice would re-home a bare ``HERMES_HOME`` (the default home), and gives a named
    profile the scrubbed environment every other profile action gets."""
    from hermes_cli.web_server_gateway import _spawn_hermes_action
    selector = ["-p", target.profile] if target.profile else []
    proc = _spawn_hermes_action(selector + argv, ACTION_NAME, env_overrides={"HERMES_HOME": str(target.home)},
                                output=output)
    with _LAUNCHERS_LOCK:
        _LAUNCHERS[str(target.home)] = proc
    return proc


def _launcher_report(home: Path, output) -> Optional[dict]:
    """The launcher's own ``--json`` answer for *home*, from this launch's own output (a start names its
    run log; a refusal names none)."""
    try:
        output.seek(0)
        tail = output.read().decode("utf-8", "replace")
    except OSError:
        return None
    mine = str(_log_path(home))
    started = refused = None
    for line in tail.splitlines():
        if line.startswith("{"):
            try:
                data = json.loads(line)
            except ValueError:
                continue
            # The engine's ``manifest_unreadable`` refusal carries no ``mode``.
            if isinstance(data, dict) and ("mode" in data or data.get("error") == "manifest_unreadable"):
                if data.get("log") == mine:
                    started = data
                elif "log" not in data:
                    refused = data
    # A start line wins over a busy line in the same output; a refusal counts only without a start.
    return started or refused


@dataclass(frozen=True)
class Launch:
    """What the launcher said. ``started``: the background run holds this home's run lock. ``busy``:
    another run holds it (nothing changed). ``refused``: it was not started (nothing changed; ``message``
    says why). ``unsupported``: exit 6, the launcher's plan is manual-only (nothing started). ``unconfirmed``: no answer either way (exit 7, a timeout, or no confirmation): the run
    may or may not be going."""

    pid: int
    outcome: str
    exit_code: Optional[int]
    message: str = ""


def _launch(argv: list[str], target: Target) -> Launch:
    """Start the ``--detach`` launcher and wait for its answer: it exits once the background run reports
    whether it holds this home's run lock (exit 0 and ``confirmed: true``), or that another run holds it
    (exit 3).

    The race between two clicks is decided by that kernel lock in the background runs, not here, so two
    requests (to this process or to another server of this home) can never both start a migration."""
    import subprocess
    from hermes_cli.gateway_migrate_backends_lock import ACK_WAIT_SECONDS
    import tempfile
    # The launcher's output goes to an anonymous file only this request holds (no name in any home, where
    # the Files browser could read it; on Windows a delete-on-close file in the system temp directory).
    with tempfile.TemporaryFile(buffering=0) as output:
        proc = _spawn(argv, target, output)
        _forget_scan(target.home)
        try:
            code = proc.wait(timeout=ACK_WAIT_SECONDS + 30)
        except subprocess.TimeoutExpired:
            return Launch(proc.pid, "unconfirmed", None)
        report = _launcher_report(target.home, output) or {}
    if code == 6:  # the launcher's own plan is unsupported (manual steps only): it started nothing
        return Launch(proc.pid, "unsupported", code)
    if code == 3:  # nothing changed: another run holds the lock, or the run could not be started
        if report.get("outcome") == "refused" or report.get("error") == "manifest_unreadable":
            return Launch(proc.pid, "refused", code, str(report.get("message") or ""))
        return Launch(proc.pid, "busy", code)
    confirmed = code == 0 and report.get("confirmed") is True
    return Launch(proc.pid, "started" if confirmed else "unconfirmed", code)


def _launched(host, target: Target, launch: Launch, what: str, **started: Any) -> JSONResponse:
    """The answer for a launch: 200 started, 202 unconfirmed (look at the status: never "did not start"),
    409 another run holds the lock."""
    from hermes_cli.gateway_migrate_backends_lock import busy_message
    if launch.outcome == "busy":
        return _refuse(host, 409, "already_running", f"A {what} is already running for this home.",
                       detail=busy_message(Path(host.home)), status_command=target.command("--status"))
    if launch.outcome == "refused":
        return _refuse(host, 409, "not_started", launch.message or f"The {what} could not be started; nothing "
                       "changed.", status_command=target.command("--status"))
    confirmed = launch.outcome == "started"
    return _respond({"ok": True, "name": ACTION_NAME, "pid": launch.pid, "confirmed": confirmed,
                     "outcome": launch.outcome, "launcher_exit_code": launch.exit_code,
                     "status_command": target.command("--status"), "profile": target.profile, **started},
                    host, 200 if confirmed else 202)


# --------------------------------------------------------------------------- routes


@router.get("/api/gateway/migrate/backends/plan")
async def backend_migration_plan(address: Optional[str] = None, keep_webapp_address: bool = False,
                                 force_interrupt: bool = False, profile: Optional[str] = None):
    """What ``--apply --dry-run --json`` would do for the button's run (``--detach``), the read-only
    ``--backends`` report, and the last run's status. Changes nothing."""
    try:
        target = await _target(profile)
    except UnknownProfile as exc:
        return _unknown_profile(str(exc))
    host = _host(target.home)
    body = ApplyBody(address=address, keep_webapp_address=keep_webapp_address, force_interrupt=force_interrupt)
    plan, items, run = await _off_loop(target, lambda: _plan_read(host, body, target.profile))
    return _respond({"applied": False, "dry_run": True, "exit_code": plan.exit_code(),
                     "plan": _plan_view(plan, target.profile), "report": _report_view(items), "run": run,
                     "profile": target.profile, "commands": target.commands()}, host)


@router.post("/api/gateway/migrate/backends/apply")
async def backend_migration_apply(body: Optional[ApplyBody] = None, profile: Optional[str] = None):
    """Start ``hermes [-p <profile>] gateway migrate --backends --apply --yes --detach --json``; refused
    (nothing started) when the plan is blocked, unsupported or empty, differs from the confirmed one
    (``expect``), or a run is live or unfinished."""
    body = body or ApplyBody()
    try:
        target = await _target(profile)
    except UnknownProfile as exc:
        return _unknown_profile(str(exc))
    host = _host(target.home)
    if _launcher_busy(target.home):
        return _refuse(host, 409, "already_running", "A move is already starting.",
                       status_command=target.command("--status"))
    if (refusal := await _off_loop(target, lambda: _unfinished(host, target))) is not None:
        return refusal
    plan = await _off_loop(target, lambda: _build_plan(host, body))
    if plan.nothing_to_do:
        return _refuse(host, 409, "nothing_to_do", "No older Hermes server serves this home: nothing to move.")
    view = _plan_view(plan, target.profile)
    from hermes_cli.gateway_migrate_backends_apply import plan_token
    if body.expect is not None and plan_token(plan) != body.expect:
        # The plan to confirm again is the whole fresh one: filtered to the confirmed address, a backend that
        # moved to another port would show only a blocker and nothing to move.
        fresh = plan if body.address is None else await _off_loop(
            target, lambda: _build_plan(host, body.model_copy(update={"address": None})))
        return _refuse(host, 409, "plan_changed", "What this server would move changed since it was shown. "
                       "Check the new details and confirm again.", plan=_plan_view(fresh, target.profile))
    if plan.unsupported:
        return _refuse(host, 409, "unsupported", "This server cannot be moved automatically.",
                       unsupported=view["unsupported"])
    if plan.blockers:
        return _refuse(host, 409, "blocked", "The move cannot start yet.", blockers=list(plan.blockers))
    requested_at = time.time()
    try:
        launch = await asyncio.to_thread(_launch, _apply_argv(body), target)
    except (OSError, RuntimeError) as exc:
        return _refuse(host, 500, "spawn_failed", f"Could not start the move: {exc}")
    if launch.outcome == "unsupported":
        # Exit 6: the backend became manual-only between this route's plan and the launcher's (nothing
        # started). The same answer as the pre-check above, its steps from a fresh plan (sanitized and
        # profile-qualified like every view; the launcher's raw plan JSON is not passed on).
        fresh = _plan_view(await _off_loop(target, lambda: _build_plan(host, body)), target.profile)
        return _refuse(host, 409, "unsupported", "This server cannot be moved automatically.",
                       unsupported=fresh["unsupported"], launcher_exit_code=launch.exit_code)
    return _launched(host, target, launch, "move", requested_at=requested_at,
                     old_address=view.get("old_address"), new_address=view.get("new_address"))


RESTORE_ARGV = ["gateway", "migrate", "--backends", "--restore", "--yes", "--detach", "--json"]


@router.post("/api/gateway/migrate/backends/restore")
async def backend_migration_restore(profile: Optional[str] = None):
    """Start ``hermes [-p <profile>] gateway migrate --backends --restore --yes --detach --json`` (undo a
    finished move).

    The restore stops the gateway, which on the Webapp's listener is the process answering this request:
    ``--detach`` runs it as its own transient unit / session, so it outlives that process."""
    from hermes_cli.gateway_migrate_backends_apply import read_manifest
    try:
        target = await _target(profile)
    except UnknownProfile as exc:
        return _unknown_profile(str(exc))
    host = _host(target.home)
    if _launcher_busy(target.home):
        return _refuse(host, 409, "already_running", "A move is already starting.",
                       status_command=target.command("--status"))
    if (refusal := await _off_loop(target, lambda: _unfinished(host, target))) is not None:
        return refusal
    m = await asyncio.to_thread(read_manifest, host.home)
    if m is None or m.get("state") != "done":
        return _refuse(host, 409, "nothing_to_restore", "There is no finished move to undo.")
    requested_at = time.time()
    try:
        launch = await asyncio.to_thread(_launch, list(RESTORE_ARGV), target)
    except (OSError, RuntimeError) as exc:
        return _refuse(host, 500, "spawn_failed", f"Could not start the undo: {exc}")
    return _launched(host, target, launch, "move or undo", requested_at=requested_at)


@router.get("/api/gateway/migrate/backends/status")
async def backend_migration_status(request: Request, profile: Optional[str] = None):
    """``--status --json`` (rule 13 shape) plus ``manual_steps`` (state ``failed``), ``alive`` and ``reason``,
    the run log's tail, the launcher this process started, and which server answered (``served_by``)."""
    try:
        target = await _target(profile)
    except UnknownProfile as exc:
        return _unknown_profile(str(exc))
    host = _host(target.home)
    return _respond(await _off_loop(target, lambda: _status_view(host, target, request)), host)
