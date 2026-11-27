"""Home scoping for ``hermes update``'s fleet restart (#93349).

The restart phase enumerates ``hermes-gateway*``/``hermes-serve*`` units, ``ai.hermes.gateway*``
LaunchAgents and every ``gateway run`` process on the host. Those are HOST-wide namespaces: a
second Hermes install (another ``HERMES_HOME`` root under the same account, its own checkout and
venv) shares them, and the update used to restart that install's gateway too — including the
account's real ``hermes-gateway.service`` when a scratch home ran ``hermes update``.

The fleet an update owns is the set of homes its plan inventories: the updating root plus every
``<root>/profiles/<name>``. Ownership is judged from what a runtime actually runs on — the live
process environment (``HERMES_HOME``/``HOME``, via ``_hermes_home_for_pid``), the unit's declared
``Environment=`` or the plist's pinned ``HERMES_HOME`` — never from a unit label or an argv
substring. Unknown ownership is left alone: a restart we cannot prove is ours is somebody else's
outage.

A runtime is also owned only when it runs the CHECKOUT this update is updating: a worktree's
update shares the account's homes and units, and used to repoint the live unit at the worktree's
launcher and restart the gateway another checkout serves. A runtime whose code root is not
provably this checkout is left alone; a shared home is not proof of code ownership.
"""

from __future__ import annotations

import plistlib
import re
import shlex
from contextlib import suppress
from pathlib import Path


def _resolved(path) -> Path | None:
    try:
        return Path(str(path)).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def update_scope_homes() -> set[Path]:
    """Resolved homes the running update owns: the invoking home, the install root and its profiles.

    A home NESTED inside an inventoried home (a scratch ``HERMES_HOME`` under ``~/.hermes/cache``
    resolves the account's root) is not that install: it owns only itself."""
    invoking: Path | None = None
    with suppress(Exception):
        from hermes_constants import get_hermes_home
        invoking = _resolved(get_hermes_home())
    inventory: set[Path] = set()
    with suppress(Exception):
        from hermes_cli.update_receipt import _profile_homes
        inventory = {resolved for _profile, home in _profile_homes() if (resolved := _resolved(home)) is not None}
    if invoking is None:
        return inventory
    if invoking not in inventory and any(invoking.is_relative_to(home) for home in inventory):
        return {invoking}
    return {invoking, *inventory}


def home_in_update_scope(home, scope: set[Path] | None = None) -> bool:
    """True when *home* (a path or string) is one of the homes this update owns."""
    if not home:
        return False
    resolved = _resolved(home)
    if resolved is None:
        return False
    return resolved in (update_scope_homes() if scope is None else scope)


def _ledger_gateway_home(pid: int) -> str | None:
    """Home this install's spawn ledger records for gateway *pid* (``register_self`` at gateway boot),
    only for an exact live ``(pid, create_time)`` match. Proof for a gateway whose environment is
    unreadable (elevated, or another account's service) while its create time still is."""
    with suppress(Exception):
        from hermes_cli.process_identity import ledger_entries
        for entry in ledger_entries(verified_only=True):
            if entry.get("pid") == pid and entry.get("purpose") == "gateway":
                return entry.get("hermes_home") or None
    return None


def gateway_pid_home(pid: int) -> str | None:
    """The home gateway *pid* runs on: its live environment, else the exact-birth ledger witness;
    ``None`` when neither proves one. Discovery, the Windows pause record and its readiness check
    share it, so the home discovery proved is the home the restart debt keeps."""
    from hermes_cli.dashboard_procs import _hermes_home_for_pid
    try:
        home = _hermes_home_for_pid(pid)
    except Exception:  # health: allow BLE001 -- tri-state resolver: any environment-read failure is "unknown" and the ledger witness decides
        home = None
    return home or _ledger_gateway_home(pid)


def gateway_pid_in_update_scope(pid: int, scope: set[Path] | None = None) -> bool | None:
    """Does gateway *pid* run on a home this update owns? ``None`` when its home cannot be read."""
    home = gateway_pid_home(pid)
    if home is None:
        return None
    return home_in_update_scope(home, scope)


def partition_gateway_pids_by_scope(pids, scope: set[Path] | None = None) -> tuple[list[int], list[tuple[int, str | None]]]:
    """``(owned, foreign)`` split of *pids*; ``foreign`` pairs each PID with its home (None = unreadable)."""
    scope = update_scope_homes() if scope is None else scope
    owned: list[int] = []
    foreign: list[tuple[int, str | None]] = []
    for pid in pids:
        verdict = gateway_pid_in_update_scope(pid, scope)
        if verdict:
            owned.append(pid)
        else:
            home = None
            if verdict is False:
                with suppress(Exception):
                    from hermes_cli.dashboard_procs import _hermes_home_for_pid
                    home = _hermes_home_for_pid(pid)
            foreign.append((pid, home))
    return owned, foreign


def systemd_unit_hermes_home(scope_cmd: list, svc_name: str) -> str | None:
    """Home the systemd unit *svc_name* runs on: its live MainPID's environment first, then the
    unit's declared ``Environment=HERMES_HOME``; for a user-scope unit that declares none, the
    user's own default home. ``None`` when nothing readable names a home."""
    from hermes_cli.update_cmd_fleet import _systemctl, _unit_main_pid

    pid = _unit_main_pid(scope_cmd, svc_name)
    if pid > 0:
        with suppress(Exception):
            from hermes_cli.dashboard_procs import _hermes_home_for_pid
            if (home := _hermes_home_for_pid(pid)) is not None:
                return home
    try:
        shown = _systemctl(list(scope_cmd) + ["show", svc_name, "--property=Environment", "--value"], timeout=10)
    except Exception:
        return None
    if getattr(shown, "returncode", 1) != 0:
        return None
    env_line = (getattr(shown, "stdout", "") or "").strip()
    try:
        tokens = shlex.split(env_line)
    except ValueError:
        tokens = env_line.split()
    for token in tokens:
        key, sep, value = token.partition("=")
        if sep and key == "HERMES_HOME" and value.strip():
            return value.strip()
    if "--user" in scope_cmd:
        return str(Path.home() / ".hermes")
    return None


def systemd_unit_in_update_scope(scope_cmd: list, svc_name: str, scope: set[Path] | None = None) -> bool | None:
    """Does unit *svc_name* belong to this update? ``None`` = ownership unreadable (leave it alone)."""
    home = systemd_unit_hermes_home(scope_cmd, svc_name)
    if home is None:
        return None
    return home_in_update_scope(home, scope)


def launchd_label_foreign_home(label: str, scope: set[Path] | None = None) -> str | None:
    """The HERMES_HOME a derived launchd label's installed plist pins when that home is NOT one of
    this update's — labels are account-global, so root B's default profile derives the same bare
    ``ai.hermes.gateway`` root A installed. ``None`` = ours, or no/unreadable plist (the locate step
    decides whether a job exists; only a proven foreign home is refused)."""
    pinned = launchd_label_hermes_home(label)
    return None if pinned is None or home_in_update_scope(pinned, scope) else pinned


def _launchd_label_details(label: str) -> tuple[str | None, set[Path]]:
    """The pinned home and launched checkout from a label's installed plist."""
    try:
        from hermes_cli.gateway import get_launchd_plist_path
        path = get_launchd_plist_path().with_name(f"{label}.plist")
        data = plistlib.loads(path.read_bytes())
        env = data.get("EnvironmentVariables")
        home = env.get("HERMES_HOME") if isinstance(env, dict) else None
        return (str(home) if home else None), definition_checkout_roots(path)
    except (OSError, ValueError, KeyError, TypeError, plistlib.InvalidFileException):
        return None, set()


def launchd_label_hermes_home(label: str) -> str | None:
    """The pinned Hermes home for a loaded launchd label, or None when it is unproven."""
    return _launchd_label_details(label)[0]


def launchd_label_checkout_roots(label: str) -> set[Path]:
    """The checkout(s) a launchd label's installed plist launches."""
    return _launchd_label_details(label)[1]


# A path inside an argument that is code (``python -I -c "...sys.path.insert(0, '<root>')..."``):
# a quoted literal first, so a checkout path with spaces is read whole; bare tokens as a fallback.
_EMBEDDED_PATH = re.compile(r"""'(/[^']+)'|"(/[^"]+)"|(/[^\s'"]+)""")


def _checkouts_named(strings) -> set[Path]:
    """Every Hermes checkout an argv names: as an argument, or as a path inside one (the
    ``python -I -c "...sys.path.insert(0, '<root>')..."`` launcher form)."""
    from hermes_cli.update_receipt import _code_root_for_path
    roots: set[Path] = set()
    for raw in strings:
        embedded = (next(group for group in match if group) for match in _EMBEDDED_PATH.findall(str(raw)))
        for candidate in (str(raw), *embedded):
            if (root := _code_root_for_path(candidate)) is not None:
                roots.add(root)
    return roots


def _source_checkouts(argv, interpreter=()) -> set[Path]:
    """The checkout(s) whose CODE an argv runs.

    The interpreter (a ``python*`` argv[0], ``VIRTUAL_ENV``) is weaker evidence than the source it
    runs: a worktree started through the runtime checkout's venv names both, and runs the worktree.
    It decides only when the argv names no source checkout at all."""
    argv = [str(arg) for arg in argv]
    head = argv[:1] if argv and Path(argv[0]).name.startswith("python") else []
    return _checkouts_named(argv[len(head):]) or _checkouts_named([*head, *interpreter])


def _foreign_of(roots: set[Path]) -> str | None:
    """A checkout from *roots* when none of them is the one running this update, else None."""
    from hermes_cli.update_receipt import _updater_code_root
    mine = _updater_code_root()
    if mine is None or not roots or mine in roots:
        return None
    return str(min(roots))


def gateway_pid_foreign_checkout(pid: int) -> str | None:
    """The other checkout gateway *pid* runs, or None (this update's, or unreadable)."""
    return _foreign_of(gateway_pid_checkout_roots(pid))


def gateway_pid_checkout_roots(pid: int) -> set[Path]:
    """The Hermes checkout(s) whose code a live gateway process runs, or an empty set if unknown."""
    argv: list[str] = []
    interpreter: list[str] = []
    with suppress(Exception):
        import psutil  # type: ignore
        process = psutil.Process(pid)
        with suppress(Exception):
            interpreter.append(process.environ().get("VIRTUAL_ENV") or "")
        with suppress(Exception):
            argv.extend(process.cmdline() or [])
    if (home := gateway_pid_home(pid)) is not None:
        with suppress(Exception):
            from gateway.status import read_runtime_status
            record = read_runtime_status(Path(home) / "gateway_state.json") or {}
            if record.get("pid") == pid:
                # sys.argv: its first item is the resolved module path, i.e. source evidence.
                argv.extend(str(arg) for arg in record.get("argv") or [])
    return _source_checkouts(argv, interpreter)


def _definition_command(path: Path) -> list[str]:
    if path.suffix == ".plist":
        data = plistlib.loads(path.read_bytes())
        args = data.get("ProgramArguments")
        command = [str(arg) for arg in args] if isinstance(args, list) else []
        if working_directory := data.get("WorkingDirectory"):
            command.append(str(working_directory))
        return command
    command: list[str] = []
    working_directory = None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("ExecStart=") and not command:
            command.extend(shlex.split(line[len("ExecStart="):].lstrip("-@:+!")))
        elif line.startswith("WorkingDirectory="):
            working_directory = line[len("WorkingDirectory="):].strip()
    if working_directory:
        command.append(working_directory)
    return command


def definition_foreign_checkout(path) -> str | None:
    """The other checkout a systemd unit or launchd plist at *path* launches, or None (this
    update's, absent, unreadable, or a command naming no checkout)."""
    return _foreign_of(definition_checkout_roots(path))


def definition_checkout_roots(path) -> set[Path]:
    """The Hermes checkout(s) a systemd unit or launchd plist at *path* launches."""
    try:
        argv = _definition_command(Path(path))
    except (OSError, ValueError, plistlib.InvalidFileException):
        return set()
    return _source_checkouts(argv)


def systemd_unit_checkout_roots(scope_cmd: list, svc_name: str) -> set[Path]:
    """Both the live process and the effective (including drop-ins) launch command must agree."""
    from hermes_cli.gateway_migrate_backends_host import exec_start_argv
    from hermes_cli.update_cmd_fleet import _systemctl, _unit_main_pid

    pid = _unit_main_pid(scope_cmd, svc_name)
    live_roots = gateway_pid_checkout_roots(pid) if pid > 0 else None
    try:
        shown = _systemctl(list(scope_cmd) + ["show", svc_name, "--property=ExecStart", "--value"], timeout=10)
    except Exception:  # health: allow BLE001 -- an unreadable effective definition cannot confer ownership
        return set()
    value = (shown.stdout or "").strip() if shown.returncode == 0 else ""
    if not value.startswith("{ path=") or value.count("argv[]=") != 1:
        return set()
    executable = value.removeprefix("{ path=").split(" ;", 1)[0]
    # The separate path field preserves a launcher path containing spaces. argv supplies the
    # script path when the executable is Python rather than a launcher belonging to the checkout.
    definition_roots = _source_checkouts([executable, *exec_start_argv(value)[1:]])
    if live_roots is None:
        return definition_roots
    roots = live_roots | definition_roots
    if live_roots and definition_roots:
        return roots
    from hermes_cli.update_receipt import _updater_code_root
    # A known foreign checkout vetoes even when the other witness is unavailable.
    return roots - {_updater_code_root()}


def systemd_unit_foreign_checkout(scope_cmd: list, svc_name: str) -> str | None:
    """The other checkout unit *svc_name* runs (its MainPID) or would start (its unit file)."""
    return _foreign_of(systemd_unit_checkout_roots(scope_cmd, svc_name))


def launchd_label_skip_notice(label: str, scope: set[Path] | None = None) -> str | None:
    """The left-alone notice for launchd job *label* when this update does not own it (its plist pins
    another home, or launches another checkout), else None. The one ownership decision every launchd
    restart and pause path shares."""
    home = launchd_label_hermes_home(label)
    if home is None:
        return describe_skipped_runtime("launchd job", label, None)
    if not home_in_update_scope(home, scope):
        return describe_skipped_runtime("launchd job", label, home)
    from hermes_cli.update_receipt import _updater_code_root
    mine = _resolved(_updater_code_root())
    roots = launchd_label_checkout_roots(label)
    if mine is None or roots != {mine}:
        foreign = sorted(roots - ({mine} if mine is not None else set()))
        return describe_skipped_runtime("launchd job", label, None,
                                        checkout=str(foreign[0]) if foreign else None)
    return None


def describe_skipped_runtime(kind: str, name: str, home: str | None, *, checkout: str | None = None) -> str:
    """One notice line for a runtime the update leaves alone (foreign home or checkout, or
    unreadable ownership)."""
    if checkout is not None:
        return f"  ↷ {name}: {kind} of another Hermes checkout ({checkout}) — left alone (not restarted)"
    if home is None:
        return f"  ↷ {name}: {kind} ownership could not be proved — left alone (not restarted)"
    return f"  ↷ {name}: {kind} of another Hermes home ({home}) — left alone (not restarted)"


def announce_skipped_runtime(kind: str, name: str, home: str | None, *, checkout: str | None = None) -> None:
    """Print :func:`describe_skipped_runtime` once per update run, and name the runtime on its receipt.

    The pre-swap pause and the post-swap restart phase each scope the same host-wide gateway list,
    and the restart phase runs in a fresh completion process, so only the run's receipt (handed to
    that process) knows what the pause already announced. No open receipt: every call prints.
    """
    from hermes_cli import update_receipt
    entry = [kind, name, checkout or home]
    current = update_receipt._current.get()
    announced = list((current.data if current is not None else {}).get("runtimes_left_alone") or [])
    if entry in announced:
        return
    print(describe_skipped_runtime(kind, name, home, checkout=checkout))
    update_receipt.record_fact("runtimes_left_alone", [*announced, entry])
