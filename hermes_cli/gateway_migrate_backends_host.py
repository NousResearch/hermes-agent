"""Host operations for ``hermes gateway migrate --backends --apply/--restore``.

The plan (:mod:`hermes_cli.gateway_migrate_backends_plan`) and the engine
(:mod:`hermes_cli.gateway_migrate_backends_apply`) reach the machine only through :class:`SystemHost`:
reads of the old backend (``collect_backends``, ``systemctl show``, ``/proc``, a launchd plist), the
home's config and ``.env``, the gateway's user service, lingering, the Webapp bundle and
lightweight health and interactive-provider reads on the old address; and the changes the engine makes (config/``.env`` writes, the gateway service, the
old backend's supervisor, a detached run). This module owns no decisions.

Process identity comes from Hermes' own detection (``gateway_migrate_backends.collect_backends``,
``gateway.status``, ``dashboard_procs``), never from text matching on ``ps`` output.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import stat
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

_ENV_FILE_ENTRY = re.compile(r"(\S+) \(ignore_errors=(yes|no)\)")
logger = logging.getLogger(__name__)

_SHOW_PROPS = ("MainPID", "ActiveState", "UnitFileState", "FragmentPath", "ExecStart", "ControlGroup",
               "Environment", "EnvironmentFiles", "User")
_TIMEOUT = 30
#: The transient unit a ``--detach`` run is (``<prefix><seconds>.service``): what it starts dies with it.
DETACHED_UNIT_PREFIX = "hermes-backend-migration-"
HEALTH_TIMEOUT = 3  # seconds for the by-hand plan's one GET /api/health


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # a 3xx is an answer about another URL: the health read stops there


def _systemctl(scope: str, args: list[str], *, timeout: int = _TIMEOUT) -> subprocess.CompletedProcess:
    """``systemctl [--user] <args>`` through the gateway's own wrapper (user bus env included)."""
    from hermes_cli import gateway as gw
    return gw._run_systemctl(args, system=scope != "user", check=False, timeout=timeout,
                             capture_output=True, text=True, encoding="utf-8", errors="replace")


def _launchctl(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=_TIMEOUT, check=False)


def plist_environment(path: str) -> dict[str, str]:
    """``EnvironmentVariables`` of the launchd plist at *path* (values stay in memory); {} when unreadable."""
    import plistlib
    from xml.parsers.expat import ExpatError
    try:
        with open(path, "rb") as fh:
            data = plistlib.load(fh)
    except (OSError, ValueError, plistlib.InvalidFileException, ExpatError):
        return {}
    env = data.get("EnvironmentVariables") if isinstance(data, dict) else None
    return {str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {}


def parse_show(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in (stdout or "").splitlines() if "=" in line)


def exec_start_argv(value: str) -> list[str]:
    """argv of a ``systemctl show -p ExecStart`` value (``{ path=... ; argv[]=a b c ; ... }``)."""
    marker = "argv[]="
    if marker not in value:
        return []
    tail = value.split(marker, 1)[1].split(" ;", 1)[0]
    try:
        return shlex.split(tail)
    except ValueError:
        return tail.split()


def _dotenv_file(path: str) -> dict[str, str]:
    from agent.secret_scope import load_env_file
    with contextlib.suppress(Exception):
        return dict(load_env_file(Path(path)))
    return {}


def _env_tokens(environment: str) -> dict[str, str]:
    try:
        tokens = shlex.split(environment)
    except ValueError:
        tokens = environment.split()
    return dict(t.split("=", 1) for t in tokens if "=" in t)


_HOME_KEYS = ("HERMES_HOME", "HOME", "HERMES_DATA_DIR_SUFFIX")  # the variables that decide a unit's home
_HOME_LINE = re.compile(r"^\s*(export\s+)?(" + "|".join(_HOME_KEYS) + r")\s*=(.*)$")


def _alive_markers(path, pid: int) -> Optional[int]:
    """Live turn markers *pid* wrote in the marker file *path* (0 when it does not exist), or None when
    that cannot be told: the file cannot be read or parsed, is not a JSON object, or one of the markers
    it could be from has no writer whose state can be checked."""
    from tui_gateway import turn_marker
    try:
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return 0
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    count = 0
    for entry in data.values():
        if not isinstance(entry, dict):
            return None
        writer_pid = entry.get("writer_pid")
        if isinstance(writer_pid, int) and not isinstance(writer_pid, bool) and writer_pid > 0 and writer_pid != pid:
            continue
        state = turn_marker.marker_writer_state(entry)
        if state == "unknown":
            return None
        count += state == "alive"
    return count


def _unit_file_syntax(path: str) -> Optional[str]:
    """Why *path* cannot be read the way systemd reads it, or None. systemd's EnvironmentFile= rules
    (systemd.exec(5)) differ from a Hermes ``.env`` for continuation lines, backslashes, quotes and a
    ``#`` after the value, so a HERMES_HOME written with any of those is not guessed."""
    with contextlib.suppress(OSError, UnicodeDecodeError):
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
        if any(line.rstrip().endswith("\\") for line in lines):
            return "continues a line with a backslash"
        for line in lines:
            match = _HOME_LINE.match(line)
            if match and match.group(1):  # systemd does not strip `export`: the variable is not set for it
                return f"writes `export {match.group(2)}=`, which systemd does not read as that variable"
            if match and any(c in match.group(3) for c in "#\\\"'$"):
                return (f"sets {match.group(2)} with quotes, a backslash, a $ or a # that systemd reads "
                        f"differently")
        return None
    return "cannot be read by this user"


def run_quietly(fn):
    """Call *fn* with everything it prints captured: Python output and its children's (fds 1 and 2).

    The captured text is shown only when *fn* fails, before its exception propagates, so a step such
    as ``hermes gateway install`` cannot fill the migration's output with its own "Next steps"."""
    import io
    import sys
    import tempfile
    sys.stdout.flush()
    sys.stderr.flush()
    py = io.StringIO()
    with tempfile.TemporaryFile() as raw:
        saved = (os.dup(1), os.dup(2))
        ok = False
        try:
            os.dup2(raw.fileno(), 1)
            os.dup2(raw.fileno(), 2)
            with contextlib.redirect_stdout(py), contextlib.redirect_stderr(py):
                result = fn()
            ok = True
            return result
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, copy in zip((1, 2), saved):
                os.dup2(copy, fd)
                os.close(copy)
            if not ok:
                raw.seek(0)
                text = py.getvalue() + raw.read().decode("utf-8", "replace")
                if text.strip():
                    print(text.rstrip("\n"))


def profile_args(home) -> list[str]:
    """``["-p", NAME]`` for the profile that owns *home* (``default`` for the root home), [] for a home
    that is no profile. Every re-launch and printed command carries it: without it a ``hermes`` started
    with the root home re-reads the sticky ``hermes profile use`` choice and acts on another profile."""
    from hermes_constants import profile_name_for_home
    name = profile_name_for_home(home) if home else None
    return ["-p", name] if name else []


def _with_profile(argv: list[str], home: Path) -> list[str]:
    """*argv* of a ``hermes dashboard``/``serve`` with this home's ``-p NAME`` before the subcommand
    (unless it names a profile already): replayed without it, a sticky ``hermes profile use`` choice
    would start it for another profile."""
    from hermes_cli._parser import scan_profile_flag
    extra = profile_args(home)
    if not extra or scan_profile_flag(argv[1:])[0] is not None:
        return argv
    for i, word in enumerate(argv):
        if i and word in ("dashboard", "serve"):
            return [*argv[:i], *extra, *argv[i:]]
    return argv


class FileChangedSinceBackup(RuntimeError):
    """``config.yaml`` / ``.env`` no longer holds the bytes the migration backed up."""


@contextlib.contextmanager
def _staged_copy(path: Path, expect: Optional[str], *, private: bool = False):
    """A private (0600) copy of *path* beside it, yielded only when its bytes hash to *expect* (None: the
    file must not exist), then given *path*'s own mode unless *private* (a copy that receives secrets stays
    0600 until it is renamed). Removed again unless :func:`_publish` renamed it over *path*."""
    from hermes_cli.gateway_migrate_backends_apply import write_private_file
    staged = path.with_name(f".{path.name}.{os.getpid()}.migrate")
    try:
        data: Optional[bytes] = path.read_bytes()
    except FileNotFoundError:
        data = None
    if (hashlib.sha256(data).hexdigest() if data is not None else None) != expect:
        raise FileChangedSinceBackup(f"{path} changed while the migration was running (after its backup was "
                                     f"taken), so the migration did not write it and your edit was kept")
    staged.unlink(missing_ok=True)
    if data is not None:
        write_private_file(staged, data)
    if data is not None and not private:
        with contextlib.suppress(OSError):
            os.chmod(staged, stat.S_IMODE(os.stat(path).st_mode))
    try:
        yield staged
    finally:
        staged.unlink(missing_ok=True)


def _publish(staged: Path, path: Path) -> bytes:
    """Rename *staged* over *path*; returns the bytes published (read from the private copy before)."""
    from utils.files import fsync_directory
    data = staged.read_bytes()
    os.replace(staged, path)
    fsync_directory(path.parent)
    return data


class SystemHost:
    """The real machine, for the home this process runs in."""

    def __init__(self, home: Path):
        self.home = Path(home)

    # ------------------------------------------------------------------ discovery

    def backends(self):
        from hermes_cli.gateway_migrate_backends import collect_backends
        items = collect_backends()
        self._mark_respawned_backend(items)
        return items

    def _mark_respawned_backend(self, items) -> None:
        """The hand-started backend a rollback or --restore started again is this home's, hand-started.

        It was started in a new session by a process that has exited, so on macOS its parent is launchd
        (ppid 1) and the scan cannot name a supervisor ("unknown", which --apply refuses with exit 6).
        Only the process whose pid *and* start time match the manifest's recorded replacement is renamed,
        only when the manifest records this home (a ``profile create --clone-all`` copy names its source),
        and never one the scan found serving another home."""
        from hermes_cli.gateway_migrate_backends_apply import read_manifest
        try:
            m = read_manifest(self.home) or {}
        except Exception:  # an unreadable manifest is reported by the engine itself
            logger.debug("manifest unreadable while marking a respawned backend", exc_info=True)
            return
        again = m.get("old_backend_respawned") or {}
        if not again.get("pid") or again.get("start_time") is None or not m.get("home"):
            return
        if not self.same_home(m["home"]):
            return
        for item in items:
            if (item.supervisor == "unknown" and item.this_home is not False and item.pid == again["pid"]
                    and self.start_time(item.pid) == again["start_time"]):
                item.supervisor = "manual"
                item.this_home = True

    def os_family(self) -> str:
        from hermes_platform.host.facts import os_family
        return os_family()

    def username(self) -> str:
        import getpass
        return getpass.getuser()

    def same_home(self, path: str) -> bool:
        from hermes_cli.dashboard_procs import _normalized_home_for_compare
        return _normalized_home_for_compare(path) == _normalized_home_for_compare(str(self.home))

    def gateway_owner(self) -> tuple[str, Optional[str]]:
        """Which gateway serves this home, answered by the gateway's own ownership checks:
        ``("own", None)`` for the default home, a profile with its own installed service, or a profile
        whose own process hosts the multiplexer; ``("other", description)`` when the live host gateway
        of another home serves it; ``("unknown", None)`` otherwise. Nothing is guessed."""
        from hermes_constants import profile_name_for_home
        name = profile_name_for_home(self.home)
        if name in (None, "default"):
            return "own", None
        from hermes_cli import gateway as gw
        try:
            if gw._is_service_installed():
                return "own", None
            other = gw._served_by_another_host_gateway(name)
            if other is not None:
                return "other", other.describe()
            if gw.host_multiplexer_serving(name) is not None:
                return "own", None  # the live host gateway is this profile's own process
        except Exception:  # an unreadable host record or service state: the owner is not known
            import logging
            logging.getLogger(__name__).debug("gateway ownership could not be read", exc_info=True)
            return "unknown", None
        from hermes_cli.profiles import profile_is_standalone
        return ("own", None) if profile_is_standalone(self.home) else ("unknown", None)

    def default_home(self, env: Optional[dict] = None) -> Optional[bool]:
        """Whether this home is the default home of a process with *env* running as the caller: the
        platform default (``hermes_constants._platform_default_hermes_home``) of *env*'s own ``HOME`` and
        ``HERMES_DATA_DIR_SUFFIX``, its ``HOME`` defaulting to the account's passwd home, never this
        shell's HOME or HERMES_HOME root (a unit started by systemd does not inherit them). None when
        *env*'s ``HOME`` is not a plain absolute path, so the default cannot be told."""
        from hermes_constants import _get_platform_default_hermes_home, _platform_default_hermes_home
        env = env or {}
        account = None
        with contextlib.suppress(ImportError, KeyError):
            import pwd
            account = Path(pwd.getpwnam(self.username()).pw_dir)
        home = env.get("HOME")
        if home is not None and (not home.startswith("/") or any(c in home for c in "$~")):
            return None
        if home is None and account is None:
            return self.same_home(str(_get_platform_default_hermes_home()))
        user_home = Path(home) if home else account
        return self.same_home(str(_platform_default_hermes_home(env, lambda: user_home)))

    def unit_props(self, unit: str, scope: str) -> dict[str, str]:
        res = _systemctl(scope, ["show", unit, "--no-pager", *(f"--property={p}" for p in _SHOW_PROPS)])
        return parse_show(res.stdout) if res.returncode == 0 else {}

    def unit_environment(self, unit: str, scope: str) -> tuple[dict[str, str], Optional[str]]:
        """The environment systemd gives *unit*, and why it is incomplete (None when complete). systemd
        applies ``Environment=`` first, then every ``EnvironmentFile=`` in order, a later value winning
        (systemd.exec(5)); ``systemctl show`` prints one ``EnvironmentFiles=`` line per file. A required
        file that is missing, or any file this user cannot read, makes the environment unknown."""
        res = _systemctl(scope, ["show", unit, "--no-pager", "--property=Environment", "--property=EnvironmentFiles"])
        if res.returncode != 0:
            return {}, f"systemctl show {unit} failed"
        env: dict[str, str] = {}
        manager = _systemctl(scope, ["show-environment"])  # the manager's own environment comes first
        if manager.returncode == 0:
            env.update(_env_tokens(" ".join(shlex.quote(line) for line in (manager.stdout or "").splitlines())))
        files: list[tuple[str, bool]] = []
        for line in (res.stdout or "").splitlines():
            key, _, value = line.partition("=")
            if key == "Environment":
                env.update(_env_tokens(value))
            elif key == "EnvironmentFiles":
                files += [(path, flag == "yes") for path, flag in _ENV_FILE_ENTRY.findall(value)]
        for path, optional in files:
            if any(c in path for c in "*?["):  # systemd expands it; which files it matches is not read here
                return env, f"its EnvironmentFile={path} is a pattern, so the files that set its HERMES_HOME are unknown"
            try:
                Path(path).read_bytes()
            except FileNotFoundError:
                if optional:
                    continue
                return env, f"its EnvironmentFile={path} does not exist"
            except OSError:
                return env, f"its EnvironmentFile={path} cannot be read by this user"
            if unreadable := _unit_file_syntax(path):
                return env, f"its EnvironmentFile={path} {unreadable}"
            env.update(_dotenv_file(path))
        return env, None

    def unit_launcher_env(self, unit: str, scope: str) -> dict[str, str]:
        """:meth:`unit_environment` without the reason (values stay in memory)."""
        return self.unit_environment(unit, scope)[0]

    def process_env(self, pid: int) -> Optional[dict[str, str]]:
        from hermes_cli.dashboard_procs import _pid_environ
        return _pid_environ(pid)

    def process_cwd(self, pid: int) -> Optional[str]:
        """The working directory of *pid* (``/proc`` on Linux, psutil elsewhere: macOS, Windows), else None.
        A hand-started ``python -m hermes_cli.main`` only finds its package from there."""
        with contextlib.suppress(OSError):
            return os.readlink(f"/proc/{pid}/cwd")
        import psutil
        with contextlib.suppress(psutil.Error, OSError):
            return psutil.Process(pid).cwd() or None
        return None

    def same_network_namespace(self, pid: int, *, container: bool) -> bool:
        """Whether a request to an address on this machine reaches *pid*'s listener: on Linux its network
        namespace is the planner's own (``/proc/PID/ns/net``); a pid that is not visible or a link that
        cannot be read counts as another namespace. Elsewhere there are no per-process network
        namespaces: a container is not local, any other pid this machine can see is."""
        import sys
        if sys.platform.startswith("linux"):
            try:
                return os.readlink(f"/proc/{pid}/ns/net") == os.readlink("/proc/self/ns/net")
            except OSError:
                return False
        return not container and self.start_time(pid) is not None

    def start_time(self, pid: int) -> Optional[int]:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)

    def same_process(self, pid: Optional[int], start_time) -> bool:
        """*pid* is alive and is still the process recorded with *start_time* (reuse guard). Without a
        recorded start time nothing proves it, so the answer is False."""
        if not pid or start_time is None:
            return False
        now = self.start_time(int(pid))
        return now is not None and now == start_time

    def running_turns(self, pid: int, start_time) -> Optional[int]:
        """Turns the old backend *pid* is running now: its live turn markers in this home, or None when
        activity cannot be determined (an existing marker file cannot be read or parsed, is not a JSON
        object, or a marker's writer state is unknown; :meth:`unknown_turn_files` names those files).

        The old backend writes a durable marker at every turn start and clears it on any conclusion
        (``tui_gateway.turn_marker``), stamped with its own pid and start time, so a marker whose
        writer is that process and still alive is a turn in flight. No sign-in, no socket."""
        return self._turn_activity(pid)[0]

    def unknown_turn_files(self, pid: int, start_time) -> list[str]:
        """The marker files that make :meth:`running_turns` unknown for *pid*, sorted: each one that cannot
        be read or parsed, or holds a marker without a writer whose state can be checked. Paths only:
        a marker holds the turn's prompt."""
        return self._turn_activity(pid)[1]

    def _turn_activity(self, pid: int) -> tuple[Optional[int], list[str]]:
        from tui_gateway import turn_marker
        from hermes_cli.profiles import profiles_to_serve
        from hermes_constants import hermes_home_key
        try:
            profiles = profiles_to_serve(True, include_standalone=True, include_parked=True)
        except OSError:
            return None, []
        homes = {hermes_home_key(self.home), *(hermes_home_key(h) for _, h in profiles)}
        count, unknown = 0, []
        for h in homes:
            path = turn_marker._marker_path(h)
            alive = _alive_markers(path, pid)
            if alive is None:
                unknown.append(str(path))
            else:
                count += alive
        return (None if unknown else count), sorted(unknown)

    # ------------------------------------------------------------------ config / env

    def read_config(self) -> dict:
        from hermes_cli.config import read_user_config_raw
        return read_user_config_raw(self.home / "config.yaml") or {}

    def config_problem(self) -> Optional[str]:
        """Why config.yaml cannot be read as written, or None. The gateway falls back to its last good
        config then; the dry run does not guess what that is. The error's text is not repeated: a
        parser message quotes the offending line, which may hold a secret."""
        from ruamel.yaml.error import YAMLError
        try:
            self.read_config()
        except (OSError, ValueError, YAMLError) as exc:
            return f"{self.home / 'config.yaml'} cannot be read or parsed ({type(exc).__name__})"
        return None

    def dotenv_exists(self) -> bool:
        return (self.home / ".env").exists()

    def managed_env(self) -> dict[str, str]:
        """The administrator-managed ``.env`` the backend applies last (read only)."""
        from hermes_cli import managed_scope
        return dict(managed_scope.load_managed_env() or {})

    def dotenv(self) -> dict[str, str]:
        return _dotenv_file(str(self.home / ".env"))

    def effective_config(self) -> dict:
        """This home's config as the gateway sees it (defaults, ``${VAR}`` expansion, the managed
        overlay), built from the file as written: ``load_config``'s merge without its writes (it
        creates the home layout and backs up config.yaml on a cache miss)."""
        import copy
        from hermes_cli.config import (
            DEFAULT_CONFIG, _canonicalize_config, _deep_merge, _expand_env_vars, _merge_managed_overlay,
        )
        merged = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), self.read_config())
        return _merge_managed_overlay(_expand_env_vars(_canonicalize_config(merged)))[0]
    def set_config(self, values: dict, *, expect: object = False) -> Optional[bytes]:
        """Set *values* in ``config.yaml``; returns the exact bytes published (None on a non-file config
        backend, which keeps its own write path). With *expect* (the sha256 of the backup the migration took,
        None: there was no file) the file is copied to a private file beside it and compared with it first,
        the change is made on that copy, and the copy is renamed over ``config.yaml``: an edit made since
        the backup is never folded in (``FileChangedSinceBackup``)."""
        from hermes_cli.config import _set_nested, _write_user_config, require_readable_config_before_write
        from hermes_cli.config_backend import _FILE_BACKEND, drop_other_access, get_config_backend
        path = self.home / "config.yaml"
        if expect is False or get_config_backend() is not _FILE_BACKEND:
            cfg = require_readable_config_before_write(path)
            for key, value in values.items():
                _set_nested(cfg, key, value)
            _write_user_config(path, cfg)
            return None
        with _staged_copy(path, expect) as staged:
            cfg = require_readable_config_before_write(staged)
            for key, value in values.items():
                _set_nested(cfg, key, value)
            _write_user_config(staged, cfg)  # the writer keeps the copy's mode: the file's own (below)
            drop_other_access(staged)
            return _publish(staged, path)

    def set_env(self, values: dict[str, str], *, expect: object = False) -> Optional[bytes]:
        """Write sign-in keys into ``.env``, which is private (0600) before the first secret byte lands;
        returns the exact bytes published. With *expect* it works on a private copy checked against the
        backup first, as :meth:`set_config` does."""
        from hermes_cli.config import _env_write_lock, require_env_writable, save_env_value
        for key in values:
            require_env_writable(key, "set")
        env = self.home / ".env"
        if expect is False:
            if env.exists():
                os.chmod(env, 0o600)
            for key, value in values.items():
                save_env_value(key, value)
            os.chmod(env, 0o600)
            return None
        with _env_write_lock(env), _staged_copy(env, expect, private=True) as staged:  # Hermes' .env writers wait
            for key, value in values.items():
                save_env_value(key, value, env_path=staged)  # it keeps the copy's 0600
            os.chmod(staged, 0o600)  # a .env that did not exist: the writer's new file
            return _publish(staged, env)

    def webapp_settings(self):
        """The gateway's own ``webapp_settings`` judged on :meth:`effective_config`."""
        from gateway.webapp_listener import webapp_settings
        return webapp_settings(self.home, document=self.effective_config())

    def proposed_webapp_reason(self, host: str, port: int, public_url: Optional[str]) -> Optional[str]:
        """Why the gateway would keep the Webapp off with ``webapp.enabled: true`` and this *host*/*port*
        (and *public_url* when the move sets one) over the rest of the block as configured, or None."""
        import copy
        from gateway.webapp_listener import webapp_settings
        document = copy.deepcopy(self.effective_config())
        block = document.get("webapp") if isinstance(document.get("webapp"), dict) else {}
        document["webapp"] = {**block, "enabled": True, "host": host, "port": port,
                              **({"public_url": public_url} if public_url else {})}
        return webapp_settings(self.home, document=document).reason

    def accepted_webapp_host(self, host: str) -> Optional[str]:
        """The gateway's own ``webapp.host`` validation: the canonical host, or None when refused."""
        from gateway.webapp_listener import _configured_address
        return _configured_address({"host": host})[0]

    # ------------------------------------------------------------------ webapp bundle

    def webapp_dist(self) -> Path:
        from hermes_cli.config import get_project_root
        from hermes_cli.webapp import webapp_dist_dir
        return webapp_dist_dir(get_project_root())

    def bundle_missing(self) -> Optional[str]:
        from gateway.webapp_listener import bundle_missing_reason
        return bundle_missing_reason(self.webapp_dist())

    def webapp_page_problem(self, host: str, port: int) -> Optional[str]:
        """None when ``GET /`` on *host:port* serves this install's Webapp bundle; else why not.

        Without sign-in ``/`` must be the bundle's own index.html (it references the build's entry
        asset), never the "Hermes is unavailable" page. Behind sign-in ``/`` redirects to ``/login``,
        so the bundle is proven through its entry asset, which the gate serves publicly: it must come
        back byte for byte as built."""
        import hashlib
        import http.client
        import re
        if (reason := self.bundle_missing()) is not None:
            return f"the Webapp bundle cannot be served: {reason}"
        dist = self.webapp_dist()
        index = (dist / "index.html").read_text(encoding="utf-8-sig", errors="replace")
        found = re.search(r'(?:src|href)="\.?(/assets/[^"?#]+\.js)"', index)
        if found is None:
            return f"{dist / 'index.html'} references no /assets/*.js entry: not a Webapp build"
        asset = found.group(1)

        def get(path: str) -> tuple[int, dict, bytes]:
            conn = http.client.HTTPConnection(host, port, timeout=10)
            try:
                conn.request("GET", path, headers={"Accept": "text/html,*/*"})
                resp = conn.getresponse()
                return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
            finally:
                conn.close()

        from hermes_cli.url_utils import format_url_host
        where = f"http://{format_url_host(host)}:{port}"
        try:
            code, headers, body = get("/")
            if code == 200:
                return None if asset.encode() in body else f"GET {where}/ is not the Webapp page"
            if code not in (302, 303, 307) or "/login" not in headers.get("location", ""):
                return f"GET {where}/ answered {code}, not the Webapp page"
            code, _headers, body = get(asset)
        except (OSError, http.client.HTTPException) as exc:
            return f"GET {where}/ failed: {exc}"
        built = (dist / asset.lstrip("/")).read_bytes()
        if code != 200 or hashlib.sha256(body).digest() != hashlib.sha256(built).digest():
            return f"GET {where}{asset} answered {code} without this install's Webapp bundle"
        return None

    def build_bundle(self) -> None:
        from hermes_cli.config import get_project_root
        from hermes_cli.webapp import prepare_webapp_renderer
        prepare_webapp_renderer(get_project_root(), explicit=True)

    # ------------------------------------------------------------------ gateway service

    def gateway_kind(self) -> Optional[str]:
        from hermes_cli import gateway as gw
        if gw._running_under_s6():
            return None
        if gw.supports_systemd_services():
            return "systemd"
        return "launchd" if gw.is_macos() else None

    def gateway_state(self) -> dict:
        from gateway.status import live_gateway_pid_for_home
        from hermes_cli import gateway as gw
        kind = self.gateway_kind()
        state = {"kind": kind, "installed": False, "enabled": False, "active": False, "running_pid": None,
                 "unit": None}
        with contextlib.suppress(Exception):
            state["running_pid"] = live_gateway_pid_for_home(self.home)
        if kind == "systemd":
            name = gw.get_service_name()
            state["unit"] = name
            state["installed"] = gw.get_systemd_unit_path(system=False).exists()
            if state["installed"]:
                state["enabled"] = _systemctl("user", ["is-enabled", name]).stdout.strip() == "enabled"
                state["active"] = _systemctl("user", ["is-active", name]).stdout.strip() == "active"
        elif kind == "launchd":
            # A LaunchAgent plist is loaded at every login: installed is enabled. Active = launchd runs it.
            state["unit"] = gw.get_launchd_label()
            state["installed"] = state["enabled"] = gw.get_launchd_plist_path().exists()
            state["active"] = state["installed"] and gw._launchctl_supervised_pid(state["unit"]) is not None
        return state

    def foreign_gateway_home(self) -> Optional[str]:
        """The HERMES_HOME pinned by the gateway's unit or LaunchAgent when it belongs to ANOTHER home (two
        homes can share ``hermes-gateway-NAME``: ``~/.hermes/profiles/NAME`` and ``/srv/x/profiles/NAME``),
        else None. Driving it would restart, stop or remove that home's gateway."""
        from hermes_cli import gateway as gw
        from hermes_cli import gateway_service_owner as owner
        kind = self.gateway_kind()
        if kind == "systemd":
            return owner.foreign_systemd_unit_home(False)
        if kind == "launchd":
            pinned = owner.pinned_home(gw.get_launchd_plist_path()) if gw.get_launchd_plist_path().exists() else None
            if pinned and owner._resolve(pinned) != owner._resolve(gw.get_hermes_home()):
                return pinned
        return None

    def legacy_gateway_units(self) -> bool:
        """Whether a legacy ``hermes.service`` exists: ``hermes gateway install`` would stop, disable and
        delete it, which no rollback or --restore could bring back."""
        from hermes_cli import gateway as gw
        return self.gateway_kind() == "systemd" and gw.has_legacy_hermes_units()

    def _refuse_foreign_gateway(self, action: str) -> None:
        """Raise instead of driving another home's same-named gateway unit or LaunchAgent."""
        from hermes_cli import gateway_service_owner as owner
        try:
            if self.gateway_kind() == "launchd":
                run_quietly(lambda: owner.refuse_foreign_launchd_plist(action))
            else:
                run_quietly(lambda: owner.refuse_foreign_systemd_unit(action, False))
        except SystemExit:
            raise RuntimeError(f"refusing to {action} the gateway service: it belongs to another Hermes home "
                               f"({self.foreign_gateway_home()})") from None

    def gateway_preflight(self) -> Optional[str]:
        """Why the gateway's user service cannot be driven from here, else None."""
        from hermes_cli import gateway as gw
        if self.gateway_kind() != "systemd":
            return None
        try:
            gw._preflight_user_systemd(auto_enable_linger=False)
        except gw.UserSystemdUnavailableError as exc:
            return str(exc).splitlines()[0]
        return None

    def install_gateway(self) -> None:
        """``hermes gateway install`` for the user (unit + enable + lingering), non-interactive."""
        from hermes_cli import gateway as gw
        if self.gateway_kind() == "launchd":
            # Written, not loaded: the agent is RunAtLoad, so loading it here would start the gateway
            # beside the old backend. The start step loads it once the old backend is out.
            run_quietly(lambda: gw.launchd_install(start_now=False))
            return

        def install() -> None:
            try:
                gw.systemd_install(system=False, non_interactive=True)
            except SystemExit as exc:
                raise RuntimeError(f"hermes gateway install refused (exit {exc.code})") from None
        run_quietly(install)  # its own "Next steps" would contradict the migration's; shown on failure
        self._require(_systemctl("user", ["enable", gw.get_service_name()]), "enable the gateway service")

    def gateway_verb(self, verb: str) -> None:
        """``start`` / ``restart`` / ``stop`` / ``enable`` / ``disable`` on the gateway's user service."""
        from hermes_cli import gateway as gw
        self._refuse_foreign_gateway(verb)
        if self.gateway_kind() == "launchd":
            # launchd has no separate enablement (an installed agent loads at login), so only the
            # start/stop/restart verbs reach it; their own "✓ Service started" stays out of the report.
            run_quietly(lambda: gw._service_call("launchd", verb, False))
            return
        if verb == "stop":
            gw._mark_planned_stop()
        self._require(_systemctl("user", [verb, gw.get_service_name()], timeout=90), f"{verb} the gateway service")

    def uninstall_gateway(self) -> None:
        from hermes_cli import gateway as gw
        self._refuse_foreign_gateway("uninstall")
        if self.gateway_kind() == "launchd":
            run_quietly(gw.launchd_uninstall)
            return
        run_quietly(lambda: gw.systemd_uninstall(system=False))

    def gateway_listens(self, port: int) -> Optional[bool]:
        """Whether this home's gateway listens on *port*: True / False when its live pid's own sockets
        prove it, None otherwise. No pid is not proof of absence (the lookup also answers None for an
        unreadable or invalid identity record): the engine then needs a socket-level proof that nothing
        listens on the port at all, or keeps the old backend down."""
        import psutil
        from gateway.status import live_gateway_pid_for_home
        try:
            pid = live_gateway_pid_for_home(self.home)
        except Exception:  # an unreadable runtime record is "unknown", never "absent"
            logger.debug("gateway pid lookup failed", exc_info=True)
            return None
        if pid is None:
            return None
        try:
            conns = psutil.Process(pid).net_connections(kind="tcp")
        except (psutil.Error, OSError):
            logger.debug("gateway listener lookup failed", exc_info=True)
            return None
        return any(c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port for c in conns)

    def live_manual_backend(self, port: int) -> Optional[dict]:
        """``{"pid", "start_time"}`` of a hand-started old backend of this home that listens on *port*
        now (found by Hermes' own backend scan and the listener's pid), else None."""
        from hermes_cli.gateway_migrate_backends_plan import backend_address
        for item in self.backends():
            address = backend_address(list(item.argv))  # None: not a backend command line
            if (item.running and item.pid and item.supervisor in ("manual", "detached") and item.this_home
                    and address is not None and address[1] == port and self.pid_listens_on(item.pid, port)):
                return {"pid": item.pid, "start_time": self.start_time(item.pid)}
        return None

    def gateway_pid(self) -> Optional[int]:
        from gateway.status import live_gateway_pid_for_home
        with contextlib.suppress(Exception):
            return live_gateway_pid_for_home(self.home)
        return None

    def linger(self) -> Optional[bool]:
        from hermes_cli import gateway as gw
        return gw.get_systemd_linger_status()[0]

    def set_linger(self, enabled: bool) -> None:
        import getpass
        verb = "enable-linger" if enabled else "disable-linger"
        cmd = ["loginctl", verb, getpass.getuser()]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=_TIMEOUT, check=False)
        self._require(res, f"loginctl {verb}")

    # ------------------------------------------------------------------ old backend supervisor

    def old_stop(self, rec: dict) -> None:
        sup = rec["supervisor"]
        if sup == "systemd":
            self._require(_systemctl(rec["scope"], ["stop", rec["unit"]], timeout=90), f"stop {rec['unit']}")
        elif sup == "launchd":
            if self.launchd_loaded(rec):
                self._require(_launchctl(["bootout", f"{rec['scope']}/{rec['unit']}"]), f"bootout {rec['unit']}")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and self.launchd_loaded(rec):
                time.sleep(0.2)
        else:
            self._stop_manual(rec)

    # launchd: the job is unloaded (bootout) AND marked disabled in the user's launchd database, so
    # neither the next login nor a later `launchctl bootstrap` of the plist starts it again (KeepAlive
    # would otherwise restart the old command every 10 s while the gateway holds its port). The plist
    # is also renamed to *.disabled; restore and rollback undo both.

    def launchd_loaded(self, rec: dict) -> bool:
        from hermes_cli.gateway import _launchd_print_service_pid
        return _launchd_print_service_pid(rec["scope"], rec["unit"])[0]

    def _stop_manual(self, rec: dict) -> None:
        pid, started = rec.get("pid"), rec.get("start_time")
        if pid is None:
            return
        if started is None:
            raise RuntimeError(f"the old backend (pid {pid}) has no recorded start time; refusing to signal it")
        if not self.same_process(pid, started):
            return  # already gone (or the pid now names another process: never signal it)
        self._terminate_same_process(int(pid), started)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and self.same_process(pid, rec.get("start_time")):
            time.sleep(0.2)
        if self.same_process(pid, rec.get("start_time")):
            raise RuntimeError(f"the old backend (pid {pid}) did not exit within 30s")

    def _terminate_same_process(self, pid: int, started) -> None:
        """SIGTERM *pid* only while it is still the recorded process. On Linux the identity is bound to
        a pidfd opened BEFORE the check, so a pid reused after the check is never signalled; elsewhere
        the check runs immediately before the signal."""
        import signal
        if hasattr(os, "pidfd_open"):
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                return
            except OSError:
                fd = None  # kernel without pidfd: the check-then-signal path below
            if fd is not None:
                try:
                    if self.same_process(pid, started):
                        signal.pidfd_send_signal(fd, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(fd)
                return
        if self.same_process(pid, started):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)

    #: systemd unit-file states the migration can put back exactly; any other one is refused before
    #: anything changes (``linked``: disabling drops the link; ``generated``/``transient``/``masked``/
    #: ``alias``/``indirect``: not this command's to change).
    RESTORABLE_UNIT_STATES = frozenset({"enabled", "enabled-runtime", "static", "disabled"})

    @staticmethod
    def _enable_args(state: Optional[str]) -> Optional[list[str]]:
        return {"enabled": [], "enabled-runtime": ["--runtime"]}.get(state or "")

    def old_disable(self, rec: dict) -> None:
        if rec["supervisor"] == "systemd":
            extra = self._enable_args(rec.get("unit_file_state"))
            if extra is not None:  # static / disabled: nothing enables it, nothing to undo
                self._require(_systemctl(rec["scope"], ["disable", *extra, rec["unit"]]), f"disable {rec['unit']}")
        elif rec["supervisor"] == "launchd":
            self._require(_launchctl(["disable", f"{rec['scope']}/{rec['unit']}"]), f"disable {rec['unit']}")
            path = Path(rec["unit_path"]) if rec.get("unit_path") else None
            if path is not None and path.exists():
                path.rename(path.with_name(path.name + ".disabled"))

    def old_enable(self, rec: dict) -> None:
        if rec["supervisor"] == "systemd":
            want = rec.get("unit_file_state")
            extra = self._enable_args(want)
            if extra is not None:
                self._require(_systemctl(rec["scope"], ["enable", *extra, rec["unit"]]), f"enable {rec['unit']}")
            now = _systemctl(rec["scope"], ["is-enabled", rec["unit"]]).stdout.strip()
            if want and now != want:
                raise RuntimeError(f"{rec['unit']} is {now or 'unknown'} after the restore, not {want} as before")
        elif rec["supervisor"] == "launchd":
            if rec.get("unit_path"):
                path = Path(rec["unit_path"])
                disabled = path.with_name(path.name + ".disabled")
                if disabled.exists() and not path.exists():
                    disabled.rename(path)
            self._require(_launchctl(["enable", f"{rec['scope']}/{rec['unit']}"]), f"enable {rec['unit']}")

    def old_start(self, rec: dict, launcher_env: Optional[dict], argv: Optional[list[str]] = None) -> Optional[int]:
        """Start the old backend again; returns the respawned pid for a manual process, which is started
        from its recorded raw command line *argv* (``rec["argv"]`` is the redacted one, for display)."""
        sup = rec["supervisor"]
        if sup == "systemd":
            self._require(_systemctl(rec["scope"], ["start", rec["unit"]], timeout=90), f"start {rec['unit']}")
            return None
        if sup == "launchd":
            if self.launchd_loaded(rec):  # never unloaded (the run failed before its stop step)
                return None
            # Right after a bootout launchd can still refuse the label (EIO) for a moment: retry briefly.
            for _attempt in range(10):
                res = _launchctl(["bootstrap", rec["scope"], rec["unit_path"]])
                if res.returncode == 0 or self.launchd_loaded(rec):
                    return None
                time.sleep(1)
            self._require(_launchctl(["bootstrap", rec["scope"], rec["unit_path"]]), f"bootstrap {rec['unit']}")
            return None
        return self._respawn_manual(rec, launcher_env, argv)

    def _respawn_manual(self, rec: dict, launcher_env: Optional[dict], argv: Optional[list[str]]) -> int:
        from hermes_cli.main_dashboard import _respawnable_command_for_current_install
        if not argv:
            raise RuntimeError("the old backend's command line was not recorded; start it again by hand")
        command = _respawnable_command_for_current_install(_with_profile(list(argv), self.home))
        if launcher_env is None:  # never the caller's environment: the old one carried its sign-in settings
            raise RuntimeError("the old backend's environment was not recorded; start it again by hand")
        env = dict(launcher_env)
        log = self.home / "logs" / "backend-migration-respawn.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as fh:
            proc = subprocess.Popen(command, cwd=rec.get("cwd") or None, env=env, stdin=subprocess.DEVNULL,
                                    stdout=fh, stderr=subprocess.STDOUT,
                                    # detached-spawn: ok — restores a hand-started backend the migration stopped
                                    start_new_session=True, close_fds=True)
        return proc.pid

    # The old unit stays installed (disabled) after a migration. Started again by hand, the old command
    # sees the gateway on its port and exits 78 (configuration error); this drop-in makes systemd stop
    # there instead of looping under Restart=always.
    DROPIN_NAME = "hermes-migrated.conf"
    DROPIN_TEXT = ("# Written by `hermes gateway migrate --backends`: the gateway serves this backend's address now.\n"
                   "# Started by hand, the old command exits 78 (configuration error) and is not restarted.\n"
                   "# `hermes gateway migrate --backends --restore` removes this file.\n"
                   "[Service]\nRestartPreventExitStatus=78\n")

    def user_unit_dir(self) -> Path:
        from hermes_cli.gateway import user_systemd_unit_dir
        return user_systemd_unit_dir()

    def old_unit_dropin_path(self, rec: dict) -> Optional[str]:
        """Where the restart guard for the old unit goes: systemd USER units only, else None."""
        if rec.get("supervisor") != "systemd" or rec.get("scope") != "user" or not rec.get("unit"):
            return None
        return str(self.user_unit_dir() / f"{rec['unit']}.d" / self.DROPIN_NAME)

    def write_old_unit_dropin(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.DROPIN_TEXT, encoding="utf-8")
        self.daemon_reload("user")

    def remove_old_unit_dropin(self, path: str) -> None:
        target = Path(path)
        if not target.exists():
            return
        target.unlink()
        with contextlib.suppress(OSError):
            target.parent.rmdir()  # only when empty: the operator's own drop-ins stay
        self.daemon_reload("user")

    def daemon_reload(self, scope: str) -> None:
        self._require(_systemctl(scope, ["daemon-reload"]), "reload the systemd unit files")

    def spawn_detached(self, flags: list[str], log: Path) -> dict:
        """Run ``hermes <flags>`` outside this process's session and systemd cgroup; returns at once.

        Under a systemd user manager it becomes its own transient unit, so stopping the old backend's
        unit (which may be the caller, e.g. the dashboard's button) cannot take it down; elsewhere it is
        a new session. Output is appended to *log*. Returns ``{"unit": name}`` or ``{"pid": n}``."""
        from hermes_cli._launchers import runtime_command
        from hermes_cli.config import get_project_root
        argv = runtime_command(get_project_root(), flags, home=self.home)
        log.parent.mkdir(parents=True, exist_ok=True)
        from hermes_platform.resolver import locate_command
        from tools.environments.local import build_subprocess_env
        # The caller's env unscrubbed (the run reads the same credentials), pinned to this home.
        env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False,
                                   extra={"HERMES_HOME": str(self.home)})
        unit = f"{DETACHED_UNIT_PREFIX}{int(time.time())}"
        systemd_run = locate_command("systemd-run").command
        if systemd_run and self.gateway_kind() == "systemd":
            cmd = [systemd_run, "--user", "--collect", f"--unit={unit}", "--quiet",
                   f"--property=StandardOutput=append:{log}", f"--property=StandardError=append:{log}",
                   *(f"--setenv={k}={env[k]}" for k in ("HERMES_HOME", "PATH") if k in env),
                   *argv]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=30, check=False)
            if res.returncode == 0:
                return {"unit": f"{unit}.service"}
        if (inside := self.own_unit()) is not None:
            # A new session is still in this unit's cgroup: stopping the unit (the old backend's, or the
            # gateway's on restore) would end the run half-way. Refuse instead of starting it there.
            raise RuntimeError(f"cannot start the run outside {inside} (systemd-run is unavailable or failed); "
                               "run the command from a terminal instead")
        with open(log, "ab") as fh:
            proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                    # detached-spawn: ok — the migration must outlive the caller it may stop
                                    start_new_session=True, close_fds=True)
        return {"pid": proc.pid}

    def own_unit(self) -> Optional[str]:
        """The systemd service whose cgroup this process is in (not the user manager itself), or None:
        read from ``/proc/self/cgroup``, never inferred from how the process was started."""
        from hermes_cli.update_cmd_posix_pause import _pid_cgroup
        units = [part for part in (_pid_cgroup() or "").split("/")
                 if part.endswith(".service") and not part.startswith("user@")]
        return units[-1] if units else None

    def old_unit_active(self, rec: dict) -> Optional[bool]:
        if rec["supervisor"] == "launchd":
            from hermes_cli.gateway import _launchd_print_service_pid
            loaded, pid = _launchd_print_service_pid(rec["scope"], rec["unit"])
            return loaded and pid is not None
        if rec["supervisor"] != "systemd":
            return None
        return _systemctl(rec["scope"], ["is-active", rec["unit"]]).stdout.strip() == "active"

    # ------------------------------------------------------------------ network

    def health_status(self, host: str, port: int) -> Optional[dict]:
        """One GET /api/health with one bounded timeout and no redirect followed: ``{"health": "missing"}``
        on 401, 403 or 404 (a Hermes older than v2026.7.30), the JSON object on 200, else None."""
        from hermes_cli.url_utils import format_url_host
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        request = urllib.request.Request(f"http://{format_url_host(host)}:{port}/api/health",
                                         headers={"Accept": "application/json"})
        try:
            with opener.open(request, timeout=HEALTH_TIMEOUT) as resp:
                data = json.load(resp) if resp.status == 200 else None
        except urllib.error.HTTPError as exc:
            return {"health": "missing"} if exc.code in (401, 403, 404) else None
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def http_status(self, host: str, port: int) -> Optional[dict]:
        """Read sign-in state without invoking /api/status's install-ID or stale-record maintenance.
        ``{"health": "missing"}`` when the backend answers 401, 403 or 404 there (a Hermes older than v2026.7.30)."""
        from hermes_cli.url_utils import format_url_host
        origin = f"http://{format_url_host(host)}:{port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def read(path):
            request = urllib.request.Request(origin + path, headers={"Accept": "application/json"})
            with opener.open(request, timeout=5) as resp:
                return json.load(resp) if resp.status == 200 else None
        try:
            try:
                data = read("/api/health")
            except urllib.error.HTTPError as exc:
                # Hermes before v2026.7.30 has no /api/health: v2026.7.1-v2026.7.20 401 every /api/* path
                # outside their public list before routing (signed in or not), older ones 404 it. Since
                # v2026.7.30 /api/health is in dashboard_auth.public_paths, which both auth gates use.
                if exc.code in (401, 403, 404):
                    return {"health": "missing"}
                raise
            if not isinstance(data, dict) or not isinstance(data.get("auth_required"), bool):
                return None
            if data["auth_required"]:
                try:
                    providers = read("/api/auth/providers")
                except (OSError, ValueError, urllib.error.URLError):
                    providers = None
                data["auth_providers"] = providers.get("providers") if isinstance(providers, dict) else None
        except (OSError, ValueError, urllib.error.URLError):
            return None
        return data

    def gateway_status(self, host: str, port: int) -> Optional[dict]:
        """The migrated gateway's ``GET /api/status`` JSON on *host:port* (its ``gateway_state``, sign-in
        state and providers), or None when it does not answer 200. Only for the post-move verification:
        the plan's probe of an old backend is :meth:`http_status`, which never calls /api/status."""
        from hermes_cli.url_utils import format_url_host
        url = f"http://{format_url_host(host)}:{port}/api/status"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # on-machine: never a proxy
        try:
            with opener.open(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=5) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except (OSError, ValueError, urllib.error.URLError):
            return None
        return data if isinstance(data, dict) else None

    def pid_listens_on(self, pid: int, port: int) -> bool:
        import psutil
        with contextlib.suppress(Exception):
            return any(c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port
                       for c in psutil.Process(pid).net_connections(kind="tcp"))
        return False

    def port_in_use(self, host: str, port: int) -> bool:
        import socket
        with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=1):
            return True
        return False

    @staticmethod
    def _require(res: subprocess.CompletedProcess, what: str) -> None:
        if res.returncode != 0:
            detail = (res.stderr or res.stdout or f"exit {res.returncode}").strip().splitlines()
            raise RuntimeError(f"could not {what}: {detail[-1] if detail else res.returncode}")

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def now(self) -> float:
        return time.time()
