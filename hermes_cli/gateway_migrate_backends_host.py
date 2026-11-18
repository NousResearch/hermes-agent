"""Host reads for ``hermes gateway migrate --backends --apply --dry-run``.

The plan (:mod:`hermes_cli.gateway_migrate_backends_plan`) asks the machine everything through
:class:`SystemHost`: the old backend (``collect_backends``, ``systemctl show``, ``/proc``, a launchd
plist), the home's config and ``.env``, the gateway's user service, lingering, the Webapp bundle and
lightweight health and interactive-provider reads on the old address. None starts, stops, installs or
writes anything. This module owns no decisions.

Process identity comes from Hermes' own detection (``gateway_migrate_backends.collect_backends``,
``gateway.status``, ``dashboard_procs``), never from text matching on ``ps`` output.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

_ENV_FILE_ENTRY = re.compile(r"(\S+) \(ignore_errors=(yes|no)\)")
_SHOW_PROPS = ("MainPID", "ActiveState", "UnitFileState", "FragmentPath", "ExecStart", "ControlGroup",
               "Environment", "EnvironmentFiles", "User")
_TIMEOUT = 30


def _systemctl(scope: str, args: list[str], *, timeout: int = _TIMEOUT) -> subprocess.CompletedProcess:
    """``systemctl [--user] <args>`` through the gateway's own wrapper (user bus env included)."""
    from hermes_cli import gateway as gw
    return gw._run_systemctl(args, system=scope != "user", check=False, timeout=timeout,
                             capture_output=True, text=True, encoding="utf-8", errors="replace")


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


class SystemHost:
    """The real machine, for the home this process runs in."""

    def __init__(self, home: Path):
        self.home = Path(home)

    # ------------------------------------------------------------------ discovery

    def backends(self):
        from hermes_cli.gateway_migrate_backends import collect_backends
        return collect_backends()

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
        with contextlib.suppress(OSError):
            return os.readlink(f"/proc/{pid}/cwd")
        return None

    def start_time(self, pid: int) -> Optional[int]:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)

    def running_turns(self, pid: int, start_time) -> int:
        """Turns the old backend *pid* is running now: its live turn markers in this home.

        The old backend writes a durable marker at every turn start and clears it on any conclusion
        (``tui_gateway.turn_marker``), stamped with its own pid and start time, so a marker whose
        writer is that process and still alive is a turn in flight. No sign-in, no socket."""
        from tui_gateway import turn_marker
        from hermes_cli.profiles import profiles_to_serve
        from hermes_constants import hermes_home_key
        profiles = profiles_to_serve(True, include_standalone=True, include_parked=True)
        homes = {hermes_home_key(self.home), *(hermes_home_key(h) for _, h in profiles)}
        return sum(1 for h in homes for e in turn_marker._load(turn_marker._marker_path(h)).values()
                   if e.get("writer_pid") == pid and turn_marker.marker_writer_state(e) == "alive")

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

    def linger(self) -> Optional[bool]:
        from hermes_cli import gateway as gw
        return gw.get_systemd_linger_status()[0]

    # ------------------------------------------------------------------ network

    def http_status(self, host: str, port: int) -> Optional[dict]:
        """Read sign-in state without invoking /api/status's install-ID or stale-record maintenance."""
        from hermes_cli.url_utils import format_url_host
        origin = f"http://{format_url_host(host)}:{port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def read(path):
            request = urllib.request.Request(origin + path, headers={"Accept": "application/json"})
            with opener.open(request, timeout=5) as resp:
                return json.load(resp) if resp.status == 200 else None
        try:
            data = read("/api/health")
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

