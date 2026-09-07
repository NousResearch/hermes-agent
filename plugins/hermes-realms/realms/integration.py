"""Trusted session ownership and plugin integration; no focused-session globals."""

from contextlib import contextmanager
from pathlib import Path
import os
import sqlite3


class OwnerError(PermissionError):
    """The supplied identifiers do not name one registered conversation."""


class OwnershipStore:
    """Profile-local cross-process aliases, populated only by trusted host hooks."""

    def __init__(self, home):
        self.root = Path(home).resolve() / "realms"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.root / "sessions.sqlite3"
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS owners (id TEXT PRIMARY KEY, mode TEXT);
                CREATE TABLE IF NOT EXISTS aliases (
                    kind TEXT NOT NULL, value TEXT NOT NULL, owner TEXT NOT NULL,
                    PRIMARY KEY(kind, value));
            """)
        self.path.chmod(0o600)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def identifiers(values):
        result = []
        for kind, value in values.items():
            if value is None or value == "":
                continue
            if kind not in (
                "session_id",
                "runtime_session_id",
                "stored_session_id",
                "task_id",
            ):
                raise OwnerError("Unsupported identity field")
            if (
                not isinstance(value, str)
                or value != value.strip()
                or "\0" in value
                or len(value) > 256
            ):
                raise OwnerError("Invalid session identity")
            result.append((kind, value))
        if not result:
            raise OwnerError("An active conversation is required")
        return result

    def bind(self, **values):
        identifiers = self.identifiers(values)
        if not values.get("session_id"):
            raise OwnerError("A trusted conversation identity is required")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            owners = {
                row[0]
                for kind, value in identifiers
                if (
                    row := db.execute(
                        "SELECT owner FROM aliases WHERE kind=? AND value=?",
                        (kind, value),
                    ).fetchone()
                )
            }
            if len(owners) > 1:
                raise OwnerError("Conflicting session owners")
            owner = next(iter(owners)) if owners else values["session_id"]
            db.execute("INSERT OR IGNORE INTO owners(id) VALUES (?)", (owner,))
            db.executemany(
                "INSERT OR IGNORE INTO aliases(kind,value,owner) VALUES (?,?,?)",
                [(kind, value, owner) for kind, value in identifiers],
            )
            return owner

    def resolve(self, *, allow_missing=False, **values):
        identifiers = self.identifiers(values)
        with self.connection() as db:
            rows = [
                db.execute(
                    "SELECT owner FROM aliases WHERE kind=? AND value=?", (kind, value)
                ).fetchone()
                for kind, value in identifiers
            ]
        if allow_missing and all(row is None for row in rows):
            return None
        if any(row is None for row in rows) or len({row[0] for row in rows}) != 1:
            raise OwnerError("Unregistered or conflicting session owners")
        return rows[0][0]

    def mode(self, owner, default):
        with self.connection() as db:
            row = db.execute("SELECT mode FROM owners WHERE id=?", (owner,)).fetchone()
        return row[0] or default if row else default

    def aliases(self, owner):
        with self.connection() as db:
            return tuple(
                row[0]
                for row in db.execute(
                    "SELECT DISTINCT value FROM aliases WHERE owner=?", (owner,)
                )
            )

    def set_mode(self, owner, mode):
        if mode not in ("realm", "host", "ask"):
            raise ValueError("Invalid realm mode")
        with self.connection() as db:
            db.execute("UPDATE owners SET mode=? WHERE id=?", (mode, owner))


class SetupError(ValueError):
    """Missing prerequisites, safe to expose with administrative recovery steps."""


def setup_status(*, driver_executable=None):
    import shutil
    import sys

    from .config import driver_path
    from .install_driver import execution_verified

    driver = driver_executable if driver_executable is not None else driver_path()
    missing = []
    if sys.platform != "linux":
        missing.append("Linux")
    if not execution_verified(driver):
        missing.append("pinned Cua driver (run hermes realms install-driver in this profile)")
    missing.extend(
        name for name in (
                "labwc",
                "Xwayland",
                "wayvnc",
                "dbus-daemon",
                "systemd-run",
                "systemd-inhibit",
                "grim",
                "wlr-randr",
                "gdbus",
                "bwrap",
        ) if not shutil.which(name)
    )
    missing.extend(
        path for path in ("/usr/lib/at-spi-bus-launcher", "/usr/lib/at-spi2-registryd")
        if not os.access(path, os.X_OK)
    )
    return {
        "ready": not missing,
        "message": ("Realms setup required: " + "; ".join(missing)
                    + ". Install missing system packages with your distribution's package manager; "
                    "then run hermes realms doctor in this profile; host fallback is disabled.")
        if missing else "Realms dependencies are available.",
    }


def requirements_available(*, driver_executable=None):
    return setup_status(driver_executable=driver_executable)["ready"]


# Profile-keyed infrastructure only; current/focused identity never lives here.
_services = {}


def get_integration(home=None):
    from hermes_constants import get_hermes_home

    home = Path(home if home is not None else get_hermes_home()).resolve()
    if home not in _services:
        _services[home] = RealmIntegration(home)
    return _services[home]


# Keys stripped even if absent at registration: later ambient/shell snapshots
# must not resurrect a host display, portal, bus, driver or input endpoint.
HOST_ENV_KEYS = frozenset(
    {
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
        "DBUS_STARTER_ADDRESS",
        "DBUS_STARTER_BUS_TYPE",
        "AT_SPI_BUS_ADDRESS",
        "HYPRLAND_INSTANCE_SIGNATURE",
        "HYPRLAND_CMD",
        "SWAYSOCK",
        "I3SOCK",
        "YDOTOOL_SOCKET",
        "CUA_INJECT_SOCKET",
        "CUA_DRIVER_SOCKET",
        "CUA_DRIVER_RS_SOCKET",
        "CUA_DRIVER_RS_SESSION",
        "CUA_DRIVER_RS_STATE_DIR",
        "CUA_DRIVER_RS_PERMISSION_MODE",
        "CUA_DRIVER_PERMISSION_MODE",
        "CUA_DRIVER_RS_BYPASS_APPROVALS",
        "SESSION_MANAGER",
        "XAUTHORITY",
        "DESKTOP_STARTUP_ID",
        "XDG_ACTIVATION_TOKEN",
        "GTK_USE_PORTAL",
        "GTK_MODULES",
        "GTK3_MODULES",
        "GIO_EXTRA_MODULES",
        "QT_QPA_PLATFORMTHEME",
        "QT_PLUGIN_PATH",
        "GTK_PATH",
        "GDK_DEBUG",
        "PIPEWIRE_REMOTE",
        "PULSE_SERVER",
        "SSH_AUTH_SOCK",
        "GPG_AGENT_INFO",
    }
)


class RealmIntegration:
    def __init__(self, home):
        from .manager import Manager
        import threading

        self.manager = Manager(home)
        self.home = self.manager.home
        self.owners = OwnershipStore(self.home)
        from .config import driver_path

        self.driver_executable = driver_path(self.home)
        self._attachments = {}
        self._lock = threading.RLock()
        from .windows import WindowCounter

        self._window_counter = WindowCounter()

    def bind(self, *, hermes_home=None, profile=None, surface=None, **identity):
        if hermes_home is not None and Path(hermes_home).resolve() != self.home:
            raise OwnerError("Profile ownership mismatch")
        return self.owners.bind(
            **{
                key: identity.get(key)
                for key in (
                    "session_id",
                    "stored_session_id",
                    "runtime_session_id",
                    "task_id",
                )
            }
        )

    def records(self, owner):
        return [
            record for record in self.manager.list() if record["session_id"] == owner
        ]

    def status(self, owner):
        from .bridge import get_profile_viewer

        viewer = get_profile_viewer(self.home)
        rows = [
            {
                "id": r["id"],
                "state": "live" if r["status"] == "running" else r["status"],
                "size": r["size"],
                "window_count": self._window_counter.count(r),
                "controlled": viewer.is_controlled(r["id"]),
                "error": None,
            }
            for r in self.records(owner)
        ]
        return {
            "mode": self.owners.mode(owner, self.manager.config.default_mode),
            "realms": rows,
            "setup": setup_status(driver_executable=self.driver_executable),
        }

    def watch(self, owner, realm_id):
        from .bridge import get_profile_viewer
        from .lifecycle import validate_live

        record = next((r for r in self.records(owner) if r["id"] == realm_id), None)
        if record is None:
            raise OwnerError("Realm ownership mismatch")
        validate_live(record)
        viewer = get_profile_viewer(self.home).start()
        token = viewer.issue(realm_id, can_control=True, ttl=300)
        return {"url": viewer.origin + "/realms/" + realm_id + "/view#ticket=" + token}

    def command(self, raw, **identity):
        owner = self.bind(**identity)
        parts = raw.split()
        action = parts[0] if parts else "status"
        if action in ("on", "off"):
            if action == "on":
                setup = setup_status(driver_executable=self.driver_executable)
                if not setup["ready"]:
                    raise SetupError(setup["message"])
            if action == "off":
                self.stop(owner)
            self.owners.set_mode(owner, {"on": "realm", "off": "host"}[action])
        elif action == "stop":
            self.stop(owner)
        elif action == "size":
            if len(parts) != 2:
                raise ValueError("Use /realm size WIDTHxHEIGHT")
            records = self.records(owner)
            if not records:
                raise ValueError(
                    "Realm is not running; enable it and run a desktop action first"
                )
            self.manager.resize(records[0]["id"], parts[1])
        elif action == "shot":
            if len(parts) != 1:
                raise ValueError("Use /realm shot (own-session capture only)")
            records = self.records(owner)
            if not records:
                raise ValueError("Realm is not running; host fallback is disabled")
            record = records[0]
            self.manager.env(record["id"])  # Validate before creating any artifact.
            import hashlib
            import shutil
            import struct
            import tempfile

            directory = Path(
                tempfile.mkdtemp(prefix="shot-", dir=self.manager.registry.root)
            )
            try:
                target = Path(self.manager.shot(record["id"], directory / "screen.png"))
                target.chmod(0o600)
                data = target.read_bytes()
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError(
                        "Realm capture did not return a PNG; host fallback is disabled"
                    )
                width, height = struct.unpack(">II", data[16:24])
                return {
                    "realm_id": record["id"],
                    "path": str(target),
                    "mime_type": "image/png",
                    "width": width,
                    "height": height,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "capture": "grim",
                    "fallback": True,
                }
            except BaseException:
                shutil.rmtree(directory)
                raise
        elif action == "watch":
            records = self.records(owner)
            if not records:
                raise ValueError("Realm is not running")
            if identity.get("surface") == "gateway":
                raise ValueError(
                    "Remote viewers require an explicit viewer tunnel; use the local desktop Watch action"
                )
            return self.watch(owner, records[0]["id"])
        elif action != "status":
            raise ValueError(
                "Use /realm on|off|status|size WIDTHxHEIGHT|stop|watch|shot"
            )
        return self.status(owner)

    def _valid(self, record, owner):
        from .lifecycle import validate_live

        if self.owners.mode(owner, self.manager.config.default_mode) != "realm":
            return False
        current = next(
            (r for r in self.records(owner) if r["id"] == record["id"]), None
        )
        if current is None or current["generation"] != record["generation"]:
            return False
        validate_live(current)
        from .driver import create_driver_launcher

        create_driver_launcher(self.manager, current["id"], self.driver_executable)
        return True

    def ready(self, owner):
        from hermes_cli.session_execution import (
            SessionExecutionContext,
            ComputerUseLaunchContext,
            register_session_execution_context,
            resolve_session_execution_context,
        )
        from .driver import create_driver_launcher
        import sys

        with self._lock:
            setup = setup_status(driver_executable=self.driver_executable)
            if not setup["ready"]:
                raise SetupError(setup["message"])
            record = self.manager.start(owner)
            aliases = self.owners.aliases(owner)
            current = self._attachments.get(owner)
            if current and current[:2] == (record["generation"], aliases):
                resolve_session_execution_context(session_id=owner).check()
                return record
            env = self.manager.env(record["id"])
            launcher = create_driver_launcher(
                self.manager, record["id"], self.driver_executable
            )
            context = SessionExecutionContext(
                env_set=env,
                env_unset=HOST_ENV_KEYS - env.keys(),
                command_prefix=(
                    sys.executable,
                    str(Path(__file__).with_name("launch.py")),
                    str(self.home),
                    record["id"],
                    "--",
                ),
                computer_use=ComputerUseLaunchContext(
                    driver_command=launcher,
                    private_daemon=True,
                    desktop_only=True,
                    runtime_dir=record["runtime_dir"],
                    no_overlay=not record["overlay"],
                    session_name=record["id"],
                    theme=record["cursor_theme"],
                    allow_input=lambda: self.input_allowed(record["id"]),
                ),
                validate=lambda: self._valid(record, owner),
            )
            register_session_execution_context(owner, context, task_ids=aliases)
            self._attachments[owner] = (record["generation"], aliases, record["id"])
            return record

    def input_allowed(self, realm_id):
        # A shared profile viewer authority must be available before input.
        from .bridge import get_profile_viewer

        return not get_profile_viewer(self.home).is_controlled(realm_id)

    def stop(self, owner):
        from hermes_cli.session_execution import remove_session_execution_context

        with self._lock:
            remove_session_execution_context(owner)
            self._attachments.pop(owner, None)
            for record in self.records(owner):
                from .bridge import get_profile_viewer

                get_profile_viewer(self.home).revoke(record["id"])
                self.manager.stop(record["id"])

    def finalize(self, **identity):
        keys = {
            key: identity.get(key)
            for key in (
                "session_id",
                "stored_session_id",
                "runtime_session_id",
                "task_id",
            )
        }
        owner = self.owners.resolve(allow_missing=True, **keys)
        if owner is not None:
            self.stop(owner)

    def unload(self):
        from .bridge import close_profile_viewer

        try:
            for owner in list(self._attachments):
                self.stop(owner)
        finally:
            close_profile_viewer(self.home)

    def pre_tool(self, *, tool_name, args, **identity):
        if tool_name not in ("terminal", "computer_use", "realm"):
            return None
        try:
            owner = self.bind(**identity)
            mode = self.owners.mode(owner, self.manager.config.default_mode)
            if tool_name == "realm" or mode == "host":
                return None
            if mode == "ask":
                return {
                    "action": "block",
                    "message": "Choose /realm on for a private desktop or /realm off for explicit host access. Approvals still apply.",
                }
            self.ready(owner)
            return None
        except SetupError as exc:
            return {"action": "block", "message": str(exc)}
        except Exception:
            # Upstream hook exceptions fail open; return a concrete veto instead.
            import logging

            logging.getLogger(__name__).exception("Realm execution preparation refused")
            return {
                "action": "block",
                "message": "Realm setup or ownership validation failed; host fallback is disabled.",
            }
