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
            # Added after the initial schema: an existing profile's store has no
            # realm_kind column, and a migration is cheaper than a rebuild that
            # would orphan live realms.
            if "realm_kind" not in {
                row[1] for row in db.execute("PRAGMA table_info(owners)")
            }:
                db.execute("ALTER TABLE owners ADD COLUMN realm_kind TEXT")
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

    def kind(self, owner, default):
        with self.connection() as db:
            row = db.execute(
                "SELECT realm_kind FROM owners WHERE id=?", (owner,)
            ).fetchone()
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

    def set_kind(self, owner, kind):
        from .config import KINDS

        if kind not in KINDS:
            raise ValueError("Invalid realm kind")
        with self.connection() as db:
            db.execute("UPDATE owners SET realm_kind=? WHERE id=?", (kind, owner))


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


def vm_setup_status(home=None):
    """Readiness of the Omarchy VM kind, reported separately from the labwc one.

    A profile with a working labwc realm and no VM base image is *ready* — it
    just cannot use the VM kind yet. Folding the two together would block the
    default realm on a 5 GB download nobody asked for.
    """
    from .vm_manager import VmManager

    try:
        report = VmManager(home).doctor()
    except (ValueError, OSError) as exc:
        return {"ready": False, "message": "Omarchy VM realms unavailable: " + str(exc)}
    if report["ok"]:
        return {"ready": True, "message": "Omarchy VM realm dependencies are available."}
    return {
        "ready": False,
        "message": (
            "Omarchy VM realm setup required: "
            + "; ".join(report["missing"])
            + ". Install missing system packages with your distribution's package "
            "manager, then run hermes realms vm install to build the base image; "
            "host fallback is disabled."
        ),
    }


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
        self._vm = None

    @property
    def vm(self):
        """The VM kind's manager, built on first use.

        Deferred because the labwc realm must keep working on a host with no
        QEMU at all, and because reading VM config costs a config load that
        every non-VM session would otherwise pay.
        """
        with self._lock:
            if self._vm is None:
                from .vm_manager import VmManager

                self._vm = VmManager(self.home)
            return self._vm

    def kind(self, owner):
        return self.owners.kind(owner, self.manager.config.default_kind)

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

    def vm_records(self, owner):
        return [r for r in self.vm.list() if r["session_id"] == owner]

    def status(self, owner):
        from .bridge import get_profile_viewer

        viewer = get_profile_viewer(self.home)
        rows = [
            {
                "id": r["id"],
                "kind": "realm",
                "state": "live" if r["status"] == "running" else r["status"],
                "size": r["size"],
                "window_count": self._window_counter.count(r),
                "controlled": viewer.is_controlled(r["id"]),
                "error": None,
            }
            for r in self.records(owner)
        ]
        kind = self.kind(owner)
        if kind == "omarchy-vm" or self._vm is not None:
            try:
                rows.extend(
                    {
                        "id": r["id"],
                        "kind": "omarchy-vm",
                        "state": "live" if r["status"] == "running" else r["status"],
                        "size": None,
                        "window_count": None,
                        "controlled": viewer.is_controlled(r["id"]),
                        "stats": self.vm.stats(r["id"]),
                        "memory_mb": r.get("memory"),
                        "network": r.get("network"),
                        "error": None,
                    }
                    for r in self.vm_records(owner)
                )
            except (OwnerError, ValueError, OSError):
                # A status read must never take down the labwc rows with it.
                pass
        return {
            "mode": self.owners.mode(owner, self.manager.config.default_mode),
            "kind": kind,
            "realms": rows,
            "setup": setup_status(driver_executable=self.driver_executable),
            "vm_setup": vm_setup_status(self.home) if kind == "omarchy-vm" else None,
        }

    def watch(self, owner, realm_id):
        from .bridge import get_profile_viewer
        from .lifecycle import validate_live

        record = next((r for r in self.records(owner) if r["id"] == realm_id), None)
        if record is None:
            # A VM realm's viewer is QEMU's own VNC socket, which speaks the same
            # RFB the bridge already proxies for wayvnc.
            record = next(
                (r for r in self.vm_records(owner) if r["id"] == realm_id), None
            )
            if record is None:
                raise OwnerError("Realm ownership mismatch")
            self.vm.validate(realm_id)
        else:
            validate_live(record)
        viewer = get_profile_viewer(self.home).start()
        token = viewer.issue(realm_id, can_control=True, ttl=300)
        return {"url": viewer.origin + "/realms/" + realm_id + "/view#ticket=" + token}

    USAGE = (
        "Use /realm on [omarchy]|off|status|size WIDTHxHEIGHT|stop|watch|shot"
        "|push SOURCE [DEST]|pull GUEST_PATH LOCAL_PATH"
    )

    def command(self, raw, **identity):
        owner = self.bind(**identity)
        parts = raw.split()
        action = parts[0] if parts else "status"
        handler = self._COMMANDS.get(action)
        if handler is None:
            raise ValueError(self.USAGE)
        result = handler(self, owner, parts[1:], identity)
        return self.status(owner) if result is None else result

    def _command_on(self, owner, arguments, identity):
        """``/realm on`` selects private routing; ``/realm on omarchy`` its kind.

        The kind is a property of this conversation, so a later plain
        ``/realm on`` in the same chat keeps the kind that was chosen.
        """
        from .config import KINDS

        if len(arguments) > 1:
            raise ValueError(self.USAGE)
        if arguments:
            requested = {"omarchy": "omarchy-vm"}.get(arguments[0], arguments[0])
            if requested not in KINDS:
                raise ValueError("Use /realm on [omarchy]")
            if requested != self.kind(owner):
                # Switching kind mid-conversation would leave the previous
                # desktop running with nothing routed to it.
                self.stop(owner)
            self.owners.set_kind(owner, requested)
        if self.kind(owner) == "omarchy-vm":
            setup = vm_setup_status(self.home)
        else:
            setup = setup_status(driver_executable=self.driver_executable)
        if not setup["ready"]:
            raise SetupError(setup["message"])
        self.owners.set_mode(owner, "realm")
        if self.kind(owner) == "omarchy-vm":
            # Every other kind starts lazily on first tool use, inside
            # ``pre_tool``. A guest boots in tens of seconds and that hook is
            # bounded by ``plugins.hook_callback_timeout`` (30s), which fails
            # CLOSED: the boot outran the budget and the tool was blocked with
            # "pre_tool_call plugin callback timed out". The slash command is
            # not on that path, and it is where the user asked for the realm,
            # so a VM realm boots here and the first tool call finds it live.
            self.ready(owner)

    def _command_off(self, owner, arguments, identity):
        if arguments:
            raise ValueError(self.USAGE)
        self.stop(owner)
        self.owners.set_mode(owner, "host")

    def _command_stop(self, owner, arguments, identity):
        self.stop(owner)

    def _command_status(self, owner, arguments, identity):
        return None

    def _command_size(self, owner, arguments, identity):
        if len(arguments) != 1:
            raise ValueError("Use /realm size WIDTHxHEIGHT")
        records = self.records(owner)
        if not records:
            raise ValueError(
                "Realm is not running; enable it and run a desktop action first"
            )
        self.manager.resize(records[0]["id"], arguments[0])

    def _command_shot(self, owner, arguments, identity):
        if arguments:
            raise ValueError("Use /realm shot (own-session capture only)")
        import hashlib
        import shutil
        import struct
        import tempfile

        vm_records = self.vm_records(owner) if self.kind(owner) == "omarchy-vm" else []
        records = self.records(owner)
        if not records and not vm_records:
            raise ValueError("Realm is not running; host fallback is disabled")
        realm_id = (records or vm_records)[0]["id"]
        capture = "grim"
        if records:
            self.manager.env(realm_id)  # Validate before creating any artifact.
        else:
            self.vm.validate(realm_id)
            capture = "grim-in-guest"
        directory = Path(
            tempfile.mkdtemp(prefix="shot-", dir=self.manager.registry.root)
        )
        try:
            source = self.manager if records else self.vm
            target = Path(source.shot(realm_id, directory / "screen.png"))
            target.chmod(0o600)
            data = target.read_bytes()
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(
                    "Realm capture did not return a PNG; host fallback is disabled"
                )
            width, height = struct.unpack(">II", data[16:24])
            return {
                "realm_id": realm_id,
                "path": str(target),
                "mime_type": "image/png",
                "width": width,
                "height": height,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "capture": capture,
                "fallback": True,
            }
        except BaseException:
            shutil.rmtree(directory)
            raise

    def _command_watch(self, owner, arguments, identity):
        records = self.records(owner) or self.vm_records(owner)
        if not records:
            raise ValueError("Realm is not running")
        if identity.get("surface") == "gateway":
            raise ValueError(
                "Remote viewers require an explicit viewer tunnel; use the local desktop Watch action"
            )
        return self.watch(owner, records[0]["id"])

    def _command_push(self, owner, arguments, identity):
        """Copy a host path into the VM guest. Copying IS the boundary."""
        if not 1 <= len(arguments) <= 2:
            raise ValueError("Use /realm push SOURCE [GUEST_PATH]")
        record = self._vm_record(owner)
        return self.vm.push(record["id"], arguments[0], *arguments[1:])

    def _command_pull(self, owner, arguments, identity):
        if len(arguments) != 2:
            raise ValueError("Use /realm pull GUEST_PATH LOCAL_PATH")
        record = self._vm_record(owner)
        return self.vm.pull(record["id"], arguments[0], arguments[1])

    def _vm_record(self, owner):
        records = self.vm_records(owner)
        if not records:
            raise ValueError(
                "No Omarchy VM realm is running for this conversation; "
                "use /realm on omarchy first"
            )
        return records[0]

    _COMMANDS = {
        "on": _command_on,
        "off": _command_off,
        "stop": _command_stop,
        "status": _command_status,
        "size": _command_size,
        "shot": _command_shot,
        "watch": _command_watch,
        "push": _command_push,
        "pull": _command_pull,
    }

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
        """Start (or reuse) this conversation's desktop and route execution to it.

        Two kinds, one contract: whatever comes back, the session's execution
        context routes every terminal child through it and re-validates on each
        use. The VM kind passes ``computer_use=None`` — the guest has no Cua
        driver in it yet, and offering desktop control that would silently act
        on the *host* is exactly the failure this whole feature exists to stop.
        """
        if self.kind(owner) == "omarchy-vm":
            return self._ready_vm(owner)
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

    def _vm_valid(self, record, owner):
        from .vm_manager import VmError

        if self.owners.mode(owner, self.manager.config.default_mode) != "realm":
            return False
        if self.kind(owner) != "omarchy-vm":
            return False
        try:
            current = self.vm.validate(record["id"])
        except (OwnerError, VmError, OSError):
            return False
        return current["generation"] == record["generation"]

    def _ready_vm(self, owner):
        from hermes_cli.session_execution import (
            SessionExecutionContext,
            register_session_execution_context,
            resolve_session_execution_context,
        )

        with self._lock:
            setup = vm_setup_status(self.home)
            if not setup["ready"]:
                raise SetupError(setup["message"])
            record = self.vm.start(owner)
            aliases = self.owners.aliases(owner)
            current = self._attachments.get(owner)
            if current and current[:2] == (record["generation"], aliases):
                resolve_session_execution_context(session_id=owner).check()
                return record
            env = self.vm.env(record["id"])
            context = SessionExecutionContext(
                env_set=env,
                # The guest is a different machine; every host display, bus and
                # input handle in this environment is meaningless there and must
                # not survive into a command that believes it is in the realm.
                env_unset=HOST_ENV_KEYS - env.keys(),
                command_prefix=self.vm.command_prefix(record["id"]),
                computer_use=None,
                # Shell state lives in the guest, where no host temp directory
                # exists. Without this the session snapshot is written to a host
                # path the guest cannot see and env vars stop persisting.
                backend_temp_dir="/tmp",
                # A routed session starts in the guest user's home. The host cwd
                # this profile is configured with does not exist in there.
                backend_cwd=self.vm.guest_home(record["id"]),
                validate=lambda: self._vm_valid(record, owner),
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
            if self._vm is None and self.kind(owner) != "omarchy-vm":
                return
            from .vm_manager import VmError

            try:
                for record in self.vm_records(owner):
                    from .bridge import get_profile_viewer

                    get_profile_viewer(self.home).revoke(record["id"])
                    self.vm.stop(record["id"])
            except (VmError, OwnerError, OSError, ValueError):
                import logging

                logging.getLogger(__name__).exception("VM realm teardown failed")

    def reset(self, **identity):
        """A context reset is not a teardown boundary — keep the realm.

        ``on_session_reset`` means "same session, fresh context" (``/new``,
        ``/clear``, or simply the gateway building this session's agent). The
        realm was asked for explicitly and may hold minutes of work, so it
        outlives the transcript that happens to be in front of it; only
        ``finalize`` ends it. Reaping is never lost by keeping it: real
        finalize, the per-guest systemd owner watcher, and idle expiry all
        still apply.

        Reconciliation is the one thing worth doing here, and only when a
        ``VmManager`` already exists: a reset on a labwc-only host must not be
        what finally constructs one, because the ``vm`` property is deferred
        precisely so a host with no QEMU never pays for it.
        """
        if self._vm is None:
            return
        from .vm_manager import VmError

        try:
            self.vm.list()
        except (VmError, OwnerError, OSError, ValueError):
            import logging

            logging.getLogger(__name__).warning(
                "VM realm reconciliation on reset failed", exc_info=True)

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
            if tool_name == "terminal":
                from .host_guard import host_escape

                escape = host_escape(args.get("command"))
                if escape is not None:
                    return {"action": "block", "message": escape}
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
