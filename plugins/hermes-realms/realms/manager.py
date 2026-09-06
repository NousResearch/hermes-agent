"""Profile-scoped headless desktops. GUI separation, not a hostile-code sandbox."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import shlex
import socket
import struct
import subprocess
import sys
import time
import uuid

from .config import Config, effective_home
from .lifecycle import (
    Registry,
    RealmError,
    OwnershipError,
    RPC_RESPONSE_LIMIT_BYTES,
    alive,
    atomic_json,
    identity,
    remove_runtime,
    scope_info,
    stop_scope,
    validate_live,
    validate_environment,
)


class Manager:
    def __init__(self, home=None, *, realm_id=None):
        self.home = effective_home(home)
        if realm_id is None:
            self.config = Config.load(self.home)
        self.registry = Registry(self.home)
        if realm_id is not None:
            # Payload-only launchers attach to an already owned generation.
            # Its validated launch spec, not the host's current config, governs
            # that desktop; importing the host core here would break -P routing.
            with self.registry.lock():
                record = self.registry.get(realm_id)
                validate_live(record)
                spec = Path(record["runtime_dir"]) / "spec.json"
                self.config = Config(**json.loads(spec.read_text(encoding="utf-8")))

    def list(self):
        with self.registry.lock():
            self._reconcile_locked()
            return self.registry.records()

    def _reconcile_locked(self):
        for record in self.registry.records():
            if record["status"] == "starting":
                owner = Path(record["runtime_dir"]) / "owner.json"
                if owner.exists():
                    candidate = json.loads(owner.read_text(encoding="utf-8"))
                    if any(
                        candidate.get(key) != record.get(key)
                        for key in (
                            "id",
                            "generation",
                            "uid",
                            "home",
                            "session_id",
                            "runtime_dir",
                            "scope",
                            "guardian_unit",
                        )
                    ):
                        raise OwnershipError("startup ownership receipt changed")
                    if alive(candidate.get("supervisor")):
                        validate_live(candidate)
                        guardian = scope_info(record["guardian_unit"])
                        if (
                            guardian["ActiveState"] != "active"
                            or guardian.get("InvocationID")
                            != candidate.get("guardian_invocation_id")
                            or Path(f"/proc/{candidate['supervisor']['pid']}/cgroup")
                            .read_text(encoding="utf-8")
                            .strip()
                            != "0::" + guardian.get("ControlGroup", "")
                        ):
                            raise OwnershipError("startup guardian invocation changed")
                        self.registry.put(candidate)
                        continue
                # A launcher can outlive the caller. Reserve its owner through
                # the startup deadline; absence must be observed on both units.
                if time.time() - record["created_at"] < 35:
                    continue
                guardian = scope_info(record["guardian_unit"])
                scope = scope_info(record["scope"])
                if any(
                    info["ActiveState"] not in ("inactive", "failed")
                    for info in (guardian, scope)
                ):
                    continue
                stop_scope(record)
                remove_runtime(self.registry, record)
            elif record["status"] == "cleanup_failed" or (
                record["status"] in ("running", "stopping")
                and (
                    not alive(record.get("supervisor"))
                    or not Path(record["runtime_dir"]).exists()
                )
            ):
                stop_scope(record)
                remove_runtime(self.registry, record)

    def start(self, session_id):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
            raise ValueError("session_id must contain 1 to 256 characters")
        with self.registry.lock():
            self._reconcile_locked()
            for existing in self.registry.records():
                if (
                    existing["session_id"] == session_id
                    and existing["status"] != "running"
                ):
                    raise RealmError(
                        "realm is "
                        + existing["status"]
                        + "; retry after reconciliation"
                    )
                if (
                    existing["session_id"] == session_id
                    and existing["status"] == "running"
                ):
                    validate_live(existing)
                    existing["last_activity"] = time.time()
                    self.registry.put(existing)
                    return existing
            generation = uuid.uuid4().hex
            realm_id = "r-" + generation[:24]
            runtime = Path(f"/run/user/{os.getuid()}/hr-{generation[:16]}")  # windows-footgun: ok — runtime package rejects non-Linux hosts
            runtime.mkdir(mode=0o700)
            logs = self.registry.root / "logs"
            logs.mkdir(mode=0o700, exist_ok=True)
            logfile = logs / (realm_id + ".log")
            record = {
                "id": realm_id,
                "session_id": session_id,
                "generation": generation,
                "uid": os.getuid(),  # windows-footgun: ok — runtime package rejects non-Linux hosts
                "home": str(self.home),
                "runtime_dir": str(runtime),
                "scope": "hermes-realm-" + generation + ".scope",
                "status": "starting",
                "guardian_unit": "hermes-realm-" + generation + "-guard.service",
                "created_at": time.time(),
                "last_activity": time.time(),
                "size": self.config.size,
                "renderer": self.config.renderer,
                "overlay": self.config.overlay,
                "cursor_theme": self.config.cursor_theme,
                "idle_ttl": self.config.idle_ttl,
                "log_path": str(logfile),
            }
            atomic_json(runtime / "spec.json", asdict(self.config))
            self.registry.put(record)
            env = dict(os.environ)
            # Fresh -P children resolve the CLI package only from this payload,
            # never the systemd working directory or caller PYTHONPATH.
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path=/run/user/{os.getuid()}/bus"  # windows-footgun: ok — runtime package rejects non-Linux hosts
            env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"  # windows-footgun: ok — runtime package rejects non-Linux hosts
            with logfile.open("w") as log:
                os.chmod(logfile, 0o600)
                cleanup = shlex.join(
                    [
                        sys.executable,
                        "-P",
                        "-m",
                        "realms.supervisor",
                        "cleanup",
                        str(self.home),
                        realm_id,
                    ]
                ).replace("%", "%%")
                command = [
                    "systemd-run",
                    "--user",
                    "--quiet",
                    "--collect",
                    "--pipe",
                    "--service-type=exec",
                    "--expand-environment=no",
                    "--unit=" + record["guardian_unit"],
                    "--setenv=PYTHONPATH=" + env["PYTHONPATH"],
                    "--setenv=PATH=" + env["PATH"],
                    "--property=ExecStopPost=:" + cleanup,
                    sys.executable,
                    "-P",
                    "-m",
                    "realms.supervisor",
                    str(self.home),
                    realm_id,
                ]
                try:
                    monitor = subprocess.Popen(
                        command,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        close_fds=True,
                        start_new_session=True,
                    )
                except BaseException:
                    remove_runtime(self.registry, record)
                    raise
            import threading

            threading.Thread(
                target=monitor.wait, name="realm-launcher-reaper", daemon=True
            ).start()
            deadline = time.monotonic() + 30
            try:
                while not (runtime / "ready.json").exists():
                    if (runtime / "error.json").exists() or monitor.poll() is not None:
                        raise RealmError(
                            "realm startup failed; see "
                            + str(logfile)
                            + "\n"
                            + logfile.read_text(encoding="utf-8")[-5000:]
                        )
                    if time.monotonic() > deadline:
                        raise RealmError("realm startup timed out; see " + str(logfile))
                    time.sleep(0.1)
                ready = json.loads((runtime / "ready.json").read_text(encoding="utf-8"))
                record["supervisor"] = json.loads(
                    (runtime / "supervisor.json").read_text(encoding="utf-8")
                )
                info = scope_info(record["scope"])
                if info.get("ActiveState") != "active":
                    raise RealmError("realm scope is not active")
                record.update(
                    status="running",
                    last_activity=time.time(),
                    processes=ready["processes"],
                    vnc_port=None,
                    vnc_socket=ready["vnc_socket"],
                    guardian_invocation_id=scope_info(record["guardian_unit"])[
                        "InvocationID"
                    ],
                    invocation_id=info["InvocationID"],
                    cgroup=info["ControlGroup"],
                )
                atomic_json(runtime / "owner.json", record)
                self.registry.put(record)
                return record
            except BaseException:
                stop_scope(record)
                remove_runtime(self.registry, record)
                raise

    def env(self, realm_id):
        with self.registry.lock():
            record = self.registry.get(realm_id)
            if record["status"] != "running":
                raise RealmError("realm is not running")
            validate_live(record)
            env = json.loads((Path(record["runtime_dir"]) / "ready.json").read_text(encoding="utf-8"))[
                "env"
            ]
            validate_environment(record, env)
            record["last_activity"] = time.time()
            self.registry.put(record)
            return env

    def _rpc(self, realm_id, *, fds=(), **request):
        self.env(realm_id)
        with self.registry.lock():
            record = self.registry.get(realm_id)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(15)
            connection.connect(str(Path(record["runtime_dir"]) / "control"))
            pid, uid, _ = struct.unpack(
                "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            if uid != os.getuid() or identity(pid) != record["processes"]["worker"]:  # windows-footgun: ok — runtime package rejects non-Linux hosts
                raise OwnershipError("realm control endpoint ownership mismatch")
            payload = json.dumps(request).encode() + b"\n"
            if fds:
                import array

                sent = connection.sendmsg(
                    [payload],
                    [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))],
                )
                if sent < len(payload):
                    connection.sendall(payload[sent:])
            else:
                connection.sendall(payload)
            with connection.makefile("rb") as stream:
                response = json.loads(stream.readline(RPC_RESPONSE_LIMIT_BYTES))
        if "error" in response:
            raise RealmError(response["error"])
        return response["result"]

    def shot(self, realm_id, path):
        return self._rpc(
            realm_id, op="shot", path=str(Path(path).expanduser().resolve())
        )

    def resize(self, realm_id, size):
        from .config import parse_size

        parse_size(size)
        self._rpc(realm_id, op="resize", size=size)
        with self.registry.lock():
            record = self.registry.get(realm_id)
            record["size"] = size
            self.registry.put(record)
            return record

    def command_prefix(self, realm_id):
        """Prefix actual Popen argv; caller must use env(id) and sanitized env.

        An absolute script path avoids installing or injecting PYTHONPATH into
        the caller. Equivalent module CLI: python -m realms.launch HOME ID --.
        """
        self.env(realm_id)
        return (
            sys.executable,
            str(Path(__file__).with_name("launch.py").resolve()),
            str(self.home),
            realm_id,
            "--",
        )

    def doctor(self, realm_id=None):
        from .lifecycle import host_control_env

        tools = {
            name: shutil.which(name)
            for name in (
                "labwc",
                "wayvnc",
                "wlr-randr",
                "grim",
                "dbus-daemon",
                "gdbus",
                "Xwayland",
                "systemd-run",
                "systemd-inhibit",
            )
        }
        for path in ("/usr/lib/at-spi-bus-launcher", "/usr/lib/at-spi2-registryd"):
            tools[Path(path).name] = path if os.access(path, os.X_OK) else None
        try:
            user_manager = scope_info("basic.target").get("ActiveState") == "active"
        except (RealmError, OSError, subprocess.SubprocessError):
            user_manager = False
        report = {
            "ok": sys.platform == "linux" and all(tools.values()) and user_manager,
            "platform": sys.platform,
            "tools": tools,
            "systemd_user": user_manager,
            "security_boundary": "GUI separation and owned process lifecycle; not a hostile-code sandbox",
        }
        if realm_id is not None:
            try:
                self.env(realm_id)
                with self.registry.lock():
                    record = self.registry.get(realm_id)
                pids = {
                    int(pid)
                    for pid in (
                        Path("/sys/fs/cgroup")
                        / record["cgroup"].lstrip("/")
                        / "cgroup.procs"
                    )
                    .read_text(encoding="utf-8")
                    .split()
                }
                inhibitors = json.loads(
                    subprocess.run(
                        ["systemd-inhibit", "--list", "--json=short"],
                        env=host_control_env(),
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5,
                    ).stdout
                )
                own = [
                    item
                    for item in inhibitors
                    if item["pid"] in pids and item["who"] == "Hermes Realms"
                ]
                guarded = bool(own) and all(
                    item["what"] == "sleep" and item["mode"] == "block" for item in own
                )
                detail = next(
                    (
                        line.split("GL renderer:", 1)[1].strip()
                        for line in Path(record["log_path"]).read_text(encoding="utf-8").splitlines()
                        if "GL renderer:" in line
                    ),
                    record["renderer"],
                )
                outputs = self._rpc(realm_id, op="outputs")
                for output in outputs:
                    output["current_mode"] = next(
                        (mode for mode in output["modes"] if mode["current"]), None
                    )
                report.update(
                    realm=record,
                    scope_owned=True,
                    sleep_only_inhibitor=guarded,
                    outputs=outputs,
                    renderer_detail=detail,
                )
                report["ok"] = report["ok"] and guarded
            except (RealmError, OSError, ValueError, subprocess.SubprocessError) as exc:
                report.update(ok=False, error=str(exc))
        return report

    def exec(self, realm_id, command, *, cwd=None, wait=False, timeout=30):
        """Run a captured command in the realm.

        Capture keeps the first 256 KiB per stream, drains overflow without
        breaking writes, and reports *_truncated plus output_complete. Results
        survive five minutes after exit and capture EOF; live output never
        expires. At 128 retained/active jobs, admission fails rather than
        evicting results. FD-proxied command_prefix output has no capture cap.
        wait waits for the direct child, not daemonized descendants.
        """
        if (
            not isinstance(command, (list, tuple))
            or not command
            or not all(isinstance(arg, str) and "\x00" not in arg for arg in command)
        ):
            raise ValueError("command must be a nonempty argv list")
        job = self._rpc(
            realm_id,
            op="exec",
            command=list(command),
            cwd=str(Path(cwd or Path.cwd()).resolve()),
        )
        if not wait:
            return job
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self._rpc(realm_id, op="job", job_id=job["job_id"])
            if status["returncode"] is not None:
                return status
            time.sleep(0.05)
        raise RealmError(
            "command wait timed out; job remains owned by realm: " + job["job_id"]
        )

    def stop(self, realm_id):
        with self.registry.lock():
            if not self.registry.path(realm_id).exists():
                return False
            record = self.registry.get(realm_id)
            stop_scope(record)
            remove_runtime(self.registry, record)
        deadline = time.monotonic() + 8
        while alive(record.get("supervisor")) and time.monotonic() < deadline:
            time.sleep(0.1)
        return True
