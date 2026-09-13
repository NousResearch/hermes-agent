"""Private desktop worker. All components and RPC-launched apps inherit its scope."""

import ast
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

if __package__ in (None, ""):
    import runpy

    __package__ = runpy.run_path(str(Path(__file__).resolve().with_name("_binding.py")))["load_runtime"]().__name__
from .lifecycle import atomic_json, identity, RealmError, OUTPUT_LIMIT_BYTES


# Admission is bounded rather than evicting results a caller may still need.
# Completed results have a guaranteed five-minute retrieval window; active
# jobs never expire. Limits apply equally to FD-proxied and captured commands.
JOB_LIMIT = 128
JOB_RETENTION_SECONDS = 300
job_clock = time.monotonic


def drain_job_output(process, metadata):
    """Keep the first bounded bytes, but drain overflow so writers succeed.

    Only captured exec has pipes. FD-proxied launch remains an exact passthrough.
    No existing file is ever truncated and no background reader threads are
    introduced (launch's controlling-tty preexec runs in this single thread).
    """
    if "stdout_path" not in metadata:
        return
    for key in ("stdout", "stderr"):
        pipe = getattr(process, key)
        if pipe.closed:
            continue
        path = Path(metadata[key + "_path"])
        remaining = max(0, OUTPUT_LIMIT_BYTES - path.stat().st_size)
        with path.open("ab") as output:
            for _ in range(16):
                try:
                    chunk = os.read(pipe.fileno(), 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    pipe.close()
                    break
                output.write(chunk[:remaining])
                if len(chunk) > remaining:
                    metadata[key + "_truncated"] = True
                remaining = max(0, remaining - len(chunk))
    metadata["output_complete"] = process.stdout.closed and process.stderr.closed


def reap_jobs(jobs, finished):
    now = job_clock()
    for job_id, (process, metadata) in list(jobs.items()):
        drain_job_output(process, metadata)
        if process.poll() is None or not metadata.get("output_complete", True):
            continue
        finished.setdefault(job_id, now)
        if now - finished[job_id] >= JOB_RETENTION_SECONDS:
            for key in ("stdout_path", "stderr_path"):
                if key in metadata:
                    Path(metadata[key]).unlink(missing_ok=True)
            del jobs[job_id]
            del finished[job_id]


def private_environment(runtime):
    runtime = Path(runtime)
    home = runtime / "home"
    home.mkdir(mode=0o700, exist_ok=True)
    env = {
        k: os.environ[k]
        for k in ("PATH", "LANG", "LC_ALL", "USER", "LOGNAME")
        if k in os.environ
    }
    env.update(
        HOME=str(home),
        XDG_RUNTIME_DIR=str(runtime),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_DATA_HOME=str(home / ".local/share"),
        XDG_CACHE_HOME=str(home / ".cache"),
        XDG_STATE_HOME=str(home / ".local/state"),
        XDG_CONFIG_DIRS=str(runtime / "empty"),
        XDG_DATA_DIRS="/usr/local/share:/usr/share",
        XDG_SESSION_TYPE="wayland",
        XDG_CURRENT_DESKTOP="labwc",
        GTK_USE_PORTAL="0",
        ATSPI_DBUS_IMPLEMENTATION="dbus-daemon",
        LABWC_UPDATE_ACTIVATION_ENV="false",
        WLR_BACKENDS="headless",
        WLR_HEADLESS_OUTPUTS="1",
        WLR_LIBINPUT_NO_DEVICES="1",
        CUA_DRIVER_RS_ENABLE_WAYLAND="1",
        XAUTHORITY=str(runtime / "Xauthority"),
        QT_QPA_PLATFORM="wayland",
        GDK_BACKEND="wayland,x11",
        MOZ_ENABLE_WAYLAND="1",
        ELECTRON_OZONE_PLATFORM_HINT="wayland",
        XCURSOR_SIZE="24",
    )
    Path(env["XAUTHORITY"]).touch(mode=0o600)
    (runtime / "empty").mkdir(exist_ok=True)
    return env


def wait_until(check, children, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in children.values()):
            raise RealmError("desktop component exited during startup")
        result = check()
        if result:
            return result
        time.sleep(0.05)
    raise RealmError("desktop startup timed out")


def worker(runtime):
    runtime = Path(runtime)
    spec = json.loads((runtime / "spec.json").read_text(encoding="utf-8"))
    env = private_environment(runtime)
    env["WLR_RENDERER"] = spec["renderer"]
    children = {}
    jobs = {}
    finished_jobs = {}

    def spawn(name, command, *, pass_fds=()):
        children[name] = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=pass_fds,
        )
        return children[name]

    bus = runtime / "bus"
    config = runtime / "bus.conf"
    config.write_text(
        '<!DOCTYPE busconfig PUBLIC "-//freedesktop//DTD D-Bus Bus Configuration 1.0//EN" "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">\n'
        "<busconfig><type>session</type><listen>unix:path="
        + str(bus)
        + "</listen><auth>EXTERNAL</auth>"
        '<policy context="default"><allow send_destination="*" eavesdrop="true"/>'
        '<allow eavesdrop="true"/><allow own="*"/></policy></busconfig>'
    )
    env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=" + str(bus)
    spawn("bus", ["dbus-daemon", "--nofork", "--config-file=" + str(config)])
    wait_until(bus.is_socket, children)
    labwc = runtime / "labwc"
    labwc.mkdir()
    (labwc / "rc.xml").write_text(
        "<labwc_config><core><xwaylandPersistence>yes</xwaylandPersistence></core>"
        "<theme><name></name></theme></labwc_config>"
    )
    for name in ("autostart", "environment", "shutdown"):
        (labwc / name).write_text("", encoding="utf-8")
    publish = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), "publish", str(runtime)]
    )
    spawn("compositor", ["labwc", "-V", "-C", str(labwc), "-s", publish])
    published = runtime / "display.json"
    wait_until(published.exists, children)
    env.update(json.loads(published.read_text(encoding="utf-8")))
    if not env.get("DISPLAY") or not (runtime / env["WAYLAND_DISPLAY"]).is_socket():
        raise RealmError("compositor did not publish a private display")
    subprocess.run(
        ["wlr-randr", "--output", "HEADLESS-1", "--custom-mode", spec["size"] + "@60"],
        env=env,
        check=True,
        timeout=8,
    )
    spawn("atspi", ["/usr/lib/at-spi-bus-launcher", "--launch-immediately"])

    def a11y_address():
        result = subprocess.run(
            [
                "gdbus",
                "call",
                "--address",
                env["DBUS_SESSION_BUS_ADDRESS"],
                "--dest",
                "org.a11y.Bus",
                "--object-path",
                "/org/a11y/bus",
                "--method",
                "org.a11y.Bus.GetAddress",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return ast.literal_eval(result.stdout)[0] if result.returncode == 0 else None

    address = wait_until(a11y_address, children)
    if not address.startswith("unix:path=" + str(runtime) + "/"):
        raise RealmError("accessibility bus is not realm-local")
    env["AT_SPI_BUS_ADDRESS"] = address
    spawn("registry", ["/usr/lib/at-spi2-registryd"])
    vnc_socket = runtime / "wayvnc.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(vnc_socket))
    os.chmod(vnc_socket, 0o600)
    listener.listen(16)
    vnc_config = runtime / "wayvnc.conf"
    vnc_config.write_text("", encoding="utf-8")
    spawn(
        "vnc",
        [
            "wayvnc",
            "-r",
            "-R",
            "-C",
            str(vnc_config),
            "-S",
            str(runtime / "vncctl"),
            "-o",
            "HEADLESS-1",
            "-x",
            str(listener.fileno()),
        ],
        pass_fds=(listener.fileno(),),
    )
    listener.close()
    wait_until(lambda: (runtime / "vncctl").is_socket(), children)
    processes = {name: identity(p.pid) for name, p in children.items()}
    processes["worker"] = identity(os.getpid())
    # Persistent Xwayland is compositor-owned; only inspect this scope's PIDs.
    cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8").strip().split("::", 1)[1]
    for pid in (
        (Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "cgroup.procs")
        .read_text(encoding="utf-8")
        .split()
    ):
        try:
            if Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip() == "Xwayland":
                processes["xwayland"] = identity(int(pid))
        except FileNotFoundError:
            pass
    control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    control.bind(str(runtime / "control"))
    os.chmod(runtime / "control", 0o600)
    control.listen(16)
    control.settimeout(0.2)
    atomic_json(
        runtime / "ready.json",
        {
            "env": env,
            "processes": processes,
            "vnc_port": None,
            "vnc_socket": str(vnc_socket),
        },
    )
    while True:
        reap_jobs(jobs, finished_jobs)
        if any(p.poll() is not None for p in children.values()):
            raise RealmError("desktop component crashed")
        import select

        readers = [
            pipe
            for process, _ in jobs.values()
            for pipe in (process.stdout, process.stderr)
            if pipe is not None and not pipe.closed
        ]
        readable, _, _ = select.select([control, *readers], [], [], 0.2)
        if control not in readable:
            continue
        try:
            connection, _ = control.accept()
        except socket.timeout:
            continue
        with connection:
            connection.settimeout(5)
            received_fds = []
            try:
                import struct

                _, uid, _ = struct.unpack(
                    "3i",
                    connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12),
                )
                if uid != os.getuid():  # windows-footgun: ok — runtime package rejects non-Linux hosts
                    raise RealmError("foreign RPC peer")
                import array

                line, ancillary, flags, _ = connection.recvmsg(
                    1024 * 1024, socket.CMSG_SPACE(12)
                )
                for level, kind, data in ancillary:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        descriptors = array.array("i")
                        descriptors.frombytes(
                            data[: len(data) - len(data) % descriptors.itemsize]
                        )
                        received_fds.extend(descriptors)
                if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                    raise RealmError("truncated control message")
                while not line.endswith(b"\n") and len(line) < 1024 * 1024:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    line += chunk
                if len(line) >= 1024 * 1024:
                    raise RealmError("control message too large")
                request = json.loads(line)
                reap_jobs(jobs, finished_jobs)
                if request["op"] in ("launch", "exec") and len(jobs) >= JOB_LIMIT:
                    raise RealmError(
                        "realm job capacity reached; retry after completed results expire"
                    )
                if (
                    request["op"] in ("job", "signal")
                    and request.get("job_id") not in jobs
                ):
                    raise RealmError("unknown or expired realm job")
                if request["op"] == "outputs":
                    result = json.loads(
                        subprocess.run(
                            ["wlr-randr", "--json"],
                            env=env,
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        ).stdout
                    )
                elif request["op"] == "shot":
                    subprocess.run(
                        ["grim", "-o", "HEADLESS-1", request["path"]],
                        env=env,
                        check=True,
                        timeout=10,
                    )
                    result = request["path"]
                elif request["op"] == "resize":
                    from .config import parse_size

                    parse_size(request["size"])
                    subprocess.run(
                        [
                            "wlr-randr",
                            "--output",
                            "HEADLESS-1",
                            "--custom-mode",
                            request["size"] + "@60",
                        ],
                        env=env,
                        check=True,
                        timeout=8,
                    )
                    result = request["size"]
                elif request["op"] == "launch":
                    import uuid
                    from .lifecycle import validate_environment

                    if len(received_fds) != 3:
                        raise RealmError(
                            "launch requires exactly stdin, stdout and stderr FDs"
                        )
                    owner = json.loads((runtime / "owner.json").read_text(encoding="utf-8"))
                    validate_environment(owner, request["env"])

                    def acquire_tty():
                        if os.isatty(0):
                            import fcntl
                            import termios

                            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

                    process = subprocess.Popen(
                        request["command"],
                        env=request["env"],
                        cwd=request["cwd"],
                        stdin=received_fds[0],
                        stdout=received_fds[1],
                        stderr=received_fds[2],
                        close_fds=True,
                        start_new_session=True,
                        preexec_fn=acquire_tty,
                    )
                    job_id = uuid.uuid4().hex
                    metadata = {
                        "job_id": job_id,
                        "pid": process.pid,
                        "result_retention_seconds": JOB_RETENTION_SECONDS,
                        "start_time": (identity(process.pid) or {}).get("start_time"),
                    }
                    jobs[job_id] = (process, metadata)
                    result = metadata
                elif request["op"] == "signal":
                    import signal

                    process, metadata = jobs[request["job_id"]]
                    if process.poll() is None:
                        number = int(request["signal"])
                        if number not in (
                            signal.SIGINT,
                            signal.SIGTERM,
                            signal.SIGHUP,  # windows-footgun: ok — runtime package rejects non-Linux hosts
                            signal.SIGWINCH,
                            signal.SIGQUIT,  # windows-footgun: ok — runtime package rejects non-Linux hosts
                            signal.SIGKILL,  # windows-footgun: ok — runtime package rejects non-Linux hosts
                        ):
                            raise RealmError("unsupported forwarded signal")
                        os.killpg(process.pid, number)  # windows-footgun: ok — runtime package rejects non-Linux hosts
                    result = True
                elif request["op"] == "exec":
                    import uuid

                    job_id = uuid.uuid4().hex
                    stdout_path = runtime / (job_id + ".stdout")
                    stderr_path = runtime / (job_id + ".stderr")
                    try:
                        stdout_path.touch(mode=0o600)
                        stderr_path.touch(mode=0o600)
                        process = subprocess.Popen(
                            request["command"],
                            env=env,
                            cwd=request["cwd"],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            close_fds=True,
                            start_new_session=True,
                        )
                    except BaseException:
                        stdout_path.unlink(missing_ok=True)
                        stderr_path.unlink(missing_ok=True)
                        raise
                    os.set_blocking(process.stdout.fileno(), False)
                    os.set_blocking(process.stderr.fileno(), False)
                    metadata = {
                        "job_id": job_id,
                        "pid": process.pid,
                        "result_retention_seconds": JOB_RETENTION_SECONDS,
                        "start_time": (identity(process.pid) or {}).get("start_time"),
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                        "output_limit_bytes": OUTPUT_LIMIT_BYTES,
                        "output_complete": False,
                        "stdout_truncated": False,
                        "stderr_truncated": False,
                    }
                    jobs[job_id] = (process, metadata)
                    result = metadata
                elif request["op"] == "job":
                    process, metadata = jobs[request["job_id"]]
                    result = dict(metadata, returncode=process.poll())
                    if result["returncode"] is not None and "stdout_path" in metadata:
                        for key in ("stdout", "stderr"):
                            with open(metadata[key + "_path"], "rb") as output:
                                result[key] = output.read(OUTPUT_LIMIT_BYTES).decode(
                                    "utf-8", errors="replace"
                                )
                else:
                    raise RealmError("unknown realm operation")
                response = {"result": result}
            except Exception as exc:
                response = {"error": str(exc)}
            finally:
                for fd in received_fds:
                    os.close(fd)
            try:
                connection.sendall(json.dumps(response).encode() + b"\n")
            except BrokenPipeError:
                pass


if __name__ == "__main__":
    if sys.argv[1] == "publish":
        # Executed as a file by labwc, before package imports are needed.
        atomic_json(
            Path(sys.argv[2]) / "display.json",
            {
                key: os.environ[key]
                for key in ("WAYLAND_DISPLAY", "DISPLAY")
                if key in os.environ
            },
        )
    else:
        worker(sys.argv[2])
