"""Bounded setup subprocesses; never commit conversation routing here."""
import json
import os
from pathlib import Path
import subprocess
import sys

from .setup_plan import PACKAGES, confined, setup_module


def run_child(argv, *, env, timeout, lock_fd=None, privileged=False, cleanup_unit=None):
    """Retain exclusion in an independent process, never a backend thread."""
    from .setup_process import open_pidfd
    owner_fd = open_pidfd(os.getpid())
    try:
        inherited = (owner_fd,) + (() if lock_fd is None else (lock_fd,))
        process = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).with_name("setup_process.py")), str(owner_fd)],
            env=dict(env), cwd=str(Path(__file__).resolve().parents[3]),
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, pass_fds=inherited,
        )
    finally:
        os.close(owner_fd)
    payload = {"argv": argv, "env": dict(env), "timeout": timeout,
               "privileged": privileged, "cleanup_unit": cleanup_unit}
    process.communicate(json.dumps(payload).encode())
    if process.returncode == 124:
        raise ValueError("Setup operation timed out; retry after checking prerequisites")
    if process.returncode:
        raise ValueError("Setup operation failed; check prerequisites and administrator authorization")


def worker_env(home):
    uid = os.getuid()  # windows-footgun: ok — Linux-only plugin
    return {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "HERMES_HOME": str(home),
            "LANG": "C.UTF-8", "XDG_RUNTIME_DIR": f"/run/user/{uid}",
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{uid}/bus"}


def install_plan(plan, phase, *, lock_fd=None):
    from dataclasses import asdict
    from .config import Config

    home = Path(plan["home"])
    confined(home, home / "plugin-data" / "hermes-realms")
    env = worker_env(home)
    packages = plan["packages"]
    if any(package not in set(PACKAGES.values()) for package in packages):
        raise ValueError("Unsupported prerequisite")
    if "config" not in plan or asdict(Config.load(home)) != plan["config"]:
        raise ValueError("Setup configuration changed; prepare the proposal again")
    if packages:
        phase("packages")
        run_child(["/usr/bin/pkexec", "/usr/bin/timeout", "--signal=TERM",
                   "--kill-after=10s", "1800s", "/usr/bin/pacman", "-S",
                   "--needed", "--noconfirm", "--", *packages],
                  env=env, timeout=1820, lock_fd=lock_fd, privileged=True)
    if asdict(Config.load(home)) != plan["config"]:
        raise ValueError("Setup configuration changed; prepare the proposal again")
    phase("install")
    # -I excludes user site/PYTHONPATH; add only the owning product checkout.
    root = str(Path(__file__).resolve().parents[3])
    binding = str(Path(__file__).with_name("_binding.py"))
    if plan["kind"] == "realm":
        source = str(Path(__file__).resolve().parents[1] / "setup.py")
        code = "import sys,runpy; sys.path.insert(0,sys.argv[1]); runpy.run_path(sys.argv[2])['run'](sys.argv[3])"
        run_child([sys.executable, "-I", "-c", code, root, source, str(home)],
                  env=env, timeout=900, lock_fd=lock_fd)
    elif plan["action"] == "install":
        import uuid
        generation = uuid.uuid4().hex
        # Do not pass --iso: the vendor's default path downloads AND verifies.
        code = ("import sys,runpy,json; sys.path.insert(0,sys.argv[1]); "
                "runpy.run_path(sys.argv[2])['load_runtime']('setup_worker')"
                "._install_vm(sys.argv[3],json.loads(sys.argv[4]),sys.argv[5])")
        run_child([sys.executable, "-I", "-c", code, root, binding, str(home),
                   json.dumps(plan["config"]), generation],
                  env=env, timeout=5700, lock_fd=lock_fd,
                  cleanup_unit=f"hermes-vm-base-{generation}.service")


def _install_vm(home, snapshot, generation):
    from .config import Config
    from .vm_manager import VmManager

    frozen = Config(**snapshot)
    manager = VmManager(home)
    if manager.config != frozen:
        raise ValueError("Setup configuration changed; prepare the proposal again")
    manager.config = frozen
    manager.install_base(generation=generation)


def verify_ready(home, kind):
    """Reverify in the backend before it takes ownership of the new realm."""
    if kind == "realm":
        if not setup_module()["describe"](home)["ready"]:
            raise ValueError("Driver verification incomplete")
        from .manager import Manager
        if not Manager(home).doctor()["ok"]:
            raise ValueError("Realms prerequisites are not ready")
    else:
        from .vm_manager import VmManager
        if not VmManager(home).doctor()["ok"]:
            raise ValueError("Omarchy VM prerequisites are not ready")
