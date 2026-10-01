"""A recorded stand-in for the commands litco-supervisor runs as root.

It models just enough of useradd/userdel/getent/id, loginctl and systemctl for the
supervisor's handler to be driven end to end without root or systemd: users
exist once useradd ran, units are active once started, and every argv is kept
in ``calls`` in order.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import threading
from pathlib import Path

HOST = Path(__file__).resolve().parents[2] / "deploy" / "host"


def _load_supervisor():
    """Import deploy/host/litco-supervisor (no .py suffix) as a module.

    Registered in sys.modules before it runs, which its dataclasses need.
    """
    name = "litco_supervisor"
    if name in sys.modules:
        return sys.modules[name]
    loader = importlib.machinery.SourceFileLoader(name, str(HOST / "litco-supervisor"))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    sys.modules[name] = module
    loader.exec_module(module)
    return module


sup = _load_supervisor()


class FakeRunner:
    def __init__(self, *, users=(), active=(), fail=None, uid_base=20000):
        self.calls: list = []
        self.users = {name: uid_base + i for i, name in enumerate(users)}
        self.units = {unit: "active" for unit in active}
        self.fail = dict(fail or {})          # command name -> exit code
        self.uid_base = uid_base + len(self.users)
        self.stop_gate: threading.Event | None = None   # set() to let a stop finish
        self.on_stop = None                   # callback(unit) run as the stop begins
        self._lock = threading.Lock()

    def names(self) -> list:
        return [" ".join(argv[:2]) for argv in self.calls]

    def __call__(self, argv):
        argv = list(argv)
        with self._lock:
            self.calls.append(argv)
        key = " ".join(argv[:2])
        for prefix, code in self.fail.items():
            if key.startswith(prefix):
                return sup.CommandResult(code, "", f"{prefix}: simulated failure")
        cmd = argv[0]
        if cmd == "getent":
            return sup.CommandResult(0 if argv[2] in self.users else 2)
        if cmd == "useradd":
            with self._lock:
                self.users[argv[-1]] = self.uid_base
                self.uid_base += 1
            return sup.CommandResult(0)
        if cmd == "id":
            uid = self.users.get(argv[-1])
            return sup.CommandResult(0, f"{uid}\n") if uid is not None else sup.CommandResult(1, "", "no such user")
        if cmd == "userdel":
            with self._lock:
                gone = self.users.pop(argv[-1], None) is None
            return sup.CommandResult(6, "", "user does not exist") if gone else sup.CommandResult(0)
        if cmd in ("install", "loginctl"):
            return sup.CommandResult(0)
        if cmd == "systemctl":
            verb, unit = argv[1], argv[-1]
            state = self.units.get(unit, "inactive")
            if verb == "is-active":
                return sup.CommandResult(0 if state == "active" else 3, f"{state}\n")
            if verb == "start":
                self.units[unit] = "active"
                return sup.CommandResult(0)
            if verb == "reset-failed":
                self.units[unit] = "inactive"
                return sup.CommandResult(0)
            if verb == "stop":
                self.units[unit] = "deactivating"
                if self.on_stop:
                    self.on_stop(unit)
                if self.stop_gate is not None:
                    self.stop_gate.wait(10)
                self.units[unit] = "inactive"
                return sup.CommandResult(0)
        raise AssertionError(f"unexpected command: {argv}")
