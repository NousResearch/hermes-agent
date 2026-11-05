"""The ``sitecustomize`` a crash-point test writes into its subprocess's path.

``FAULT_POINT`` names ``module:qualname`` on the tree the subprocess imports. When that function is called (the
call number ``FAULT_SKIP`` + 1 among the calls whose arguments' ``repr`` contains ``FAULT_MATCH``), the process
writes ``FAULT_MARKER`` and ends with ``os._exit(FAULT_EXIT)``, before the function body (``FAULT_WHEN=before``) or
once it returns (``after``). The module is patched as it is imported, or at once when it already is. Every
connection to an address off this host is refused before it opens, and logged with its stack to ``FAULT_NET_LOG``.
"""

import importlib.abc
import importlib.machinery
import inspect
import os
import socket
import sys
import traceback

FAULT_EXIT = 86


def _net_guard():
    log = os.environ.get("FAULT_NET_LOG")
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def local(address):
        if not isinstance(address, tuple):
            return True
        host = str(address[0])
        return host in ("localhost", "::1") or host.startswith("127.")

    def refuse(address):
        if log:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"{address!r}\n{''.join(traceback.format_stack()[:-2])}\n")
        raise ConnectionRefusedError(f"crash-point test: no connection off this host ({address!r})")

    def connect(self, address):
        if not local(address):
            refuse(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        if not local(address):
            refuse(address)
        return real_connect_ex(self, address)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex


def _die(point, frames):
    marker = os.environ.get("FAULT_MARKER")
    if marker:
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, (point + "\n" + "".join(frames)).encode())
        os.close(fd)
    os._exit(FAULT_EXIT)


def _wrap(point, func, when, match, skip):
    calls = [0]

    def wrapper(*args, **kwargs):
        hit = not match or match in repr((args, kwargs))
        if hit:
            calls[0] += 1
        if hit and calls[0] > skip and when == "before":
            _die(point, traceback.format_stack()[:-1])
        result = func(*args, **kwargs)
        if hit and calls[0] > skip and when == "after":
            _die(point, traceback.format_stack()[:-1])
        return result

    wrapper.__wrapped__ = func
    return wrapper


def _install(module, point, qualname, when, match, skip):
    *path, name = qualname.split(".")
    owner = module
    for part in path:
        owner = getattr(owner, part)
    raw = inspect.getattr_static(owner, name)
    if isinstance(raw, staticmethod):
        setattr(owner, name, staticmethod(_wrap(point, raw.__func__, when, match, skip)))
    elif isinstance(raw, classmethod):
        setattr(owner, name, classmethod(_wrap(point, raw.__func__, when, match, skip)))
    else:
        setattr(owner, name, _wrap(point, raw, when, match, skip))


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Patches the target module right after it executes."""

    def __init__(self, module_name, install):
        self.module_name, self.install = module_name, install

    def find_spec(self, name, path=None, target=None):
        if name != self.module_name:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module, install = spec.loader.exec_module, self.install

        def exec_and_patch(module):
            exec_module(module)
            install(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def _arm():
    point = os.environ.get("FAULT_POINT")
    if not point:
        return
    module_name, _, qualname = point.partition(":")
    when, match = os.environ.get("FAULT_WHEN", "before"), os.environ.get("FAULT_MATCH", "")
    skip = int(os.environ.get("FAULT_SKIP", "0"))

    def install(module):
        _install(module, point, qualname, when, match, skip)

    if module_name in sys.modules:
        install(sys.modules[module_name])
    else:
        sys.meta_path.insert(0, _PatchOnImport(module_name, install))


_net_guard()
_arm()
