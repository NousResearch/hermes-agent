"""Per-process witness for the classic-CLI live session (``test_classic_cli_live_session.py``).

Loaded as ``sitecustomize`` by every Python process the CLI test spawns (the ``PYTHONPATH`` entry
for this directory is set only in that test's child env). Appends one JSON line per event to
``$HERMES_E2E_WITNESS``; never changes behaviour. Events:

* ``startup <argv>``         emitted once, at interpreter start, by every instrumented process: the
                             attestation that this pid is witnessed. A spawned CLI with no
                             ``startup`` line was not instrumented, and its silence proves nothing.
* ``agent_init <class>``     an ``AIAgent`` was constructed in this process
* ``tool <name>``            ``tools.registry`` dispatched a tool in this process
* ``sqlite_write <file>:<table>``  an INSERT/UPDATE/DELETE on a ``*.db`` file opened by this process

``role`` is ``gateway`` for ``gateway run`` (the daemon), else ``cli``; ``ppid`` lets the test tie
a line to the process it spawned. The test asserts the contract of the one-gateway design: every
CLI process it started attests, builds no agent, runs no tool and writes no ``state.db`` row; the
gateway does all three.
"""

import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading

_PATH = os.environ.get("HERMES_E2E_WITNESS")
_lock = threading.Lock()


def _role():
    argv = list(getattr(sys, "argv", None) or [])
    for i, word in enumerate(argv[:-1]):
        if word == "gateway" and argv[i + 1] == "run":
            return "gateway"
    return "cli"


def _emit(kind, detail=""):
    try:
        line = json.dumps({"pid": os.getpid(), "ppid": os.getppid(), "role": _role(), "kind": kind,
                           "detail": str(detail)[:200]})
        with _lock, open(_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # health: allow BLE001 -- a witness must never break the process it watches
        pass


if _PATH:
    # The attestation. At sitecustomize time ``sys.argv`` is already the command's arguments
    # (``-m`` stands in for the module path until runpy fills it in).
    _emit("startup", " ".join(getattr(sys, "argv", None) or []))

    import sqlite3

    _connect = sqlite3.connect

    def _sqlite_connect(database, *args, **kwargs):
        conn = _connect(database, *args, **kwargs)
        name = str(database).split("?", 1)[0].rsplit("/", 1)[-1]
        if name.endswith(".db"):
            def authorizer(action, table, *_rest):
                if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE) \
                        and table and not str(table).startswith("sqlite_"):
                    _emit("sqlite_write", f"{name}:{table}")
                return sqlite3.SQLITE_OK
            try:
                conn.set_authorizer(authorizer)
            except Exception:  # health: allow BLE001 -- witness only
                pass
        return conn

    sqlite3.connect = _sqlite_connect

    def _patch_run_agent(mod):
        cls = getattr(mod, "AIAgent", None)
        if cls is None:
            return
        original = cls.__init__

        def init(self, *args, **kwargs):
            _emit("agent_init", type(self).__name__)
            return original(self, *args, **kwargs)
        cls.__init__ = init

    def _patch_registry(mod):
        cls = getattr(mod, "ToolRegistry", None)
        if cls is None or not hasattr(cls, "dispatch"):
            return
        original = cls.dispatch

        def dispatch(self, name, args, *rest, **kwargs):
            _emit("tool", name)
            return original(self, name, args, *rest, **kwargs)
        cls.dispatch = dispatch

    _PATCHES = {"run_agent": _patch_run_agent, "tools.registry": _patch_registry}

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner, patch):
            self._inner, self._patch = inner, patch

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            try:
                self._patch(module)
            except Exception as exc:  # health: allow BLE001 -- witness only; recorded for the test
                _emit("witness_error", f"{module.__name__}: {exc!r}")

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            patch = _PATCHES.get(name)
            if patch is None:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return None
            spec.loader = _Loader(spec.loader, patch)
            return spec

    sys.meta_path.insert(0, _Finder())
