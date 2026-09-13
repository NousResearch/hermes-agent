"""Exact-path runtime binding for directory plugins and file launchers.

Never claim the generic ``realms`` name or change the application's sys.path.
A path-derived package keeps native and dashboard state shared while allowing
independent installed copies to coexist. Call this file via runpy.run_path.
"""
import _imp
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys


def load_runtime(module=""):
    root = Path(__file__).resolve().parent
    name = "_hermes_realms_" + hashlib.sha256(str(root).encode()).hexdigest()
    origin = str(root / "__init__.py")
    # run_path callers have separate globals, so use Python's import lock rather
    # than a per-copy threading lock while publishing the package.
    _imp.acquire_lock()
    try:
        package = sys.modules.get(name)
        if package is not None:
            if (getattr(package, "__file__", None) != origin
                    or list(getattr(package, "__path__", ())) != [str(root)]):
                raise ImportError("Realms runtime namespace is already occupied")
        else:
            spec = importlib.util.spec_from_file_location(
                name, origin, submodule_search_locations=[str(root)]
            )
            if spec is None or spec.loader is None:
                raise ImportError("Cannot bind Realms runtime package")
            package = importlib.util.module_from_spec(spec)
            sys.modules[name] = package
            try:
                spec.loader.exec_module(package)
            except BaseException:
                del sys.modules[name]
                raise
    finally:
        _imp.release_lock()
    return importlib.import_module("." + module, name) if module else package
