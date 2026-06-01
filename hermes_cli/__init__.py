"""
Hermes CLI - Unified command-line interface for Hermes Agent.

Provides subcommands for:
- hermes chat          - Interactive chat (same as ./hermes)
- hermes gateway       - Run gateway in foreground
- hermes gateway start - Start gateway service
- hermes gateway stop  - Stop gateway service
- hermes setup         - Interactive setup wizard
- hermes status        - Show status of all components
- hermes cron          - Manage cron jobs
"""

import os
import sys

__version__ = "0.15.1"
__release_date__ = "2026.5.29"


def _ensure_utf8():
    """Force UTF-8 stdout/stderr on Windows to prevent UnicodeEncodeError.

    Windows services and terminals default to cp1252, which cannot encode
    box-drawing characters used in CLI output. This causes unhandled
    UnicodeEncodeError crashes on gateway startup.
    """
    if sys.platform != "win32":
        return
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            if getattr(stream, "encoding", "").lower().replace("-", "") != "utf8":
                new_stream = open(
                    stream.fileno(), "w", encoding="utf-8",
                    buffering=1, closefd=False,
                )
                setattr(sys, stream_name, new_stream)
        except (AttributeError, OSError):
            pass


_ensure_utf8()


def _install_doctor_docker_backend_diagnostics():
    """Patch hermes_cli.doctor with Docker-backend diagnostics on import.

    This keeps the check isolated and best-effort: if the import hook cannot be
    installed, doctor still runs with its existing behavior.
    """
    if "hermes_cli.doctor" in sys.modules:
        try:
            from .doctor_docker_backend import patch_doctor_module
            patch_doctor_module(sys.modules["hermes_cli.doctor"])
        except Exception:
            pass
        return

    try:
        import importlib.abc
        import importlib.machinery
    except Exception:
        return

    class _DoctorDockerDiagnosticsLoader(importlib.abc.Loader):
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def create_module(self, spec):
            create_module = getattr(self._wrapped, "create_module", None)
            if create_module is None:
                return None
            return create_module(spec)

        def exec_module(self, module):
            self._wrapped.exec_module(module)
            try:
                from .doctor_docker_backend import patch_doctor_module
                patch_doctor_module(module)
            except Exception:
                pass

    class _DoctorDockerDiagnosticsFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "hermes_cli.doctor":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            if isinstance(spec.loader, _DoctorDockerDiagnosticsLoader):
                return spec
            spec.loader = _DoctorDockerDiagnosticsLoader(spec.loader)
            return spec

    if not any(f.__class__.__name__ == "_DoctorDockerDiagnosticsFinder" for f in sys.meta_path):
        sys.meta_path.insert(0, _DoctorDockerDiagnosticsFinder())


_install_doctor_docker_backend_diagnostics()
