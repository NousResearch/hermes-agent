"""Solstice provider must load in the plugin-host runtime (Task 1 probe).

Runtime identity (evidence 2026-10-15, failing run):
- Plugin-host interpreter: the PM store tool python, read from the
  checkout's own published launcher (``.hermes/bin/hermes`` embeds exactly
  what production execs):
  ``/home/lorenzo/.hermes/tools/python-3.14.7+20260901-linux-x64/bin/python3``
  (CPython 3.14.7, run with ``-I`` isolated mode, matching
  ``runtime_command``'s ``[python, "-I", "-c", ...]`` shape; identical to
  ``hermes_cli._launchers.resolve_store_python`` for this checkout).
- Its ``sys.path`` under ``-I`` is the bare stdlib only
  (``lib/python314.zip``, ``lib/python3.14``, ``lib-dynload``,
  ``lib/python3.14/site-packages``); ``import httpx`` raises
  ``ModuleNotFoundError: No module named 'httpx'`` there.
- Verdict: ``PLUGIN_HOST = "tool-python"`` (the store tool python, NOT a
  venv python). The probe below spawns exactly that interpreter with ``-I``
  and loads the bundled solstice plugin through the production discovery
  path (``providers._import_plugin_dir(..., "bundled")``); before the fix it
  fails with ``Failed to load bundled provider plugin solstice: No module
  named 'httpx'`` (the production ``errors.log`` signature), because the
  import chain ``transport.py -> agent/gemini_native_adapter.py:20
  (import httpx)`` cannot resolve under ``-I``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Probe executed inside the plugin-host interpreter. It loads the bundled
# solstice plugin exactly the way production discovery does (file-location
# import under the stable ``plugins.model_providers.<name>`` module name,
# which pulls ``transport.py -> gemini_native_adapter (import httpx)``) and
# reports success only when the provider registers with its transport class.
_PROBE = (
    "import sys; "
    "repo_root = sys.argv[1]; "
    "sys.path.insert(0, repo_root); "
    "from pathlib import Path; "
    "from providers import _REGISTRY, _import_plugin_dir; "
    "_import_plugin_dir(Path(repo_root) / 'plugins' / 'model-providers' / 'solstice', 'bundled'); "
    "profile = _REGISTRY.get('solstice'); "
    "assert profile is not None, 'solstice provider did not register'; "
    "module = sys.modules.get('plugins.model_providers.solstice'); "
    "assert module is not None and getattr(module, 'SolsticeClient', None) is not None, "
    "'solstice transport class missing'; "
    "print('SOLSTICE_PLUGIN_OK')"
)


def _resolve_plugin_host_interpreter() -> tuple[Path, bool]:
    """The plugin-host interpreter, resolved the same way the launcher does.

    Returns ``(interpreter, is_store_tool_python)``. Primary source is the
    checkout's own published launcher (``.hermes/bin/hermes``): its embedded
    interpreter is exactly what production execs (``exec <python> -I -c
    ...``), and reading it is a pure checkout-file read, so the test-session
    ``HERMES_HOME`` sandbox cannot divert it the way a home-derived store
    lookup would. Falls back to the PM store resolution and finally to the
    current interpreter (the launcher's own fallback when no store exists).
    """
    from hermes_cli._launchers import _launcher_python, resolve_store_python

    launcher = REPO_ROOT / ".hermes" / "bin" / ("hermes.exe" if os.name == "nt" else "hermes")
    python = _launcher_python(launcher)
    if python is not None:
        try:
            if python.is_file():
                return (python, True)
        except OSError:
            pass
    store_python = resolve_store_python(REPO_ROOT)
    if store_python is not None:
        return (store_python, True)
    return (Path(sys.executable), False)


def test_solstice_loads_in_plugin_host_runtime(tmp_path):
    interpreter, is_store = _resolve_plugin_host_interpreter()
    home = tmp_path / "home"
    home.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    env["HERMES_HOME"] = str(home)
    completed = subprocess.run(
        [str(interpreter), "-I", "-c", _PROBE, str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert "SOLSTICE_PLUGIN_OK" in completed.stdout, (
        f"solstice plugin failed to load in the plugin-host runtime "
        f"(interpreter={interpreter}, store_tool_python={is_store}): "
        f"rc={completed.returncode} stdout={completed.stdout[-1500:]!r} "
        f"stderr={completed.stderr[-2000:]!r}"
    )
