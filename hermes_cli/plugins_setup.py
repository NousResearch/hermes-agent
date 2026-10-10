"""Explicit native-plugin setup. Never called by discovery or runtime loading.

Setup is trusted plugin code, not a sandbox. Authors must keep describe read-only.
The subprocess avoids importing the plugin into (or changing env in) a live gateway.
"""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path
import subprocess
import sys

from hermes_constants import get_hermes_home

DESCRIBE_TIMEOUT = 15
RUN_TIMEOUT = 300

# run_path loads only the reviewed root setup.py, never the plugin __init__.py.
# The run child checks revision in that same module instance, guarding a package
# replacement between the host description and subprocess launch.
_RUNNER = """
import contextlib, json, pathlib, runpy, sys
with contextlib.redirect_stdout(sys.stderr):
    module = runpy.run_path(sys.argv[1])
    home = pathlib.Path(sys.argv[3])
    if sys.argv[2] == "run":
        current = module["describe"](home)
        if current.get("revision") != sys.argv[4]:
            raise ValueError("Setup revision changed before execution; review and retry enable.")
    result = module[sys.argv[2]](home)
print(json.dumps(result if sys.argv[2] == "describe" else None))
"""


def _entrypoint(entry):
    from hermes_cli.plugins_cmd import _read_manifest
    if not entry[4]:
        return None
    root = Path(entry[4]).absolute()
    manifest = _read_manifest(root)
    if "setup" not in manifest:
        return None
    setup = manifest["setup"]
    if not isinstance(setup, dict) or setup.get("entrypoint") != "setup.py":
        raise ValueError("Native setup.entrypoint must be the plugin-owned root setup.py.")
    path = root / "setup.py"
    if any(p.is_symlink() for p in (path, *root.parents, root)) or not path.is_file():
        raise ValueError("Native setup.py must be a regular, non-symlink file in the plugin package.")
    if entry[3] not in {"user", "git", "bundled"}:
        raise ValueError("Setup requires a trusted installed or bundled native plugin package.")
    return path


def _await_group_exit(pgid, timeout=5.0):
    """SIGKILL delivery is asynchronous: wait until no live member of the setup group remains."""
    import time

    import psutil

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = False
        for proc in psutil.process_iter():
            try:
                if os.getpgid(proc.pid) == pgid and proc.status() != psutil.STATUS_ZOMBIE:
                    alive = True
                    break
            except (OSError, psutil.Error):
                continue
        if not alive:
            return
        time.sleep(0.05)


def _setup_python():
    """The interpreter of the environment PM committed for this install.

    Boot activation puts that environment's site-packages on ``sys.path`` without changing
    ``sys.executable``, so an isolated (``-I``) child of ``sys.executable`` would lose the
    plugin's declared dependencies. With nothing committed (a developer venv), the running
    interpreter already carries its own packages.
    """
    from pm.environments import committed_venv, venv_python
    from pm.paths import repo_root

    try:
        environment = committed_venv(repo_root())
    except RuntimeError as exc:  # a malformed committed selection fails closed
        raise ValueError(f"Cannot select the dependency environment for plugin setup: {exc}.") from exc
    return str(venv_python(environment)) if environment is not None else sys.executable


def _prepare_dependencies(path):
    """Install the plugin's declared Python dependencies before its setup runs.

    The enable admission installs them only after setup succeeds, so a first-time setup would
    otherwise run without them. PM prepares the enabled members plus this candidate, as for a
    memory provider's setup; plugin enablement is still committed only by the admission.
    """
    from pm.client import sync_venv
    from pm.plugin_declarations import read_python_declaration
    from pm.plugin_inputs import Candidates

    try:
        if read_python_declaration(path.parent).is_member:
            sync_venv(explicit=True, plugins=Candidates([path.parent]))
    except Exception as exc:
        raise ValueError(f"Plugin dependencies could not be prepared for setup: {exc}.") from exc


def _invoke(path, action, home, *, revision=""):
    timeout = DESCRIBE_TIMEOUT if action == "describe" else RUN_TIMEOUT
    with subprocess.Popen(
        [_setup_python(), "-I", "-B", "-c", _RUNNER, str(path), action, str(home), revision],
        cwd=path.parent, env={**os.environ, "HERMES_HOME": str(home)},
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", start_new_session=os.name != "nt",
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)  # windows-footgun: ok — POSIX-only branch
            else:
                process.kill()
            process.wait()
            if os.name != "nt":
                _await_group_exit(process.pid)
            raise ValueError(f"Plugin setup {action} timed out after {timeout}s. Check prerequisites and retry.") from None
        if process.returncode:
            raise ValueError(f"Plugin setup {action} failed: {stderr[-2000:].strip()}")
    return json.loads(stdout)


def _describe(path, home):
    value = _invoke(path, "describe", home)
    if (not isinstance(value, dict) or not isinstance(value.get("revision"), str)
            or not value["revision"] or type(value.get("ready")) is not bool
            or not isinstance(value.get("summary"), str)
            or not isinstance(value.get("details"), list)
            or not all(isinstance(d, str) for d in value["details"])):
        raise ValueError("setup.describe must return revision, ready, summary and details.")
    return {k: value[k] for k in ("revision", "ready", "summary", "details")}


def prepare_plugin_setup(entry, *, setup_consent=None):
    """Return a refusal or None when ready; caller holds the profile mutation lock."""
    key = entry[5]
    home = get_hermes_home().resolve()
    try:
        path = _entrypoint(entry)
        if path is None:
            return None
        description = _describe(path, home)
        if description["ready"]:
            return None
        consent = {"key": key, "hermes_home": str(home), "revision": description["revision"]}
        if setup_consent != consent:
            return {"ok": False, "status": "consent_required", "name": key,
                    "error": "Review native setup and explicitly consent before enabling this plugin.",
                    "setup": description, "consent": consent}
        _prepare_dependencies(path)
        _invoke(path, "run", home, revision=description["revision"])
        after = _describe(path, home)
        if not after["ready"] or after["revision"] != description["revision"]:
            raise ValueError("Setup did not verify readiness at the reviewed revision. Review and retry enable.")
        return None
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"ok": False, "status": "setup_failed", "name": key,
                "error": str(exc) + " Enablement was not changed; resolve the prerequisite and retry."}
