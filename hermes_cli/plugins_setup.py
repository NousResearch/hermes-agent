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


def _invoke(path, action, home, *, revision=""):
    timeout = DESCRIBE_TIMEOUT if action == "describe" else RUN_TIMEOUT
    with subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", _RUNNER, str(path), action, str(home), revision],
        cwd=path.parent, env={**os.environ, "HERMES_HOME": str(home)},
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=os.name != "nt",
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
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
        _invoke(path, "run", home, revision=description["revision"])
        after = _describe(path, home)
        if not after["ready"] or after["revision"] != description["revision"]:
            raise ValueError("Setup did not verify readiness at the reviewed revision. Review and retry enable.")
        return None
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"ok": False, "status": "setup_failed", "name": key,
                "error": str(exc) + " Enablement was not changed; resolve the prerequisite and retry."}
