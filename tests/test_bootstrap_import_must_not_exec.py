"""Contract: importing a hermes module must never replace the importing process's image.

``hermes_bootstrap`` finishes an armed source update and ``os.execv``-s the managed
interpreter **from its module body**, which every entry-point module imports. A plain
library import therefore reached that relaunch and lost its image mid-run — no traceback,
no summary, nothing on stdout or stderr, just ``rc=1``. Two independent reports and one
in-tree trigger:

* ``agent/process_bootstrap.py`` does ``from hermes_bootstrap import
  _happy_eyeballs_create_connection`` for one socket helper (#122160), reached at run
  time from ``agent/auxiliary_client.py`` and, via the deferred import in
  ``agent/context_compressor.py``, from an ordinary state-db write;
* an external consumer importing ``hermes_cli`` at module level (#122160).

The answer these tests pin: **no.** Only a process that is *becoming* a Hermes entry
point may re-exec itself; repair, PM activation, the Happy Eyeballs shim and the scratch
TMPDIR keep running for every importer.

Each case runs in a child process with an audit hook (stdlib, no new dependency) that
appends every image-replacing or process-killing request to a file and flushes it
immediately — so a child that *does* get exec'd has already left its record behind,
which is exactly the failure this contract forbids.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Audit events that would replace or end the importing process. ``subprocess.Popen`` is
# the Windows shape of the same relaunch (``hermes_bootstrap`` has no exec there).
FATAL_EVENTS = ("os.exec", "os.execv", "os._exit", "os.fork", "subprocess.Popen")

_CHILD_PREAMBLE = f"""
import json, os, sys, traceback
sys.path.insert(0, {str(REPO_ROOT)!r})
events_path = os.environ["HERMES_PROBE_EVENTS"]
def _hook(event, args):
    if event in {FATAL_EVENTS!r}:
        with open(events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({{"event": event, "args": repr(args)[:400],
                                 "stack": [f.filename.rsplit("/", 1)[-1] + ":" + str(f.lineno)
                                           for f in traceback.extract_stack()[:-2]][-12:]}}) + "\\n")
            fh.flush()
            os.fsync(fh.fileno())
sys.addaudithook(_hook)
"""


def _run_child(tmp_path: Path, body: str) -> dict:
    """Run *body* in a fresh interpreter; return its report plus the audit events it saw."""
    events = tmp_path / "events.jsonl"
    events.write_text("", encoding="utf-8")
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"), HERMES_PROBE_EVENTS=str(events))
    env.pop("HERMES_DISABLE_LAZY_INSTALLS", None)  # the guard under test must be active
    script = tmp_path / "child.py"
    script.write_text(_CHILD_PREAMBLE + body, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-B", str(script)],
        capture_output=True, text=True, timeout=180, env=env, cwd=tmp_path,
    )
    report = {"returncode": completed.returncode, "stdout": completed.stdout,
              "stderr": completed.stderr,
              "events": [json.loads(line) for line in events.read_text().splitlines() if line]}
    return report


ARM_LAUNCH = """
import hermes_cli.venv_sync as _vs
# Arm the branch under test: the launch decision answers "restart cleanly into the
# managed interpreter", which is what prepare_launch() returns when a source update is
# owed. Supplying that answer is what makes the invariant observable without leaving a
# real install on a stale dependency generation.
_vs.prepare_launch = lambda root, argv: __import__("pathlib").Path(sys.executable)
"""


def _arm_and_import(tmp_path: Path, target: str) -> dict:
    return _run_child(tmp_path, ARM_LAUNCH + f"""
import {target}
print("IMPORT-OK {target}")
""")


def test_library_import_does_not_replace_the_process_image(tmp_path):
    """An ordinary ``import agent.process_bootstrap`` may not exec, fork or exit."""
    report = _run_child(tmp_path, """
import agent.process_bootstrap
print("IMPORT-OK agent.process_bootstrap")
""")
    assert report["returncode"] == 0, report["stderr"][-2000:]
    assert "IMPORT-OK agent.process_bootstrap" in report["stdout"]
    assert report["events"] == [], (
        "importing a hermes module replaced or ended the importing process: "
        f"{json.dumps(report['events'], indent=2)}"
    )


def test_armed_launch_does_not_exec_a_library_importer(tmp_path):
    """With the relaunch armed, a library importer still keeps its image.

    Without this case the test above only passes because the dependency tree happens to
    be current: ``prepare_launch()`` returns ``None`` and the exec branch is never
    reached. Arming it is what pins the contract instead of the current install state.
    """
    report = _arm_and_import(tmp_path, "agent.process_bootstrap")
    assert report["returncode"] == 0, report["stderr"][-2000:]
    assert "IMPORT-OK agent.process_bootstrap" in report["stdout"], report["stdout"][-2000:]
    assert report["events"] == [], (
        "a library import replaced the caller's image while finishing a source update "
        f"(#122160): {json.dumps(report['events'], indent=2)}"
    )


@pytest.mark.parametrize("target", ["agent.conversation_loop", "hermes_state"])
def test_armed_launch_does_not_exec_other_library_surfaces(tmp_path, target):
    """Same contract for the other deferred-import paths that reached the relaunch."""
    report = _arm_and_import(tmp_path, target)
    assert report["returncode"] == 0, report["stderr"][-2000:]
    assert report["events"] == [], f"import {target} took a launch action: {report['events']}"


@pytest.mark.platforms("posix")
def test_entry_point_still_relaunches_when_the_launch_is_armed(tmp_path):
    """The gate must not cost a real launch its tail.

    ``relaunch_command`` has a first-class ``argv[0] == "-c"`` branch (the published
    launchers boot that way), so a ``-c`` launch is an entry point and must still exec.
    Intercepted rather than performed: Windows has no exec and takes the
    ``subprocess.call`` arm instead, which the armed tests above already watch.
    """
    report = _run_child(tmp_path, """
import os
attempts = []
def _intercept(path, argv):
    attempts.append([str(path), [str(a) for a in argv]])
    raise RuntimeError("probe: os.execv intercepted (no image replacement performed)")
os.execv = _intercept
import hermes_cli.venv_sync as _vs
_vs.prepare_launch = lambda root, argv: __import__("pathlib").Path(sys.executable)
import hermes_bootstrap
print("ENTRY-ATTEMPTS", len(attempts))
print("WOULD-BECOME", attempts[0][1][-1] if attempts else "")
""")
    assert "ENTRY-ATTEMPTS 1" in report["stdout"], (
        "an entry-point launch lost its relaunch: "
        f"stdout={report['stdout'][-1500:]} stderr={report['stderr'][-1500:]}"
    )
    assert "runpy" in report["stdout"], report["stdout"][-1500:]
