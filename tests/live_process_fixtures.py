"""Sleeper script used by the live process-topology E2Es to stand in for real Hermes processes.

The fixtures spawn ``python <sleeper.py> <argv tail...>``: the tail is inert to the child but
fully visible to psutil / ``Win32_Process`` cmdline scans, which is what the detection and
classification code reads.

It must NOT be ``python -c "import time; time.sleep(...)" <tail>``. A ``-c`` command line is an
interpreter running inline source, and the identity matchers deliberately refuse to read the
trailing argv off one — that tail belongs to a program the inline source may spawn LATER, which is
how the post-update gateway restart watcher was mistaken for a live gateway (#107002). A ``-c``
fixture therefore no longer stands in for anything.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_SLEEPER_SOURCE = "import time\ntime.sleep(300)\n"
_sleeper_script: Path | None = None
_hermes_module_root: Path | None = None

#: Substring a caller can wait for in the spawned process's *command line* to know the argv is
#: visible to a cmdline scan. It must name the SCRIPT, not its source text: the source now lives in
#: a file and never appears in the command line the way a ``-c`` snippet used to.
SLEEPER_MARKER = "sleeper.py"

#: Marker for :func:`hermes_module_root` — the entry module's own dotted name, as it appears in the
#: spawned process's command line.
HERMES_ENTRYPOINT_MARKER = "hermes_cli.main"


def sleeper_script_path() -> str:
    """Path to the sleeper script, created once per test session."""
    global _sleeper_script
    if _sleeper_script is None:
        path = Path(tempfile.mkdtemp(prefix="hermes-live-sleeper-")) / "sleeper.py"
        path.write_text(_SLEEPER_SOURCE, encoding="utf-8")
        _sleeper_script = path
    return str(_sleeper_script)


def hermes_module_root() -> str:
    """Directory to run ``python -m hermes_cli.main <sub>`` from, for fixtures that must BE Hermes.

    ``sleeper_script_path()`` stands a process up with an inert Hermes argv TAIL
    (``python sleeper.py -m hermes_cli.main serve``). That is deliberately NOT a Hermes process to
    the identity matchers: the interpreter's selected script is ``sleeper.py``, and a Hermes-looking
    tail behind an unrelated script is the exact false positive #121156 exists to stop — it is how
    ``hermes update`` came to reap processes whose argv merely contained ``hermes serve``.

    A fixture standing in for the DESKTOP BACKEND must therefore carry the Desktop's real spawn
    shape, ``python -m hermes_cli.main <subcommand>``, with no inert tail at all. This returns a
    throwaway package root holding a sleeping ``hermes_cli.main``.

    Spawn it with ``-P`` and ``PYTHONPATH`` set to this root, and leave ``cwd`` at the project root
    (see :func:`hermes_backend_spawn_kwargs`). Both halves matter and pull against each other:

    * ``-P`` keeps the interpreter from prepending ``cwd`` to ``sys.path``, so ``-m hermes_cli.main``
      resolves to this sleeping stub instead of the repo's real entry point -- which would launch an
      actual backend inside the E2E.
    * ``cwd`` must nevertheless stay at the project root, because the Windows venv holder scan only
      keeps a ``-m hermes_cli.main`` process whose command line or CWD is under the project root
      (``_detect_venv_python_processes``). Spawning from the stub directory makes the holder invisible
      to the very scan the fixture exists to feed.
    """
    global _hermes_module_root
    if _hermes_module_root is None:
        root = Path(tempfile.mkdtemp(prefix="hermes-live-entry-"))
        package = root / "hermes_cli"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "main.py").write_text(_SLEEPER_SOURCE, encoding="utf-8")
        _hermes_module_root = root
    return str(_hermes_module_root)


def hermes_backend_spawn_kwargs(project_root) -> dict:
    """``subprocess.Popen`` kwargs that make ``-m hermes_cli.main`` a visible, stubbed backend.

    Keeps the two constraints from :func:`hermes_module_root` in one place so a caller cannot satisfy
    one and silently break the other.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = hermes_module_root()
    return {"cwd": str(project_root), "env": env}
