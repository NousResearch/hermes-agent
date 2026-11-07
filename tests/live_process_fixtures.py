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

import tempfile
from pathlib import Path

_SLEEPER_SOURCE = "import time\ntime.sleep(300)\n"
_sleeper_script: Path | None = None

#: Substring a caller can wait for in the spawned process's *command line* to know the argv is
#: visible to a cmdline scan. It must name the SCRIPT, not its source text: the source now lives in
#: a file and never appears in the command line the way a ``-c`` snippet used to.
SLEEPER_MARKER = "sleeper.py"


def sleeper_script_path() -> str:
    """Path to the sleeper script, created once per test session."""
    global _sleeper_script
    if _sleeper_script is None:
        path = Path(tempfile.mkdtemp(prefix="hermes-live-sleeper-")) / "sleeper.py"
        path.write_text(_SLEEPER_SOURCE, encoding="utf-8")
        _sleeper_script = path
    return str(_sleeper_script)
