"""The SDK's ``client.chat`` / ``client.responses`` properties import on first touch.

Two threads first-touching different parts of the SDK deadlock on its import graph
(``openai.resources`` and ``resources.beta`` and ``resources.chat``, and ``resources.chat`` and
``openai.types.chat``): each holds one module lock and waits for the other's, and CPython
raises ``_DeadlockError`` in the loser. The auto-title thread and the agent-build thread lose
this race on cold starts. ``prewarm_openai_resource_modules`` imports the subtrees once,
single-threaded, so the later first touch never takes a module lock.

Each case runs in a subprocess so it starts with a clean ``sys.modules``.
"""

import subprocess
import sys

import pytest

RACE_SCRIPT = """
import importlib, sys, threading

use_barrier = sys.argv[1] == "barrier"
names = sys.argv[2:]
errors = []
barrier = threading.Barrier(len(names))

def touch(name):
    def run():
        if use_barrier:
            barrier.wait()
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}")
    return run

threads = [threading.Thread(target=touch(name)) for name in names]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=30)
print(";".join(errors) if errors else "CLEAN")
"""

PREWARM = "from agent.process_bootstrap import prewarm_openai_resource_modules; prewarm_openai_resource_modules()\n"


def _run(code: str, *argv: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", code, *argv],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr[-800:]}"
    return proc.stdout.strip().splitlines()[-1]


# Each pair closes the cycle only under one start timing, so the timing is part of the case.
@pytest.mark.parametrize(
    ("start", "pair"),
    [
        ("free", ("openai.resources", "openai.resources.chat")),
        ("barrier", ("openai.types.chat", "openai.resources.chat")),
    ],
    ids=["resources-vs-chat", "types-chat-vs-resources-chat"],
)
def test_concurrent_first_touch_after_prewarm_does_not_deadlock(start, pair):
    assert _run(PREWARM + RACE_SCRIPT, start, *pair) == "CLEAN"
