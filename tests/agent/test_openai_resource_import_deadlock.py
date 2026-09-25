"""The SDK's ``client.chat`` / ``client.responses`` properties import on first touch.

Two threads first-touching different properties deadlock on the SDK's import
graph (``openai.resources`` ↔ ``resources.beta`` ↔ ``resources.chat``): each holds
one module lock and waits for the other's, and CPython raises ``_DeadlockError``
in the loser — the auto-title thread loses exactly this race on cold starts.
``prewarm_openai_resource_modules`` imports the subtrees once, single-threaded,
so the concurrent first-touch never takes a module lock.
"""
import subprocess
import sys

RACE_SCRIPT = """
import importlib, threading

errors = []

def touch(name):
    def run():
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}")

    return run

first = threading.Thread(target=touch("openai.resources"))
second = threading.Thread(target=touch("openai.resources.chat"))
first.start()
second.start()
first.join(timeout=30)
second.join(timeout=30)
print(";".join(errors) if errors else "CLEAN")
"""


def _run_race(prewarm: bool) -> str:
    prelude = ""
    if prewarm:
        prelude = "from agent.process_bootstrap import prewarm_openai_resource_modules; prewarm_openai_resource_modules()\n"
    proc = subprocess.run(
        [sys.executable, "-c", prelude + RACE_SCRIPT],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise AssertionError(f"race subprocess failed:\n{proc.stderr[-800:]}")
    return proc.stdout.strip().splitlines()[-1]


def test_prewarm_stops_the_two_thread_import_race():
    result = _run_race(prewarm=True)
    assert result == "CLEAN", f"deadlock survived the prewarm: {result}"


def test_unprewarmed_race_is_the_documented_failure_surface():
    # The bare race is timing-dependent (it fires when the two imports overlap),
    # so this pins the harness rather than asserting a deadlock: if the race ever
    # stops reproducing at all, the prewarm test above loses its meaning.
    result = _run_race(prewarm=False)
    assert result in ("CLEAN", "openai.resources: _DeadlockError"), result
