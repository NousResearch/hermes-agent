"""_evict_modules must not crash when an unrelated thread mutates sys.modules concurrently.

Regression for #125746: on a long-lived process, background plugin discovery, per-plugin
deadline workers and ordinary imports on other threads all touch sys.modules at the same time
as a plugin load. Filtering it with a live `for name in sys.modules` comprehension iterates the
dict bytecode-by-bytecode, so a concurrent insert/delete on another thread can raise
"RuntimeError: dictionary changed size during iteration" mid-comprehension -- reported by the
loader as the *target* plugin failing, even though that plugin never touched sys.modules.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROGRAM = '''
import sys
import threading
import time

from hermes_cli.plugins_loader import _evict_modules

stop = threading.Event()

def churn():
    i = 0
    while not stop.is_set():
        sys.modules[f"_evict_modules_race_churn_{i}"] = None
        sys.modules.pop(f"_evict_modules_race_churn_{i - 1}", None)
        i += 1

threads = [threading.Thread(target=churn, daemon=True) for _ in range(4)]
for t in threads:
    t.start()

deadline = time.monotonic() + 2.5
try:
    while time.monotonic() < deadline:
        _evict_modules("hermes_plugins.does_not_exist")
finally:
    stop.set()
    for t in threads:
        t.join(timeout=5)

print("OK")
'''


def test_evict_modules_survives_concurrent_sys_modules_mutation():
    result = subprocess.run(
        [sys.executable, "-c", PROGRAM],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
