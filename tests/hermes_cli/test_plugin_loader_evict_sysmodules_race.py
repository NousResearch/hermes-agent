"""Regression: _evict_modules must not raise while another thread mutates sys.modules.

The background-discovery thread (start_background_plugin_discovery) and the foreground
loader run ``_evict_modules`` and ordinary imports concurrently. The pre-fix body
comprehended over the live ``sys.modules`` dict; a concurrent insert/delete between two
``__next__`` calls raised ``RuntimeError: dictionary changed size during iteration``,
which the per-plugin catch site swallowed as
"Failed to load plugin '<name>': dictionary changed size during iteration" — the plugin
was silently lost for the process (see SEARCHER-SEAT-DIAGNOSTICS R1b).
"""

import sys
import threading
import time

from hermes_cli.plugins_loader import _evict_modules


def _churn(evict, stop, tid, errors):
    """Mimic one loader thread: evict → insert (import) → insert submodule → delete.

    Each thread owns its slug — the production bug lives in the evict comprehension
    itself, which iterates the WHOLE live sys.modules dict and is therefore disturbed by
    every other thread's inserts/deletes, exactly like the background-discovery sweep
    racing foreground CLI-startup imports."""
    i = 0
    while not stop.is_set():
        slug = f"hermes_plugins.race_{tid}"
        try:
            evict(slug)
            sys.modules[slug] = object()
            sys.modules[slug + ".sub"] = object()
            del sys.modules[slug + ".sub"]
            del sys.modules[slug]
            i += 1
        except RuntimeError as exc:  # the production failure mode
            errors.append(repr(exc))
            return


def _noise(stop, errors, tid):
    """Mimic the rest of CLI startup: ordinary imports mutate sys.modules."""
    i = 0
    while not stop.is_set():
        name = f"race_startup_noise_{tid}_{i % 50}"
        try:
            sys.modules[name] = object()
            sys.modules.pop(name, None)
            i += 1
        except RuntimeError as exc:
            errors.append(repr(exc))
            return


def test_evict_modules_survives_concurrent_sys_modules_mutation():
    errors: list[str] = []
    stop = threading.Event()
    threads = [threading.Thread(target=_churn, args=(_evict_modules, stop, n, errors))
               for n in range(6)]
    threads += [threading.Thread(target=_noise, args=(stop, errors, 100 + n))
                for n in range(2)]
    for t in threads:
        t.daemon = True
        t.start()
    deadline = time.monotonic() + 5.0
    # Join promptly when a thread already failed; otherwise run the full window.
    while time.monotonic() < deadline and not errors and any(t.is_alive() for t in threads):
        time.sleep(0.05)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert not errors, f"_evict_modules raced sys.modules: {errors[:3]}"
    for name in [n for n in list(sys.modules) if n.startswith(("hermes_plugins.race",
                                                               "race_startup_noise_"))]:
        sys.modules.pop(name, None)


def test_evict_modules_removes_module_and_submodules():
    sys.modules["hermes_plugins.evict_probe"] = object()
    sys.modules["hermes_plugins.evict_probe.sub"] = object()
    sys.modules["hermes_plugins.evict_probeother"] = object()  # must survive (no dot match)
    try:
        _evict_modules("hermes_plugins.evict_probe")
        assert "hermes_plugins.evict_probe" not in sys.modules
        assert "hermes_plugins.evict_probe.sub" not in sys.modules
        assert "hermes_plugins.evict_probeother" in sys.modules
    finally:
        sys.modules.pop("hermes_plugins.evict_probeother", None)
