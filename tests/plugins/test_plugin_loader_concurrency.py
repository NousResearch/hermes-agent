"""Regression: two concurrent first-time loads of one plugin must both yield a provider.

The directory-plugin loader (``plugins/plugin_loader.py::load_plugin_module``)
hand-executes plugin modules on the calling thread, outside the import system's
own per-module locks. The module name is reserved in ``sys.modules`` *before*
exec and the cache check returns that reservation (``__file__`` is set
pre-exec), so a second thread racing the first load used to observe a
partially-executed module, extract no provider instance, and get ``None``.

Real impact (2026-09-10): two desktop chats restored concurrently after a
restart raced ``load_memory_provider("openviking")``; the losing build ran
without the provider's six tools for the whole conversation ("Tool
'viking_search' does not exist") because a session's toolset is frozen for the
life of the conversation.

This test holds one module exec open (the fixture blocks until the test
releases it) and runs a second load inside that window. Both loads must return
a provider instance; on unfixed code the second returns ``None``.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

_FIXTURE_NAME = "racefixture"
_USER_NAMESPACE_PREFIX = "_hermes_user_memory"

# Everything (class + register) is defined only AFTER the block, so a load
# that observes this module mid-exec can extract nothing from it.
_FIXTURE_INIT = '''\
"""Concurrency fixture memory provider (register_memory_provider / MemoryProvider markers).

Exec is held open until the test releases it, so a concurrent load can only
ever observe a fully-executed module once first-time loads are serialized.
"""
import time
from pathlib import Path

from agent.memory_provider import MemoryProvider

_DIR = Path(__file__).parent
_MARKER = _DIR / "exec_started"
_RELEASE = _DIR / "exec_release"

_MARKER.write_text("started", encoding="utf-8")
_deadline = time.monotonic() + 15.0
while not _RELEASE.exists() and time.monotonic() < _deadline:
    time.sleep(0.01)


class RaceFixtureProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "racefixture"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []


def register(ctx):
    ctx.register_memory_provider(RaceFixtureProvider())
'''


@pytest.fixture
def race_plugin_dir():
    """A user-installed memory provider whose module exec the test controls."""
    home = Path(os.environ["HERMES_HOME"])
    plugin_dir = home / "plugins" / _FIXTURE_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(_FIXTURE_INIT, encoding="utf-8")
    try:
        yield plugin_dir
    finally:
        for name in [n for n in list(sys.modules) if n.startswith(_USER_NAMESPACE_PREFIX)]:
            sys.modules.pop(name, None)


def test_concurrent_first_loads_both_yield_a_provider(race_plugin_dir):
    from plugins.memory import load_memory_provider

    marker = race_plugin_dir / "exec_started"
    release = race_plugin_dir / "exec_release"
    results = {}
    errors = {}

    def _load(key):
        try:
            results[key] = load_memory_provider(_FIXTURE_NAME)
        except Exception as exc:  # surfaced via the assertions below
            errors[key] = exc

    first = threading.Thread(target=_load, args=("first",), daemon=True)
    first.start()
    deadline = time.monotonic() + 10.0
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert marker.exists(), "fixture module exec never started"

    second = threading.Thread(target=_load, args=("second",), daemon=True)
    second.start()
    # Give the second load the chance to land inside the first load's exec
    # window; once loads are serialized it blocks here and is released below.
    time.sleep(0.3)
    release.write_text("go", encoding="utf-8")

    first.join(timeout=10.0)
    second.join(timeout=10.0)
    assert not first.is_alive() and not second.is_alive(), "loads did not finish"
    assert not errors, f"load raised: {errors!r}"

    assert results.get("first") is not None, "first load returned no provider"
    assert results.get("second") is not None, (
        "the concurrent second load returned no provider instance: it observed the "
        "partially-executed plugin module (regressed plugins/plugin_loader.py first-load lock)"
    )
    assert results["first"].name == _FIXTURE_NAME
    assert results["second"].name == _FIXTURE_NAME
