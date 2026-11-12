"""The embedded dispatcher re-reads its concurrency caps per tick.

``_DispatcherSettings`` was documented as "read once at boot (restart to
apply)", so editing ``kanban.max_in_progress`` in config.yaml did nothing until
the gateway restarted — and a restart strands in-flight workers, which is how a
capacity knob became an outage generator. ``_cmd_dispatch`` (CLI) already
resolves config per invocation, so the embedded loop was the odd one out.

Contracts here: a cap edit applies on the next refresh, the memory-derived
default is honoured exactly as at boot, and a config read failure keeps the last
good values while warning once instead of every tick.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gateway import kanban_watchers_dispatcher as kwd


class _StubKbd:
    """Stand-in for kanban_db_dispatch so the derived default is deterministic."""

    def __init__(self, derived: int = 7) -> None:
        self.derived = derived
        self.saw_explicit: list = []

    def resolve_max_in_progress(self, explicit):
        self.saw_explicit.append(explicit)
        return self.derived if explicit is None else explicit


def _dispatcher(monkeypatch, *, host: int = 4, per_profile: int = 2, derived: int = 7):
    stub = _StubKbd(derived=derived)
    monkeypatch.setattr(kwd, "_kbd", lambda: stub)
    settings = kwd._DispatcherSettings(
        interval=60.0,
        max_spawn=None,
        max_in_progress=host,
        failure_limit=2,
        stale_timeout_seconds=0,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=per_profile,
    )
    return kwd._KanbanDispatcher(MagicMock(), settings), stub


def _cfg(**kanban):
    return lambda: {"kanban": dict(kanban)}


def test_raising_the_host_cap_applies_without_a_restart(monkeypatch):
    dispatcher, _ = _dispatcher(monkeypatch, host=4)

    dispatcher.refresh_caps(_cfg(max_in_progress=12, max_in_progress_per_profile=4))

    assert dispatcher.settings.max_in_progress == 12
    assert dispatcher.settings.max_in_progress_per_profile == 4


def test_lowering_the_host_cap_applies_without_a_restart(monkeypatch):
    dispatcher, _ = _dispatcher(monkeypatch, host=12)

    dispatcher.refresh_caps(_cfg(max_in_progress=3, max_in_progress_per_profile=1))

    assert dispatcher.settings.max_in_progress == 3
    assert dispatcher.settings.max_in_progress_per_profile == 1


def test_unset_host_cap_follows_the_memory_derived_default_like_boot(monkeypatch):
    """Boot resolves the derived default when the operator set nothing; so must the tick.

    Divergence here would let a tick quietly widen an intentionally-throttled
    host back to unbounded fan-out — the OOF-30/OOF-77 failure shape.
    """
    dispatcher, stub = _dispatcher(monkeypatch, host=4, derived=7)

    dispatcher.refresh_caps(_cfg(max_in_progress_per_profile=2))

    assert stub.saw_explicit == [None]
    assert dispatcher.settings.max_in_progress == 7


def test_config_read_failure_keeps_the_last_good_caps(monkeypatch):
    dispatcher, _ = _dispatcher(monkeypatch, host=8, per_profile=3)

    def boom():
        raise RuntimeError("config.yaml is mid-write")

    dispatcher.refresh_caps(boom)

    assert dispatcher.settings.max_in_progress == 8
    assert dispatcher.settings.max_in_progress_per_profile == 3


def test_config_read_failure_warns_once_not_every_tick(monkeypatch):
    dispatcher, _ = _dispatcher(monkeypatch, host=8)
    warnings = []
    monkeypatch.setattr(kwd.logger, "warning", lambda msg, *a, **kw: warnings.append(msg % a))

    def boom():
        raise RuntimeError("config.yaml is mid-write")

    dispatcher.refresh_caps(boom)
    dispatcher.refresh_caps(boom)
    dispatcher.refresh_caps(boom)

    assert len(warnings) == 1
    assert "keeping max_in_progress=8" in warnings[0]


def test_a_recovered_config_clears_the_warning_state(monkeypatch):
    dispatcher, _ = _dispatcher(monkeypatch, host=8)

    def boom():
        raise RuntimeError("mid-write")

    dispatcher.refresh_caps(boom)
    dispatcher.refresh_caps(_cfg(max_in_progress=11, max_in_progress_per_profile=2))
    dispatcher.refresh_caps(boom)
    dispatcher.refresh_caps(boom)

    assert dispatcher.settings.max_in_progress == 11


def test_non_positive_cap_falls_to_the_derived_default_like_boot(monkeypatch):
    """`max_in_progress: 0` is dropped by `_positive_int_setting` and resolves to the
    memory-derived default — the same thing boot does. Tested for parity, not preference.
    """
    dispatcher, stub = _dispatcher(monkeypatch, host=6, per_profile=3, derived=7)

    dispatcher.refresh_caps(_cfg(max_in_progress=0, max_in_progress_per_profile="4"))

    assert stub.saw_explicit == [None]
    assert dispatcher.settings.max_in_progress == 7
    assert dispatcher.settings.max_in_progress_per_profile == 4


def test_a_real_config_edit_is_obeyed_without_a_restart(tmp_path, monkeypatch):
    """E2E through the real loader and a real config.yaml on disk.

    The unit tests above stub the config source; this one proves the thing the
    fix exists for — an operator edits ``kanban.max_in_progress`` and the running
    dispatcher obeys, which previously required a gateway restart (and a restart
    is what strands in-flight workers).
    """
    import os
    from unittest.mock import MagicMock

    from hermes_cli import kanban_db as kb
    from hermes_cli.config import load_config

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    cfg_path = home / "config.yaml"
    cfg_path.write_text("kanban:\n  max_in_progress: 3\n  max_in_progress_per_profile: 2\n")

    settings = kwd._resolve_dispatcher_settings(load_config().get("kanban", {}), kb)
    dispatcher = kwd._KanbanDispatcher(MagicMock(), settings)
    assert (dispatcher.settings.max_in_progress,
            dispatcher.settings.max_in_progress_per_profile) == (3, 2)

    cfg_path.write_text("kanban:\n  max_in_progress: 12\n  max_in_progress_per_profile: 5\n")
    st = os.stat(cfg_path)
    os.utime(cfg_path, (st.st_atime, st.st_mtime + 5))  # real loaders may cache by mtime

    dispatcher.refresh_caps(load_config)
    assert dispatcher.settings.max_in_progress == 12
    assert dispatcher.settings.max_in_progress_per_profile == 5

    # Tightening obeys too, and dropping a key returns it to the boot meaning
    # (no per-profile cap), not to the last value.
    cfg_path.write_text("kanban:\n  max_in_progress: 1\n")
    st = os.stat(cfg_path)
    os.utime(cfg_path, (st.st_atime, st.st_mtime + 5))

    dispatcher.refresh_caps(load_config)
    assert dispatcher.settings.max_in_progress == 1
    assert dispatcher.settings.max_in_progress_per_profile is None
