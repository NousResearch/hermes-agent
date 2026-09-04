"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import functools
import sys

import pytest

#: Package prefixes that ``update_cmd._purge_stale_hermes_modules`` evicts from
#: ``sys.modules``. Mirrors ``_STALE_PURGE_PREFIXES`` — kept as a literal so the
#: fixture below stays usable even when the purge itself dropped ``update_cmd``.
_HERMES_MODULE_PREFIXES = (
    "hermes_cli",
    "gateway",
    "tools",
    "tui_gateway",
    "agent",
)


@pytest.fixture(autouse=True)
def _restore_purged_hermes_modules():
    """Undo ``sys.modules`` eviction performed by ``cmd_update`` under test.

    ``hermes update`` keeps running in the pre-pull interpreter, so
    ``update_cmd._purge_stale_hermes_modules`` drops every cached
    ``hermes_cli``/``gateway``/``tools``/``agent`` module to force later
    imports to rebuild from the new checkout. In production that process is
    about to exit; inside pytest it poisons the rest of the session.

    A test module that did ``from hermes_cli.curses_ui import curses_radiolist``
    at import time keeps the OLD module's function, while any later
    ``import hermes_cli.curses_ui`` builds a SECOND module object with its own
    ``ContextVar`` instances. A menu navigation handler installed through one
    copy is then invisible to the other, ``allow_back`` reads ``False``, Left
    arrow is dropped, and ``curses_radiolist`` loops on ``getch() == -1``
    forever while appending to the fake screen's write log.

    That is not hypothetical: running ``tests/hermes_cli/`` as a whole,
    ``test_cmd_update.py`` sorts before ``test_curses_arrow_keys.py``, and the
    resulting spin grew one pytest process to 47 GB before the kernel's OOM
    killer took the machine down.

    Restoring only entries whose object identity changed keeps genuinely new
    imports made by the test, and costs one dict copy per test.
    """
    snapshot = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".", 1)[0] in _HERMES_MODULE_PREFIXES
    }
    yield
    _restore_module_snapshot(snapshot)


def _restore_module_snapshot(snapshot: dict) -> None:
    """Put back the snapshotted modules — through BOTH names each one has.

    A submodule is reachable two ways: ``sys.modules["pkg.mod"]`` and the
    ``pkg.mod`` attribute the import system binds on the parent package.
    Rewriting only the dict re-creates the very split identity this fixture
    exists to prevent — just one level up. Observed in this suite: after
    ``test_state_db_guard`` deletes ``hermes_cli.config*`` and re-imports it
    under a temporary ``HERMES_HOME``, restoring the dict alone leaves
    ``from hermes_cli import config`` on the tmp-era copy with its own
    module-level locks and caches, while ``hermes_cli.config`` as a target of
    ``monkeypatch.setattr`` resolves to the original — so a later test patches
    a copy the code under test never looks at.

    Kept as a module-level function rather than an inline loop so the contract
    is directly testable; see ``test_update_stale_module_purge``.
    """
    for name, module in snapshot.items():
        if sys.modules.get(name) is module:
            continue
        sys.modules[name] = module
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name) if parent_name else None
        # `getattr` first: a package whose attribute already agrees needs no
        # write, and some parents (namespace shims, mocks) reject setattr.
        if parent is not None and getattr(parent, child, None) is not module:
            try:
                setattr(parent, child, module)
            except (AttributeError, TypeError):
                pass


@functools.lru_cache(maxsize=None)
def _module_drives_cmd_update(path: str) -> bool:
    """True when a test module's source calls ``cmd_update(``.

    Deliberately a source scan and not a hand-written list of module names. A
    list is a thing that goes stale silently: the next ``test_update_*`` module
    to grow one step further down ``cmd_update`` loses the seat belt without
    anybody noticing, and it stays green in CI because CI has no gateways to
    kill. The predicate that actually matters — "does this module drive
    ``cmd_update``" — is right there in the source, so read it.

    The two modules that must NOT be patched (``test_update_launchd_*``
    exercises the real ``find_gateway_pids``, ``test_update_stale_module_purge``
    asserts ``hermes_cli.gateway`` IS evicted) fall out for free: neither calls
    ``cmd_update``. So this needs no opt-out list either.

    Cached per file path — 13 modules match today, each read once per session.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return "cmd_update(" in handle.read()
    except OSError:
        # Unreadable module: fail SAFE (patch it) rather than let an
        # end-to-end update test loose on this machine's gateways.
        return True


@pytest.fixture(autouse=True)
def _guard_live_gateway_discovery(request):
    """Give every module that drives ``cmd_update`` the discovery seat belt.

    Lives here rather than as a ``pytestmark`` in each test module so that a
    new ``test_update_*`` module gets the seat belt automatically instead of
    having to remember to opt in.
    """
    path = getattr(request.module, "__file__", None)
    if path and _module_drives_cmd_update(path):
        request.getfixturevalue("patch_gateway_discovery")


@pytest.fixture
def patch_gateway_discovery(monkeypatch):
    """Keep ``cmd_update``'s gateway auto-restart phase off this machine's fleet.

    Since the restart phase was surfaced (#78574: an aborted restart fails the
    update), any end-to-end ``cmd_update`` test that leaves gateway discovery
    unmocked finds the developer's REAL gateways, tries to ``os.kill`` them,
    trips the ``tests/conftest.py`` live-system guard and turns into a spurious
    ``sys.exit(1)``. Discovery returning nothing makes the phase a clean no-op.

    Patching the module attributes is not enough on its own — and that is why
    the upstream per-module attempts at this (``test_update_head_moved_gate``,
    ``test_update_fleet_restart_pending``, ``test_update_autostash`` each
    already mock all three functions "to keep the phase away from this
    machine's real gateways") still shipped SIGTERM at the live fleet.
    Right before the restart phase does its function-level ``from
    hermes_cli.gateway import ...``, ``cmd_update`` calls
    ``_purge_stale_hermes_modules()``, which drops ``hermes_cli.gateway`` out
    of ``sys.modules``. The import then re-executes the module from disk and
    binds the REAL functions; the patches are still attached to the evicted
    module object, so they no longer apply. Pinning ``hermes_cli.gateway`` into
    the purge's protected set keeps the mocked module object in ``sys.modules``
    so the re-import resolves to it, while leaving the purge itself running for
    every other module (its own behaviour is covered by
    ``test_update_stale_module_purge.py``, which asserts ``hermes_cli.gateway``
    IS evicted — hence this pin is opt-in per module, not autouse here).

    A test that wants the restart phase to see gateways re-patches these three
    inside its own body; the inner patch wins for its duration.

    ``monkeypatch`` — not ``mock.patch`` — on purpose, and it is not a style
    preference. Three of the covered modules ``monkeypatch.setattr`` these same
    three names inside their own bodies. ``monkeypatch`` is function-scoped, so
    this fixture and the test body get the SAME ``MonkeyPatch`` object: both
    sets of writes land on one undo stack and unwind in LIFO order, which is
    what makes the nesting compose. Two INDEPENDENT undo stacks do not compose
    — whichever unwinds last wins — and a ``mock.patch`` version of this
    fixture did exactly that: it left a live ``MagicMock`` on
    ``hermes_cli.gateway.find_gateway_pids`` for the rest of the session, so
    ``test_update_launchd_fleet_restart``'s own ``monkeypatch.setattr``
    silently did nothing. It failed only when it ran after
    ``test_update_head_moved_gate`` and passed on its own.
    """
    import hermes_cli.gateway as hermes_gateway
    import hermes_cli.update_cmd as update_cmd

    monkeypatch.setattr(
        update_cmd,
        "_STALE_PURGE_PROTECTED",
        frozenset(update_cmd._STALE_PURGE_PROTECTED | {"hermes_cli.gateway"}),
    )
    monkeypatch.setattr(hermes_gateway, "find_gateway_pids", lambda *a, **k: [])
    monkeypatch.setattr(
        hermes_gateway, "supports_systemd_services", lambda *a, **k: False
    )
    monkeypatch.setattr(
        hermes_gateway, "find_profile_gateway_processes", lambda *a, **k: []
    )


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )
