"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import pytest


_LIFECYCLE_TEST_FILENAME = "test_kanban_lifecycle.py"


def _install_permissive_missing_registry_check():
    """Monkeypatch (by hand — see below) ``check_dispatch_eligibility`` to
    treat a completely-missing P0-G-B1 lifecycle registry as pre-rollout.

    Returns a ``(module, real_check)`` pair for manual restoration, or
    ``None`` if the module can't be imported (never block a test on this).
    """
    try:
        from hermes_cli import kanban_lifecycle as lc
    except Exception:
        return None
    real_check = lc.check_dispatch_eligibility

    def _permissive_check(slug):
        try:
            registry_missing = not lc.registry_path().exists()
        except OSError:
            registry_missing = False
        if registry_missing:
            return lc.EligibilityResult(
                eligible=True,
                slug=slug,
                state="LEGACY_ACTIVE",
                reason=(
                    "test-fixture: no P0-G-B1 registry provisioned for "
                    "this HERMES_HOME; treating as pre-rollout (see "
                    "conftest.py::pytest_runtest_call)"
                ),
                registry_ok=True,
            )
        return real_check(slug)

    lc.check_dispatch_eligibility = _permissive_check
    return lc, real_check


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Treat a completely-missing P0-G-B1 lifecycle registry as pre-rollout
    for ``dispatch_once``/gateway/CLI dispatch eligibility, for every kanban
    test EXCEPT the containment layer's own test suite.

    Background: P0-G-B1 round 2 pushed lifecycle-eligibility gating down
    into the core functions themselves (``dispatch_once`` in
    ``hermes_cli/kanban_db.py``, in addition to the gateway tick and the
    manual CLI guard that already had it) so no caller can bypass it — not
    even the dashboard plugin, which used to call ``dispatch_once()``
    directly with zero lifecycle awareness. ``check_dispatch_eligibility``
    fails CLOSED whenever the external lifecycle registry
    (``<HERMES_HOME>/kanban-control/boards.json``) is entirely missing —
    the same rule the gateway tick and the CLI dispatch guard already
    enforced pre-round-2 at their own call sites. The overwhelming
    majority of kanban tests in this directory build their own
    ``HERMES_HOME`` (via this directory's shared ``_hermetic_environment``
    fixture, or a local per-test ``tempfile.mkdtemp``/``monkeypatch``
    variant) with no such registry, and dispatch real work through
    ``dispatch_once`` expecting it to behave exactly as it did before this
    round — without this, EVERY one of those calls would suddenly return
    an all-zero, ``skipped_lifecycle=True`` result, which is not what any
    of them are testing.

    Implementation notes:

    * This is a ``pytest_runtest_call`` hookwrapper, NOT a plain
      autouse fixture, and that is load-bearing: several test files in
      this directory (e.g. ``test_kanban_per_profile_cap.py``,
      ``test_kanban_default_assignee.py``) build their own isolated
      ``HERMES_HOME`` via a local, non-autouse fixture that deliberately
      deletes every ``hermes_cli.*``/``hermes_state``/``hermes_constants``
      entry from ``sys.modules`` before re-importing them — including
      ``hermes_cli.kanban_lifecycle``. An autouse fixture (which pytest
      instantiates BEFORE a test's own explicitly-requested fixtures at
      the same scope) would patch the module object that exists at that
      point, only for that local fixture to immediately replace it with a
      freshly re-imported module object with no patch on it — silently
      undoing the patch before the test body ever runs (this was
      confirmed the hard way: an earlier autouse-fixture version of this
      exact patch had zero effect on any test using such a fixture).
      ``pytest_runtest_call`` fires strictly after ALL fixture setup
      (autouse and explicit) has completed, immediately before the test
      function itself executes, so it patches whatever module object is
      ACTUALLY current at that moment — surviving any amount of
      sys.modules churn during fixture setup.
    * Patches ``hermes_cli.kanban_lifecycle.check_dispatch_eligibility``
      directly (module attribute, not ``monkeypatch.setattr`` — this runs
      outside the `monkeypatch` fixture's scope) — the single choke point
      ``dispatch_once``, ``gateway_lifecycle_gate``, and the CLI guard all
      call via a fresh ``from hermes_cli import kanban_lifecycle`` at call
      time, so patching the module attribute reaches every one of them
      regardless of which ``HERMES_HOME`` is active when the call happens.
    * When (and only when) no registry file exists at all for the CURRENT
      ``HERMES_HOME`` at check time, this reports the board eligible (as
      if pre-rollout) instead of the real fail-closed answer; once a real
      registry exists (a test wrote one), every call defers to the real,
      unmodified function — so this can never mask an actual
      QUARANTINED/ARCHIVED/INACTIVE decision.
    * Manually restored in a ``finally`` (not ``monkeypatch``, which is
      function-fixture-scoped and not available/appropriate at this
      hook level) so the patched module doesn't leak the stub into a
      later test that reuses the same module object.

    ``test_kanban_lifecycle.py`` — the P0-G-B1 containment layer's own
    test suite — asserts the real fail-closed-on-missing-registry behavior
    directly and is excluded outright (by filename) rather than via an
    opt-out marker on every test, since that IS what the whole file is
    about. Any other test that needs the real fail-closed behavior can opt
    out with ``@pytest.mark.no_default_lifecycle_registry``.
    """
    if (
        item.path.name == _LIFECYCLE_TEST_FILENAME
        or item.get_closest_marker("no_default_lifecycle_registry")
    ):
        yield
        return

    patched = _install_permissive_missing_registry_check()
    try:
        yield
    finally:
        if patched is not None:
            lc_module, real_check = patched
            try:
                lc_module.check_dispatch_eligibility = real_check
            except Exception:
                pass


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
