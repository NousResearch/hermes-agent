"""CLI dispatch exit-status tests.

``main()`` propagates a subcommand handler's return value to the process exit
status: an exact non-zero ``int`` becomes ``sys.exit(rc)``, while ``None`` (and,
critically, ``bool``) are treated as success.  ``bool`` is a subclass of
``int``, so a bare ``isinstance(rc, int)`` check would map a handler returning
``True`` to ``sys.exit(True)`` -> exit status ``1``.  ``args.func`` handlers are
unconstrained ``Callable``s (notably external-plugin CLI handlers) that
routinely return ``True``/``False`` to signal success, so these pin that a
boolean result never becomes a spurious failure.
"""

from __future__ import annotations

import pytest


def test_dispatch_exits_nonzero_on_integer_failure(monkeypatch):
    """A handler returning a non-zero int must ``sys.exit`` with that status."""
    from hermes_cli import main as main_mod

    monkeypatch.setattr(main_mod, "cmd_kanban", lambda args: 7)
    monkeypatch.setattr("sys.argv", ["hermes", "kanban", "list"])

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    assert exc.value.code == 7


def test_dispatch_success_on_zero_and_none(monkeypatch):
    """Zero and ``None`` returns are success — no ``sys.exit`` is raised."""
    from hermes_cli import main as main_mod

    for result in (0, None):
        monkeypatch.setattr(main_mod, "cmd_kanban", lambda args, r=result: r)
        monkeypatch.setattr("sys.argv", ["hermes", "kanban", "list"])
        # Must not raise SystemExit; a bare return is treated as exit 0.
        main_mod.main()


def test_dispatch_treats_boolean_result_as_success(monkeypatch):
    """A handler returning ``True``/``False`` must not become exit status 1.

    Regression: ``bool`` is an ``int`` subclass, so ``isinstance(rc, int) and
    rc != 0`` mapped ``True`` to ``sys.exit(True)`` (a spurious exit status 1).
    Plugin handlers commonly return booleans to signal success.
    """
    from hermes_cli import main as main_mod

    for result in (True, False):
        monkeypatch.setattr(main_mod, "cmd_kanban", lambda args, r=result: r)
        monkeypatch.setattr("sys.argv", ["hermes", "kanban", "list"])
        # A boolean must flow through as success — no SystemExit.
        main_mod.main()


def test_dispatch_preserves_system_exit_from_handler(monkeypatch):
    """Handlers that already raise ``SystemExit`` keep owning their status."""
    from hermes_cli import main as main_mod

    def fake(args):
        raise SystemExit(3)

    monkeypatch.setattr(main_mod, "cmd_kanban", fake)
    monkeypatch.setattr("sys.argv", ["hermes", "kanban", "list"])

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    assert exc.value.code == 3
