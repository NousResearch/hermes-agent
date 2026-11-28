"""``main()`` turns a handler's ``int`` return into the process exit status.

``bool`` subclasses ``int``, so a handler returning ``True`` for success must not
exit 1 (#62810).
"""

from __future__ import annotations

from enum import IntEnum

import pytest

from hermes_cli import main as main_mod


def _dispatch(monkeypatch, handler):
    monkeypatch.setattr(main_mod, "cmd_kanban", handler)
    monkeypatch.setattr("sys.argv", ["hermes", "kanban", "list"])
    main_mod.main()


@pytest.mark.parametrize("result", [True, False, None, 0, "done", {"ok": True}])
def test_non_status_results_exit_success(monkeypatch, result):
    _dispatch(monkeypatch, lambda args: result)


@pytest.mark.parametrize("status", [1, 2, 7])
def test_nonzero_int_result_becomes_exit_status(monkeypatch, status):
    with pytest.raises(SystemExit) as exc:
        _dispatch(monkeypatch, lambda args: status)
    assert exc.value.code == status


def test_int_enum_result_becomes_exit_status(monkeypatch):
    class Status(IntEnum):
        FAILED = 3

    with pytest.raises(SystemExit) as exc:
        _dispatch(monkeypatch, lambda args: Status.FAILED)
    assert exc.value.code == 3


def test_handler_raised_system_exit_keeps_its_status(monkeypatch):
    def handler(args):
        raise SystemExit(9)

    with pytest.raises(SystemExit) as exc:
        _dispatch(monkeypatch, handler)
    assert exc.value.code == 9
