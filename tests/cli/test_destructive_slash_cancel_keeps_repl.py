"""Cancelling a destructive-slash confirmation must keep the REPL alive.

``process_command`` is declared ``-> bool`` and the REPL loop treats a falsy
return as an exit signal:

    if not self.process_command(user_input):
        self._should_exit = True
        app.exit()

The destructive-command handlers (``/clear``, ``/new``/``/reset``, ``/undo``)
ask ``_confirm_destructive_slash`` for approval, which returns ``None`` when the
user picks **Cancel**.  Cancel's stated purpose is "keep current conversation",
so the command must be a no-op that returns ``True`` (continue) — never ``None``,
which would tear down the whole interactive session.

These tests drive ``process_command`` directly via ``__get__`` on a
SimpleNamespace stand-in (constructing a full HermesCLI requires extensive
setup) and assert the truthy return on the cancel path.
"""

from __future__ import annotations

from types import SimpleNamespace


def _bound(fn, instance):
    """Bind an unbound method to a stand-in instance."""
    return fn.__get__(instance, type(instance))


def _make_self(confirm_result):
    """Minimal stand-in 'self' for process_command's destructive branches.

    ``_confirm_destructive_slash`` is stubbed to return ``confirm_result`` so the
    test exercises the handler's branching without rendering a real modal.  The
    mutating helpers (``new_session``/``undo_last``) explode if called, proving
    the cancel path never touches conversation state.
    """
    from cli import HermesCLI

    def _no_mutate(*_a, **_kw):
        raise AssertionError("conversation state must not be mutated on cancel")

    self_ = SimpleNamespace(
        _pending_resume_sessions=None,
        _confirm_destructive_slash=lambda *a, **kw: confirm_result,
        _split_destructive_skip=HermesCLI._split_destructive_skip,
        new_session=_no_mutate,
        undo_last=_no_mutate,
    )
    return self_


def test_clear_cancel_returns_true():
    """Cancelling ``/clear`` keeps the session — process_command returns True."""
    from cli import HermesCLI

    self_ = _make_self(confirm_result=None)
    result = _bound(HermesCLI.process_command, self_)("/clear")

    assert result is True


def test_new_cancel_returns_true():
    """Cancelling ``/new`` keeps the session — process_command returns True."""
    from cli import HermesCLI

    self_ = _make_self(confirm_result=None)
    result = _bound(HermesCLI.process_command, self_)("/new My Session")

    assert result is True


def test_undo_cancel_returns_true():
    """Cancelling ``/undo`` keeps the session — process_command returns True."""
    from cli import HermesCLI

    self_ = _make_self(confirm_result=None)
    result = _bound(HermesCLI.process_command, self_)("/undo 2")

    assert result is True


def test_undo_invalid_count_returns_true():
    """A bad ``/undo`` count is rejected without exiting the REPL."""
    from cli import HermesCLI

    # confirm_result is irrelevant — the invalid-count guard returns before the
    # confirmation prompt is reached.
    self_ = _make_self(confirm_result="once")
    result = _bound(HermesCLI.process_command, self_)("/undo abc")

    assert result is True
