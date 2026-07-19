"""Oneshot (-z) must reject --resume/--continue instead of silently dropping it.

The oneshot path never hydrates a resumed session, so combining it with
--resume/--continue used to run a context-less session with no warning
(#49195). The guard fails loud instead.
"""

from types import SimpleNamespace

import pytest

from hermes_cli.main import _reject_resume_in_oneshot


def test_resume_in_oneshot_exits_with_error(capsys):
    args = SimpleNamespace(resume="abc123", continue_last=None)
    with pytest.raises(SystemExit) as exc:
        _reject_resume_in_oneshot(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--resume/--continue is not supported" in err


def test_continue_in_oneshot_exits_with_error(capsys):
    args = SimpleNamespace(resume=None, continue_last=True)
    with pytest.raises(SystemExit) as exc:
        _reject_resume_in_oneshot(args)
    assert exc.value.code == 2


def test_plain_oneshot_is_allowed():
    # No resume/continue → no exit, no output.
    args = SimpleNamespace(resume=None, continue_last=None)
    _reject_resume_in_oneshot(args)


def test_missing_attrs_are_treated_as_unset():
    # Defensive: args without the attributes must not raise.
    _reject_resume_in_oneshot(SimpleNamespace())
