"""Regression: `hermes vault add` must not crash-traceback when stdin is
exhausted (cron / ``</dev/null`` / closed-pipe invocations).

Before the fix, ``_cmd_add`` called bare ``input()`` — with stdin at
/dev/null the first prompt raised ``EOFError`` (visible traceback, rc 1),
and the ``while kind not in VAULT_KINDS`` re-prompt loop over an exhausted
stdin would spin forever on an invalid answer.
"""

import builtins
from types import SimpleNamespace

from hermes_cli import vault as vault_mod


def test_add_aborts_cleanly_on_empty_stdin(tmp_path, monkeypatch, capsys):
    """EOF at any prompt: one clean 'Error: …' line, no traceback, and a
    re-prompt loop (kind) cannot spin on an exhausted stdin."""

    def _eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", _eof)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    args = SimpleNamespace(vault_action="add", kind=None,
                           _vault_handler=vault_mod._cmd_add)
    vault_mod.vault_command(args)  # must not raise

    out = capsys.readouterr().out
    assert "stdin is closed or exhausted" in out
    assert "Traceback" not in out
