"""Render copy-pasteable CLI hints that survive the shell and argparse verbatim.

An error message that tells the operator what to type is a remedy only if the
printed string WORKS when pasted. Interpolating a user-controlled token (a
filesystem path, a repository key) bare into a printed command line breaks in
two layers, and the SHELL layer runs first:

* the SHELL re-lexes the printed text, so a token containing whitespace splits
  into two words, a token containing ``;&|()`` is read as a control operator or
  a syntax error, ``{a,b}`` brace-expands, ``*?[]`` glob against the operator's
  CWD, and ``$VAR`` / ``` `cmd` ``` / ``$(cmd)`` / ``~`` EXPAND OR EXECUTE --
  the printed remedy runs something the tool never intended;
* ARGPARSE then binds a token beginning with ``-`` as an option rather than as
  the value and refuses with ``expected one argument``.

Either way the state the message exists to escape ends up with no accepted
input at all, and the glob case is worse than broken: it binds a DIFFERENT
value, silently, only when a matching name happens to exist in the CWD.

``hint_arg`` is the single place that knows those rules, so a hint site gets
them by calling it instead of re-deriving them.
"""
from __future__ import annotations

import shlex

__all__ = ["hint_arg"]


def _is_shell_literal(value: str) -> bool:
    """True when a real shell passes `value` through as exactly itself.

    Decided by ``shlex.quote``, which is the INVERSE of the question: it
    returns the token unchanged exactly when no shell metacharacter needs
    escaping. Deciding instead by ``shlex.split`` -- "does it tokenize to one
    word" -- under-approximates a shell, because ``shlex.split`` models only
    quoting and whitespace: it reports ``$HOME``, ``` `id` ``` and ``star*glob``
    as single clean words that a real ``/bin/bash`` expands, executes or globs.
    A metacharacter denylist is the same guess that produced the original bug.
    """
    return shlex.quote(value) == value


def hint_arg(flag: str, value: str) -> str:
    """Render ``flag``/``value`` as a shell- and argparse-safe argument pair.

    Returns the plain ``--flag value`` form when `value` is a shell literal --
    that is the spelling an operator expects to read, and it is what ordinary
    paths get, so existing guidance is unchanged for them. Otherwise returns
    the quoted ``'--flag=value'`` form: the attached ``=`` is the only spelling
    argparse accepts for a value beginning with ``-``, and the quoting is what
    stops the shell splitting, expanding or executing the rest.
    """
    value = str(value)
    if not value.startswith("-") and _is_shell_literal(value):
        return f"{flag} {value}"
    return shlex.quote(f"{flag}={value}")
