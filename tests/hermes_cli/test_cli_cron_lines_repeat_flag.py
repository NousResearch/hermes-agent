"""Unit tests for the ``/cron`` slash-command flag parser's ``--repeat`` handling.

Regression for #150986: ``_parse_cron_flags`` used to ``int()``-cast ``--repeat`` and bail out
with ``None`` on anything else, so the user-facing strings the API's ``normalize_repeat_value()``
already accepts (``'forever'``, ``'once'``, ...) never reached it.
"""

from __future__ import annotations

from hermes_cli.cli_cron_lines import _parse_cron_flags


def test_repeat_accepts_user_facing_strings():
    for tokens, expected in [
        (["in 1h", "x", "--repeat", "forever"], "forever"),
        (["in 1h", "x", "--repeat", "once"], "once"),
        (["in 1h", "x", "--repeat", "3"], "3"),
    ]:
        opts = _parse_cron_flags(tokens)
        assert opts is not None
        assert opts["repeat"] == expected


def test_repeat_absent_stays_none():
    opts = _parse_cron_flags(["in 1h", "x"])
    assert opts is not None
    assert opts["repeat"] is None


def test_repeat_garbage_string_is_not_rejected_at_parse_time():
    # Validation now happens downstream in normalize_repeat_value(), not here.
    opts = _parse_cron_flags(["in 1h", "x", "--repeat", "bogus"])
    assert opts is not None
    assert opts["repeat"] == "bogus"
