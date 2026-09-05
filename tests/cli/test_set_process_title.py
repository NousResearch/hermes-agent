"""Regression tests for the best-effort process title helper."""

import builtins
from unittest.mock import patch

from hermes_cli.main import _set_process_title


def test_set_process_title_survives_unicode_decode_error_during_import():
    real_import = builtins.__import__

    def import_with_decode_failure(name, *args, **kwargs):
        if name == "setproctitle":
            raise UnicodeDecodeError(
                "utf-8", b"\xe4", 0, 1, "unexpected end of data"
            )
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=import_with_decode_failure),
        patch("platform.system", return_value="Windows"),
    ):
        _set_process_title()


def test_set_process_title_logs_once_when_bytecode_is_corrupt(caplog):
    """The fallback must not be silent — a permanent no-op is how #98593 was found."""
    import logging

    real_import = builtins.__import__

    def import_with_decode_failure(name, *args, **kwargs):
        if name == "setproctitle":
            raise UnicodeDecodeError("utf-8", b"\xe4", 0, 1, "unexpected end of data")
        return real_import(name, *args, **kwargs)

    with (
        caplog.at_level(logging.DEBUG, logger="hermes_cli.main"),
        patch("builtins.__import__", side_effect=import_with_decode_failure),
        patch("platform.system", return_value="Windows"),
    ):
        _set_process_title()

    assert any("setproctitle" in r.message for r in caplog.records)
