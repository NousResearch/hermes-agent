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
