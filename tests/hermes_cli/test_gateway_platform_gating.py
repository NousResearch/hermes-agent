"""Host-specific gating in ``hermes_cli.gateway._all_platforms()``.

Some messaging platforms can't function on every host. The gate lives
in one place — ``_all_platforms()`` — so the setup wizard, the curses
gateway-config menu, and any future picker all see the same filtered
list.

Matrix used to be hidden on Windows because ``mautrix[encryption]`` pulled
``python-olm``, which had no Windows wheel and needed ``make`` + libolm to
build. E2EE now runs on vodozemac, which ships Windows wheels, so Matrix
must be offered there like everywhere else.
"""

import pytest


class TestMatrixOfferedOnWindows:

    @pytest.mark.platforms("windows")
    def test_matrix_present_on_windows(self):
        """On a real Windows host the picker the user sees must include Matrix."""
        import hermes_cli.gateway as gateway_mod

        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        assert "matrix" in keys, "matrix must be offered on Windows (vodozemac has Windows wheels)"
