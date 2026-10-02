"""Host-specific gating in ``hermes_cli.gateway._all_platforms()``.

Some messaging platforms can't function on every host. The gate lives
in one place — ``_all_platforms()`` — so the setup wizard, the curses
gateway-config menu, and any future picker all see the same filtered
list.

Currently:
- Matrix is hidden wherever pm's ``matrix`` gate
  (``[tool.hermes.extras-platforms]``) refuses the extra, which today means
  native Windows: asyncpg has no win_arm64 wheel and plaintext Matrix is
  unverified there.
"""

import pytest

class TestMatrixHiddenOnWindows:

    @pytest.mark.platforms("windows")
    def test_matrix_absent_on_windows(self):
        """The gate itself: matrix must be dropped on a real Windows host.

        A patched ``sys.platform`` proved only that the ``if`` branch runs;
        on native Windows this also proves the picker the user actually sees
        omits the platform whose dependency cannot build here.
        """
        import hermes_cli.gateway as gateway_mod

        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        assert "matrix" not in keys, "matrix must be hidden on Windows"
