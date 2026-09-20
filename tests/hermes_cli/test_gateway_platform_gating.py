"""Host-specific gating in ``hermes_cli.gateway._all_platforms()``.

Some messaging platforms can't function on every host. The gate lives
in one place — ``_all_platforms()`` — so the setup wizard, the curses
gateway-config menu, and any future picker all see the same filtered
list.

Currently:
- Matrix is hidden on Windows. The ``[matrix]`` extra pulls
  ``mautrix[encryption]`` -> ``python-olm``, which has no Windows wheel
  and needs ``make`` + libolm to build from sdist. There's no native
  Windows path that works.
"""

import pytest


class TestMatrixHiddenOnWindows:
    @pytest.mark.linux_only
    def test_matrix_present_on_linux(self):
        """Sanity: matrix is still in the picker on Linux.

        Linux-gated because the assertion is the negative of the Windows
        gate — it only means anything when the host really is not Windows.
        """
        import hermes_cli.gateway as gateway_mod

        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        assert "matrix" in keys, "matrix must be available on Linux"

    @pytest.mark.windows_only
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

    @pytest.mark.windows_only
    def test_other_platforms_unaffected_on_windows(self):
        """Gating must only drop matrix, not collateral damage."""
        import hermes_cli.gateway as gateway_mod

        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        # A representative sample of platforms that have no Windows
        # blockers — picker should still surface them.
        for must_have in ("telegram", "discord", "slack", "mattermost"):
            assert must_have in keys, (
                f"{must_have} disappeared from Windows picker — gate is "
                "over-filtering"
            )
