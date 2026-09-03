"""Host-specific gating in ``hermes_cli.gateway._all_platforms()``.

Some messaging platforms can't function on every host. The gate lives
in one place — ``_all_platforms()`` — so the setup wizard, the curses
gateway-config menu, and any future picker all see the same filtered
list.

Currently nothing is gated here.

Matrix used to be dropped on Windows: ``LAZY_DEPS["platform.matrix"]``
pinned ``mautrix[encryption]``, which pulls ``python-olm``, which has no
Windows wheel and needs ``make`` + libolm to build from sdist. #62401
split the crypto packages into ``platform.matrix.e2ee``; plain
``mautrix`` is a pure-python wheel, so the plaintext adapter installs on
every host and the picker no longer hides it (#76092). The E2EE gate
moved to ``tools/lazy_deps.py::_unsupported_feature_reason`` and is
covered by ``tests/tools/test_lazy_deps_matrix_split.py``.
"""

import pytest


class TestMatrixAvailableEverywhere:
    def test_matrix_present_in_picker(self):
        """Matrix is offered on whatever host the suite runs on."""
        import hermes_cli.gateway as gateway_mod

        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        assert "matrix" in keys, "matrix must be available in the platform picker"

    @pytest.mark.windows_only
    def test_matrix_present_on_windows(self):
        """The regression that matters: a real Windows host must offer Matrix.

        A patched ``sys.platform`` would only prove the branch was removed;
        on native Windows this proves the picker the user actually sees now
        includes the platform whose plaintext deps do install here (#76092).
        """
        import hermes_cli.gateway as gateway_mod

        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        assert "matrix" in keys, "matrix must no longer be hidden on Windows"

    @pytest.mark.windows_only
    def test_other_platforms_unaffected_on_windows(self):
        """Removing the gate must not disturb the rest of the picker."""
        import hermes_cli.gateway as gateway_mod

        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        for must_have in ("telegram", "discord", "slack", "mattermost"):
            assert must_have in keys, (
                f"{must_have} disappeared from the Windows picker"
            )
