"""Host-specific gating in ``hermes_cli.gateway._all_platforms()``.

Some messaging platforms can't function on every host. The gate lives
in one place — ``_all_platforms()`` — so the setup wizard, the curses
gateway-config menu, and any future picker all see the same list.

Currently:
- Matrix is visible but marked ``unavailable on this OS`` on native
  Windows. The ``[matrix]`` extra pulls ``mautrix[encryption]`` ->
  ``python-olm``, which has no Windows wheel and needs ``make`` +
  libolm to build from sdist. There's no native Windows path that
  works, so the picker keeps the row with an actionable explanation
  (pointing at WSL) instead of silently dropping it.
"""


class TestMatrixGatedOnWindows:
    def test_matrix_present_on_linux(self, monkeypatch):
        """Sanity: matrix is still in the picker on Linux/macOS."""
        import hermes_cli.gateway as gateway_mod

        monkeypatch.setattr(gateway_mod.sys, "platform", "linux")
        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        assert "matrix" in keys, "matrix must be available on Linux"

    def test_other_platforms_unaffected_on_windows(self, monkeypatch):
        """Gating must only gate matrix, not collateral damage."""
        import hermes_cli.gateway as gateway_mod

        monkeypatch.setattr(gateway_mod.sys, "platform", "win32")
        platforms = gateway_mod._all_platforms()
        keys = {p["key"] for p in platforms}
        # A representative sample of platforms that have no Windows
        # blockers — picker should still surface them.
        for must_have in ("telegram", "discord", "slack", "mattermost"):
            assert must_have in keys, (
                f"{must_have} disappeared from Windows picker — gate is "
                "over-filtering"
            )


class TestUnavailableRowsAreNotSetupProgress:
    """The post-setup gateway offer must ignore explainer-only rows.

    A row that exists purely to explain why a platform can't be
    configured on this host (``unavailable on this OS``) must never
    make ``any_configured`` true, otherwise the wizard offers gateway
    installation/restart although nothing was configured.
    """

    def test_unavailable_status_is_not_progress(self):
        import hermes_cli.gateway as gateway_mod

        assert gateway_mod._is_setup_progress("unavailable on this OS") is False
        assert gateway_mod._is_setup_progress("unavailable") is False

    def test_configured_statuses_still_count_as_progress(self):
        import hermes_cli.gateway as gateway_mod

        assert gateway_mod._is_setup_progress("configured") is True
        assert gateway_mod._is_setup_progress("enabled, not paired") is True

    def test_wizard_offers_nothing_when_only_placeholder_is_present(
        self, monkeypatch, capsys
    ):
        """Wizard-level: a lone unavailable placeholder must not trigger the
        install/restart offer."""
        import hermes_cli.gateway as gateway_mod

        placeholder = gateway_mod._windows_matrix_placeholder()
        monkeypatch.setattr(
            gateway_mod, "_all_platforms", lambda: [placeholder]
        )
        monkeypatch.setattr(gateway_mod, "_is_service_installed", lambda: False)
        monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)
        # First select the Matrix row (index 0), then "Done" (index 1).
        choices = iter([0, 1])
        monkeypatch.setattr(gateway_mod, "prompt_choice", lambda *a, **k: next(choices))
        # Any invocation of the install/start offer prompts is a failure:
        # with only an unavailable row, the wizard must not offer anything.
        offered = []

        def _fail_if_offered(prompt, default=True):
            offered.append(prompt)
            return False

        monkeypatch.setattr(gateway_mod, "prompt_yes_no", _fail_if_offered)
        monkeypatch.setattr(gateway_mod, "is_macos", lambda: False)
        monkeypatch.setattr(gateway_mod, "is_linux", lambda: True)

        gateway_mod.gateway_setup()

        out = capsys.readouterr().out
        # The unavailable row is shown with its reason…
        assert "Matrix" in out
        assert "native Windows" in out
        # …but no install/start offer prompt is ever raised.
        assert offered == [], f"wizard offered setup for unavailable-only rows: {offered}"
