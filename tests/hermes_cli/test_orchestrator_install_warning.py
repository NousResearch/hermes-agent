"""RED test for issue #87287.

A plugin that auto-spawns worker processes (e.g. an orchestrator) installs
silently today — agents see only a description and have no signal at install
time. This test pins the expected installer warning: when the manifest declares
``kind: orchestrator``, ``auto_dispatches: true``, or ``spawns_workers: true``,
the install flow must surface a warning that names gateway + dispatch so the
user knows to install and enable them before the plugin does anything.

Tests target the shared display helper directly — adding it once in the
installer pipeline warns every entry point (cmd_install, dashboard_install,
update, enable).
"""

from __future__ import annotations

from unittest.mock import MagicMock


def _console_with_buffer():
    console = MagicMock()

    def _record(*args, **kwargs):
        console._transcript.append(" ".join(str(a) for a in args))

    console._transcript = []
    console.print.side_effect = _record
    return console


class TestOrchestratorInstallWarning:
    """PIN: orchestrator plugin manifests must surface a gateway/dispatch warning."""

    def _call_helper(self, manifest):
        from hermes_cli.plugins_cmd_install import _display_orchestrator_warning
        console = _console_with_buffer()
        _display_orchestrator_warning(manifest, console)
        return console

    def test_kind_orchestrator_triggers_warning(self):
        console = self._call_helper({
            "name": "acme_orchestrator",
            "kind": "orchestrator",
            "description": "Spreads work across workers",
        })
        joined = "\n".join(console._transcript)
        assert "gateway" in joined.lower(), (
            f"Installer must mention 'gateway' for orchestrator plugins; got: {joined!r}"
        )
        assert "dispatch" in joined.lower(), (
            f"Installer must mention 'dispatch' for orchestrator plugins; got: {joined!r}"
        )

    def test_auto_dispatches_true_triggers_warning(self):
        console = self._call_helper({
            "name": "acme_dispatcher",
            "auto_dispatches": True,
        })
        joined = "\n".join(console._transcript)
        assert "gateway" in joined.lower(), (
            f"Installer must warn when auto_dispatches=true; got: {joined!r}"
        )

    def test_spawns_workers_true_triggers_warning(self):
        console = self._call_helper({
            "name": "acme_spawner",
            "spawns_workers": True,
        })
        joined = "\n".join(console._transcript)
        assert "gateway" in joined.lower(), (
            f"Installer must warn when spawns_workers=true; got: {joined!r}"
        )

    def test_passive_plugin_does_not_warn(self):
        console = self._call_helper({
            "name": "acme_passive",
            "kind": "standalone",
            "description": "Just a passive skill",
        })
        joined = "\n".join(console._transcript).lower()
        assert "auto-spawn" not in joined, (
            f"Passive plugin should not emit orchestrator warning; got: {joined!r}"
        )
        assert console.print.call_count == 0, (
            f"Passive plugin should call console.print 0 times; got {console.print.call_count}"
        )

    def test_no_orchestrator_signals_silently_skip(self):
        console = self._call_helper({
            "name": "acme_unknown",
            "kind": "backend",
        })
        assert console.print.call_count == 0, (
            f"Non-orchestrator kind should not emit orchestrator warning; "
            f"got {console.print.call_count} prints: {console._transcript!r}"
        )


def test_dashboard_install_surfaces_orchestrator_warning(tmp_path, monkeypatch):
    """The dashboard/TUI install returns ``warnings`` instead of a console, so
    the warning has to be appended there too — otherwise installs from the
    Desktop/TUI surface stay silent, which is the bug #87287 is about."""
    from hermes_cli import plugins_cmd_catalog as catalog
    from hermes_cli import plugins_cmd_install as m

    fake = MagicMock()
    fake._resolve_git_url.return_value = ("https://example.com/acme.git", None)
    fake._install_plugin_core.return_value = (
        tmp_path, {"name": "acme_orchestrator", "kind": "orchestrator"}, "acme_orchestrator",
    )
    fake._python_dependency_summary.return_value = []
    fake._missing_env_specs.return_value = []
    monkeypatch.setattr(m, "_pc", lambda: fake)
    monkeypatch.setattr(catalog, "raise_if_removed", lambda *a, **k: None)

    out = m.dashboard_install_plugin("https://example.com/acme.git", force=False, enable=False)

    assert out["ok"] is True, out
    assert any(
        "gateway" in w.lower() and "dispatch" in w.lower() for w in out["warnings"]
    ), f"dashboard warnings must name gateway + dispatch; got {out['warnings']!r}"


def test_orchestrator_warning_text_is_none_for_passive_manifest():
    from hermes_cli.plugins_cmd_install import _orchestrator_warning_text

    assert _orchestrator_warning_text({"name": "x", "kind": "standalone"}) is None
    text = _orchestrator_warning_text({"name": "x", "kind": "ORCHESTRATOR"})
    assert text is not None and "gateway" in text.lower()