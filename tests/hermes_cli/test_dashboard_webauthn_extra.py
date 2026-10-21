"""dashboard.webauthn configured after a fresh base-extras install provisions
its `webauthn` extra itself (the chicken-and-egg: the block activates the
feature whose dependency is missing). Covers the start-up hook
(hermes_cli.dashboard_webauthn_extra) and the update-repair row
(_configured_features_missing_deps, #124228 parity).
"""

from __future__ import annotations

import importlib.util

import pytest


@pytest.fixture
def no_fido2(monkeypatch):
    """The testenv HAS fido2 (webauthn extra); make find_spec lie for fido2 only."""
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        if name == "fido2":
            return None
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


@pytest.fixture
def sync_calls(monkeypatch):
    import pm

    calls: list[list[str]] = []
    monkeypatch.setattr(pm, "sync_venv", lambda extras, **k: calls.append(list(extras)))
    return calls


_CONFIG = {"dashboard": {"webauthn": {
    "rp_id": "dash.example.com", "rp_name": "Hermes",
    "origins": ["https://dash.example.com"]}}}


def _load_cfg(monkeypatch, cfg):
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config_readonly", lambda: cfg)


class TestMaybeSyncWebauthnExtra:
    def test_configured_and_missing_installs_the_extra(
            self, monkeypatch, tmp_path, no_fido2, sync_calls):
        from hermes_cli.dashboard_webauthn_extra import maybe_sync_webauthn_extra

        _load_cfg(monkeypatch, _CONFIG)
        maybe_sync_webauthn_extra()
        assert sync_calls == [["webauthn"]]

    def test_unconfigured_is_inert(self, monkeypatch, tmp_path, no_fido2, sync_calls):
        from hermes_cli.dashboard_webauthn_extra import maybe_sync_webauthn_extra

        _load_cfg(monkeypatch, {"dashboard": {"webauthn": {"rp_id": "x"}}})  # no origins
        maybe_sync_webauthn_extra()
        assert sync_calls == []

    def test_fido2_present_is_inert(self, monkeypatch, tmp_path, sync_calls):
        """The real testenv has fido2 installed: find_spec finds it -> no sync."""
        from hermes_cli.dashboard_webauthn_extra import maybe_sync_webauthn_extra

        _load_cfg(monkeypatch, _CONFIG)
        maybe_sync_webauthn_extra()
        assert sync_calls == []

    def test_failure_never_blocks_dashboard_start(
            self, monkeypatch, tmp_path, no_fido2, sync_calls, capsys):
        import pm

        from hermes_cli.dashboard_webauthn_extra import maybe_sync_webauthn_extra

        _load_cfg(monkeypatch, _CONFIG)

        def boom(extras, **k):
            raise RuntimeError("pm down")

        monkeypatch.setattr(pm, "sync_venv", boom)
        maybe_sync_webauthn_extra()  # no raise
        assert sync_calls == []


class TestUpdateRepairRow:
    def test_configured_webauthn_row_is_added(self, monkeypatch, no_fido2):
        from hermes_cli.main_install_repair import _configured_features_missing_deps

        _load_cfg(monkeypatch, _CONFIG)
        rows = _configured_features_missing_deps()
        assert ("Dashboard passkeys (WebAuthn)",
                "Run `hermes pm install --extra webauthn` to install fido2.",
                "webauthn") in rows

    def test_unconfigured_webauthn_row_is_absent(self, monkeypatch, no_fido2):
        from hermes_cli.main_install_repair import _configured_features_missing_deps

        _load_cfg(monkeypatch, {"dashboard": {"webauthn": {}}})
        rows = _configured_features_missing_deps()
        assert not any(r[2] == "webauthn" for r in rows)

    def test_row_absent_when_fido2_present(self, monkeypatch):
        """Without the no_fido2 fixture the real import finds fido2 (webauthn extra)."""
        from hermes_cli.main_install_repair import _configured_features_missing_deps

        _load_cfg(monkeypatch, _CONFIG)
        rows = _configured_features_missing_deps()
        assert not any(r[2] == "webauthn" for r in rows)