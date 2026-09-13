"""Google Chat dependency installation crosses PM's declared-feature interface."""

import pm

import pytest

from plugins.platforms.google_chat import oauth


def test_installer_syncs_declared_features_without_rechecking_old_interpreter(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(pm, "sync_venv", lambda extras, **kwargs: calls.append((extras, kwargs)))
    assert oauth.install_deps() is True
    assert calls == [(["google", "google-chat"], {"explicit": True})]
    assert "restart" in capsys.readouterr().out.lower()


def test_auth_stops_when_pm_requires_a_restart(monkeypatch, capsys):
    def unavailable(extra):
        raise RuntimeError(f"{extra} installed; restart Hermes to activate")

    monkeypatch.setattr(pm, "ensure_import", unavailable)
    with pytest.raises(SystemExit) as failure:
        oauth._ensure_deps()
    assert failure.value.code == 1
    assert "restart Hermes" in capsys.readouterr().out
