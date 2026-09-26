"""Telegram lazy-install failures must say why (#124228): the adapter swallowed
the refusal and the gateway only logged "requirements not met" with a stale hint."""

import logging

import pm
from pm import install_hint


def _adapter_module():
    import plugins.platforms.telegram.adapter as adapter
    return adapter


def test_lazy_install_refusal_logs_the_reason(monkeypatch, caplog):
    """A declined/disabled lazy install returns False AND logs the cause, so
    "requirements not met" is diagnosable."""
    adapter = _adapter_module()
    monkeypatch.setattr(adapter, "TELEGRAM_AVAILABLE", False)  # take the missing-SDK path
    from pm.package import InstallError
    monkeypatch.setattr(pm, "ensure_import",
                        lambda extra: (_ for _ in ()).throw(InstallError("venv", "installation declined")))
    with caplog.at_level(logging.WARNING, logger=adapter.logger.name):
        assert adapter.check_telegram_requirements() is False
    assert "declined" in caplog.text


def test_registry_hint_names_the_pm_remedy():
    """The registry hint must agree with pm's one command for a missing extra."""
    adapter = _adapter_module()
    seen = {}
    adapter.register(type("Ctx", (), {"register_platform": lambda self, **kw: seen.update(kw)})())
    assert install_hint("telegram") in seen["install_hint"]
