"""Security-floor tests for the Google Chat runtime installer."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

from plugins.platforms.google_chat import oauth


def test_stale_google_transitives_are_reported_missing(monkeypatch):
    installed = {
        "google-cloud-pubsub": "2.39.0",
        "google-api-python-client": "2.194.0",
        "google-auth": "2.55.0",
        "google-auth-oauthlib": "1.3.1",
        "google-auth-httplib2": "0.3.1",
        "httplib2": "0.31.2",
        "pyasn1": "0.6.3",
    }

    def fake_version(name):
        try:
            return installed[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    monkeypatch.setattr(oauth, "_distribution_version", fake_version)

    stale = {spec.split("==")[0] for spec in oauth._missing_required_packages()}
    assert {"google-auth", "httplib2", "pyasn1"} <= stale


def test_installer_repairs_stale_transitives(monkeypatch):
    states = iter(
        [
            [
                "google-auth==2.55.1",
                "httplib2==0.32.0",
                "pyasn1==0.6.4",
            ],
            [],
        ]
    )
    monkeypatch.setattr(oauth, "_missing_required_packages", lambda: next(states))
    calls = []
    pip_calls = []

    def fake_ensure(feature, prompt=False):
        calls.append((feature, prompt))

    monkeypatch.setattr("tools.lazy_deps.ensure", fake_ensure)
    monkeypatch.setattr(
        "hermes_cli.tools_config._pip_install",
        lambda argv: pip_calls.append(argv) or SimpleNamespace(returncode=0, stderr=""),
    )

    assert oauth.install_deps() is True
    assert calls == [("platform.google_chat", False)]
    assert pip_calls == []


def test_ensure_deps_surfaces_install_reason(monkeypatch):
    """A blocked lazy install must reach the registry's log with its reason, not a bare False."""
    from tools.lazy_deps import FeatureUnavailable
    import pytest
    from plugins.platforms.google_chat import adapter

    monkeypatch.setattr(adapter, "GOOGLE_CHAT_AVAILABLE", False)

    def blocked(feature, prompt=False):
        raise FeatureUnavailable(feature, ("google-cloud-pubsub==2.39.0",), "lazy install target /x is not writable")

    monkeypatch.setattr("tools.lazy_deps.ensure", blocked)
    with pytest.raises(FeatureUnavailable, match="not writable"):
        adapter.ensure_google_chat_deps()
