"""Google Chat dependency installation crosses PM's declared-feature interface."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import pm
from plugins.platforms.google_chat import oauth

ROOT = Path(__file__).resolve().parents[2]


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


def test_installer_repairs_stale_transitives_through_pm(monkeypatch, capsys):
    states = iter([["google-auth==2.55.1", "httplib2==0.32.0", "pyasn1==0.6.4"], []])
    monkeypatch.setattr(oauth, "_missing_required_packages", lambda: next(states))
    calls = []
    monkeypatch.setattr(pm, "sync_venv", lambda extras, **kwargs: calls.append((extras, kwargs)))

    assert oauth.install_deps() is True
    assert calls == [(["google", "google-chat"], {"explicit": True})]
    assert "restart" in capsys.readouterr().out.lower()


def test_ensure_deps_surfaces_install_reason(monkeypatch):
    """A refused install must reach the registry's log with its reason, not a bare False."""
    from plugins.platforms.google_chat import adapter

    monkeypatch.setattr(adapter, "GOOGLE_CHAT_AVAILABLE", False)

    def blocked(extra):
        raise RuntimeError(f"extra {extra!r}: lazy installs are disabled")

    monkeypatch.setattr(pm, "ensure_import", blocked)
    with pytest.raises(RuntimeError, match="lazy installs are disabled"):
        adapter.ensure_google_chat_deps()


def test_ensure_deps_requests_every_extra_before_reporting_a_restart(monkeypatch):
    """A successful install of the FIRST extra raises InstallError("restart Hermes…"); stopping
    there left the second extra uninstalled, so the restart landed right back here. Both are
    requested in one pass and the first failure is what the registry sees."""
    from plugins.platforms.google_chat import adapter

    monkeypatch.setattr(adapter, "GOOGLE_CHAT_AVAILABLE", False)
    requested = []

    def installs_then_needs_restart(extra):
        requested.append(extra)
        raise pm.InstallError("venv", f"{extra} installed; restart Hermes to activate the new dependency environment")

    monkeypatch.setattr(pm, "ensure_import", installs_then_needs_restart)
    with pytest.raises(pm.InstallError, match="google installed"):
        adapter.ensure_google_chat_deps()
    assert requested == ["google", "google-chat"]


def test_probe_pins_match_the_pins_pyproject_declares():
    """Each probe pin must equal the pin its declared extra carries in pyproject.

    ``_REQUIRED_PACKAGES`` doubles as the staleness probe, so an exact ``==`` pin that no
    longer matches what the resolver installs keeps ``_missing_required_packages()``
    non-empty forever: ``install_deps()`` reinstalls the version pyproject asks for, the
    re-check still reports the old pin, and ``_ensure_deps()`` exits. Users saw that as
    "Couldn't generate the OAuth URL" on a healthy install — a lockstep drift, not a
    broken interpreter or client secret.
    """
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    declared_pins = {
        canonicalize_name(requirement.name): str(requirement.specifier)
        for extra in oauth._DEPENDENCY_EXTRAS
        for spec in declared.get(extra, [])
        if (requirement := Requirement(spec)).specifier
    }

    drift = {}
    for spec in oauth._REQUIRED_PACKAGES:
        requirement = Requirement(spec)
        name = canonicalize_name(requirement.name)
        if declared_pins.get(name) != str(requirement.specifier):
            drift[name] = {"probe": str(requirement.specifier), "pyproject": declared_pins.get(name)}

    assert not drift, f"probe pins disagree with the {oauth._DEPENDENCY_EXTRAS} extras: {drift}"

