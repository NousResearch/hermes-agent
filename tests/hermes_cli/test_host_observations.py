"""Permission-aware device observations exposed to plugins."""

from __future__ import annotations

from types import SimpleNamespace


def test_windows_location_returns_granted_coordinates(monkeypatch):
    from hermes_cli import host_observations

    monkeypatch.setattr(host_observations.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        host_observations.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"latitude":31.2304,"longitude":121.4737}',
            stderr="",
        ),
    )

    result = host_observations.observe_location()

    assert result == {
        "status": "granted",
        "permission": "os_managed",
        "source": "windows_location_api",
        "latitude": 31.2304,
        "longitude": 121.4737,
    }


def test_windows_location_distinguishes_denied_from_unavailable(monkeypatch):
    from hermes_cli import host_observations

    monkeypatch.setattr(host_observations.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        host_observations.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Location permission denied",
        ),
    )

    denied = host_observations.observe_location()
    monkeypatch.setattr(host_observations.platform, "system", lambda: "Linux")
    unavailable = host_observations.observe_location()

    assert denied["status"] == "denied"
    assert denied["permission"] == "denied"
    assert unavailable["status"] == "unavailable"
    assert unavailable["source"] == "unsupported_os"


def test_plugin_context_exposes_location_without_secrets(monkeypatch):
    from hermes_cli import host_observations
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    monkeypatch.setattr(
        host_observations,
        "observe_location",
        lambda: {
            "status": "granted",
            "permission": "os_managed",
            "source": "test_location",
            "latitude": 1.0,
            "longitude": 2.0,
        },
    )
    context = PluginContext(
        PluginManifest(name="location-consumer", key="location-consumer"),
        PluginManager(),
    )

    result = context.get_location_observation()

    assert result["status"] == "granted"
    assert set(result) == {
        "status",
        "permission",
        "source",
        "latitude",
        "longitude",
    }
