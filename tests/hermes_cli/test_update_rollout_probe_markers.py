"""S2.2 marker truth and profile-bound deployment eligibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import update_rollout_probe as probe


SHA = "a" * 40


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, "absent"),
        ("started=1700000000.0\npid=1234\n", "live"),
        ("started=\npid=1234\n", "malformed"),
        ("started=1700000000.0\n", "malformed"),
        ("started=1700000000.0\npid=0\n", "malformed"),
        ("started=1700000000.0\npid=1234\nextra=yes\n", "malformed"),
    ],
)
def test_marker_state_validates_the_actual_generic_producer_format(tmp_path, text, expected):
    path = tmp_path / "marker"
    if text is not None:
        _write(path, text)
    assert probe._marker_state(path)["state"] == expected


def test_marker_state_reports_present_unreadable_as_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "marker"
    path.write_bytes(b"started=1700000000.0\npid=1234\n")

    def unreadable(*args, **kwargs):
        raise OSError("read blocked")

    monkeypatch.setattr(Path, "read_text", unreadable)
    result = probe._marker_state(path)
    assert result["state"] == "unavailable"
    assert result["reason"] == "OSError"


def test_fleet_marker_accepts_only_strict_optional_producer_fields(tmp_path):
    path = tmp_path / "fleet_restart_pending"
    inventory = {"version": 1, "runtimes": []}
    _write(
        path,
        "started=1700000000.0\npid=1234\n"
        f"expected_sha={SHA}\n"
        f"inventory={json.dumps(inventory, separators=(',', ':'))}\n",
    )
    assert probe._marker_state(path, fleet=True)["state"] == "live"

    _write(path, "started=1700000000.0\npid=1234\ninventory={\"version\":2,\"runtimes\":[]}\n")
    assert probe._marker_state(path, fleet=True)["state"] == "malformed"


def test_recovery_uses_marker_owners_and_active_profile_home(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    home = tmp_path / "active-profile"
    home.mkdir()
    update = checkout / ".update-incomplete"
    lazy = checkout / ".lazy-refresh-incomplete"
    fleet = home / "fleet_restart_pending"
    _write(update, "started=1700000000.0\npid=10\n")

    import hermes_cli.main as main_module
    import hermes_cli.update_cmd as update_cmd

    monkeypatch.setattr(main_module, "PROJECT_ROOT", checkout)
    monkeypatch.setattr(probe, "_CODE_ROOT", checkout)
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: home)
    result = probe._read_recovery()

    assert result["markers"]["updateIncomplete"]["path"] == str(update)
    assert result["markers"]["lazyRefreshIncomplete"]["path"] == str(lazy)
    assert result["markers"]["fleetRestartPending"]["path"] == str(fleet)
    assert result["state"] == "live"


def test_recovery_is_clear_only_when_every_marker_is_verified_absent(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    import hermes_cli.main as main_module
    import hermes_cli.update_cmd as update_cmd

    monkeypatch.setattr(main_module, "PROJECT_ROOT", checkout)
    monkeypatch.setattr(probe, "_CODE_ROOT", checkout)
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: home)
    assert probe._read_recovery()["state"] == "clear"

    _write(home / "fleet_restart_pending", "not producer data\n")
    result = probe._read_recovery()
    assert result["state"] == "malformed"
    assert result["markers"]["fleetRestartPending"]["state"] == "malformed"


def test_image_absence_is_distinct_and_only_absence_is_deployment_eligible(monkeypatch):
    import hermes_cli.image_provenance as image_provenance

    monkeypatch.setattr(image_provenance, "read_image_provenance", lambda: None)
    assert probe._read_image_marker() == {"state": "absent"}

    invalid = type("Invalid", (), {"valid": False, "marker_path": "marker", "error": "bad"})()
    monkeypatch.setattr(image_provenance, "read_image_provenance", lambda: invalid)
    assert probe._read_image_marker()["state"] == "malformed"

    monkeypatch.setattr(image_provenance, "read_image_provenance", lambda: (_ for _ in ()).throw(ImportError("missing")))
    assert probe._read_image_marker()["state"] == "unavailable"


def test_collect_observation_fails_closed_for_unavailable_image(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr(probe, "_CODE_ROOT", tmp_path / "checkout")
    monkeypatch.setattr(probe, "_read_git_metadata", lambda root: {"checkoutState": "clean"})
    monkeypatch.setattr(probe, "_read_recovery", lambda: {"state": "clear", "markers": {}})
    monkeypatch.setattr(probe, "_read_runtime_evidence", lambda: {"processGeneration": {"state": "unknown"}})
    monkeypatch.setattr(probe, "_dependency_evidence", lambda: {"state": "ready"})
    monkeypatch.setattr(probe, "_read_image_marker", lambda: {"state": "unavailable"})

    observation = probe.collect_observation()["observation"]
    assert observation["deployment"]["eligible"] is False
    assert observation["deployment"]["reason"] == "unavailable-image-marker"

    monkeypatch.setattr(probe, "_read_image_marker", lambda: {"state": "absent"})
    observation = probe.collect_observation()["observation"]
    assert observation["deployment"]["eligible"] is True
