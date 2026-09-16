"""Fleet matrix coverage for a multiplexed gateway (#113350).

One multiplexer serves several profiles, so only its own home holds a live
gateway pid. Enumerating profile homes therefore produced a single row, and
a receipt owing an identity per profile could never be discharged: the
pending fleet-restart warning stayed forever and `hermes gateway restart`,
the remedy the warning names, could not clear it. The served profiles are
already recorded in runtime status, so each one gets a row.
"""

import json
import os
from pathlib import Path

import hermes_cli.update_receipt as ur


def _setup(monkeypatch, tmp_path, record: dict, profiles: list[str]):
    """Default home holds the multiplexer; the named profiles have homes but no gateway."""
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir(exist_ok=True)
    for profile in profiles:
        (profiles_root / profile).mkdir(exist_ok=True)

    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "HEADSHA", "version": "1.0"},
    )
    monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr("gateway.control_socket.identify_gateway", lambda h, **k: None)
    monkeypatch.setattr("gateway.status.live_gateway_pid_for_home", lambda h: os.getpid())
    (home / "gateway_state.json").write_text(json.dumps(record), encoding="utf-8")
    return home


def _multiplexer_record(served: list[str]) -> dict:
    return {
        "kind": "hermes-gateway",
        "pid": os.getpid(),
        "gateway_state": "running",
        "code_sha": "HEADSHA",
        "code_version": "1.0",
        "served_profiles": served,
    }


def test_a_multiplexer_covers_every_profile_it_serves(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _multiplexer_record(["default", "athena"]), ["athena"])

    rows = ur.collect_fleet_versions()

    by_profile = {row["profile"]: row for row in rows}
    assert set(by_profile) == {"default", "athena"}
    assert by_profile["athena"]["state"] == "current"
    assert by_profile["athena"]["code_sha"] == "HEADSHA"
    assert by_profile["athena"]["pid"] == os.getpid()


def test_a_served_profile_inherits_a_stale_multiplexer(monkeypatch, tmp_path):
    """A served profile is exactly as stale as the process serving it."""
    record = _multiplexer_record(["default", "athena"]) | {"code_sha": "OLDSHA"}
    _setup(monkeypatch, tmp_path, record, ["athena"])

    rows = ur.collect_fleet_versions()

    assert {row["profile"]: row["state"] for row in rows} == {
        "default": "stale",
        "athena": "stale",
    }


def test_a_profile_with_its_own_gateway_is_left_alone(monkeypatch, tmp_path):
    """A profile that answers for itself keeps its own row, not the multiplexer's."""
    home = _setup(
        monkeypatch, tmp_path, _multiplexer_record(["default", "athena"]), ["athena"]
    )
    own = tmp_path / "profiles" / "athena" / "gateway_state.json"
    own.write_text(
        json.dumps(
            {
                "kind": "hermes-gateway",
                "pid": os.getpid(),
                "gateway_state": "running",
                "code_sha": "OLDSHA",
                "code_version": "0.9",
            }
        ),
        encoding="utf-8",
    )

    rows = ur.collect_fleet_versions()

    athena = next(row for row in rows if row["profile"] == "athena")
    assert athena["code_sha"] == "OLDSHA"
    assert athena["state"] == "stale"
    assert Path(home).exists()


def test_nothing_changes_without_served_profiles(monkeypatch, tmp_path):
    record = _multiplexer_record(["default", "athena"])
    del record["served_profiles"]
    _setup(monkeypatch, tmp_path, record, ["athena"])

    rows = ur.collect_fleet_versions()

    assert [row["profile"] for row in rows] == ["default"]
