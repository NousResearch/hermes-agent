"""Regression coverage for the unified runtime's ``served_profiles`` shape (#147868).

The gateway's ``identify`` answer merges the runner's session runtime descriptor over
``build_identify_payload()``, and ``SessionAuthorities.served_profiles()`` publishes each served
profile as ``{"profile_id", "home"}``. ``gateway_state.json`` (the older surface) publishes bare
names. ``_fleet_row`` kept the field only when EVERY entry was a non-empty ``str``, so against the
current shape it dropped the roster entirely and ``_fleet_covered_gateways`` credited only the row's
own profile: every other profile a multiplexer demonstrably serves stayed "not yet restarted", and
the "a previous `hermes update` ... did not restart running gateways" warning never cleared.

Restart obligations are keyed by profile NAME, so both shapes must reduce to names — and an entry
that cannot name a profile home must be dropped rather than defaulted, because crediting a profile
the gateway does not serve is what discharges an obligation nobody met.
"""

from hermes_cli import update_cmd_fleet, update_receipt
from hermes_constants import get_default_hermes_root


def _home(name: str) -> str:
    root = get_default_hermes_root()
    return str(root if name == "default" else root / "profiles" / name)


def _row(served_profiles, profile: str = "default"):
    return update_receipt._fleet_row(
        profile, 4242, "sha", "0.0.0", "sha", served_profiles=served_profiles,
    )


def test_runtime_dict_entries_credit_every_served_profile():
    served = [
        {"profile_id": _home("default"), "home": _home("default")},
        {"profile_id": _home("accountant"), "home": _home("accountant")},
    ]

    row = _row(served)

    assert row["served_profiles"] == ["default", "accountant"]
    covered = update_cmd_fleet._fleet_covered_gateways([row])
    assert covered is not None and ("gateway", "accountant") in covered


def test_legacy_name_list_still_covers_and_dedupes():
    assert _row(["default", "coder", "coder"])["served_profiles"] == ["default", "coder"]


def test_dict_entries_fall_back_to_profile_id_and_keep_first_seen_order():
    served = [
        {"profile_id": _home("coder"), "home": ""},
        {"home": _home("accountant")},
        {"home": _home("coder")},
    ]

    assert _row(served)["served_profiles"] == ["coder", "accountant"]


def test_unresolvable_entry_is_dropped_instead_of_widening_coverage():
    served = [
        {"home": _home("accountant")},
        {"home": "relative/not-a-home"},
        {"home": ""},
        "",
        7,
    ]

    row = _row(served)

    assert row["served_profiles"] == ["accountant"]
    assert update_cmd_fleet._fleet_covered_gateways([row]) == {
        ("gateway", "default"), ("gateway", "accountant"),
    }


def test_no_usable_entry_keeps_the_field_absent():
    row = _row([{"home": "relative/not-a-home"}, 7, ""])

    assert "served_profiles" not in row
    assert update_cmd_fleet._fleet_covered_gateways([row]) == {("gateway", "default")}


def test_non_list_payload_is_still_refused():
    assert "served_profiles" not in _row("default")
    assert "served_profiles" not in _row(None)


def test_roster_names_the_other_profiles_not_just_the_launch_one():
    # The row's own profile is the launch profile; the roster must still name the others.
    served = [{"home": _home("default")}, {"home": _home("saber")}]

    row = _row(served, profile="default")

    assert row["served_profiles"] == ["default", "saber"]
