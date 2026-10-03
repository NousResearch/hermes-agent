"""Public names for a doctor section banner and the named-profile listing.

A plugin that adds its own ``hermes doctor`` output wants its section to look
like every other one; a plugin that offers per-profile features needs the same
list of live named profiles the profile commands use (identity marker present,
tombstones skipped). Each public name is the SAME object as the private
spelling, so doctor and the profile commands are unchanged.
"""

import pytest

from hermes_cli import doctor_report, profiles


@pytest.mark.parametrize(
    ("module", "public", "private"),
    [
        (doctor_report, "section", "_section"),
        (profiles, "iter_named_profile_dirs", "_iter_named_profile_dirs"),
    ],
)
def test_public_name_is_the_private_helper(module, public, private):
    assert getattr(module, public) is getattr(module, private)


def test_section_prints_a_titled_banner(capsys):
    doctor_report.section("Plugin checks")
    assert "Plugin checks" in capsys.readouterr().out
