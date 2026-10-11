"""Public names for a doctor section banner and the named-profile listing.

A plugin that adds its own ``hermes doctor`` output wants its section to look
like every other one; a plugin that offers per-profile features needs the same
list of live named profiles the profile commands use (identity marker present,
tombstones skipped). The private spellings stay as aliases bound to the same
objects; doctor and the profile commands read the public names, so overriding
a public name is what takes effect.
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


def test_overriding_iter_named_profile_dirs_reaches_list_profile_names(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(profiles, "iter_named_profile_dirs", lambda: [SimpleNamespace(name="zeta")])
    assert profiles.list_profile_names() == ["default", "zeta"]


def test_overriding_section_reaches_the_live_doctor_checks(monkeypatch):
    from hermes_cli import doctor_live

    titles = []
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    monkeypatch.setattr(doctor_live, "_KEYED_PROBES", {})
    monkeypatch.setattr(doctor_live, "_probe_browser",
                        lambda timeout: doctor_live.ProbeResult("Browser", "skip", ""))
    monkeypatch.setattr(doctor_live, "section", titles.append)
    doctor_live.run_live_checks([])
    assert titles == ["Live Backend Probes (opt-in, real calls)"]
