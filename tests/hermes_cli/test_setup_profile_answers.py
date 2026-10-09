"""Setup keeps what is already there: a re-sent /initiate-setup keeps the picks, and a Reset run from the setup
chat itself still gets the launch profile's skills."""

from agent import initiate_setup_prompt
from hermes_cli import profiles, setup_profile


def test_resent_setup_prompt_keeps_the_recorded_picks(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(setup_profile, "read_state", dict)
    monkeypatch.setattr("hermes_cli.anon_auth.free_tier_route", lambda: None)
    setup_profile.record_cards("s1", {"pick_ids": ["github"], "picks": {"plugins": ["blender"]}})

    initiate_setup_prompt.build_initiate_setup_prompt("desktop", [], "default", session_id="s1")

    cards = setup_profile.read_cards("s1")
    assert cards["pick_ids"] == ["github"]
    assert cards["picks"] == {"plugins": ["blender"]}
    assert "fork" in cards


def test_reset_run_from_the_setup_profile_copies_the_launch_profiles_skills(tmp_path, monkeypatch):
    launch, setup = tmp_path / "launch", tmp_path / "hermes-setup"
    for home in (launch, setup):
        (home / "skills" / "notes").mkdir(parents=True)
        (home / "skills" / "notes" / "SKILL.md").write_text("notes", encoding="utf-8")
    (setup / profiles.SETUP_PROFILE_MARKER).write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(setup))
    monkeypatch.setattr(setup_profile, "find_setup_profile", lambda: ("hermes-setup", setup))

    setup_profile.reset_setup_profile(launch)

    assert (setup / "skills" / "notes" / "SKILL.md").read_text(encoding="utf-8") == "notes"
