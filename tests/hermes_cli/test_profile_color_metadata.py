from hermes_cli.profiles import read_profile_meta, write_profile_meta


def test_profile_color_round_trips_without_replacing_bot_title(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "profile.yaml").write_text(
        "ui_meta:\n  hermes-bots:\n    title: Build bot\n",
        encoding="utf-8",
    )

    write_profile_meta(profile, profile_color="#e91e63")

    assert read_profile_meta(profile)["profile_color"] == "#e91e63"
    assert "title: Build bot" in (profile / "profile.yaml").read_text(encoding="utf-8")


def test_profile_color_can_be_cleared(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()

    write_profile_meta(profile, profile_color="#e91e63")
    write_profile_meta(profile, profile_color="")

    assert read_profile_meta(profile)["profile_color"] == ""


def test_profile_color_survives_metadata_reload(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()

    write_profile_meta(profile, profile_color="hsl(42 68% 58%)")

    assert read_profile_meta(profile)["profile_color"] == "hsl(42 68% 58%)"


def test_profile_color_survives_profile_directory_rename(tmp_path):
    profile = tmp_path / "builder"
    profile.mkdir()
    write_profile_meta(profile, profile_color="#e91e63")

    renamed = tmp_path / "build-bot"
    profile.rename(renamed)

    assert read_profile_meta(renamed)["profile_color"] == "#e91e63"
