from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from hermes_cli.main import cmd_profile
from hermes_cli.profiles import ProfileInfo
from hermes_cli import profiles as profiles_mod


def _profile(name: str, *, nickname: str = "", is_default: bool = False) -> ProfileInfo:
    return ProfileInfo(
        name=name,
        path=Path(f"/tmp/{name}"),
        is_default=is_default,
        gateway_running=True,
        model="gpt-5.4",
        provider="openai-codex",
        skill_count=3,
        nickname=nickname,
    )


def test_write_profile_meta_preserves_nickname_when_updating_other_fields(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()

    profiles_mod.write_profile_meta(
        profile_dir,
        nickname="Clovis",
        description="original",
        description_auto=True,
    )
    profiles_mod.write_profile_meta(profile_dir, description="edited")

    meta = profiles_mod.read_profile_meta(profile_dir)
    assert meta["nickname"] == "Clovis"
    assert meta["description"] == "edited"
    assert meta["description_auto"] is True


def test_cmd_profile_list_omits_nickname_column_when_empty(monkeypatch, capsys):
    monkeypatch.setattr(
        profiles_mod,
        "list_profiles",
        lambda: [_profile("default", is_default=True), _profile("coder")],
    )
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")

    cmd_profile(Namespace(profile_action="list"))

    out = capsys.readouterr().out
    assert "Nickname" not in out
    assert "Profile" in out
    assert "Alias" in out


def test_cmd_profile_list_shows_nickname_column_when_any_profile_has_one(monkeypatch, capsys):
    monkeypatch.setattr(
        profiles_mod,
        "list_profiles",
        lambda: [
            _profile("default", nickname="Noame", is_default=True),
            _profile("coder", nickname="Clovis"),
        ],
    )
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")

    cmd_profile(Namespace(profile_action="list"))

    out = capsys.readouterr().out
    assert "Nickname" in out
    assert "Noame" in out
    assert "Clovis" in out


def test_cmd_profile_show_includes_nickname_when_present(monkeypatch, tmp_path, capsys):
    profile_dir = tmp_path / "profiles" / "coder"
    profile_dir.mkdir(parents=True)

    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name == "coder")
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda name: profile_dir)
    monkeypatch.setattr(profiles_mod, "_read_config_model", lambda path: ("gpt-5.4", "openai-codex"))
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda path: True)
    monkeypatch.setattr(profiles_mod, "_count_skills", lambda path: 7)
    monkeypatch.setattr(profiles_mod, "_read_distribution_meta", lambda path: (None, None, None))
    monkeypatch.setattr(
        profiles_mod,
        "read_profile_meta",
        lambda path: {"nickname": "Clovis", "description": "", "description_auto": False},
    )
    monkeypatch.setattr(profiles_mod, "_get_wrapper_dir", lambda: tmp_path / "bin")

    cmd_profile(Namespace(profile_action="show", profile_name="coder"))

    out = capsys.readouterr().out
    assert "Profile: coder" in out
    assert "Nickname: Clovis" in out


def test_cmd_profile_show_omits_nickname_when_empty(monkeypatch, tmp_path, capsys):
    profile_dir = tmp_path / "profiles" / "coder"
    profile_dir.mkdir(parents=True)

    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name == "coder")
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda name: profile_dir)
    monkeypatch.setattr(profiles_mod, "_read_config_model", lambda path: ("gpt-5.4", "openai-codex"))
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda path: True)
    monkeypatch.setattr(profiles_mod, "_count_skills", lambda path: 7)
    monkeypatch.setattr(profiles_mod, "_read_distribution_meta", lambda path: (None, None, None))
    monkeypatch.setattr(
        profiles_mod,
        "read_profile_meta",
        lambda path: {"nickname": "", "description": "", "description_auto": False},
    )
    monkeypatch.setattr(profiles_mod, "_get_wrapper_dir", lambda: tmp_path / "bin")

    cmd_profile(Namespace(profile_action="show", profile_name="coder"))

    out = capsys.readouterr().out
    assert "Profile: coder" in out
    assert "Nickname:" not in out
