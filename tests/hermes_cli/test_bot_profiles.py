from pathlib import Path

import pytest
import yaml

from hermes_cli import profiles as profiles_mod
from hermes_cli.bot_profiles import (
    configure_bot_profile,
    create_bot_profile,
    get_bot_profile,
    list_bot_profiles,
    remove_bot_profile,
    rename_bot_profile,
    resolve_bot_chain,
)


@pytest.fixture()
def bot_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n  provider: nous\n  default: default/model\n",
        encoding="utf-8",
    )
    (home / "SOUL.md").write_text("Default system prompt\n", encoding="utf-8")
    (home / ".env").write_text("TEST_API_KEY=secret\n", encoding="utf-8")
    (home / "auth.json").write_text('{"token": "test"}\n', encoding="utf-8")

    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _path: False)
    monkeypatch.setattr(profiles_mod, "_cleanup_gateway_service", lambda *_args: None)
    monkeypatch.setattr(profiles_mod, "_maybe_unregister_gateway_service", lambda *_args: None)
    monkeypatch.setattr(profiles_mod, "_stop_profile_backends", lambda *_args: None)
    monkeypatch.setattr(profiles_mod, "remove_wrapper_script", lambda *_args: False)
    monkeypatch.setattr(profiles_mod, "_migrate_honcho_profile_host", lambda *_args: None)
    monkeypatch.setattr(
        profiles_mod,
        "check_alias_collision",
        lambda _name: "wrapper disabled in test",
    )
    return home


def test_bot_profile_crud_persists_model_prompt_enabled_and_credentials(bot_home):
    created = create_bot_profile(
        "DeepSeek",
        provider="deepseek",
        model="deepseek-v4-flash",
        system_prompt="Solve carefully.",
        seed_skills=False,
    )

    assert created.name == "deepseek"
    assert created.provider == "deepseek"
    assert created.model == "deepseek-v4-flash"
    assert created.system_prompt == "Solve carefully."
    assert created.enabled is True
    assert (created.path / ".env").read_text() == "TEST_API_KEY=secret\n"
    assert (created.path / "auth.json").is_file()

    config = yaml.safe_load((created.path / "config.yaml").read_text())
    assert config["model"]["provider"] == "deepseek"
    assert config["model"]["default"] == "deepseek-v4-flash"
    metadata = yaml.safe_load((created.path / "profile.yaml").read_text())
    assert metadata["bot"]["enabled"] is True

    configured = configure_bot_profile(
        "deepseek",
        model="deepseek-v4",
        system_prompt="Reason before answering.",
        enabled=False,
    )
    assert configured.model == "deepseek-v4"
    assert configured.system_prompt == "Reason before answering."
    assert configured.enabled is False

    renamed = rename_bot_profile("deepseek", "researcher")
    assert renamed.name == "researcher"
    assert renamed.enabled is False
    assert not (bot_home / "profiles" / "deepseek").exists()

    removed = remove_bot_profile("researcher", confirmed=True)
    assert removed.name == "researcher"
    assert not removed.exists()


def test_list_and_resolution_include_default_and_reject_unknown_or_disabled(bot_home):
    create_bot_profile(
        "worker",
        provider="openrouter",
        model="vendor/model",
        system_prompt="Work.",
        seed_skills=False,
    )

    assert [profile.name for profile in list_bot_profiles()] == ["default", "worker"]
    assert [profile.name for profile in resolve_bot_chain(["Worker", "Default"])] == [
        "worker",
        "default",
    ]

    with pytest.raises(ValueError, match=r"Unknown bot '\$missing'.*\$default.*\$worker"):
        resolve_bot_chain(["missing"])

    configure_bot_profile("worker", enabled=False)
    with pytest.raises(ValueError, match=r"Bot '\$worker' is disabled"):
        resolve_bot_chain(["worker"])


def test_remove_requires_explicit_confirmation(bot_home):
    create_bot_profile(
        "worker",
        provider="nous",
        model="test/model",
        system_prompt="Work.",
        seed_skills=False,
    )

    with pytest.raises(ValueError, match="without --yes"):
        remove_bot_profile("worker")
    assert get_bot_profile("worker").name == "worker"


# ---------------------------------------------------- bot_enabled authority


def _write_bot(home, name, *, profile_yaml=None):
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text(
        yaml.safe_dump({"model": {"provider": "nous", "default": "m/1"}}),
        encoding="utf-8",
    )
    if profile_yaml is not None:
        (profile_dir / "profile.yaml").write_text(profile_yaml, encoding="utf-8")
    return profile_dir


@pytest.mark.parametrize("metadata", [None, "{}\n", "description: legacy\n"])
def test_read_profile_meta_no_bot_field_keeps_legacy_default(bot_home, metadata):
    profile_dir = _write_bot(bot_home, "legacy", profile_yaml=metadata)
    assert profiles_mod.read_profile_meta(profile_dir)["bot_enabled"] is True


@pytest.mark.parametrize("roster", [[], None, "invalid", [{"from": "default", "to": "other"}]])
def test_explicit_chain_obeys_declarative_roster(bot_home, roster):
    _write_bot(bot_home, "worker")
    config_path = bot_home / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["agent"] = {"bot_mode": {"enabled": True, "roster": roster}}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="not allowed"):
        resolve_bot_chain(["worker"])
    config["agent"]["bot_mode"]["roster"] = [{"from": "default", "to": "worker"}]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert [profile.name for profile in resolve_bot_chain(["worker"])] == ["worker"]


def test_read_profile_meta_corrupt_yaml_fails_closed(bot_home):
    profile_dir = _write_bot(bot_home, "broken", profile_yaml="bot: [unclosed\n")
    assert profiles_mod.read_profile_meta(profile_dir)["bot_enabled"] is False


@pytest.mark.parametrize("replacement", ["", "null\n", "# interrupted write\n"])
def test_indeterminate_metadata_does_not_enable_disabled_bot(bot_home, replacement):
    from tools.bot_mode_probe import allowed_local_profile_names

    directory = _write_bot(bot_home, "worker", profile_yaml="bot:\n  enabled: false\n")
    assert get_bot_profile("worker").enabled is False
    (directory / "profile.yaml").write_text(replacement, encoding="utf-8")
    assert get_bot_profile("worker").enabled is False
    assert "worker" not in allowed_local_profile_names(bot_home)
    with pytest.raises(ValueError, match="disabled"):
        resolve_bot_chain(["worker"])


@pytest.mark.parametrize("replacement", ["", "null\n", "# interrupted write\n"])
def test_indeterminate_config_keeps_roster_fail_closed(bot_home, replacement):
    from tools.bot_mode_probe import allowed_local_profile_names

    _write_bot(bot_home, "worker")
    path = bot_home / "config.yaml"
    path.write_text("agent:\n  bot_mode:\n    enabled: true\n    roster: []\n", encoding="utf-8")
    assert "worker" not in allowed_local_profile_names(bot_home)
    path.write_text(replacement, encoding="utf-8")
    assert "worker" not in allowed_local_profile_names(bot_home)
    with pytest.raises(ValueError, match="not allowed"):
        resolve_bot_chain(["worker"])


def test_read_profile_meta_non_mapping_document_fails_closed(bot_home):
    profile_dir = _write_bot(bot_home, "listdoc", profile_yaml="- a\n- b\n")
    assert profiles_mod.read_profile_meta(profile_dir)["bot_enabled"] is False


def test_read_profile_meta_non_mapping_bot_section_fails_closed(bot_home):
    profile_dir = _write_bot(bot_home, "weirdbot", profile_yaml="bot: 42\n")
    assert profiles_mod.read_profile_meta(profile_dir)["bot_enabled"] is False


def test_read_profile_meta_explicit_flag_is_authoritative(bot_home):
    off = _write_bot(bot_home, "off", profile_yaml="bot:\n  enabled: false\n")
    on = _write_bot(bot_home, "on", profile_yaml="bot:\n  enabled: true\n")
    assert profiles_mod.read_profile_meta(off)["bot_enabled"] is False
    assert profiles_mod.read_profile_meta(on)["bot_enabled"] is True


@pytest.mark.parametrize("value", ['"false"', '"true"', "1", "0", "null", "[]"])
def test_bot_enabled_requires_a_boolean(bot_home, value):
    profile_dir = _write_bot(bot_home, "invalid", profile_yaml=f"bot:\n  enabled: {value}\n")
    assert profiles_mod.read_profile_meta(profile_dir)["bot_enabled"] is False


def test_bot_gate_refuses_unreadable_metadata(bot_home, monkeypatch):
    from hermes_cli.profile_bot_policy import read_bot_enabled

    profile_dir = _write_bot(bot_home, "unreadable", profile_yaml="bot:\n  enabled: false\n")
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == profile_dir / "profile.yaml":
            raise PermissionError("metadata is not readable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert read_bot_enabled(profile_dir) is False


def test_resolve_bot_chain_refuses_corrupt_metadata_profile(bot_home):
    """A previously disabled/indeterminate profile must not become callable
    because its metadata file cannot be parsed."""
    _write_bot(bot_home, "worker")
    _write_bot(bot_home, "broken", profile_yaml="bot: [unclosed\n")

    assert get_bot_profile("worker").enabled is True
    assert get_bot_profile("broken").enabled is False
    with pytest.raises(ValueError, match="disabled"):
        resolve_bot_chain(["worker", "broken"])
