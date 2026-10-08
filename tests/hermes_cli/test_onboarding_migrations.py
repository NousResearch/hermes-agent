"""The one-shot release of the agentic onboarding's setup profiles: a marked profile becomes a plain one
(marker and setup tool limits gone; chats, credentials and memories kept), the old guide's outcome maps
onto ``onboarding.run``, and the retired onboarding skills are uninstalled."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import onboarding_migrations, profiles
from hermes_cli.config import atomic_config_write, read_user_config_raw
from hermes_state import SessionDB

MARKER = ".setup-profile.json"
SETUP_DEFER = list(onboarding_migrations._SETUP_DEFERRED_TOOLS)


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    return home


def _setup_profile(name: str, marker: dict) -> Path:
    """A profile the agentic build made: its marker, its tool limits on top of a user-disabled toolset,
    a task chat, a memory and a guest identity."""
    path = profiles.create_profile(name, no_alias=True)
    atomic_config_write(path / "config.yaml", {
        "agent": {"disabled_toolsets": ["browser", "project", "catalog"], "reasoning_effort": "low"},
        "platform_toolsets": {"cli": ["setup", "start_chat", "connections", "no_mcp"]},
        "tools": {"tool_search": {"defer": SETUP_DEFER}},
    })
    (path / MARKER).write_text(json.dumps({"setup_disabled_toolsets": ["project", "catalog"], **marker}))
    db = SessionDB(path / "state.db")
    try:
        db.create_session("task-chat", "desktop")
    finally:
        db.close()
    (path / "memories").mkdir(exist_ok=True)
    (path / "memories" / "USER.md").write_text("Likes short answers.\n")
    (path / "auth.json").write_text(json.dumps({"version": 1, "providers": {"nous": {"anon_token": "anon_guest"}}}))
    return path


def _sessions(path: Path) -> int:
    db = SessionDB(path / "state.db")
    try:
        return len(db._read_all("SELECT id FROM sessions"))
    finally:
        db.close()


def test_a_finished_guide_releases_its_profile_and_keeps_its_chats(root):
    setup = _setup_profile("hermes-setup", {"intro": "seen", "completed_at": "2026-10-07T10:00:00+00:00"})
    work = profiles.create_profile("work", no_alias=True)
    atomic_config_write(work / "config.yaml", {"platform_toolsets": {"cli": ["terminal"]}})

    onboarding_migrations.release_setup_profiles()

    assert not (setup / MARKER).exists()
    config = read_user_config_raw(setup / "config.yaml")
    assert config["agent"] == {"disabled_toolsets": ["browser"], "reasoning_effort": "low"}
    assert "platform_toolsets" not in config and "tools" not in config
    assert _sessions(setup) == 1
    assert (setup / "memories" / "USER.md").read_text() == "Likes short answers.\n"
    assert json.loads((setup / "auth.json").read_text())["providers"]["nous"]["anon_token"] == "anon_guest"
    assert read_user_config_raw(work / "config.yaml") == {"platform_toolsets": {"cli": ["terminal"]}}

    root_config = read_user_config_raw(root / "config.yaml")
    assert root_config["onboarding"]["run"] is False
    assert root_config["onboarding"]["seen"]["setup_profile_released"] is True

    from hermes_cli.auth_nous import _read_shared_nous_state
    assert (_read_shared_nous_state() or {}).get("anon_token") == "anon_guest"


def test_an_unfinished_guide_opens_the_questionnaire_and_runs_once(root):
    from hermes_cli.onboarding_run import should_run

    setup = _setup_profile("hermes-setup", {"intro": "unseen", "failed_starts": 0})

    onboarding_migrations.release_setup_profiles()
    assert should_run() is True

    # Latched: a marker that reappears (an old build ran again) is left for that build.
    (setup / MARKER).write_text("{}")
    onboarding_migrations.release_setup_profiles()
    assert (setup / MARKER).exists()


def test_an_unfinished_guide_keeps_an_explicit_run(root):
    from hermes_cli.onboarding_run import set_run, should_run

    _setup_profile("hermes-setup", {"intro": "unseen", "failed_starts": 1})
    set_run(False)

    onboarding_migrations.release_setup_profiles()
    assert should_run() is False


def test_a_backend_launched_inside_the_setup_profile_does_not_release_it(root, monkeypatch):
    setup = _setup_profile("hermes-setup", {"intro": "seen"})
    monkeypatch.setenv("HERMES_HOME", str(setup))

    onboarding_migrations.release_setup_profiles()

    assert (setup / MARKER).exists()
    assert not (root / "config.yaml").exists() or "onboarding" not in read_user_config_raw(root / "config.yaml")


def _hub_install_first_task(home: Path, source: str = "official",
                            identifier: str = "official/productivity/first-task") -> None:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.skills_hub import HubLockFile

    token = set_hermes_home_override(home)
    try:
        skill_dir = home / "skills" / "productivity" / "first-task"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: first-task\n---\n")
        HubLockFile().record_install(
            "first-task", source, identifier, "builtin", "safe", "sha256:0",
            "productivity/first-task", ["SKILL.md"])
    finally:
        reset_hermes_home_override(token)


def _assert_first_task_gone(home: Path) -> None:
    assert not (home / "skills" / "productivity" / "first-task").exists()
    lock = json.loads((home / "skills" / ".hub" / "lock.json").read_text())
    assert "first-task" not in lock["installed"]


def test_hand_installed_onboarding_skills_are_uninstalled(root):
    work = profiles.create_profile("work", no_alias=True)
    _hub_install_first_task(work)

    onboarding_migrations.release_setup_profiles()

    _assert_first_task_gone(work)


def test_an_unrelated_skill_named_first_task_survives(root):
    work = profiles.create_profile("work", no_alias=True)
    _hub_install_first_task(work, "github", "someone/skills/first-task")

    onboarding_migrations.release_setup_profiles()

    assert (work / "skills" / "productivity" / "first-task" / "SKILL.md").is_file()
    lock = json.loads((work / "skills" / ".hub" / "lock.json").read_text())
    assert lock["installed"]["first-task"]["identifier"] == "someone/skills/first-task"


@pytest.mark.parametrize("cli", [["safe"], []])
def test_a_cli_toolset_list_the_user_changed_outlives_the_release(root, cli):
    setup = _setup_profile("hermes-setup", {"intro": "seen"})
    config = read_user_config_raw(setup / "config.yaml")
    config["platform_toolsets"] = {"cli": cli}
    atomic_config_write(setup / "config.yaml", config)

    onboarding_migrations.release_setup_profiles()

    assert read_user_config_raw(setup / "config.yaml")["platform_toolsets"] == {"cli": cli}


def test_the_user_s_own_tool_search_settings_outlive_the_setup_deferred_list(root):
    setup = _setup_profile("hermes-setup", {"intro": "seen"})
    config = read_user_config_raw(setup / "config.yaml")
    config["tools"] = {"tool_search": {"defer": SETUP_DEFER, "enabled": True, "listing_max_tokens": 800},
                       "web": {"backend": "exa"}}
    atomic_config_write(setup / "config.yaml", config)

    onboarding_migrations.release_setup_profiles()

    assert read_user_config_raw(setup / "config.yaml")["tools"] == {
        "tool_search": {"enabled": True, "listing_max_tokens": 800}, "web": {"backend": "exa"}}


def test_a_failed_shared_identity_write_keeps_the_profile_marked_for_the_next_boot(root, monkeypatch):
    from hermes_cli import auth
    from hermes_cli.auth_nous import _nous_shared_store_path, _read_shared_nous_state

    setup = _setup_profile("hermes-setup", {"intro": "seen"})
    save = auth._save_private_json

    def failing_shared_save(target, *args, **kwargs):
        if Path(target) == _nous_shared_store_path():
            raise OSError("disk full")
        return save(target, *args, **kwargs)

    monkeypatch.setattr(auth, "_save_private_json", failing_shared_save)
    onboarding_migrations.release_setup_profiles()

    assert (setup / MARKER).exists()
    seen = read_user_config_raw(root / "config.yaml").get("onboarding", {}).get("seen", {})
    assert "setup_profile_released" not in seen

    monkeypatch.setattr(auth, "_save_private_json", save)
    onboarding_migrations.release_setup_profiles()

    assert not (setup / MARKER).exists()
    assert (_read_shared_nous_state() or {}).get("anon_token") == "anon_guest"


def _assert_released_copy(path: Path) -> None:
    assert not (path / MARKER).exists()
    config = read_user_config_raw(path / "config.yaml")
    assert config["agent"]["disabled_toolsets"] == ["browser"]
    assert "platform_toolsets" not in config and "tools" not in config


def test_a_clone_of_an_unreleased_setup_profile_drops_its_setup_limits(root):
    setup = _setup_profile("hermes-setup", {"intro": "seen"})

    clone = profiles.create_profile("copy", clone_from="hermes-setup", no_alias=True)
    clone_all = profiles.create_profile("full-copy", clone_from="hermes-setup", clone_all=True, no_alias=True)

    _assert_released_copy(clone)
    _assert_released_copy(clone_all)
    assert (setup / MARKER).exists()  # the source is left for the boot release


def test_an_archive_of_a_setup_profile_imported_after_the_release_is_released(root, tmp_path):
    import tarfile

    onboarding_migrations.release_setup_profiles()  # latched before the archive arrives
    exported = _setup_profile("old-setup", {"intro": "seen"})
    archive = tmp_path / "old-setup.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(exported, arcname="old-setup")

    _assert_released_copy(profiles.import_profile(str(archive), name="imported"))


def test_a_distribution_that_ships_a_setup_marker_installs_released(root, tmp_path):
    from hermes_cli.profile_distribution import install_distribution

    import shutil

    staged = tmp_path / "dist"
    shutil.copytree(_setup_profile("staged-setup", {"intro": "seen"}), staged)
    (staged / "distribution.yaml").write_text("name: staged-setup\nversion: 0.1.0\n")

    _assert_released_copy(install_distribution(str(staged), name="installed").target_dir)


def test_an_archive_imported_after_the_release_arrives_without_the_retired_skills(root, tmp_path):
    import tarfile

    onboarding_migrations.release_setup_profiles()  # latched before the archive arrives
    exported = _setup_profile("old-setup", {"intro": "seen"})
    _hub_install_first_task(exported)
    archive = tmp_path / "old-setup.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(exported, arcname="old-setup")

    _assert_first_task_gone(profiles.import_profile(str(archive), name="imported"))
