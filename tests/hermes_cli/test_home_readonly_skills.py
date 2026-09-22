"""Existing read-only skills roots keep their operator-selected write restriction."""

import stat
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_constants
from hermes_cli import config
from hermes_cli.config_home import _ensure_directory, initialize_home


@pytest.mark.parametrize(
    "existing_mode,explicit_mode,preserve,expected",
    [(0o500, "", True, 0o500), (0o555, "", True, 0o500),
     (0o755, "", True, 0o700), (0o500, "701", True, 0o701),
     (0o500, "", False, 0o700)],
)
def test_secure_dir_requested_mode(monkeypatch, existing_mode, explicit_mode, preserve, expected):
    """Check requested syscalls, without pretending Windows has POSIX permissions."""
    monkeypatch.setenv("HERMES_HOME_MODE", explicit_mode)
    with patch.object(hermes_constants, "get_managed_system", return_value=None), \
         patch.object(hermes_constants, "_container_or_chmod_skipped", return_value=False), \
         patch.object(hermes_constants, "_chown_to_hermes_uid") as chown, \
         patch.object(hermes_constants.os, "stat", return_value=SimpleNamespace(st_mode=existing_mode)), \
         patch.object(hermes_constants.os, "chmod") as chmod:
        config._secure_dir("skills", preserve_readonly=preserve)
    chmod.assert_called_once_with("skills", expected)
    chown.assert_called_once_with("skills")


@pytest.mark.parametrize("existing", [False, True])
def test_only_existing_directory_opts_in(tmp_path, existing):
    skills = tmp_path / "skills"
    if existing:
        skills.mkdir()
    with patch.object(config, "_secure_dir") as secure:
        _ensure_directory(skills, create=True, secure=True, home=tmp_path, preserve_readonly=True)
    if existing:
        secure.assert_called_once_with(skills, preserve_readonly=True)
    else:
        secure.assert_called_once_with(skills)


def test_home_only_opts_in_skills(tmp_path):
    home = tmp_path / "home"
    (home / "skills").mkdir(parents=True)
    (home / "logs").mkdir()
    with patch.object(config, "is_managed", return_value=False), \
         patch.object(config, "_ensure_default_soul_md"), \
         patch.object(config, "_secure_dir") as secure:
        initialize_home(home, ("skills", "logs"), set())
    assert [call.kwargs for call in secure.call_args_list] == [
        {}, {"preserve_readonly": True}, {},
    ]


@pytest.mark.linux_only
@pytest.mark.parametrize(
    "directory,initial_mode,explicit_mode,expected",
    [("skills", 0o500, "", 0o500), ("skills", 0o555, "", 0o500),
     ("skills", None, "", 0o700), ("logs", 0o500, "", 0o700),
     ("skills", 0o500, "701", 0o701), ("skills", 0o500, "invalid", 0o700)],
)
def test_load_config_real_directory_permissions(
    tmp_path, monkeypatch, directory, initial_mode, explicit_mode, expected
):
    """Exercise real POSIX chmod/stat through the public initialization entrypoint."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    target = home / directory
    if initial_mode is not None:
        target.mkdir()
        target.chmod(initial_mode)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_MODE", explicit_mode)
    for variable in ("HERMES_UID", "HERMES_GID", "HERMES_MANAGED"):
        monkeypatch.delenv(variable, raising=False)
    # CI may itself run in a container; exercise the non-container policy while
    # retaining the host's real filesystem and permission syscalls.
    monkeypatch.setattr(hermes_constants, "_container_or_chmod_skipped", lambda: False)
    assert str(home) not in config._HERMES_HOME_ENSURED
    try:
        config.load_config()
        assert str(home) in config._HERMES_HOME_ENSURED
        assert stat.S_IMODE(target.stat().st_mode) == expected
    finally:
        if target.exists():
            target.chmod(0o700)
