from unittest.mock import patch

import pytest


def test_unknown_install_detected_when_no_git_dir(tmp_path):
    """When PROJECT_ROOT has no .git, detect as 'unknown' (not 'pip')."""
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        from hermes_cli.config import detect_install_method
        method = detect_install_method(project_root=tmp_path)
        assert method == "unknown"


def test_git_install_detected_when_git_dir_exists(tmp_path):
    """When PROJECT_ROOT has .git, detect as git install."""
    (tmp_path / ".git").mkdir()
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        from hermes_cli.config import detect_install_method
        method = detect_install_method(project_root=tmp_path)
        assert method == "git"


def test_managed_install_takes_precedence(tmp_path):
    """When HERMES_MANAGED is set, that takes precedence over git detection."""
    (tmp_path / ".git").mkdir()
    with patch("hermes_cli.config.get_managed_system", return_value="NixOS"), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        from hermes_cli.config import detect_install_method
        method = detect_install_method(project_root=tmp_path)
        assert method == "nixos"


def test_stamp_file_takes_precedence(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".install_method").write_text("docker\n")
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=tmp_path) == "docker"


@pytest.mark.parametrize("retired_method", ["pip", "homebrew"])
def test_code_scoped_retired_stamp_falls_back_to_unknown(tmp_path, retired_method):
    """Removed install methods must not survive in an upgraded code stamp."""
    (tmp_path / ".install_method").write_text(retired_method + "\n")
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=tmp_path) == "unknown"


@pytest.mark.parametrize("retired_method", ["pip", "homebrew"])
def test_home_scoped_retired_stamp_falls_back_to_unknown(tmp_path, retired_method):
    """Removed install methods must not survive in an upgraded home stamp."""
    code = tmp_path / "code"
    home = tmp_path / "home"
    code.mkdir()
    home.mkdir()
    (home / ".install_method").write_text(retired_method + "\n")
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=home):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=code) == "unknown"


def test_code_scoped_stamp_wins_over_home_stamp(tmp_path):
    """The stamp next to the running code is authoritative over $HERMES_HOME.

    Models a host git install whose $HERMES_HOME is shared with (and stamped
    'docker' by) a co-located container. The code-scoped stamp must win so the
    host install is correctly identified as 'git' and 'hermes update' works.
    """
    code = tmp_path / "code"
    home = tmp_path / "home"
    code.mkdir()
    home.mkdir()
    (code / ".install_method").write_text("git\n")
    (home / ".install_method").write_text("docker\n")  # container contamination
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=home):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=code) == "git"


def test_home_docker_stamp_ignored_when_not_containerized(tmp_path):
    """A 'docker' home stamp is ignored on a host (non-container) install.

    Self-heal path for homes already poisoned by an older image that wrote
    'docker' into the shared $HERMES_HOME. With no code-scoped stamp, a host
    git checkout must fall through to '.git' detection rather than honour the
    contaminating 'docker' value and refuse to update.
    """
    code = tmp_path / "code"
    home = tmp_path / "home"
    code.mkdir()
    home.mkdir()
    (code / ".git").mkdir()
    (home / ".install_method").write_text("docker\n")
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=home), \
         patch("hermes_cli.config._running_in_container", return_value=False):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=code) == "git"


def test_home_docker_stamp_honored_inside_container(tmp_path):
    """A 'docker' home stamp is still honoured when genuinely containerized.

    Back-compat: an older published image that only ever wrote the home-scoped
    stamp (no baked code stamp) must still resolve to 'docker' so the update
    path keeps directing the user to ``docker pull``.
    """
    code = tmp_path / "code"
    home = tmp_path / "home"
    code.mkdir()
    home.mkdir()
    (home / ".install_method").write_text("docker\n")
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=home), \
         patch("hermes_cli.config._running_in_container", return_value=True):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=code) == "docker"


def test_home_non_docker_stamp_still_honored_for_backcompat(tmp_path):
    """Legacy non-'docker' home stamps (e.g. 'git') are still respected.

    Only the 'docker' value carries the cross-contamination risk, so a host
    install that historically stamped 'git' into $HERMES_HOME keeps
    resolving from there when no code-scoped stamp exists yet.
    """
    code = tmp_path / "code"
    home = tmp_path / "home"
    code.mkdir()
    home.mkdir()
    (home / ".install_method").write_text("git\n")
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=home), \
         patch("hermes_cli.config._running_in_container", return_value=False):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=code) == "git"


def test_stamp_install_method_writes_code_scoped(tmp_path):
    """stamp_install_method writes next to the code, not into $HERMES_HOME."""
    code = tmp_path / "code"
    home = tmp_path / "home"
    code.mkdir()
    home.mkdir()
    with patch("hermes_cli.config.get_hermes_home", return_value=home):
        from hermes_cli.config import stamp_install_method
        stamp_install_method("git", project_root=code)
    assert (code / ".install_method").read_text().strip() == "git"
    assert not (home / ".install_method").exists()


def test_container_without_stamp_is_not_docker(tmp_path):
    """An unstamped install in a generic container must NOT be flagged as docker.

    Regression for issue #34397. The two supported installs both stamp
    ``.install_method`` (the curl installer -> ``git``, covered by
    ``test_stamp_file_takes_precedence``; the published image -> ``docker``),
    so neither hits this path. An unsupported manual install dropped into a
    container has no stamp and was wrongly classified as the published Docker
    image, so ``hermes update`` refused to run. With a ``.git`` checkout it
    must resolve to ``git``.
    """
    (tmp_path / ".git").mkdir()
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path), \
         patch("hermes_constants.is_container", return_value=True):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=tmp_path) == "git"


def test_container_unknown_install_without_stamp_is_unknown(tmp_path):
    """Container + no .git + no stamp -> unknown, not docker (issue #34397)."""
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path), \
         patch("hermes_constants.is_container", return_value=True):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=tmp_path) == "unknown"


def test_recommended_update_command_docker():
    from hermes_cli.config import recommended_update_command_for_method
    assert "docker pull" in recommended_update_command_for_method("docker")


def test_recommended_update_command_nix():
    from hermes_cli.config import recommended_update_command_for_method
    command = recommended_update_command_for_method("nix")
    assert "nix profile upgrade" in command
    assert "nixos-rebuild" in command


def test_nix_store_path_detected_as_nix(tmp_path, monkeypatch):
    """A code path under /nix/store/ (nix run / nix profile install) is detected
    as 'nix' even without HERMES_MANAGED or a .install_method stamp."""
    # detect_install_method checks whether the resolved root is a descendant
    # of _NIX_STORE (Path("/nix/store")). We can't create files under the real
    # /nix/store, so patch the constant to point at a temp dir and create the
    # fake install path under it.
    fake_nix_store = tmp_path / "fake-nix-store"
    fake_nix_store.mkdir(parents=True)
    fake_nix = fake_nix_store / "abc123-hermes-agent-0.19.0"
    fake_nix.mkdir(parents=True)

    monkeypatch.setattr("hermes_cli.config._NIX_STORE", fake_nix_store)

    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=fake_nix) == "nix"
