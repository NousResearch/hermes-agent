from unittest.mock import patch


def test_pip_install_detected_when_no_git_dir(tmp_path):
    """When PROJECT_ROOT has no .git, detect as pip install."""
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        from hermes_cli.config import detect_install_method
        method = detect_install_method(project_root=tmp_path)
        assert method == "pip"


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


def test_recommended_update_command_pip():
    """Pip installs recommend pip install --upgrade."""
    from hermes_cli.config import recommended_update_command_for_method
    cmd = recommended_update_command_for_method("pip")
    assert "pip install" in cmd or "uv pip install" in cmd
    assert "--upgrade" in cmd
    assert "hermes-agent" in cmd


def test_stamp_file_takes_precedence(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".install_method").write_text("docker\n")
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=tmp_path) == "docker"


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
    install that historically stamped 'git'/'pip' into $HERMES_HOME keeps
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
        stamp_install_method("pip", project_root=code)
    assert (code / ".install_method").read_text().strip() == "pip"
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


def test_container_pip_install_without_stamp_is_pip(tmp_path):
    """Container + no .git + no stamp -> pip, not docker (issue #34397)."""
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path), \
         patch("hermes_constants.is_container", return_value=True):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=tmp_path) == "pip"


def test_recommended_update_command_docker():
    from hermes_cli.config import recommended_update_command_for_method
    assert "docker pull" in recommended_update_command_for_method("docker")


def test_banner_warns_on_pip_install(tmp_path):
    """The welcome banner surfaces a warning when the install method is pip."""
    import io
    from rich.console import Console
    from hermes_cli import banner

    hh = tmp_path / ".hermes"
    hh.mkdir()
    (hh / ".install_method").write_text("pip\n")

    with patch("hermes_cli.config.get_hermes_home", return_value=hh), \
         patch("hermes_constants.get_hermes_home", return_value=hh):
        buf = io.StringIO()
        # Wide console so the warning isn't wrapped across lines in the panel.
        console = Console(file=buf, width=400, force_terminal=False, color_system=None)
        banner.build_welcome_banner(
            console, model="m", cwd="/tmp",
            tools=[{"function": {"name": "terminal"}}],
            enabled_toolsets=["terminal"],
        )
        out = buf.getvalue()

    assert "officially" in out
    assert "instability" in out


def test_banner_no_pip_warning_on_git_install(tmp_path):
    """Git installs must not show the pip-install warning."""
    import io
    from rich.console import Console
    from hermes_cli import banner

    hh = tmp_path / ".hermes"
    hh.mkdir()
    (hh / ".install_method").write_text("git\n")

    with patch("hermes_cli.config.get_hermes_home", return_value=hh), \
         patch("hermes_constants.get_hermes_home", return_value=hh):
        buf = io.StringIO()
        console = Console(file=buf, width=400, force_terminal=False, color_system=None)
        banner.build_welcome_banner(
            console, model="m", cwd="/tmp",
            tools=[{"function": {"name": "terminal"}}],
            enabled_toolsets=["terminal"],
        )
        out = buf.getvalue()

    assert "officially" not in out
