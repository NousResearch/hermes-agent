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


def test_git_dir_takes_precedence_over_managed(tmp_path):
    """When .git exists AND HERMES_MANAGED is set, .git wins (strongest signal)."""
    (tmp_path / ".git").mkdir()
    with patch("hermes_cli.config.get_managed_system", return_value="NixOS"), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        from hermes_cli.config import detect_install_method
        method = detect_install_method(project_root=tmp_path)
        assert method == "git"


def test_managed_detected_when_no_git_dir(tmp_path):
    """When HERMES_MANAGED is set and no .git, detect as managed."""
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


def test_stamp_file_takes_precedence_when_no_git(tmp_path):
    """Stamp file is honored when no .git directory exists."""
    (tmp_path / ".install_method").write_text("docker\n")
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=tmp_path) == "docker"


def test_git_dir_takes_precedence_over_stamp_file(tmp_path):
    """When .git exists AND stamp file exists, .git wins."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".install_method").write_text("docker\n")
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=tmp_path) == "git"


def test_docker_detected_via_dockerenv(tmp_path):
    with patch("hermes_cli.config.get_managed_system", return_value=None), \
         patch("hermes_cli.config.get_hermes_home", return_value=tmp_path), \
         patch("hermes_constants.is_container", return_value=True):
        from hermes_cli.config import detect_install_method
        assert detect_install_method(project_root=tmp_path) == "docker"


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

    with patch("hermes_cli.config.detect_install_method", return_value="pip"), \
         patch("hermes_cli.config.get_hermes_home", return_value=hh), \
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

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch("hermes_cli.config.get_hermes_home", return_value=hh), \
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
