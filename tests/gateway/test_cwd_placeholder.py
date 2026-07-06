"""Unit tests for gateway.cwd_placeholder.resolve_placeholder_terminal_cwd."""

from gateway.cwd_placeholder import resolve_placeholder_terminal_cwd


class TestResolvePlaceholderTerminalCwd:
    def test_local_placeholder_uses_messaging_cwd(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd=".",
            terminal_backend="local",
            messaging_cwd="/home/user/project",
            docker_mount_cwd_to_workspace=False,
            home_fallback="/home/user",
        ) == "/home/user/project"

    def test_local_placeholder_falls_back_to_home(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd="auto",
            terminal_backend="local",
            messaging_cwd=None,
            docker_mount_cwd_to_workspace=False,
            home_fallback="/home/user",
        ) == "/home/user"

    def test_docker_placeholder_mount_off_unset(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd=".",
            terminal_backend="docker",
            messaging_cwd="/home/user",
            docker_mount_cwd_to_workspace=False,
            home_fallback="/home/user",
        ) is None

    def test_docker_placeholder_mount_on_preserves_host_path(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd=".",
            terminal_backend="docker",
            messaging_cwd="/host/project",
            docker_mount_cwd_to_workspace=True,
            home_fallback="/home/user",
        ) == "/host/project"

    def test_docker_placeholder_mount_on_without_messaging_cwd_unset(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd=".",
            terminal_backend="docker",
            messaging_cwd=None,
            docker_mount_cwd_to_workspace=True,
            home_fallback="/home/user",
        ) is None

    def test_ssh_placeholder_unset(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd="cwd",
            terminal_backend="ssh",
            messaging_cwd="/home/user",
            docker_mount_cwd_to_workspace=False,
            home_fallback="/home/user",
        ) is None

    def test_explicit_configured_cwd_passthrough(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd="/explicit/path",
            terminal_backend="docker",
            messaging_cwd="/home/user",
            docker_mount_cwd_to_workspace=False,
            home_fallback="/home/user",
        ) == "/explicit/path"
