"""The worker client's environment is an allowlist of host plumbing (``kanban_worker_env``)."""

from hermes_cli.kanban_worker_env import worker_client_env


def test_keeps_host_plumbing_and_hermes_location(tmp_path):
    env = {"PATH": "/bin", "HOME": "/home/u", "LC_ALL": "C.UTF-8", "HTTPS_PROXY": "http://p",
           "https_proxy": "http://p", "HERMES_HOME": "/h", "XDG_RUNTIME_DIR": "/run/user/1",
           "XDG_STATE_HOME": "/s", "HERMES_GATEWAY_LOCK_DIR": "/locks", "HERMES_DISABLE_LAZY_INSTALLS": "1"}
    assert worker_client_env(env, tmp_path / "assignee") == env


def test_drops_everything_else(tmp_path):
    env = {"PATH": "/bin", "SERVICE_TOKEN": "t", "GITHUB_TOKEN": "g", "HERMES_MODEL": "m",
           "TERMINAL_ENV": "docker", "HERMES_TUI": "1", "HERMES_SESSION_PLATFORM": "telegram",
           "MY_PASSTHROUGH_VAR": "v", "XDG_ACTIVATION_TOKEN": "a", "XDG_SERVICE_TOKEN": "x"}
    assert worker_client_env(env, tmp_path / "assignee") == {"PATH": "/bin"}
