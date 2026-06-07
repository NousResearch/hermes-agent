"""Worker arg builder: tokenless api_server, never --replace."""

from hermes_cli.gateway import _worker_run_args_for_profile


def test_argv_has_profile_run_no_replace():
    argv, _ = _worker_run_args_for_profile("research", 51234, "k" * 64)
    assert "--profile" in argv and argv[argv.index("--profile") + 1] == "research"
    assert argv[-2:] == ["gateway", "run"]
    assert "--replace" not in argv


def test_env_pins_loopback_api_server():
    _, env = _worker_run_args_for_profile("research", 51234, "secret-key")
    assert env["HERMES_GATEWAY_ONLY_PLATFORMS"] == "api_server"
    assert env["API_SERVER_ENABLED"] == "true"
    assert env["API_SERVER_HOST"] == "127.0.0.1"
    assert env["API_SERVER_PORT"] == "51234"
    assert env["API_SERVER_KEY"] == "secret-key"
