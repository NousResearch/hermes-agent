"""Tests for `--resume latest` and `--in DIR` launch sugar.

`hermes --tui --resume latest --in ./dir` (and the classic-CLI equivalents)
resolve "latest" through the same workspace-scoped MRU lookup as `-c`, with
`--in` re-homing the process before any session resolution happens.
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from hermes_constants import HERMES_EXPLICIT_CWD_PIN, HERMES_EXPLICIT_CWD_PIN_VALUE


def _args(**overrides):
    base = {
        "cli": False,
        "continue_last": None,
        "in_dir": None,
        "model": None,
        "no_restore_cwd": False,
        "provider": None,
        "query": None,
        "resume": None,
        "safe_mode": False,
        "toolsets": None,
        "tui": True,
        "tui_dev": False,
        "worktree": False,
    }
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(mod, "_sync_bundled_skills_for_startup", lambda: False)
    monkeypatch.setattr(mod, "_pin_kanban_board_env", lambda: None)
    return mod


@pytest.fixture
def launched(main_mod, monkeypatch):
    """Capture the _launch_tui call instead of exec'ing Node."""
    captured = {}

    def fake_launch(resume_session_id=None, **kwargs):
        captured["resume"] = resume_session_id
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)
    return captured


# ---------------------------------------------------------------------------
# argparse surface
# ---------------------------------------------------------------------------


def test_top_level_parser_accepts_in_and_resume_latest():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(["--tui", "--resume", "latest", "--in", "./dir"])
    assert args.tui is True
    assert args.resume == "latest"
    assert args.in_dir == "./dir"


def test_chat_subparser_accepts_in_flag():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(["chat", "--in", "/tmp", "--resume", "latest"])
    assert args.in_dir == "/tmp"
    assert args.resume == "latest"


def test_top_level_in_value_not_mistaken_for_subcommand(monkeypatch):
    # `hermes --in chat` — "chat" is the flag's value, not the subcommand.
    import sys

    import hermes_cli.main as mod

    monkeypatch.setattr(sys, "argv", ["hermes", "--in", "chat", "--resume", "latest"])
    assert mod._first_positional_argv() is None


# ---------------------------------------------------------------------------
# --resume latest resolution
# ---------------------------------------------------------------------------


def test_resume_latest_resolves_to_mru_session(main_mod, launched, monkeypatch):
    monkeypatch.setattr(
        main_mod, "_resolve_last_session", lambda source="cli": "20260807_120000_abc123"
    )
    # Keyword must NOT fall through to title resolution.
    monkeypatch.setattr(
        main_mod,
        "_resolve_session_by_name_or_id",
        lambda val: val if val != "latest" else pytest.fail("'latest' hit title resolution"),
    )

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(_args(resume="latest"))
    assert exc.value.code == 0
    assert launched["resume"] == "20260807_120000_abc123"


def test_resume_latest_tui_falls_back_to_cli_source(main_mod, launched, monkeypatch):
    calls = []

    def fake_resolve(source="cli"):
        calls.append(source)
        return "cli_session_1" if source == "cli" else None

    monkeypatch.setattr(main_mod, "_resolve_last_session", fake_resolve)
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda v: v)

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(_args(resume="latest"))
    assert exc.value.code == 0
    assert calls == ["tui", "cli"]
    assert launched["resume"] == "cli_session_1"


def test_resume_latest_is_case_insensitive(main_mod, launched, monkeypatch):
    monkeypatch.setattr(main_mod, "_resolve_last_session", lambda source="cli": "sess_1")
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda v: v)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(resume="Latest"))
    assert launched["resume"] == "sess_1"


def test_resume_latest_no_sessions_exits_with_error(main_mod, monkeypatch, capsys):
    monkeypatch.setattr(main_mod, "_resolve_last_session", lambda source="cli": None)

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(_args(resume="latest"))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "No previous TUI session found" in out


def test_resume_real_id_untouched_by_latest_keyword(main_mod, launched, monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "_resolve_last_session",
        lambda source="cli": pytest.fail("MRU lookup must not run for explicit IDs"),
    )
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda v: v)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(resume="20260807_120000_abc123"))
    assert launched["resume"] == "20260807_120000_abc123"


# ---------------------------------------------------------------------------
# --in DIR
# ---------------------------------------------------------------------------


def test_in_dir_chdirs_before_session_resolution(main_mod, launched, monkeypatch, tmp_path):
    import os

    target = tmp_path / "projdir"
    target.mkdir()
    start = os.getcwd()
    seen_cwd = {}

    def fake_resolve(source="cli"):
        seen_cwd["at_resolve"] = os.getcwd()
        return "sess_scoped"

    monkeypatch.setattr(main_mod, "_resolve_last_session", fake_resolve)
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda v: v)

    try:
        with pytest.raises(SystemExit):
            main_mod.cmd_chat(_args(resume="latest", in_dir=str(target)))
    finally:
        os.chdir(start)

    assert seen_cwd["at_resolve"] == str(target.resolve())
    assert launched["resume"] == "sess_scoped"


def test_in_dir_sets_no_restore_cwd(main_mod, launched, monkeypatch, tmp_path):
    import os

    target = tmp_path / "pin-here"
    target.mkdir()
    start = os.getcwd()

    args = _args(resume=None, in_dir=str(target))
    try:
        with pytest.raises(SystemExit):
            main_mod.cmd_chat(args)
    finally:
        os.chdir(start)

    assert args.no_restore_cwd is True


def test_in_dir_missing_directory_exits(main_mod, monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(_args(in_dir=str(tmp_path / "nope")))
    assert exc.value.code == 1
    assert "--in directory not found" in capsys.readouterr().out


def test_in_dir_expands_user_home(main_mod, launched, monkeypatch, tmp_path):
    import os

    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    start = os.getcwd()

    try:
        with pytest.raises(SystemExit):
            main_mod.cmd_chat(_args(in_dir="~/proj"))
        assert os.getcwd() == str((home / "proj").resolve())
    finally:
        os.chdir(start)


@pytest.mark.parametrize(
    ("backend", "configured_cwd", "extra_yaml"),
    [
        ("local", None, ""),
        ("ssh", "~/remote-project", ""),
        ("docker", "/workspace", ""),
        ("docker", "/root", "  container_persistent: false\n"),
    ],
)
def test_oneshot_in_dir_respects_terminal_backend(
    main_mod, monkeypatch, tmp_path, backend, configured_cwd, extra_yaml
):
    import os

    from hermes_cli import oneshot
    from tools import terminal_tool

    target = tmp_path / "requested"
    stale = tmp_path / "stale"
    target.mkdir()
    stale.mkdir()
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    configured_cwd = configured_cwd or str(stale)
    (hermes_home / "config.yaml").write_text(
        f"terminal:\n  backend: {backend}\n  cwd: {configured_cwd}\n{extra_yaml}",
        encoding="utf-8",
    )
    start = os.getcwd()
    seen = {}

    class FakeAgent:
        def __init__(self, **_kwargs):
            seen["cwd"] = os.getcwd()
            import gateway.run  # noqa: F401  -- a real lazy import re-bridges terminal config
            seen["terminal_cwd"] = os.environ.get("TERMINAL_CWD")
            seen["tool_cwd"] = terminal_tool._get_env_config()["cwd"]
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, _prompt, task_id=None):
            seen["task_id"] = task_id
            return {"final_response": "ok", "failed": False, "partial": False}

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    import hermes_cli.mcp_startup as mcp_startup
    import hermes_cli.runtime_provider as runtime_provider
    import hermes_cli.tools_config as tools_config
    import run_agent

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_CWD", str(stale))
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)
    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda *_args: set())
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "key",
            "base_url": "https://example.test/v1",
            "provider": "test",
            "requested_provider": "test",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(
        mcp_startup,
        "ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )

    try:
        response, result = oneshot._run_agent("work here", in_dir=str(target))
    finally:
        terminal_tool.clear_session_cwd("default")
        os.chdir(start)

    assert response == "ok"
    assert result["failed"] is False
    assert seen["cwd"] == str(target.resolve())
    assert seen["task_id"].startswith("oneshot:")
    assert terminal_tool.get_session_cwd(seen["task_id"]) is None
    if backend == "local":
        assert seen["terminal_cwd"] == str(target.resolve())
        assert seen["tool_cwd"] == str(target.resolve())
    else:
        assert seen["terminal_cwd"] == configured_cwd
        assert seen["tool_cwd"] == configured_cwd
    assert HERMES_EXPLICIT_CWD_PIN not in os.environ


def _stub_oneshot_hard_exit(main_mod, monkeypatch):
    monkeypatch.setattr(main_mod, "_cleanup_oneshot_runtime", lambda: None)

    def fake_exit(rc):
        raise SystemExit(0 if rc is None else rc)

    monkeypatch.setattr(main_mod, "_exit_after_oneshot", fake_exit)


def test_oneshot_explicit_cwd_resolves_backend_before_chdir(monkeypatch, tmp_path):
    import os

    from hermes_cli.oneshot import _oneshot_explicit_cwd
    from tools import terminal_tool

    target = tmp_path / "requested"
    target.mkdir()
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv(HERMES_EXPLICIT_CWD_PIN, raising=False)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)

    start = os.getcwd()
    seen = {}
    real_get = terminal_tool._get_env_config

    def spy_get():
        seen["cwd_at_resolve"] = os.getcwd()
        return real_get()

    monkeypatch.setattr(terminal_tool, "_get_env_config", spy_get)

    try:
        with _oneshot_explicit_cwd(str(target), "oneshot:test"):
            seen["cwd_inside"] = os.getcwd()
            assert os.environ.get(HERMES_EXPLICIT_CWD_PIN) == HERMES_EXPLICIT_CWD_PIN_VALUE
    finally:
        os.chdir(start)

    assert seen["cwd_at_resolve"] == start
    assert seen["cwd_inside"] == str(target.resolve())
    assert HERMES_EXPLICIT_CWD_PIN not in os.environ


def test_oneshot_explicit_cwd_restores_process_cwd_and_terminal_cwd(monkeypatch, tmp_path):
    import os

    from hermes_cli.oneshot import _oneshot_explicit_cwd
    from tools import terminal_tool

    target = tmp_path / "requested"
    previous_terminal_cwd = tmp_path / "previous"
    target.mkdir()
    previous_terminal_cwd.mkdir()
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_CWD", str(previous_terminal_cwd))
    monkeypatch.delenv(HERMES_EXPLICIT_CWD_PIN, raising=False)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)
    start = os.getcwd()

    try:
        with _oneshot_explicit_cwd(str(target), "oneshot:test"):
            assert os.getcwd() == str(target.resolve())
            assert os.environ["TERMINAL_CWD"] == str(target.resolve())

        assert os.getcwd() == start
        assert os.environ["TERMINAL_CWD"] == str(previous_terminal_cwd)
    finally:
        os.chdir(start)


def test_oneshot_explicit_cwd_recovers_from_deleted_launch_cwd(monkeypatch, tmp_path):
    import os

    from hermes_cli.oneshot import _oneshot_explicit_cwd
    from tools import terminal_tool

    target = tmp_path / "requested"
    fallback = tmp_path / "fallback"
    launch = tmp_path / "deleted-launch"
    target.mkdir()
    fallback.mkdir()
    launch.mkdir()
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_CWD", str(fallback))
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)
    start = os.getcwd()

    try:
        os.chdir(launch)
        launch.rmdir()

        with _oneshot_explicit_cwd(str(target), "oneshot:test"):
            assert os.getcwd() == str(target.resolve())

        assert os.getcwd() == str(fallback)
    finally:
        os.chdir(start)


def test_oneshot_explicit_cwd_preserves_config_error(monkeypatch, tmp_path):
    import os

    from hermes_cli.oneshot import _oneshot_explicit_cwd
    from tools import terminal_tool

    target = tmp_path / "requested"
    target.mkdir()
    start = os.getcwd()
    monkeypatch.delenv(HERMES_EXPLICIT_CWD_PIN, raising=False)
    def boom():
        raise ValueError("bad terminal config")

    monkeypatch.setattr(terminal_tool, "_get_env_config", boom)

    try:
        with pytest.raises(ValueError, match="bad terminal config"):
            with _oneshot_explicit_cwd(str(target), "oneshot:test"):
                pytest.fail("must not enter after a config error")
    finally:
        os.chdir(start)

    assert os.getcwd() == start
    assert HERMES_EXPLICIT_CWD_PIN not in os.environ


def test_oneshot_explicit_cwd_restores_pin_if_record_raises(monkeypatch, tmp_path):
    import os

    from hermes_cli.oneshot import _oneshot_explicit_cwd
    from tools import terminal_tool

    target = tmp_path / "requested"
    target.mkdir()
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv(HERMES_EXPLICIT_CWD_PIN, raising=False)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)

    def boom(_task_id, _cwd):
        raise RuntimeError("record boom")

    monkeypatch.setattr(terminal_tool, "record_session_cwd", boom)
    start = os.getcwd()

    try:
        with pytest.raises(RuntimeError, match="record boom"):
            with _oneshot_explicit_cwd(str(target), "oneshot:test"):
                pytest.fail("must not enter after a record failure")
    finally:
        os.chdir(start)

    assert HERMES_EXPLICIT_CWD_PIN not in os.environ


def test_oneshot_explicit_cwd_restores_pin_if_cleanup_raises(monkeypatch, tmp_path):
    import os

    from hermes_cli.oneshot import _oneshot_explicit_cwd
    from tools import terminal_tool

    target = tmp_path / "requested"
    target.mkdir()
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv(HERMES_EXPLICIT_CWD_PIN, raising=False)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)

    def boom(_task_id):
        raise RuntimeError("cleanup boom")

    monkeypatch.setattr(terminal_tool, "clear_task_env_overrides", boom)
    start = os.getcwd()

    try:
        with _oneshot_explicit_cwd(str(target), "oneshot:test"):
            assert os.environ.get(HERMES_EXPLICIT_CWD_PIN) == HERMES_EXPLICIT_CWD_PIN_VALUE
    finally:
        os.chdir(start)

    assert HERMES_EXPLICIT_CWD_PIN not in os.environ


def test_oneshot_explicit_cwd_preserves_body_error_if_cleanup_raises(monkeypatch, tmp_path):
    import os

    from hermes_cli.oneshot import _oneshot_explicit_cwd
    from tools import terminal_tool

    target = tmp_path / "requested"
    target.mkdir()
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv(HERMES_EXPLICIT_CWD_PIN, raising=False)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)

    def boom(_task_id):
        raise RuntimeError("cleanup boom")

    monkeypatch.setattr(terminal_tool, "clear_task_env_overrides", boom)
    start = os.getcwd()

    try:
        with pytest.raises(RuntimeError, match="agent boom"):
            with _oneshot_explicit_cwd(str(target), "oneshot:test"):
                raise RuntimeError("agent boom")
    finally:
        os.chdir(start)

    assert HERMES_EXPLICIT_CWD_PIN not in os.environ


def test_oneshot_dispatch_validates_in_dir_without_chdir(main_mod, monkeypatch, tmp_path):
    import os

    target = tmp_path / "proj"
    target.mkdir()
    start = os.getcwd()
    seen = {}

    def fake_run_oneshot(*_args, **kwargs):
        seen["cwd"] = os.getcwd()
        seen["in_dir"] = kwargs.get("in_dir")
        return 0

    monkeypatch.setattr("hermes_cli.oneshot.run_oneshot", fake_run_oneshot)
    _stub_oneshot_hard_exit(main_mod, monkeypatch)

    try:
        with pytest.raises(SystemExit) as exc:
            main_mod._run_and_exit_oneshot("hi", in_dir=str(target))
    finally:
        os.chdir(start)

    assert exc.value.code == 0
    assert seen["cwd"] == start
    assert seen["in_dir"] == str(target.resolve())


def test_oneshot_dispatch_missing_in_dir_exits(main_mod, monkeypatch, tmp_path, capsys):
    _stub_oneshot_hard_exit(main_mod, monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main_mod._run_and_exit_oneshot("hi", in_dir=str(tmp_path / "nope"))

    assert exc.value.code == 1
    assert "--in directory not found" in capsys.readouterr().out


def test_oneshot_dispatch_expands_user_home(main_mod, monkeypatch, tmp_path):
    import os

    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    start = os.getcwd()
    seen = {}

    def fake_run_oneshot(*_args, **kwargs):
        seen["cwd"] = os.getcwd()
        seen["in_dir"] = kwargs.get("in_dir")
        return 0

    monkeypatch.setattr("hermes_cli.oneshot.run_oneshot", fake_run_oneshot)
    _stub_oneshot_hard_exit(main_mod, monkeypatch)

    try:
        with pytest.raises(SystemExit) as exc:
            main_mod._run_and_exit_oneshot("hi", in_dir="~/proj")
    finally:
        os.chdir(start)

    assert exc.value.code == 0
    assert seen["cwd"] == start
    assert seen["in_dir"] == str((home / "proj").resolve())


def test_resolve_in_dir_routes_through_msys_translation(main_mod, monkeypatch, tmp_path):
    seen = {}
    target = tmp_path / "proj"
    target.mkdir()

    def fake_msys(path):
        seen["path"] = path
        return path

    monkeypatch.setattr(
        "tools.environments.local._msys_to_windows_path", fake_msys
    )

    resolved = main_mod._resolve_in_dir(str(target))

    assert seen["path"] == str(target)
    assert resolved == str(target.resolve())
