from argparse import Namespace
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types

import pytest


def _args(**overrides):
    base = {
        "continue_last": None,
        "model": None,
        "provider": None,
        "resume": None,
        "toolsets": None,
        "tui": True,
        "tui_dev": False,
    }
    base.update(overrides)
    return Namespace(**base)


def _raise_exit(rc):
    raise SystemExit(rc)


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    # Reset the idempotency guard so each test starts fresh.
    monkeypatch.setattr(mod, "_oneshot_cleanup_done", False)
    return mod


def test_cmd_chat_tui_continue_uses_latest_tui_session(monkeypatch, main_mod):
    calls = []
    captured = {}

    def fake_resolve_last(source="cli"):
        calls.append(source)
        return "20260408_235959_a1b2c3" if source == "tui" else None

    def fake_launch(
        resume_session_id=None,
        tui_dev=False,
        model=None,
        provider=None,
        toolsets=None,
        **kwargs,
    ):
        captured["resume"] = resume_session_id
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_resolve_last_session", fake_resolve_last)
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda val: val)
    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(continue_last=True))

    assert calls == ["tui"]
    assert captured["resume"] == "20260408_235959_a1b2c3"


def test_cmd_chat_tui_forwards_chat_flags(monkeypatch, main_mod):
    captured = {}

    def fake_launch(resume_session_id=None, **kwargs):
        captured["resume_session_id"] = resume_session_id
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(
            _args(
                skills=["foo,bar"],
                verbose=True,
                quiet=True,
                query="hello",
                image="/tmp/cat.png",
                worktree=True,
                checkpoints=True,
                pass_session_id=True,
                max_turns=7,
                accept_hooks=True,
            )
        )

    assert captured["skills"] == ["foo,bar"]
    assert captured["verbose"] is True
    assert captured["quiet"] is True
    assert captured["query"] == "hello"
    assert captured["image"] == "/tmp/cat.png"
    assert captured["worktree"] is True
    assert captured["checkpoints"] is True
    assert captured["pass_session_id"] is True
    assert captured["max_turns"] == 7
    assert captured["accept_hooks"] is True


def test_main_top_level_tui_accepts_toolsets(monkeypatch, main_mod):
    captured = {}

    import hermes_cli.config as config_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "--tui", "--toolsets", "web,terminal"])
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=lambda: None),
    )
    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(
            register_from_config=lambda _cfg, accept_hooks=False: None
        ),
    )
    monkeypatch.setattr(
        main_mod,
        "cmd_chat",
        lambda args: captured.update({"toolsets": args.toolsets, "tui": args.tui}),
    )

    main_mod.main()

    assert captured == {"toolsets": "web,terminal", "tui": True}


def test_termux_fast_tui_launch_uses_light_parser(monkeypatch, main_mod):
    captured = {}

    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(
        sys, "argv", ["hermes", "--tui", "--toolsets", "web,terminal"]
    )
    monkeypatch.setattr(
        main_mod,
        "cmd_chat",
        lambda args: captured.update({"toolsets": args.toolsets, "tui": args.tui}),
    )

    assert main_mod._try_termux_fast_tui_launch() is True
    assert captured == {"toolsets": "web,terminal", "tui": True}


def test_fast_tui_launch_is_termux_only(monkeypatch, main_mod):
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setattr(sys, "argv", ["hermes", "--tui"])

    assert main_mod._try_termux_fast_tui_launch() is False


def test_termux_ultrafast_version_runs_before_heavy_startup(
    monkeypatch, capsys, main_mod
):
    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.delenv("HERMES_TERMUX_DISABLE_FAST_CLI", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", "--version"])

    assert main_mod._try_termux_ultrafast_version() is True

    out = capsys.readouterr().out
    assert "Hermes Agent v" in out
    assert "Install directory:" in out
    assert "Python:" in out
    assert "OpenAI SDK:" in out


def test_read_openai_version_fast(monkeypatch, tmp_path, main_mod):
    package_dir = tmp_path / "openai"
    package_dir.mkdir()
    (package_dir / "_version.py").write_text(
        '__version__ = "9.8.7"  # x-release-please-version\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path)])

    assert main_mod._read_openai_version_fast() == "9.8.7"


def test_termux_skips_bundled_skill_sync_when_stamp_fresh(monkeypatch, tmp_path, main_mod):
    calls = []

    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(main_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(main_mod, "_termux_bundled_skills_fingerprint", lambda: "fp1")
    main_mod._mark_termux_bundled_skills_synced()
    monkeypatch.setitem(
        sys.modules,
        "tools.skills_sync",
        types.SimpleNamespace(sync_skills=lambda quiet: calls.append(quiet)),
    )

    assert main_mod._sync_bundled_skills_for_startup() is False
    assert calls == []


def test_read_git_revision_fingerprint_resolves_packed_refs(tmp_path, main_mod):
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    packed_sha = "1234567890abcdef1234567890abcdef12345678"
    (git_dir / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{packed_sha} refs/heads/main\n"
        "abcdef0000000000000000000000000000000000 refs/tags/v1.0\n"
        "^99999999aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
        encoding="utf-8",
    )

    fingerprint = main_mod._read_git_revision_fingerprint(repo)

    assert fingerprint == f"git:refs/heads/main:{packed_sha}"


def test_read_git_revision_fingerprint_packed_refs_in_worktree_common_dir(
    tmp_path, main_mod
):
    main_repo = tmp_path / "repo"
    common_git = main_repo / ".git"
    common_git.mkdir(parents=True)
    packed_sha = "fedcba9876543210fedcba9876543210fedcba98"
    (common_git / "packed-refs").write_text(
        f"{packed_sha} refs/heads/main\n",
        encoding="utf-8",
    )

    worktree = tmp_path / "wt"
    worktree.mkdir()
    wt_gitdir = common_git / "worktrees" / "wt"
    wt_gitdir.mkdir(parents=True)
    (wt_gitdir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")

    fingerprint = main_mod._read_git_revision_fingerprint(worktree)

    assert fingerprint == f"git:refs/heads/main:{packed_sha}"


def test_exit_after_oneshot_flushes_stdio_and_calls_os_exit(
    monkeypatch, main_mod
):
    flushed = []
    exits = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def flush(self):
            flushed.append(self.name)

    def fake_exit(rc):
        exits.append(rc)
        raise SystemExit(rc)

    monkeypatch.setattr(main_mod.sys, "stdout", FakeStream("stdout"))
    monkeypatch.setattr(main_mod.sys, "stderr", FakeStream("stderr"))
    monkeypatch.setattr(main_mod.os, "_exit", fake_exit)
    monkeypatch.setattr("logging.shutdown", lambda: None)

    with pytest.raises(SystemExit) as exc:
        main_mod._exit_after_oneshot(17)

    assert exc.value.code == 17
    assert exits == [17]
    assert flushed == ["stdout", "stderr"]


def test_run_and_exit_oneshot_routes_system_exit_to_hard_exit(monkeypatch, main_mod):
    exits = []

    def fake_run_oneshot(*_args, **_kwargs):
        raise SystemExit(2)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.oneshot",
        types.SimpleNamespace(run_oneshot=fake_run_oneshot),
    )
    monkeypatch.setattr(main_mod, "_cleanup_oneshot_runtime", lambda: None)
    monkeypatch.setattr(main_mod, "_exit_after_oneshot", lambda rc: exits.append(rc))

    main_mod._run_and_exit_oneshot("hello")

    assert exits == [2]


def test_run_and_exit_oneshot_hard_exits_on_unexpected_exception(
    monkeypatch, main_mod, capsys
):
    # ``run_oneshot`` is contracted to convert agent failures into an int and
    # only re-raise KeyboardInterrupt / SystemExit. If it ever malfunctions and
    # lets another exception escape, the one-shot path must still hard-exit
    # (rc 1) rather than fall through to interpreter teardown — the exact path
    # that SIGABRTs on AL2023.
    exits = []
    cleaned = []

    def fake_run_oneshot(*_args, **_kwargs):
        raise RuntimeError("run_oneshot itself blew up")

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.oneshot",
        types.SimpleNamespace(run_oneshot=fake_run_oneshot),
    )
    monkeypatch.setattr(
        main_mod, "_cleanup_oneshot_runtime", lambda: cleaned.append(True)
    )
    monkeypatch.setattr(main_mod, "_exit_after_oneshot", lambda rc: exits.append(rc))

    main_mod._run_and_exit_oneshot("hello")

    assert exits == [1]
    # Global cleanup still runs on the defensive path — resources must not leak
    # just because run_oneshot malfunctioned.
    assert cleaned == [True]
    # The failure is surfaced on stderr, never swallowed silently.
    assert "run_oneshot itself blew up" in capsys.readouterr().err


def test_oneshot_subprocess_exits_without_teardown_abort():
    program = textwrap.dedent(
        """
        import hermes_cli.oneshot as oneshot
        from hermes_cli.main import _exit_after_oneshot

        oneshot._run_agent = lambda *args, **kwargs: ("ok", {"final_response": "ok"})
        _exit_after_oneshot(oneshot.run_oneshot("hello"))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == b"ok\n"
    # Don't demand byte-empty stderr — an import-time warning from the heavy
    # CLI import chain shouldn't fail this. What matters is no crash traceback.
    assert b"Traceback" not in result.stderr


def test_exit_after_oneshot_bypasses_late_atexit_abort():
    program = textwrap.dedent(
        """
        import atexit
        import os
        import sys
        from hermes_cli.main import _exit_after_oneshot

        atexit.register(os.abort)
        sys.stdout.write("done\\n")
        _exit_after_oneshot(0)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == b"done\n"


def test_run_and_exit_oneshot_passes_through_nonzero_return(monkeypatch, main_mod):
    # A non-zero rc from run_oneshot (e.g. provider-without-model → 2, or the
    # empty-response guard → 1) must reach os._exit unchanged.
    exits = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.oneshot",
        types.SimpleNamespace(run_oneshot=lambda *a, **k: 2),
    )
    monkeypatch.setattr(main_mod, "_cleanup_oneshot_runtime", lambda: None)
    monkeypatch.setattr(main_mod, "_exit_after_oneshot", lambda rc: exits.append(rc))

    main_mod._run_and_exit_oneshot("hi")

    assert exits == [2]


def test_main_oneshot_path_bypasses_late_atexit_abort():
    # End-to-end through the real top-level ``main()`` ``-z`` path: a valid
    # response prints, then a late atexit handler that would abort is bypassed
    # by the hard exit, so the process reports success (#43055).
    program = textwrap.dedent(
        """
        import atexit
        import os
        import sys
        import types

        import hermes_cli.main as main_mod

        sys.argv = ["hermes", "-z", "hello"]
        main_mod._prepare_agent_startup = lambda args: None

        def _fake_run_oneshot(prompt, **kwargs):
            print("ok")
            return 0

        sys.modules["hermes_cli.oneshot"] = types.SimpleNamespace(
            run_oneshot=_fake_run_oneshot
        )
        atexit.register(os.abort)
        main_mod.main()
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == b"ok\n"
    assert b"Traceback" not in result.stderr


def _stub_plugin_discovery(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )


def test_oneshot_reraises_keyboard_interrupt(monkeypatch):
    _stub_plugin_discovery(monkeypatch)
    import hermes_cli.oneshot as oneshot_mod
    import pytest as _pytest

    def _interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(oneshot_mod, "_run_agent", _interrupt)

    with _pytest.raises(KeyboardInterrupt):
        oneshot_mod.run_oneshot("hello")


def test_oneshot_wires_session_db_for_recall(monkeypatch):
    """hermes -z bypasses HermesCLI, but recall still needs SessionDB."""
    from hermes_cli.oneshot import _run_agent

    captured = {}
    sentinel_db = object()

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt, **_kwargs):
            captured["prompt"] = prompt
            return {"final_response": "ok", "failed": False, "partial": False}

    class FakeSessionDB:
        def __new__(cls):
            return sentinel_db

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(sys.modules, "hermes_state", mod("hermes_state", SessionDB=FakeSessionDB))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        mod("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        mod("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: {"session_search"}),
    )

    text, result = _run_agent("recall this")
    assert text == "ok"
    assert not result.get("failed")
    assert captured["session_db"] is sentinel_db
    assert captured["enabled_toolsets"] == ["session_search"]
    assert captured["prompt"] == "recall this"


def test_launch_tui_exports_model_provider_and_toolsets(monkeypatch, main_mod):
    captured = {}
    active_path_during_call = None

    monkeypatch.setattr(
        main_mod,
        "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")),
    )

    def fake_call(argv, cwd=None, env=None):
        nonlocal active_path_during_call
        captured.update({"argv": argv, "cwd": cwd, "env": env})
        active_path_during_call = Path(env["HERMES_TUI_ACTIVE_SESSION_FILE"])
        assert active_path_during_call.exists()
        return 1

    monkeypatch.setattr(main_mod.subprocess, "call", fake_call)

    with pytest.raises(SystemExit):
        main_mod._launch_tui(
            model="nous/hermes-test", provider="nous", toolsets="web, terminal"
        )

    env = captured["env"]
    assert env["HERMES_MODEL"] == "nous/hermes-test"
    assert env["HERMES_INFERENCE_MODEL"] == "nous/hermes-test"
    assert env["HERMES_TUI_PROVIDER"] == "nous"
    assert env["HERMES_INFERENCE_PROVIDER"] == "nous"
    assert env["HERMES_TUI_TOOLSETS"] == "web,terminal"
    active_path = Path(env["HERMES_TUI_ACTIVE_SESSION_FILE"])
    assert active_path.name.startswith("hermes-tui-active-session-")
    assert active_path.suffix == ".json"
    assert active_path_during_call == active_path
    assert not active_path.exists()
    assert env["NODE_ENV"] == "production"


def test_launch_tui_exit_code_42_relaunches_update(monkeypatch, main_mod):
    from unittest.mock import patch

    monkeypatch.setattr(
        main_mod,
        "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")),
    )
    monkeypatch.setattr(main_mod.subprocess, "call", lambda *args, **kwargs: 42)

    with patch("hermes_cli.relaunch.relaunch") as mock_relaunch:
        with pytest.raises(SystemExit) as exc:
            main_mod._launch_tui()

    assert exc.value.code == 42
    mock_relaunch.assert_called_once_with(["update"], preserve_inherited=False)


def test_make_tui_argv_dev_prebuilds_hermes_ink(monkeypatch, main_mod, tmp_path):
    tui_dir = tmp_path / "ui-tui"
    tsx = tui_dir / "node_modules" / ".bin" / "tsx"
    ink_dir = tui_dir / "packages" / "hermes-ink"
    tsx.parent.mkdir(parents=True)
    ink_dir.mkdir(parents=True)
    tsx.write_text("#!/usr/bin/env node\n", encoding="utf-8")

    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _tui_dir: False)
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_mod.shutil, "which", lambda bin_name: f"/usr/bin/{bin_name}")

    calls = []

    def fake_run(cmd, cwd=None, **_kwargs):
        calls.append((cmd, cwd))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    argv, cwd = main_mod._make_tui_argv(tui_dir, tui_dev=True)

    assert argv == [str(tsx), "src/entry.tsx"]
    assert cwd == tui_dir
    assert calls == [(["/usr/bin/npm", "run", "build"], str(ink_dir))]


def test_print_tui_exit_summary_includes_resume_and_token_totals(monkeypatch, capsys):
    import hermes_cli.main as main_mod

    class _FakeDB:
        def get_session(self, session_id):
            assert session_id == "20260409_000001_abc123"
            return {
                "message_count": 2,
                "input_tokens": 10,
                "output_tokens": 6,
                "cache_read_tokens": 2,
                "cache_write_tokens": 2,
                "reasoning_tokens": 1,
            }

        def get_session_title(self, _session_id):
            return "demo title"

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules, "hermes_state", types.SimpleNamespace(SessionDB=lambda: _FakeDB())
    )

    main_mod._print_tui_exit_summary("20260409_000001_abc123")
    out = capsys.readouterr().out

    assert "Resume this session with:" in out
    assert "hermes --tui --resume 20260409_000001_abc123" in out
    assert 'hermes --tui -c "demo title"' in out
    assert "Tokens:         21 (in 10, out 6, cache 4, reasoning 1)" in out


def test_print_tui_exit_summary_prefers_actual_active_session_file(
    monkeypatch, capsys, tmp_path
):
    import hermes_cli.main as main_mod

    seen = []

    class _FakeDB:
        def get_session(self, session_id):
            seen.append(session_id)
            return {
                "message_count": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
            }

        def get_session_title(self, _session_id):
            return "actual"

        def close(self):
            return None

    active = tmp_path / "active.json"
    active.write_text('{"session_id":"actual_session"}', encoding="utf-8")
    monkeypatch.setitem(
        sys.modules, "hermes_state", types.SimpleNamespace(SessionDB=lambda: _FakeDB())
    )

    main_mod._print_tui_exit_summary("startup_resume", str(active))
    out = capsys.readouterr().out

    assert seen == ["actual_session"]
    assert "hermes --tui --resume actual_session" in out
    assert "startup_resume" not in out
