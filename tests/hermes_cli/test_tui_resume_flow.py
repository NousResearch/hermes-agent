from argparse import Namespace
from pathlib import Path
import sys
import types

import pytest


def _args(**overrides):
    base = {
        "continue_last": None,
        "model": None,
        "provider": None,
        "resume": None,
        "tui": True,
        "tui_dev": False,
    }
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    return mod


def test_cmd_chat_tui_continue_uses_latest_tui_session(monkeypatch, main_mod):
    calls = []
    captured = {}

    def fake_resolve_last(source="cli"):
        calls.append(source)
        return "20260408_235959_a1b2c3" if source == "tui" else None

    def fake_launch(resume_session_id=None, tui_dev=False, model=None, provider=None):
        captured["resume"] = resume_session_id
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_resolve_last_session", fake_resolve_last)
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda val: val)
    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(continue_last=True))

    assert calls == ["tui"]
    assert captured["resume"] == "20260408_235959_a1b2c3"


def test_cmd_chat_tui_continue_falls_back_to_latest_cli_session(monkeypatch, main_mod):
    calls = []
    captured = {}

    def fake_resolve_last(source="cli"):
        calls.append(source)
        if source == "tui":
            return None
        if source == "cli":
            return "20260408_235959_d4e5f6"
        return None

    def fake_launch(resume_session_id=None, tui_dev=False, model=None, provider=None):
        captured["resume"] = resume_session_id
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_resolve_last_session", fake_resolve_last)
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda val: val)
    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(continue_last=True))

    assert calls == ["tui", "cli"]
    assert captured["resume"] == "20260408_235959_d4e5f6"


def test_cmd_chat_tui_resume_resolves_title_before_launch(monkeypatch, main_mod):
    captured = {}

    def fake_launch(resume_session_id=None, tui_dev=False, model=None, provider=None):
        captured["resume"] = resume_session_id
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda val: "20260409_000000_aa11bb")
    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(resume="my t0p session"))

    assert captured["resume"] == "20260409_000000_aa11bb"


def test_cmd_chat_tui_passes_model_and_provider(monkeypatch, main_mod):
    captured = {}

    def fake_launch(resume_session_id=None, tui_dev=False, model=None, provider=None):
        captured.update(
            {
                "model": model,
                "provider": provider,
                "resume": resume_session_id,
                "tui_dev": tui_dev,
            }
        )
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(
            _args(model="anthropic/claude-sonnet-4.6", provider="anthropic")
        )

    assert captured == {
        "model": "anthropic/claude-sonnet-4.6",
        "provider": "anthropic",
        "resume": None,
        "tui_dev": False,
    }


def test_launch_tui_exports_model_and_provider(monkeypatch, main_mod):
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
        main_mod._launch_tui(model="nous/hermes-test", provider="nous")

    env = captured["env"]
    assert env["HERMES_MODEL"] == "nous/hermes-test"
    assert env["HERMES_INFERENCE_MODEL"] == "nous/hermes-test"
    assert env["HERMES_TUI_PROVIDER"] == "nous"
    assert env["HERMES_INFERENCE_PROVIDER"] == "nous"
    active_path = Path(env["HERMES_TUI_ACTIVE_SESSION_FILE"])
    assert active_path.name.startswith("hermes-tui-active-session-")
    assert active_path.suffix == ".json"
    assert active_path_during_call == active_path
    assert not active_path.exists()
    assert env["NODE_ENV"] == "production"


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

    monkeypatch.setitem(sys.modules, "hermes_state", types.SimpleNamespace(SessionDB=lambda: _FakeDB()))

    main_mod._print_tui_exit_summary("20260409_000001_abc123")
    out = capsys.readouterr().out

    assert "Resume this session with:" in out
    assert "hermes --tui --resume 20260409_000001_abc123" in out
    assert 'hermes --tui -c "demo title"' in out
    assert "Tokens:         21 (in 10, out 6, cache 4, reasoning 1)" in out


def test_print_tui_exit_summary_prefers_actual_active_session_file(monkeypatch, capsys, tmp_path):
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
    monkeypatch.setitem(sys.modules, "hermes_state", types.SimpleNamespace(SessionDB=lambda: _FakeDB()))

    main_mod._print_tui_exit_summary("startup_resume", str(active))
    out = capsys.readouterr().out

    assert seen == ["actual_session"]
    assert "hermes --tui --resume actual_session" in out
    assert "startup_resume" not in out
