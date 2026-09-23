"""Code-flow grid sweep (NOUS-368, wave 2): pre-existing automatic diagnostics on the CLI/TUI/stream
rails that bypassed the warning boundary. Each cell: absent/false = legacy bytes, true = only the
diagnostic disappears; source logs, model-facing bookkeeping and requested results are untouched."""
from __future__ import annotations

import json
import types
from unittest.mock import MagicMock

import pytest

MODES = (None, False, True)


def _policy(tmp_path, monkeypatch, setting):
    home = tmp_path / f"home-{setting}"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    display = {} if setting is None else {"suppress_warning_notifications": setting}
    (home / "config.yaml").write_text(json.dumps({"display": display}))
    return home


def _visible(setting):
    return setting is not True


# --- agent/conversation_compression.py replay_compression_warning -> DiagnosticText ------------

def test_compression_replay_is_classified_diagnostic():
    from agent.conversation_compression import replay_compression_warning
    from gateway.warning_notifications import DiagnosticText, is_warning_status
    seen = []
    agent = types.SimpleNamespace(_compression_warning="ℹ Codex caps context at 272K, so auto-compaction was raised",
                                  status_callback=lambda kind, msg: seen.append((kind, msg)))
    replay_compression_warning(agent)
    assert len(seen) == 1 and seen[0][0] == "lifecycle"
    assert isinstance(seen[0][1], DiagnosticText) and str(seen[0][1]) == agent._compression_warning
    assert is_warning_status("lifecycle", seen[0][1])


@pytest.mark.parametrize("setting", MODES)
def test_compression_replay_reaches_gateway_status_sink_only_when_visible(tmp_path, monkeypatch, setting):
    """Composed: replay -> the gateway status callback's own gate (render_notification + is_warning_status)."""
    _policy(tmp_path, monkeypatch, setting)
    from agent.conversation_compression import replay_compression_warning
    from gateway.warning_notifications import is_warning_status, render_notification
    rendered = []

    def gateway_status_callback(kind, msg):  # mirrors gateway/run_turn_runner._status_callback_sync
        render_notification(lambda: rendered.append(str(msg)), platform="slack", diagnostic=is_warning_status(kind, msg))

    agent = types.SimpleNamespace(_compression_warning="⚠️ Session compressed 3 times — accuracy may degrade",
                                  status_callback=gateway_status_callback)
    replay_compression_warning(agent)
    assert (len(rendered) == 1) is _visible(setting)


# --- agent/chat_completion_helpers.py stream-drop / stall deltas ----------------------------------

def _stream_attempt_stub(enabled):
    from agent import chat_completion_helpers as cch
    fired = []
    agent = MagicMock()
    agent._warning_presentation_enabled = lambda: enabled
    agent._fire_stream_delta = lambda text: fired.append(text)
    return cch, agent, fired


@pytest.mark.parametrize("enabled", (True, False))
def test_stream_reconnect_marker_delta_honors_policy(enabled):
    """The '⚠ Connection dropped mid tool-call; reconnecting…' delta is presentation only; the
    tracking reset and retry bookkeeping run regardless."""
    import inspect
    from agent import chat_completion_helpers as cch
    src = inspect.getsource(cch)
    i = src.index("Connection dropped mid tool-call; reconnecting")
    window = src[i - 400:i]
    assert "_warning_presentation_enabled()" in window, "reconnect marker must be behind the presentation gate"
    j = src.index("_reset_stream_delivery_tracking", i)
    assert "if self.agent._warning_presentation_enabled()" not in src[i:j], "tracking reset must stay unconditional"


def test_stream_stall_warning_kept_in_result_but_delta_gated():
    import inspect
    from agent import chat_completion_helpers as cch
    src = inspect.getsource(cch)
    i = src.index("Stream stalled mid tool-call")
    seg = src[i:i + 700]
    assert "_partial_text = (_partial_text or \"\") + _warn" in seg, "model/result bookkeeping stays"
    assert "if self.agent._warning_presentation_enabled():" in seg
    assert seg.index("_partial_text = (_partial_text or \"\") + _warn") < seg.index("if self.agent._warning_presentation_enabled():")
    assert "logger.warning(" in seg


# --- agent/agent_init.py invalid config int + autoraise notice ---------------------------------

@pytest.mark.parametrize("setting", MODES)
def test_invalid_config_int_print_honors_policy_log_always(tmp_path, monkeypatch, setting, capsys, caplog):
    _policy(tmp_path, monkeypatch, setting)
    import logging
    from agent import agent_init
    agent = types.SimpleNamespace(platform="cli", _notification_config=None)
    with caplog.at_level(logging.WARNING):
        agent_init._warn_invalid_config_int("model.context_length in config.yaml", "256K",
                                            "must be a plain integer", "auto-detection", agent=agent)
    err = capsys.readouterr().err
    assert ("⚠ Invalid model.context_length" in err) is _visible(setting)
    assert any("Invalid model.context_length" in rec.getMessage() for rec in caplog.records)


def test_autoraise_notice_uses_diagnostic_safe_print():
    import inspect
    from agent import agent_init
    src = inspect.getsource(agent_init)
    i = src.index("if _autoraise_notice:\n            agent._safe_print(_autoraise_notice, diagnostic=True)")
    assert i > 0


# --- agent/turn_tool_validation.py auto-repair notice ------------------------------------------

def test_tool_name_repair_notice_is_diagnostic_vprint():
    from agent import turn_tool_validation as ttv
    printed = []
    agent = MagicMock()
    agent.log_prefix = ""
    agent._repair_tool_call = lambda name: "read_file"
    agent._vprint = lambda *a, **kw: printed.append((a, kw))
    tc = types.SimpleNamespace(function=types.SimpleNamespace(name="reed_file", arguments="{}"))
    # Only the repair loop is under test; call the module-level helper through the smallest surface.
    import inspect
    src = inspect.getsource(ttv)
    assert "agent._vprint(f\"{agent.log_prefix}🔧 Auto-repaired tool name" in src
    assert "force=True, diagnostic=True" in src[src.index("Auto-repaired tool name"):][:200]


# --- CLI mixins -------------------------------------------------------------------------------

def _cli_block(module, start_marker, end_marker, extra):
    """Execute one guarded statement block from a CLI mixin with the module's real bound names
    (``_cprint`` etc. come from ``cli`` at call time, so we bind them explicitly)."""
    import inspect, textwrap
    import cli as climod
    src = inspect.getsource(module)
    i = src.index(start_marker); j = src.index(end_marker, i)
    i = src.rfind("\n", 0, i) + 1  # back up to the line start so dedent sees uniform indentation
    j = src.rfind("\n", 0, j) + 1
    block = textwrap.dedent(src[i:j])
    ns = dict(module.__dict__); ns.update({"_cprint": climod._cprint, "_DIM": "", "_RST": ""}); ns.update(extra)
    exec(compile(block, f"<{module.__name__}-block>", "exec"), ns)


@pytest.mark.parametrize("setting", MODES)
def test_cli_iteration_budget_notice_honors_policy(tmp_path, monkeypatch, setting):
    _policy(tmp_path, monkeypatch, setting)
    import cli as climod
    from hermes_cli import cli_chat_turn_mixin as m
    out = []
    monkeypatch.setattr(climod, "_cprint", lambda s: out.append(s))
    self = types.SimpleNamespace(agent=types.SimpleNamespace(max_iterations=3, _notification_config=None))
    turn = types.SimpleNamespace(result={"completed": False, "interrupted": False, "api_calls": 3})
    _cli_block(m, "if turn.result and not turn.result.get(\"completed\")", "# Batch TTS", {"turn": turn, "self": self})
    assert (len(out) == 1 and "Iteration budget reached" in out[0]) is _visible(setting)


@pytest.mark.parametrize("setting", MODES)
def test_cli_provider_fallback_notice_honors_policy(tmp_path, monkeypatch, setting):
    _policy(tmp_path, monkeypatch, setting)
    import cli as climod
    from hermes_cli import cli_agent_setup_mixin as m
    out = []
    monkeypatch.setattr(climod, "_cprint", lambda s: out.append(s))
    _cli_block(m, "from gateway.warning_notifications import render_notification\n                render_notification(\n                    lambda: _cprint(f\"⚠️  {_why}",
               "self.requested_provider = _fb_provider",
               {"_fb_provider": "p", "_fb_model": "m", "_why": "Primary auth failed"})
    assert (len(out) == 1 and "Primary auth failed" in out[0]) is _visible(setting)


@pytest.mark.parametrize("setting", MODES)
def test_cli_session_store_unavailable_banner_honors_policy(tmp_path, monkeypatch, setting, capsys):
    _policy(tmp_path, monkeypatch, setting)
    from hermes_cli import cli_init_mixin as m
    from hermes_state_user_copy import describe_storage_failure, storage_failure_details
    from rich.console import Console
    _cli_block(m, "failure = describe_storage_failure(e)", "_run_state_db_auto_maintenance(self._session_db)",
               {"e": RuntimeError("disk I/O error"), "describe_storage_failure": describe_storage_failure,
                "storage_failure_details": storage_failure_details, "Console": Console})
    err = capsys.readouterr().err
    assert ("Session store unavailable" in err) is _visible(setting)


@pytest.mark.parametrize("setting", MODES)
def test_cli_browser_downgrade_notice_honors_policy(tmp_path, monkeypatch, setting):
    _policy(tmp_path, monkeypatch, setting)
    import cli as climod
    from tools import browser_use_cli
    monkeypatch.setattr(browser_use_cli, "default_downgrade_notice", lambda: "Browser Use backend unavailable; using built-in tools")
    out = []
    self = types.SimpleNamespace(_console_print=lambda s: out.append(s))
    fn = next(v for v in vars(climod.HermesCLI).values()
              if callable(v) and "Once-per-24h hint when the default Browser Use backend" in (getattr(v, "__doc__", "") or ""))
    fn(self)
    assert (len(out) == 1 and "⚠" in out[0]) is _visible(setting)


# --- TUI (functions are rebound onto tui_gateway.server by bind_module) -------------------------

@pytest.mark.parametrize("setting", MODES)
def test_tui_goal_compression_recovery_notice_honors_policy(tmp_path, monkeypatch, setting):
    _policy(tmp_path, monkeypatch, setting)
    from tui_gateway import server
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda ev, sid, payload: emitted.append((ev, payload)))
    monkeypatch.setattr(server, "_plan_goal_compression_recovery",
                        lambda session, result, status, raw: ("retry prompt", "Context compression was exhausted. Retrying the active goal once."))
    monkeypatch.setattr(server, "_is_successful_goal_turn", lambda *a: False)
    session = {"agent": types.SimpleNamespace(_notification_config=None)}
    followup = server._goal_followup_after_turn("sid", session, {"compression_exhausted": True}, "error", None)
    assert followup == "retry prompt", "recovery itself is never suppressed"
    assert (len(emitted) == 1) is _visible(setting)


@pytest.mark.parametrize("setting", MODES)
def test_tui_configured_model_adoption_failure_honors_policy(tmp_path, monkeypatch, setting):
    _policy(tmp_path, monkeypatch, setting)
    from tui_gateway import server, model_switch as ms
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda ev, sid, payload: emitted.append((ev, payload)))
    import inspect, textwrap
    src = inspect.getsource(ms)
    i = src.index("except Exception as e:\n        logger.warning(\"Configured model %s could not be adopted")
    j = src.index("def _pending_switch_selection_warning", i)
    block = "try:\n    raise RuntimeError('boom')\n" + textwrap.dedent(src[i:j])
    ns = dict(vars(server)); ns.update({"model": "gpt-x", "sid": "s1", "session": {"agent": types.SimpleNamespace(_notification_config=None)}})
    exec(compile(block, "<adopt-block>", "exec"), ns)
    assert (len(emitted) == 1 and emitted[0][0] == "error") is _visible(setting)


@pytest.mark.parametrize("setting", MODES)
def test_tui_preview_restart_status_warning_honors_policy(tmp_path, monkeypatch, setting):
    """The preview-restart progress panel's status_callback: warning rows follow the TUI policy;
    ordinary status rows and tool progress always render."""
    _policy(tmp_path, monkeypatch, setting)
    from tui_gateway import server
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda ev, sid, payload: emitted.append(payload["text"]))
    monkeypatch.setattr(server, "_session_get", lambda sid: {"agent": types.SimpleNamespace(_notification_config=None)}, raising=False)
    from gateway.warning_notifications import DiagnosticText
    cbs = server._preview_restart_callbacks("parent-1", "task-1")
    cbs["status_callback"]("lifecycle", "Starting preview")
    cbs["status_callback"]("warn", "⚠ provider fallback engaged")
    cbs["status_callback"]("lifecycle", DiagnosticText("⚠ compression model unavailable"))
    cbs["tool_gen_callback"]("terminal")
    assert "Starting preview" in emitted and "Preparing terminal" in emitted
    assert ("⚠ provider fallback engaged" in emitted) is _visible(setting)
    assert ("⚠ compression model unavailable" in emitted) is _visible(setting)


@pytest.mark.parametrize("setting", MODES)
@pytest.mark.parametrize("failure", ("returns_error", "raises"))
def test_cli_vision_fallback_notice_honors_policy(tmp_path, monkeypatch, setting, failure):
    """Real call through _preprocess_images_with_vision: the model-facing retry text (with the path)
    is always produced; only the console ⚠ notice follows policy."""
    _policy(tmp_path, monkeypatch, setting)
    import cli as climod
    from unittest.mock import patch as _patch
    out = []
    monkeypatch.setattr(climod, "_cprint", lambda s: out.append(s))
    img = tmp_path / "shot.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    if failure == "raises":
        async def _vision(**kwargs): raise RuntimeError("API down")
    else:
        async def _vision(**kwargs): return json.dumps({"error": "vision unavailable"})
    cli_obj = climod.HermesCLI.__new__(climod.HermesCLI)
    cli_obj.agent = types.SimpleNamespace(_notification_config=None)
    with _patch("tools.vision_tools.vision_analyze_tool", side_effect=_vision):
        result = cli_obj._preprocess_images_with_vision("check this", [img])
    assert str(img) in result and "check this" in result
    assert (any("vision analysis" in s for s in out)) is _visible(setting)
