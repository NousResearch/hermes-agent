"""Classic CLI must consume first-class commentary rather than the reasoning fallback."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def cli_shell(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import cli as cli_mod

    shell = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    shell.show_reasoning = False
    shell.show_timestamps = False
    shell.streaming_enabled = True
    shell.verbose = False
    shell.tool_progress_mode = "all"
    shell.final_response_markdown = "raw"
    shell._app = None
    shell._held_status_lines = []
    shell._reasoning_shown_this_turn = False
    shell._reset_stream_state()
    shell._scrollback_box_width = lambda: 100
    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda text="", **kwargs: printed.append(str(text)))
    monkeypatch.setitem(cli_mod.CLI_CONFIG, "display", {})
    return shell, printed, cli_mod


@pytest.mark.parametrize("show_reasoning", [False, True])
@pytest.mark.parametrize("streaming_enabled", [False, True])
def test_commentary_renders_as_assistant_independently_of_reasoning_and_streaming(
    cli_shell, show_reasoning, streaming_enabled,
):
    shell, printed, _ = cli_shell
    shell.show_reasoning = show_reasoning
    shell.streaming_enabled = streaming_enabled
    if show_reasoning:
        shell._stream_reasoning_delta("Reasoning summary.")

    callback = shell._current_interim_assistant_callback()
    assert callable(callback)
    callback("I'll inspect the file.")

    output = "\n".join(printed)
    assert output.count("I'll inspect the file.") == 1
    # The interim must be a normal response box, not text in the dim reasoning box.
    assert "╭" in output
    if show_reasoning:
        assert output.index("└") < output.index("I'll inspect the file.")
    else:
        assert " Reasoning " not in output
    assert shell._reasoning_box_opened is False
    assert shell._stream_started is False
    assert shell._stream_box_opened is False
    assert shell._stream_box_live is False

    # A subsequent answer must not be swallowed as an already-rendered final.
    shell._stream_delta("The file is correct.")
    shell._stream_delta(None)
    output = "\n".join(printed)
    assert output.count("I'll inspect the file.") == 1
    assert output.count("The file is correct.") == 1


def test_tool_progress_off_keeps_interim_consumer(cli_shell):
    shell, printed, _ = cli_shell
    shell.tool_progress_mode = "off"

    assert callable(shell._current_interim_assistant_callback())
    assert printed == []


def test_parseable_quiet_suppresses_interim_output(cli_shell):
    shell, printed, _ = cli_shell
    shell.agent = SimpleNamespace(suppress_status_output=True)

    shell._on_interim_assistant("Must stay out of -Q stdout.")

    assert printed == []
    assert shell._stream_started is False


def test_explicit_interim_opt_out_does_not_install_consumer(cli_shell):
    shell, printed, cli_mod = cli_shell
    cli_mod.CLI_CONFIG["display"] = {"interim_assistant_messages": False}

    assert shell._current_interim_assistant_callback() is None
    assert printed == []


def test_already_streamed_interim_is_not_printed_twice(cli_shell):
    shell, printed, _ = cli_shell
    shell._stream_delta("Already visible.")
    shell._on_interim_assistant("Already visible.", already_streamed=True)
    shell._stream_delta(None)

    assert "\n".join(printed).count("Already visible.") == 1


@pytest.mark.parametrize("text", [None, "", " \n\t"])
def test_empty_interim_does_not_open_a_box(cli_shell, text):
    shell, printed, _ = cli_shell
    shell._on_interim_assistant(text)
    assert printed == []
    assert shell._stream_started is False


def test_interim_sanitizes_terminal_control_sequences(cli_shell):
    shell, printed, _ = cli_shell
    shell._on_interim_assistant("\x1b[2JVisible progress.")
    output = "\n".join(printed)
    assert "Visible progress." in output
    assert "\x1b[2J" not in output


@pytest.mark.parametrize("streaming_enabled", [False, True])
def test_init_agent_installs_interim_consumer_even_without_reasoning(cli_shell, monkeypatch, streaming_enabled):
    shell, _, cli_mod = cli_shell
    import run_agent
    import hermes_cli.mcp_startup as mcp_startup
    import agent.credits_tracker as credits_tracker

    noop = lambda *args, **kwargs: None
    shell.agent = None
    shell.model = "gpt-5-codex"
    shell.streaming_enabled = streaming_enabled
    shell._session_db = object()
    shell._resumed = False
    shell._pending_title = None
    shell.max_turns = 4
    shell.enabled_toolsets = []
    shell.disabled_toolsets = []
    shell.system_prompt = ""
    shell.prefill_messages = []
    shell.reasoning_config = None
    shell.service_tier = None
    shell.session_id = "test-commentary"
    shell._inline_diffs_enabled = False
    shell.ignore_rules = True
    shell.pass_session_id = False
    shell.checkpoints_enabled = False
    shell.checkpoint_max_snapshots = 1
    shell.checkpoint_max_total_size_mb = 1
    shell.checkpoint_max_file_size_mb = 1
    for name in (
        "_providers_only", "_providers_ignore", "_providers_order", "_provider_sort",
        "_provider_require_params", "_provider_data_collection",
        "_openrouter_min_coding_score", "_fallback_model",
    ):
        setattr(shell, name, None)
    for name in (
        "finalize_preloaded_skills", "_install_tool_callbacks", "_ensure_tirith_security",
        "_clarify_callback", "_on_reaction",
    ):
        setattr(shell, name, noop)
    shell._ensure_runtime_credentials = lambda: True
    monkeypatch.setattr(cli_mod, "_prepare_deferred_agent_startup", noop)
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", noop)
    monkeypatch.setattr(credits_tracker, "seed_credits_at_session_start", noop)
    # _init_agent updates the process-wide cleanup reference; restore it at teardown.
    monkeypatch.setattr(cli_mod, "_active_agent_ref", None)
    captured = {}

    def fake_agent(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(run_agent, "AIAgent", fake_agent)
    assert shell._init_agent(runtime_override={
        "provider": "openai-codex", "api_mode": "codex_responses",
        "base_url": "https://chatgpt.com/backend-api/codex", "api_key": "test-key",
    }) is True
    assert captured["reasoning_callback"] is None
    assert captured["interim_assistant_callback"] == shell._on_interim_assistant
    assert (captured["stream_delta_callback"] is not None) is streaming_enabled
