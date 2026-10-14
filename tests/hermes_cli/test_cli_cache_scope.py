"""Single-query cache routing is explicit and does not turn into session sharing (#136359)."""

import io
from types import SimpleNamespace

import pytest

from hermes_cli.cli_cache_scope import (
    bind_cli_cache_scope,
    configure_cli_cache_scope,
    normalize_cache_scope,
    validate_chat_cache_scope,
    validate_cli_cache_scope,
)


def _parse(argv):
    from hermes_cli._parser import build_top_level_parser

    return build_top_level_parser()[0].parse_args(argv)


@pytest.mark.parametrize("value", ["", "  ", "a\nb", "a\x7fb", "x" * 257, "é" * 129])
def test_invalid_labels_rejected_by_parser(value):
    with pytest.raises(SystemExit) as exc:
        _parse(["chat", "-Q", "-q", "draft", "--cache-scope", value])
    assert exc.value.code == 2


def test_label_normalization_and_absent_default():
    assert normalize_cache_scope(" 文章草稿 ") == "文章草稿"
    args = _parse(["chat", "-Q", "-q", "draft"])
    assert args.cache_scope is None


@pytest.mark.parametrize("kwargs", [
    {"query": None}, {"quiet": False}, {"resume": "old"}, {"use_tui": True},
])
def test_rejects_non_fresh_or_interactive_use(monkeypatch, kwargs):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    options = {"cache_scope": "drafts", "query": "draft", "quiet": True,
               "oneshot": False, "resume": None, **kwargs}
    with pytest.raises(ValueError):
        validate_cli_cache_scope(**options)


@pytest.mark.parametrize("flags", [
    ["--resume", "old"], ["--continue"], ["--continue", "old"],
])
def test_resume_and_continue_rejected_before_session_lookup(flags):
    args = _parse(["chat", "-Q", "-q", "draft", "--cache-scope", "drafts", *flags])
    with pytest.raises(SystemExit) as exc:
        validate_chat_cache_scope(args, False)
    assert exc.value.code == 2


@pytest.mark.parametrize("flags", [["-Q"], ["--oneshot"], ["--format", "stream-json"]])
def test_supported_single_query_modes_on_a_tty(monkeypatch, flags):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    args = _parse(["chat", "-q", "draft", "--cache-scope", "drafts", *flags])
    assert validate_chat_cache_scope(args, False) == "drafts"


def test_non_tty_single_query_supported(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    args = _parse(["chat", "--query-file", "-", "--cache-scope", "drafts"])
    assert validate_chat_cache_scope(args, False) == "drafts"


def test_profile_and_workspace_namespaces_a_b_a(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    scopes = []
    for home in (tmp_path / "a", tmp_path / "b", tmp_path / "a"):
        monkeypatch.setenv("HERMES_HOME", str(home))
        cli = SimpleNamespace(session_id="unique-session")
        configure_cli_cache_scope(cli, "drafts")
        agent = SimpleNamespace(session_id=cli.session_id, _gateway_session_key=None)
        bind_cli_cache_scope(cli, agent)
        scopes.append(agent._cli_prompt_cache_scope)
        assert agent.session_id == "unique-session"
        assert agent._gateway_session_key is None
        assert str(home) not in scopes[-1] and "drafts" not in scopes[-1]
        assert len(scopes[-1]) <= 64
    assert scopes[0] == scopes[2] != scopes[1]
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    monkeypatch.chdir(other_workspace)
    configure_cli_cache_scope(cli, "drafts")
    assert cli._cli_prompt_cache_scope != scopes[0]


def test_cmd_chat_forwards_scope_and_verbatim_stdin(monkeypatch):
    import hermes_cli.main as entry
    import cli
    import hermes_cli.free_tier_bootstrap as bootstrap
    from hermes_cli.observability import shared_metrics_consent, shared_metrics_process

    monkeypatch.setattr(bootstrap, "run_bootstrap", lambda **kw: None)
    monkeypatch.setattr(entry, "_resolve_use_tui", lambda args: False)
    monkeypatch.setattr(entry, "_resolve_chat_session_args", lambda *args: None)
    monkeypatch.setattr(entry, "_warn_retired_xai_models", lambda: None)
    monkeypatch.setattr(entry, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(entry, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr(entry, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(entry, "_confirm_startup_expensive_model_override", lambda args: None)
    monkeypatch.setattr(shared_metrics_consent, "offer_consent_before_chat", lambda args: None)
    monkeypatch.setattr(shared_metrics_process, "begin_process", lambda surface: None)
    captured = {}
    monkeypatch.setattr(cli, "main", lambda **kwargs: captured.update(kwargs))
    query = 'Draft "hello"; $(never-executed)'
    monkeypatch.setattr("sys.stdin", io.StringIO(query))
    args = _parse(["chat", "-Q", "--query-file", "-", "--cache-scope", "drafts"])
    entry.cmd_chat(args)
    assert captured["cache_scope"] == "drafts"
    assert captured["query"] == query
    assert captured["quiet"] is True


def test_direct_cli_main_installs_scope_before_query(monkeypatch):
    import cli
    from hermes_cli import process_identity, shared_profile_warning

    shell = SimpleNamespace(session_id="physical-session")
    monkeypatch.setattr(process_identity, "register_self", lambda surface: None)
    monkeypatch.setattr(shared_profile_warning, "shared_profile_warning", lambda: None)
    monkeypatch.setattr(cli, "_start_worktree_setup", lambda *args: None)
    monkeypatch.setattr(cli, "_build_cli_from_args", lambda *args: shell)
    monkeypatch.setattr(cli, "_install_single_query_signal_handlers", lambda obj: None)
    monkeypatch.setattr(cli.atexit, "register", lambda *args: None)
    captured = []
    monkeypatch.setattr(cli, "_run_single_query_mode", lambda obj, *args, **kw: captured.append(obj._cli_prompt_cache_scope))
    cli.main(query="draft", quiet=True, cache_scope="drafts")
    assert captured[0].startswith("cli_")
    assert shell.session_id == "physical-session"


def test_real_cli_initialization_binds_scope_to_wire_request(tmp_path, monkeypatch):
    """Keep CLI/provider/agent initialization real; stop at the query execution boundary."""
    import copy
    import cli

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    config = copy.deepcopy(cli.CLI_CONFIG)
    config["agent"]["disabled_toolsets"] = ["all"]
    monkeypatch.setattr(cli, "CLI_CONFIG", config)
    monkeypatch.setattr(cli, "_install_single_query_signal_handlers", lambda obj: None)
    monkeypatch.setattr(cli.atexit, "register", lambda *args: None)
    captured = []

    def inspect_request(shell, query, **kwargs):
        agent = shell.agent
        agent.skip_background_review = True
        request = agent._build_api_kwargs([
            {"role": "system", "content": "Stable instructions"},
            {"role": "user", "content": query},
        ], [])
        scope = agent._cli_prompt_cache_scope
        from agent.prompt_cache_scope import resolve_prompt_cache_scope
        assert resolve_prompt_cache_scope(agent) == scope
        agent._cli_prompt_cache_scope = None
        unscoped = agent._build_api_kwargs([
            {"role": "system", "content": "Stable instructions"},
            {"role": "user", "content": query},
        ], [])
        assert request["prompt_cache_key"] != unscoped["prompt_cache_key"]
        agent._cli_prompt_cache_scope = scope
        captured.append((shell.session_id, agent.session_id, scope))
        raise SystemExit(0)

    monkeypatch.setattr(cli, "_run_quiet_single_query", inspect_request)
    with pytest.raises(SystemExit) as exc:
        cli.main(
            query="draft", quiet=True, cache_scope="drafts", ignore_rules=True,
            model="gpt-5.4", provider="custom", api_key="synthetic-test-key",
            base_url="https://api.openai.com/v1", max_turns=1,
        )
    assert exc.value.code == 0
    cli_id, agent_id, affinity = captured[0]
    assert cli_id == agent_id
    assert affinity.startswith("cli_") and affinity != agent_id
