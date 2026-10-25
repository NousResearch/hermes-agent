"""Local result management without connector login, model calls, or sends."""

import pytest


@pytest.fixture
def home(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    return profile


@pytest.fixture
def saved(home):
    from gateway.connected_assistance_store import AssistanceStore

    row = {"kind": "deadline", "entity_key": "draft-1", "title": "Review draft",
           "summary": "Draft is due Friday", "why_now": "Deadline is tomorrow",
           "suggested_action": "Read the draft", "evidence": [
               {"source_id": "gmail:account-a", "record_id": "message-1", "excerpt": "Due Friday",
                "url": "https://example.test/message-1", "data_use": ["private"]}]}
    return AssistanceStore(home).ingest_scan(
        {"findings": [row], "evidence": [], "errors": [], "connected": True})[0]


def test_default_list_and_show_use_real_local_state(home, saved, capsys):
    from hermes_cli.connected_assistance import dispatch_findings_command

    output = dispatch_findings_command(home=home)
    assert saved["id"] in output and saved["title"] in output and "open" in output
    details = dispatch_findings_command(f"show {saved['id']}", home=home)
    for field in ("summary", "why_now", "suggested_action"):
        assert saved[field] in details
    for value in saved["evidence"][0].values():
        if isinstance(value, str):
            assert value in details
    assert capsys.readouterr() == ("", "")


def test_empty_list_and_unknown_id_are_controlled(home):
    from hermes_cli.connected_assistance import dispatch_findings_command

    assert "No open findings" in dispatch_findings_command(home=home)
    assert "not found" in dispatch_findings_command("show missing", home=home).lower()


@pytest.mark.parametrize("action,status", [("dismiss", "dismissed"), ("resolve", "resolved")])
def test_management_changes_persist_locally(home, saved, action, status):
    from gateway.connected_assistance_store import AssistanceStore
    from hermes_cli.connected_assistance import dispatch_findings_command

    result = dispatch_findings_command(f"{action} {saved['id']}", home=home)
    assert saved["id"] in result and status in result
    assert AssistanceStore(home).get_finding(saved["id"])["status"] == status
    assert saved["id"] not in dispatch_findings_command(home=home)
    assert saved["id"] in dispatch_findings_command(f"list --status {status}", home=home)
    assert saved["id"] in dispatch_findings_command("list --status all", home=home)
    assert "not found" in dispatch_findings_command(f"{action} missing", home=home).lower()


def test_pause_resume_remember_only_profile_pause_state(home):
    from gateway.connected_assistance_store import AssistanceStore
    from hermes_cli.connected_assistance import dispatch_findings_command

    assert "paused" in dispatch_findings_command("pause", home=home).lower()
    assert AssistanceStore(home).is_paused()
    assert "paused" in dispatch_findings_command(home=home).lower()
    assert "resumed" in dispatch_findings_command("resume", home=home).lower()
    assert not AssistanceStore(home).is_paused()
    assert {entry.name for entry in home.iterdir()} == {"connected_assistance"}


@pytest.mark.parametrize("raw", [
    "unknown", "show", "dismiss", "resolve", "pause extra", "resume extra",
    "show id extra", "list extra", "list --limit nope", "list --limit -1",
    "list --limit 101", "list --status bad", "list --lim 2", 'show "unterminated',
    "--help extra", "--unknown", "list --help extra",
])
def test_parse_errors_return_text_without_exit_output_or_state_changes(home, raw, capsys):
    from hermes_cli.connected_assistance import dispatch_findings_command

    output = dispatch_findings_command(raw, home=home)
    assert output.startswith("Error:") and "Usage:" in output
    assert not list(home.iterdir())
    assert capsys.readouterr() == ("", "")


def test_parser_helper_and_help_do_not_create_state(home):
    from hermes_cli.connected_assistance import dispatch_findings_command, parse_findings_args

    args = parse_findings_args("list --status resolved --limit 2")
    assert (args.action, args.status, args.limit) == ("list", "resolved", 2)
    args = parse_findings_args('show "a b"')
    assert args.identifier == "a b"
    with pytest.raises(ValueError):
        parse_findings_args("dismiss")
    for raw in ("help", "--help", "-h", "list --help", "show --help"):
        assert "Usage:" in dispatch_findings_command(raw, home=home)
    assert not list(home.iterdir())


def test_list_limit_is_applied(home, saved):
    from gateway.connected_assistance_store import AssistanceStore
    from hermes_cli.connected_assistance import dispatch_findings_command

    store = AssistanceStore(home)
    second = saved | {"entity_key": "second", "title": "Second draft",
                      "evidence": [saved["evidence"][0] | {"record_id": "message-2"}]}
    created = store.ingest_scan({"findings": [second], "evidence": [], "connected": True})[0]
    output = dispatch_findings_command("list --limit 1", home=home)
    assert created["id"] in output and saved["id"] not in output


def test_default_home_resolves_context_on_each_command(home, tmp_path, saved):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.connected_assistance import dispatch_findings_command

    other = tmp_path / "other"
    other.mkdir()
    token = set_hermes_home_override(home)
    try:
        assert saved["id"] in dispatch_findings_command()
        inner = set_hermes_home_override(other)
        try:
            assert saved["id"] not in dispatch_findings_command()
            assert saved["id"] in dispatch_findings_command(home=home)
        finally:
            reset_hermes_home_override(inner)
        assert saved["id"] in dispatch_findings_command()
    finally:
        reset_hermes_home_override(token)


def test_local_commands_do_not_import_connectors(home, saved, monkeypatch):
    import builtins
    from hermes_cli.connected_assistance import dispatch_findings_command

    original_import = builtins.__import__

    def no_connectors(name, *args, **kwargs):
        if name.startswith("tools.connectors") or name.startswith("agent."):
            raise AssertionError("local management must not need apps or a model")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_connectors)
    for raw in ("", f"show {saved['id']}", f"dismiss {saved['id']}", "pause", "resume"):
        assert not dispatch_findings_command(raw, home=home).startswith("Error:")


@pytest.mark.parametrize("problem", ["missing", "corrupt"])
def test_storage_failure_is_a_redacted_controlled_error(home, problem, capsys):
    from hermes_cli.connected_assistance import dispatch_findings_command

    if problem == "missing":
        home = home / "removed"
    else:
        directory = home / "connected_assistance"
        directory.mkdir()
        (directory / "state.sqlite3").write_text("SENSITIVE_SQLITE_ERROR_CONTENT")
    result = dispatch_findings_command(home=home)
    assert result.startswith("Error:") and "local findings" in result
    assert "SENSITIVE" not in result and str(home) not in result
    assert capsys.readouterr() == ("", "")


def test_source_control_characters_cannot_inject_terminal_sequences(home, saved):
    from gateway.connected_assistance_store import AssistanceStore
    from hermes_cli.connected_assistance import dispatch_findings_command

    row = saved | {"title": "Safe\x1b[2J\u202eevil\nforged list line",
                   "summary": "safe\x00 summary", "evidence": [saved["evidence"][0] | {
                       "excerpt": "data\x1b]52;c;clipboard\x07\nsecond line"}]}
    store = AssistanceStore(home)
    changed = store.ingest_scan({"findings": [row], "evidence": [], "connected": True})[0]
    listing = dispatch_findings_command(home=home)
    assert len(listing.splitlines()) == 1
    shown = dispatch_findings_command(f"show {changed['id']}", home=home)
    assert "second line" in shown and "data" in shown
    for forbidden in ("\x1b", "\u202e", "\x07", "\x00"):
        assert forbidden not in listing and forbidden not in shown
