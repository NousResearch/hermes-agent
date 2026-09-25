"""Tests for MCP tools interactive configuration in hermes_cli.tools_config."""

from unittest.mock import patch

from hermes_cli.tools_config import _configure_mcp_tools_interactive

# Patch targets: imports happen inside the function body, so patch at source
_PROBE = "tools.mcp_tool_discovery.probe_mcp_server_tools"
_CHECKLIST = "hermes_cli.curses_ui.curses_checklist"
_SAVE = "hermes_cli.tools_config.save_config"








def test_disabling_tool_writes_include_list(capsys):
    """Unchecking a tool produces an include list of the still-chosen tools.

    Standardized on tools.include (whitelist) across the codebase — the
    catalog flow, `hermes mcp configure`, and this UI all write the same
    shape so users don\'t see config drift across UIs.
    """
    config = {
        "mcp_servers": {
            "github": {"command": "npx"},
        }
    }
    tools = [
        ("create_issue", "Create an issue"),
        ("delete_repo", "Delete a repo"),
        ("search_repos", "Search repos"),
    ]

    # User unchecks delete_repo (index 1)
    with patch(_PROBE, return_value={"github": tools}), \
         patch(_CHECKLIST, return_value={0, 2}), \
         patch(_SAVE) as mock_save:
        _configure_mcp_tools_interactive(config)

    mock_save.assert_called_once()
    tools_cfg = config["mcp_servers"]["github"]["tools"]
    assert tools_cfg["include"] == ["create_issue", "search_repos"]
    assert "exclude" not in tools_cfg








def test_empty_tools_server_skipped(capsys):
    """Server with no tools shows info message and skips checklist."""
    config = {
        "mcp_servers": {
            "empty": {"command": "npx"},
        }
    }
    checklist_calls = []

    def fake_checklist(title, labels, pre_selected, **kwargs):
        checklist_calls.append(title)
        return pre_selected

    with patch(_PROBE, return_value={"empty": []}), \
         patch(_CHECKLIST, side_effect=fake_checklist), \
         patch(_SAVE):
        _configure_mcp_tools_interactive(config)

    assert len(checklist_calls) == 0
    captured = capsys.readouterr()
    assert "no tools found" in captured.out


def test_empty_include_reopens_with_nothing_preselected():
    """``include: []`` is the runtime's block-all whitelist; the picker must not reopen it as
    "all tools enabled" and must persist it when the user keeps zero tools checked (#12865)."""
    config = {"mcp_servers": {"github": {"command": "npx", "tools": {"include": []}}}}
    tools = [("create_issue", "Create an issue"), ("search_repos", "Search repos")]

    with patch(_PROBE, return_value={"github": tools}), \
         patch(_CHECKLIST, side_effect=lambda title, labels, pre, **kw: pre) as checklist, \
         patch(_SAVE) as mock_save:
        _configure_mcp_tools_interactive(config)

    assert checklist.call_args.args[2] == set()
    mock_save.assert_not_called()
    assert config["mcp_servers"]["github"]["tools"] == {"include": []}


def test_shorthand_filter_checklist_rewrites_canonical():
    """``tools: "a,b"`` shorthand is an include whitelist the checklist must accept (PR #122381):
    it preselects the whitelist like registration, and saving rewrites the canonical dict form
    instead of crashing on item assignment into the string."""
    config = {"mcp_servers": {"github": {"command": "npx", "tools": "create_issue,search_repos"}}}
    tools = [
        ("create_issue", "Create an issue"),
        ("delete_repo", "Delete a repo"),
        ("search_repos", "Search repos"),
    ]

    # User unchecks search_repos (index 2) from the shorthand-preselected {0, 2}
    with patch(_PROBE, return_value={"github": tools}), \
         patch(_CHECKLIST, return_value={0}) as checklist, \
         patch(_SAVE) as mock_save:
        _configure_mcp_tools_interactive(config)

    assert checklist.call_args.args[2] == {0, 2}
    mock_save.assert_called_once()
    assert config["mcp_servers"]["github"]["tools"] == {"include": ["create_issue"]}


def test_apply_mcp_change_accepts_shorthand_filter():
    """``hermes tools enable|disable server:tool`` must accept a shorthand ``tools`` filter
    (same TypeError/AttributeError class as the interactive writers, PR #122381): the exclude
    entry lands beside the canonical include whitelist."""
    from hermes_cli.tools_config_mcp import _apply_mcp_change

    config = {"mcp_servers": {"github": {"command": "npx", "tools": "a,b"}}}

    failed = _apply_mcp_change(config, ["github:c"], "disable")

    assert not failed
    assert config["mcp_servers"]["github"]["tools"] == {"include": ["a", "b"], "exclude": ["c"]}
