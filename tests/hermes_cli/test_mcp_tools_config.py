"""Tests for MCP tools interactive configuration in hermes_cli.tools_config."""

from unittest.mock import patch

from hermes_cli.tools_config import _configure_mcp_tools_interactive

# Patch targets: imports happen inside the function body, so patch at source
_PROBE = "tools.mcp_tool.probe_mcp_server_tools"
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


def test_empty_include_list_preselects_none(capsys):
    """``tools.include: []`` must pre-select zero tools (block all), not all.

    Regression consolidating #13096/#7822 with tools/mcp_tool runtime semantics.
    """
    config = {
        "mcp_servers": {
            "github": {
                "command": "npx",
                "tools": {"include": []},
            },
        }
    }
    tools = [
        ("create_issue", "Create"),
        ("delete_repo", "Delete"),
        ("search", "Search"),
    ]
    captured_pre_selected = {}

    def fake_checklist(title, labels, pre_selected, **kwargs):
        captured_pre_selected["value"] = set(pre_selected)
        return pre_selected  # No changes

    with patch(_PROBE, return_value={"github": tools}), \
         patch(_CHECKLIST, side_effect=fake_checklist), \
         patch(_SAVE):
        _configure_mcp_tools_interactive(config)

    assert captured_pre_selected["value"] == set()
