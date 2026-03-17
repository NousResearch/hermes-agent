"""Tests for MCP tools interactive configuration in hermes_cli.tools_config."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli.tools_config import _configure_mcp_tools_interactive

# Patch targets: imports happen inside the function body, so patch at source
_PROBE = "tools.mcp_tool.probe_mcp_server_tools"
_CHECKLIST = "hermes_cli.curses_ui.curses_checklist"
_SAVE = "hermes_cli.tools_config.save_config"


def test_no_mcp_servers_prints_info(capsys):
    """Returns immediately when no MCP servers are configured."""
    config = {}
    _configure_mcp_tools_interactive(config)
    captured = capsys.readouterr()
    assert "No MCP servers configured" in captured.out


def test_all_servers_disabled_prints_info(capsys):
    """Returns immediately when all configured servers have enabled=false."""
    config = {
        "mcp_servers": {
            "github": {"command": "npx", "enabled": False},
            "slack": {"command": "npx", "enabled": "false"},
        }
    }
    _configure_mcp_tools_interactive(config)
    captured = capsys.readouterr()
    assert "disabled" in captured.out


def test_probe_failure_shows_warning(capsys):
    """Shows warning when probe returns no tools."""
    config = {"mcp_servers": {"github": {"command": "npx"}}}
    with patch(_PROBE, return_value={}):
        _configure_mcp_tools_interactive(config)
    captured = capsys.readouterr()
    assert "Could not discover" in captured.out


def test_probe_exception_shows_error(capsys):
    """Shows error when probe raises an exception."""
    config = {"mcp_servers": {"github": {"command": "npx"}}}
    with patch(_PROBE, side_effect=RuntimeError("MCP not installed")):
        _configure_mcp_tools_interactive(config)
    captured = capsys.readouterr()
    assert "Failed to probe" in captured.out


def test_no_changes_when_checklist_cancelled(capsys):
    """No config changes when user cancels (ESC) the checklist."""
    config = {
        "mcp_servers": {
            "github": {"command": "npx", "args": ["-y", "server-github"]},
        }
    }
    tools = [("create_issue", "Create an issue"), ("search_repos", "Search repos")]

    with patch(_PROBE, return_value={"github": tools}), \
         patch(_CHECKLIST, return_value={0, 1}), \
         patch(_SAVE) as mock_save:
        _configure_mcp_tools_interactive(config)
    mock_save.assert_not_called()
    captured = capsys.readouterr()
    assert "no changes" in captured.out.lower()


def test_disabling_tool_writes_exclude_list(capsys):
    """Unchecking a tool adds it to the exclude list."""
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
    assert tools_cfg["exclude"] == ["delete_repo"]
    assert "include" not in tools_cfg


def test_enabling_all_clears_filters(capsys):
    """Checking all tools clears both include and exclude lists."""
    config = {
        "mcp_servers": {
            "github": {
                "command": "npx",
                "tools": {"exclude": ["delete_repo"], "include": ["create_issue"]},
            },
        }
    }
    tools = [("create_issue", "Create"), ("delete_repo", "Delete")]

    # User checks all tools — pre_selected would be {0} (include mode),
    # so returning {0, 1} is a change
    with patch(_PROBE, return_value={"github": tools}), \
         patch(_CHECKLIST, return_value={0, 1}), \
         patch(_SAVE) as mock_save:
        _configure_mcp_tools_interactive(config)

    mock_save.assert_called_once()
    tools_cfg = config["mcp_servers"]["github"]["tools"]
    assert "exclude" not in tools_cfg
    assert "include" not in tools_cfg


def test_pre_selection_respects_existing_exclude(capsys):
    """Tools in exclude list start unchecked."""
    config = {
        "mcp_servers": {
            "github": {
                "command": "npx",
                "tools": {"exclude": ["delete_repo"]},
            },
        }
    }
    tools = [("create_issue", "Create"), ("delete_repo", "Delete"), ("search", "Search")]
    captured_pre_selected = {}

    def fake_checklist(title, labels, pre_selected, **kwargs):
        captured_pre_selected["value"] = set(pre_selected)
        return pre_selected  # No changes

    with patch(_PROBE, return_value={"github": tools}), \
         patch(_CHECKLIST, side_effect=fake_checklist), \
         patch(_SAVE):
        _configure_mcp_tools_interactive(config)

    # create_issue (0) and search (2) should be pre-selected, delete_repo (1) should not
    assert captured_pre_selected["value"] == {0, 2}


def test_pre_selection_respects_existing_include(capsys):
    """Only tools in include list start checked."""
    config = {
        "mcp_servers": {
            "github": {
                "command": "npx",
                "tools": {"include": ["search"]},
            },
        }
    }
    tools = [("create_issue", "Create"), ("delete_repo", "Delete"), ("search", "Search")]
    captured_pre_selected = {}

    def fake_checklist(title, labels, pre_selected, **kwargs):
        captured_pre_selected["value"] = set(pre_selected)
        return pre_selected  # No changes

    with patch(_PROBE, return_value={"github": tools}), \
         patch(_CHECKLIST, side_effect=fake_checklist), \
         patch(_SAVE):
        _configure_mcp_tools_interactive(config)

    # Only search (2) should be pre-selected
    assert captured_pre_selected["value"] == {2}


def test_multiple_servers_each_get_checklist(capsys):
    """Each server gets its own checklist."""
    config = {
        "mcp_servers": {
            "github": {"command": "npx"},
            "slack": {"url": "https://mcp.example.com"},
        }
    }
    checklist_calls = []

    def fake_checklist(title, labels, pre_selected, **kwargs):
        checklist_calls.append(title)
        return pre_selected  # No changes

    with patch(
        _PROBE,
        return_value={
            "github": [("create_issue", "Create")],
            "slack": [("send_message", "Send")],
        },
    ), patch(_CHECKLIST, side_effect=fake_checklist), \
         patch(_SAVE):
        _configure_mcp_tools_interactive(config)

    assert len(checklist_calls) == 2
    assert any("github" in t for t in checklist_calls)
    assert any("slack" in t for t in checklist_calls)


def test_failed_server_shows_warning(capsys):
    """Servers that fail to connect show warnings."""
    config = {
        "mcp_servers": {
            "github": {"command": "npx"},
            "broken": {"command": "nonexistent"},
        }
    }

    # Only github succeeds
    with patch(
        _PROBE, return_value={"github": [("create_issue", "Create")]},
    ), patch(_CHECKLIST, return_value={0}), \
         patch(_SAVE):
        _configure_mcp_tools_interactive(config)

    captured = capsys.readouterr()
    assert "broken" in captured.out


def test_description_truncation_in_labels():
    """Long descriptions are truncated in checklist labels."""
    config = {
        "mcp_servers": {
            "github": {"command": "npx"},
        }
    }
    long_desc = "A" * 100
    captured_labels = {}

    def fake_checklist(title, labels, pre_selected, **kwargs):
        captured_labels["value"] = labels
        return pre_selected

    with patch(
        _PROBE, return_value={"github": [("my_tool", long_desc)]},
    ), patch(_CHECKLIST, side_effect=fake_checklist), \
         patch(_SAVE):
        _configure_mcp_tools_interactive(config)

    label = captured_labels["value"][0]
    assert "..." in label
    assert len(label) < len(long_desc) + 30  # truncated + tool name + parens


def test_switching_from_include_to_exclude(capsys):
    """When user modifies selection, include list is replaced by exclude list."""
    config = {
        "mcp_servers": {
            "github": {
                "command": "npx",
                "tools": {"include": ["create_issue"]},
            },
        }
    }
    tools = [("create_issue", "Create"), ("search", "Search"), ("delete", "Delete")]

    # User selects create_issue and search (deselects delete)
    # pre_selected would be {0} (only create_issue from include), so {0, 1} is a change
    with patch(_PROBE, return_value={"github": tools}), \
         patch(_CHECKLIST, return_value={0, 1}), \
         patch(_SAVE):
        _configure_mcp_tools_interactive(config)

    tools_cfg = config["mcp_servers"]["github"]["tools"]
    assert tools_cfg["exclude"] == ["delete"]
    assert "include" not in tools_cfg


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
