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


def test_pre_selection_respects_glob_exclude():
    """A glob exclude must start its matching tools unchecked.

    Runtime registration matches include/exclude with fnmatch globs
    (``tools/mcp_tool.py::matches_name_filter``). Matching only exact names
    here shows every glob-excluded tool as enabled, so the checklist
    disagrees with the tool surface the agent actually gets.
    """
    config = {
        "mcp_servers": {
            "cloudflare": {
                "command": "npx",
                "tools": {"exclude": ["*_radar_*"]},
            },
        }
    }
    tools = [
        ("purge_cache", "Purge"),
        ("cf_radar_http", "Radar HTTP"),
        ("dns_records", "DNS"),
        ("cf_radar_bgp", "Radar BGP"),
    ]
    captured_pre_selected = {}

    def fake_checklist(title, labels, pre_selected, **kwargs):
        captured_pre_selected["value"] = set(pre_selected)
        return pre_selected  # No changes

    with patch(_PROBE, return_value={"cloudflare": tools}), \
         patch(_CHECKLIST, side_effect=fake_checklist), \
         patch(_SAVE):
        _configure_mcp_tools_interactive(config)

    # Only the two non-radar tools may start checked.
    assert captured_pre_selected["value"] == {0, 2}


def test_pre_selection_respects_glob_include():
    """A glob include must start only its matching tools checked."""
    config = {
        "mcp_servers": {
            "cloudflare": {
                "command": "npx",
                "tools": {"include": ["dns_*"]},
            },
        }
    }
    tools = [
        ("purge_cache", "Purge"),
        ("dns_records", "DNS"),
        ("dns_zones", "Zones"),
    ]
    captured_pre_selected = {}

    def fake_checklist(title, labels, pre_selected, **kwargs):
        captured_pre_selected["value"] = set(pre_selected)
        return pre_selected  # No changes

    with patch(_PROBE, return_value={"cloudflare": tools}), \
         patch(_CHECKLIST, side_effect=fake_checklist), \
         patch(_SAVE):
        _configure_mcp_tools_interactive(config)

    assert captured_pre_selected["value"] == {1, 2}


def test_glob_exclude_survives_an_unrelated_edit():
    """Editing one tool must not silently re-enable glob-excluded tools.

    The checklist writes its selection back as ``tools.include`` and drops
    ``exclude``. When glob-excluded tools start (wrongly) checked, the user's
    next edit persists them as enabled — the filter is gone and every
    excluded tool floods back into the agent's tool surface.
    """
    config = {
        "mcp_servers": {
            "cloudflare": {
                "command": "npx",
                "tools": {"exclude": ["*_radar_*"]},
            },
        }
    }
    tools = [
        ("purge_cache", "Purge"),
        ("cf_radar_http", "Radar HTTP"),
        ("dns_records", "DNS"),
    ]

    def fake_checklist(title, labels, pre_selected, **kwargs):
        # User unchecks one unrelated tool (dns_records) and confirms.
        return set(pre_selected) - {2}

    with patch(_PROBE, return_value={"cloudflare": tools}), \
         patch(_CHECKLIST, side_effect=fake_checklist), \
         patch(_SAVE):
        _configure_mcp_tools_interactive(config)

    written = config["mcp_servers"]["cloudflare"]["tools"].get("include", [])
    assert "cf_radar_http" not in written, (
        "an unrelated edit re-enabled a glob-excluded tool"
    )
    assert written == ["purge_cache"]
