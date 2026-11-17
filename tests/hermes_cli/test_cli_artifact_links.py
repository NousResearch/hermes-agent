"""The classic CLI prints a hosted artifact's viewer link after the call that made, changed or showed it."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tests.hermes_cli import test_tool_progress_scrollback as scrollback

VIEWER = "https://portal.example.com/a/01hx"
TICKET = "https://01hx.frame.example.com/__av?t=secret-ticket"
CREATE = "mcp__hermes_artifacts__artifact_create"


def _result(body: dict) -> str:
    return json.dumps({"result": json.dumps(body), "structuredContent": body})


OK = _result({"id": "01hx", "version": 1, "title": "Plan", "viewer_url": VIEWER, "content_url": TICKET})


def _printed(mode: str, calls) -> list[str]:
    cli = scrollback._make_cli(tool_progress=mode)
    with patch.object(scrollback._cli_mod, "_cprint") as cprint:
        for name, result in calls:
            cli._on_tool_progress("tool.started", name, "", {})
            cli._on_tool_progress("tool.completed", name, None, None, duration=0.2, is_error=False, result=result)
    return [str(call.args[0]) for call in cprint.call_args_list]


@pytest.mark.parametrize("mode", ["all", "new", "verbose", "off"])
def test_viewer_link_printed_in_every_progress_mode(mode):
    lines = _printed(mode, [(CREATE, OK)])
    links = [line for line in lines if VIEWER in line]
    assert len(links) == 1 and "\u201cPlan\u201d" in links[0]
    assert not any("secret-ticket" in line for line in lines)


def test_new_mode_repeat_still_prints_each_link():
    lines = _printed("new", [(CREATE, OK), (CREATE, OK)])
    assert sum(VIEWER in line for line in lines) == 2


@pytest.mark.parametrize("name, result", [
    (CREATE, json.dumps({"error": "stale_base: base_version 1 is not the head (2)"})),
    (CREATE, "not json"),
    ("terminal", OK),
])
def test_no_link_for_errors_or_other_tools(name, result):
    assert not any(VIEWER in line for line in _printed("all", [(name, result)]))
