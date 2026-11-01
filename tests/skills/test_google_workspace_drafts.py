"""Gmail draft-by-default safety tests for the google-workspace skill.

The skill's Rule 1 is "NEVER SEND EMAIL. Only create Gmail DRAFTS." Until these
tests passed, the CLI had no draft command at all, so an agent obeying the rule
had no affordance to obey it *with* -- its only Gmail write paths were `send`
and `reply`, both of which actually send. These tests pin the two halves of the
structural fix: the safe path must exist, and the unsafe path must be gated.
"""

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


API_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/google_api.py"
)
SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/SKILL.md"
)


@pytest.fixture
def api_module(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    spec = importlib.util.spec_from_file_location("gws_api_drafts_test", API_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # Force the gws CLI path so we never import googleapiclient (not a test dep).
    module._gws_binary = lambda: "/usr/bin/gws"
    module._ensure_authenticated = lambda: None
    return module


def _capture(stdout='{"id": "draft-1", "message": {"id": "m1", "threadId": "t1"}}'):
    """Return (recorder, fake subprocess.run) capturing the gws argv and body."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "input": kwargs.get("input", "")})
        return MagicMock(returncode=0, stdout=stdout, stderr="")

    return calls, fake_run


def _draft_args(**over):
    base = dict(
        to="alice@example.com",
        subject="Re: budget",
        body="Looks good to me.",
        cc="",
        from_header="",
        html=False,
        thread_id="",
        reply_to="",
    )
    base.update(over)
    return argparse.Namespace(**base)


def _send_args(**over):
    base = dict(
        to="alice@example.com",
        subject="Re: budget",
        body="Looks good to me.",
        cc="",
        from_header="",
        html=False,
        thread_id="",
        confirm_send=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


# --- the safe path must exist -------------------------------------------------


def test_gmail_draft_function_exists(api_module):
    """There is a draft entry point at all."""
    assert hasattr(api_module, "gmail_draft"), (
        "google_api.py has no gmail_draft -- Rule 1 mandates drafts but the CLI "
        "offers no way to create one"
    )


def test_gmail_draft_creates_draft_and_never_sends(api_module):
    """gmail draft hits drafts.create, and never the send endpoint."""
    calls, fake_run = _capture()
    with patch.object(api_module.subprocess, "run", side_effect=fake_run):
        api_module.gmail_draft(_draft_args())

    assert len(calls) == 1, "expected exactly one API call"
    cmd = calls[0]["cmd"]
    assert "drafts" in cmd, f"draft must use the drafts resource, got {cmd}"
    assert "create" in cmd, f"draft must call create, got {cmd}"
    assert "send" not in cmd, f"draft must NEVER call send, got {cmd}"


def test_gmail_draft_reports_draft_status(api_module, capsys):
    """Output says 'drafted', not 'sent', so the agent cannot misreport it."""
    calls, fake_run = _capture()
    with patch.object(api_module.subprocess, "run", side_effect=fake_run):
        api_module.gmail_draft(_draft_args())

    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "drafted"
    assert "sent" not in json.dumps(out).lower()


def test_gmail_draft_reply_threads_correctly(api_module):
    """--reply-to threads the draft onto the original message."""
    calls, fake_run = _capture(
        stdout='{"id": "d2", "message": {"id": "m2", "threadId": "thread-99"}}'
    )
    with patch.object(api_module.subprocess, "run", side_effect=fake_run):
        api_module.gmail_draft(_draft_args(thread_id="thread-99"))

    payload = json.dumps(calls[0])
    assert "thread-99" in payload, "threadId must be carried into the draft body"


# --- the unsafe path must be gated -------------------------------------------


def test_gmail_send_refuses_without_confirmation(api_module):
    """send without explicit confirmation aborts and makes no API call."""
    calls, fake_run = _capture()
    with patch.object(api_module.subprocess, "run", side_effect=fake_run):
        with pytest.raises(SystemExit) as exc:
            api_module.gmail_send(_send_args(confirm_send=False))

    assert exc.value.code != 0, "refusal must be a non-zero exit"
    assert calls == [], "no send may reach the API without confirmation"


def test_gmail_send_refusal_names_the_alternative(api_module, capsys):
    """The refusal tells the agent what to do instead -- else it will retry."""
    with patch.object(api_module.subprocess, "run", side_effect=_capture()[1]):
        with pytest.raises(SystemExit):
            api_module.gmail_send(_send_args(confirm_send=False))

    err = capsys.readouterr().err.lower()
    assert "draft" in err, "refusal must point at the draft command"


def test_gmail_send_proceeds_with_confirmation(api_module):
    """The escape hatch still works once the user has approved."""
    calls, fake_run = _capture(stdout='{"id": "m3", "threadId": "t3"}')
    with patch.object(api_module.subprocess, "run", side_effect=fake_run):
        api_module.gmail_send(_send_args(confirm_send=True))

    assert len(calls) == 1
    assert "send" in calls[0]["cmd"]


def test_gmail_reply_refuses_without_confirmation(api_module):
    """reply is a send too, and is gated identically."""
    calls, fake_run = _capture()
    args = argparse.Namespace(
        message_id="m1", body="ok", from_header="", confirm_send=False
    )
    with patch.object(api_module.subprocess, "run", side_effect=fake_run):
        with pytest.raises(SystemExit) as exc:
            api_module.gmail_reply(args)

    assert exc.value.code != 0
    assert calls == [], "no reply may reach the API without confirmation"


def test_refusal_exit_code_is_distinct_from_argparse(api_module):
    """argparse uses 2; a refusal must be tellable apart from a usage error."""
    with patch.object(api_module.subprocess, "run", side_effect=_capture()[1]):
        with pytest.raises(SystemExit) as exc:
            api_module.gmail_send(_send_args(confirm_send=False))
    assert exc.value.code == 3


# --- the documented surface must match the real one ---------------------------


def test_skill_documents_the_draft_command(api_module):
    """SKILL.md must show the draft command it mandates in Rule 1."""
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert re.search(r"gmail draft\b", text), (
        "SKILL.md mandates drafts but never documents a draft command"
    )


def test_skill_documents_the_send_gate(api_module):
    """SKILL.md must document the confirmation flag, not just forbid sending."""
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "--confirm-send" in text, "the send gate must be documented"
