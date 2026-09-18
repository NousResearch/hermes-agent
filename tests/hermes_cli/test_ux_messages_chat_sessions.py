"""Plain-language contracts for chat/session failure copy (CLI UX message campaign, cluster D).

Each test asserts the CONTRACT (what happened + the exact command pointer, raw detail demoted to a
``Details:`` line), never the whole string.
"""

import pytest

from hermes_cli.active_sessions import session_already_owned_message
from hermes_cli.cli_chat_error_copy import agent_init_failure_message, chat_error_response
from hermes_cli.cli_unknown_command import unknown_command_lines


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


# ── cli-02: failed model request in the chat panel ──────────────────────────

@pytest.mark.parametrize("exc, pointer, absent", [
    (_StatusError("HTTP 401: Invalid API key", 401), "hermes model", "HTTP 401"),
    (Exception("Error code: 402 - insufficient credits"), "/model", "402"),
    (Exception("HTTP 404: model not found"), "/model", "HTTP 404"),
    (_StatusError("rate limit exceeded", 429), "/model", "429"),
    (Exception("Unknown error"), "hermes doctor", "Unknown error"),
])
def test_chat_error_response_leads_with_plain_copy_and_pointer(exc, pointer, absent):
    text = chat_error_response(exc, provider="openrouter", model="foo/bar")
    lead = text.splitlines()[0]
    assert pointer in lead
    assert absent not in lead, "raw HTTP code/exception text must not be the lead sentence"
    assert not lead.startswith("Error:")
    assert "Details: " in text and str(exc) in text


def test_chat_error_response_names_provider_and_model():
    text = chat_error_response(Exception("HTTP 404: model not found"), provider="openrouter", model="foo/bar")
    assert "foo/bar" in text.splitlines()[0]
    assert "openrouter" in text.splitlines()[0]


def test_chat_error_response_accepts_plain_string_summary():
    text = chat_error_response("HTTP 401: Invalid API key", provider="nous", model="m")
    assert "hermes model" in text.splitlines()[0]


def test_chat_error_response_trusts_stamped_provider_verdict_over_reclassifying_text():
    # The loop stamped 'rate_limit'; the summarised text alone would classify as unknown.
    text = chat_error_response("upstream said no", provider="openrouter", model="m", failure_reason="rate_limit")
    assert "Rate limited" in text.splitlines()[0]
    assert "Details: upstream said no" in text


def test_chat_error_response_returns_site_copy_verbatim_instead_of_double_wrapping():
    curated = "The model's reply was cut off before it finished. Send `continue`."
    text = chat_error_response(curated, provider="openrouter", model="m", failure_reason="truncated")
    assert text == curated
    assert "Details:" not in text


# ── cli-06: agent could not be built on first message ──────────────────────

def test_agent_init_failure_message_says_message_not_sent_and_points_to_doctor():
    exc = RuntimeError("Unsupported api_mode 'weird' for provider x\nsecond line of traceback noise " + "x" * 300)
    text = agent_init_failure_message(exc)
    assert "not sent" in text
    assert "hermes doctor" in text and "/model" in text
    assert not text.startswith("Failed to initialize agent")
    assert "second line" not in text, "only the first line of the exception is shown"
    assert len(text) < 400, "the exception summary is truncated"


# ── cli-11: session held by another window ─────────────────────────────────

def test_owned_message_first_line_is_plain_and_details_follow():
    message = session_already_owned_message("session-1", {
        "surface": "desktop", "pid": 123, "started_at": 1,
    })
    first, *rest = message.splitlines()
    assert first == ("This chat is open in another Hermes window/terminal. "
                     "Use it there, or start a new chat here.")
    for jargon in ("lease", "pid", "owner", "takeover"):
        assert jargon not in first.lower()
    assert rest and rest[0].startswith("Details: ")
    assert "desktop" in rest[0]


# ── cli-30: hermes sessions with a bad id / unopenable DB ──────────────────

def test_sessions_not_found_points_to_list(capsys):
    from hermes_cli.sessions_cmd import _not_found
    assert _not_found("abc") == 1
    out = capsys.readouterr().out
    assert "No session 'abc'" in out
    assert "hermes sessions list" in out


def test_sessions_db_open_failure_points_to_repair(monkeypatch, capsys):
    import hermes_cli.sessions_cmd as sessions_cmd

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("database disk image is malformed")

    monkeypatch.setattr("hermes_state.SessionDB", _Boom)
    import hermes_state
    # `list` is read-only: a missing store prints "empty" instead; make the file exist so the
    # open failure is the real corrupt-database case this copy is for.
    db_path = hermes_state._default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"not a database")
    import argparse
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(sessions_action="list", session_id=None)
    code = sessions_cmd.cmd_sessions(args, parser)
    out = capsys.readouterr().out
    assert code == 1
    assert "hermes sessions repair" in out
    assert out.splitlines()[0].startswith("Could not open your session history database")
    assert "Details: database disk image is malformed" in out


# ── cli-31: unknown slash command ──────────────────────────────────────────

def test_unknown_command_suggests_close_match_and_says_nothing_sent():
    lead, pointer = unknown_command_lines("/modle", {"/model", "/help", "/quit"})
    assert "/modle" in lead
    assert "nothing was sent" in lead
    assert "Did you mean /model?" in lead
    assert "/help" in pointer


def test_unknown_command_without_close_match_has_no_suggestion():
    lead, pointer = unknown_command_lines("/zzzzzz", {"/model", "/help"})
    assert "Did you mean" not in lead
    assert "/help" in pointer
