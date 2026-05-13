"""Tests for scripts/hermes-session-client.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "hermes-session-client.py"
)
_SPEC = importlib.util.spec_from_file_location("hermes_session_client", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
hermes_session_client = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hermes_session_client)


class _FakeClient:
    def __init__(self):
        self.closed = False
        self.cancelled = None
        self.loaded = None
        self.resumed = None
        self.forked = None
        self.prompt_calls = []

    def close(self):
        self.closed = True

    def new_session(self, cwd):
        return "sid-new"

    def load_session(self, session_id, cwd):
        self.loaded = (session_id, cwd)
        return session_id

    def resume_session(self, session_id, cwd):
        self.resumed = (session_id, cwd)
        return session_id

    def fork_session(self, session_id, cwd):
        self.forked = (session_id, cwd)
        return "sid-fork"

    def list_sessions(self, cwd=None):
        return {"sessions": [{"sessionId": "sid-1", "cwd": cwd or "/tmp"}]}

    def cancel(self, session_id):
        self.cancelled = session_id

    def prompt(self, session_id, text, timeout=180.0):
        self.prompt_calls.append((session_id, text, timeout))
        return {"text": "hello from hermes", "stop_reason": "end_turn"}


def test_select_auth_method_id_uses_preferred_match():
    init_result = {
        "authMethods": [
            {"id": "openrouter"},
            {"id": "anthropic"},
        ]
    }

    selected = hermes_session_client._select_auth_method_id(
        init_result,
        preferred_method="anthropic",
    )

    assert selected == "anthropic"


def test_select_auth_method_id_falls_back_to_first_method():
    init_result = {"authMethods": [{"id": "openrouter"}, {"id": "anthropic"}]}
    assert hermes_session_client._select_auth_method_id(init_result) == "openrouter"


def test_select_auth_method_id_returns_none_without_methods():
    assert hermes_session_client._select_auth_method_id({}) is None
    assert hermes_session_client._select_auth_method_id({"authMethods": []}) is None


def test_maybe_authenticate_skips_when_server_advertises_no_auth(monkeypatch):
    client = object.__new__(hermes_session_client.HermesACPSessionClient)
    calls = []
    monkeypatch.setattr(client, "_send_rpc", lambda *args, **kwargs: calls.append((args, kwargs)))

    method_id = hermes_session_client.HermesACPSessionClient.maybe_authenticate(
        client,
        {"authMethods": []},
    )

    assert method_id is None
    assert calls == []


def test_maybe_authenticate_uses_selected_method(monkeypatch):
    client = object.__new__(hermes_session_client.HermesACPSessionClient)
    calls = []
    monkeypatch.setattr(
        client,
        "_send_rpc",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {},
    )

    method_id = hermes_session_client.HermesACPSessionClient.maybe_authenticate(
        client,
        {"authMethods": [{"id": "openrouter"}]},
    )

    assert method_id == "openrouter"
    assert calls == [
        (
            ("authenticate", {"methodId": "openrouter", "args": {}}),
            {"timeout": 30.0},
        )
    ]


def test_main_new_prints_session_id(monkeypatch, capsys):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        hermes_session_client,
        "_initialize_session",
        lambda profile, cwd=None: (fake_client, "openrouter"),
    )

    exit_code = hermes_session_client.main(["researcher", "new", "/tmp/project"])

    out = capsys.readouterr()
    assert exit_code == 0
    assert out.out.strip() == "SESSION_ID:sid-new"
    assert fake_client.closed is True


def test_main_resume_reuses_session_id(monkeypatch, capsys):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        hermes_session_client,
        "_initialize_session",
        lambda profile, cwd=None: (fake_client, "openrouter"),
    )

    exit_code = hermes_session_client.main(["work", "resume", "sid-old", "/tmp/work"])

    out = capsys.readouterr()
    assert exit_code == 0
    assert out.out.strip() == "SESSION_ID:sid-old"
    assert fake_client.resumed == ("sid-old", "/tmp/work")
    assert fake_client.closed is True


def test_main_prompt_prints_response(monkeypatch, capsys):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        hermes_session_client,
        "_initialize_session",
        lambda profile, cwd=None: (fake_client, "openrouter"),
    )

    exit_code = hermes_session_client.main(
        ["work", "prompt", "sid-123", "Ship it", "42"]
    )

    out = capsys.readouterr()
    assert exit_code == 0
    assert out.out.strip() == "hello from hermes"
    assert fake_client.prompt_calls == [("sid-123", "Ship it", 42.0)]
    assert fake_client.closed is True


def test_main_list_prints_json(monkeypatch, capsys):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        hermes_session_client,
        "_initialize_session",
        lambda profile, cwd=None: (fake_client, None),
    )

    exit_code = hermes_session_client.main(["researcher", "list", "/tmp/context"])

    out = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(out.out)
    assert payload["sessions"][0]["sessionId"] == "sid-1"
    assert payload["sessions"][0]["cwd"] == "/tmp/context"
    assert fake_client.closed is True


def test_main_cancel_prints_confirmation(monkeypatch, capsys):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        hermes_session_client,
        "_initialize_session",
        lambda profile, cwd=None: (fake_client, None),
    )

    exit_code = hermes_session_client.main(["researcher", "cancel", "sid-cancel"])

    out = capsys.readouterr()
    assert exit_code == 0
    assert out.out.strip() == "CANCELLED:sid-cancel"
    assert fake_client.cancelled == "sid-cancel"
    assert fake_client.closed is True


def test_main_unknown_command_returns_error(capsys):
    exit_code = hermes_session_client.main(["researcher", "nope"])
    out = capsys.readouterr()
    assert exit_code == 1
    assert "Unknown command" in out.err
