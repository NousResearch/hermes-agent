"""Kanban specify/decompose run headless (no agent turn), yet their auxiliary calls must still carry a
relay-affinity key — the OpenCode Go relay rejects a request without ``x-opencode-session`` with
400 MissingSessionID (#112043). ``_call_aux`` declares a per-task affinity scope unless one is bound."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import kanban_decompose as decompose
from hermes_cli import kanban_specify as specify


def _capturing_call_llm(seen: list):
    def call_llm(**kwargs):
        from agent.opencode_affinity import opencode_session_headers
        seen.append(opencode_session_headers("opencode-go", None).get("x-opencode-session"))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    return call_llm


@pytest.mark.parametrize("caller", [specify._call_aux, decompose._call_aux])
def test_headless_kanban_aux_call_declares_a_stable_per_task_affinity_key(caller):
    from agent.portal_tags import get_affinity_scope
    seen: list = []
    with patch("agent.auxiliary_client.call_llm", _capturing_call_llm(seen)):
        for task_id in ("t_123", "t_123", "t_456"):
            reply, reason = caller(
                "specify", task_id, aux_task="triage_specifier", system="s", user="u",
                max_tokens=10, timeout=5)
            assert (reply, reason) == ("ok", "")
    assert seen == ["kanban:t_123", "kanban:t_123", "kanban:t_456"]
    assert get_affinity_scope() is None  # nothing leaks past the call


def test_in_turn_caller_keeps_its_declared_affinity_key():
    from agent.portal_tags import reset_affinity_scope, set_affinity_scope
    seen: list = []
    token = set_affinity_scope("conversation-root")
    try:
        with patch("agent.auxiliary_client.call_llm", _capturing_call_llm(seen)):
            specify._call_aux("specify", "t_123", aux_task="triage_specifier", system="s", user="u",
                              max_tokens=10, timeout=5)
    finally:
        reset_affinity_scope(token)
    assert seen == ["conversation-root"]


def _dashboard_plugin_api():
    mod_name = "hermes_dashboard_plugin_kanban_aux_affinity_test"
    if mod_name not in sys.modules:
        plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
        spec = importlib.util.spec_from_file_location(mod_name, plugin_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


def test_dashboard_estimate_declares_an_affinity_key_too():
    """The dashboard estimate endpoints are the same headless aux call: per task when one exists,
    one stable key for the create dialog (no task yet)."""
    from agent.portal_tags import get_affinity_scope
    api = _dashboard_plugin_api()
    seen: list = []
    with patch("agent.auxiliary_client.call_llm", _capturing_call_llm(seen)):
        api._run_estimate("title", "body", task_id="t_1")
        api._run_estimate("title", "body", task_id=None)
    assert seen == ["kanban:t_1", "kanban:estimate"]
    assert get_affinity_scope() is None
