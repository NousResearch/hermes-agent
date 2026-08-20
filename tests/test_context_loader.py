import importlib

import pytest
from datetime import datetime, timezone
from unittest.mock import patch

from agent.modules.budgeting import budget_for_intent
from agent.modules.context_loader import assemble_context, UserMessage, ContextPackage
from agent.modules.identity import IdentityPacket


def test_budget_for_intent():
    # 3 intent fixtures produce 3 budgets
    assert budget_for_intent("ANSWER_DIRECTLY", "cloud") == 1024
    assert budget_for_intent("SUBMIT_OPENCLAW_JOB", "cloud") == 6144
    assert budget_for_intent("DELEGATE_SPECIALIST", "local") == 2048


def test_hydration_populates_panels():
    msg = UserMessage(text="hello", session_id="session-1")
    ident = IdentityPacket(
        user_id="user-1",
        company_id="comp-1",
        mode="enterprise",
        time=datetime.now(tz=timezone.utc),
    )
    
    with patch("agent.modules.context_loader._fetch_session_history", return_value=[{"role": "user", "text": "foo"}]), \
         patch("agent.modules.context_loader._fetch_memory_snippets", return_value=["mem1", "mem2"]), \
         patch("agent.modules.context_loader._fetch_active_tasks", return_value=[{"id": "task1"}]):
        
        # Give a large budget to keep everything
        pkg = assemble_context(msg, ident, token_budget=10000)
        
        assert len(pkg.session_history) == 1
        assert len(pkg.memory_snippets) == 2
        assert len(pkg.active_tasks) == 1
        assert pkg.company_context == {"company_id": "comp-1"}
        assert pkg.token_estimate > 0


def test_over_budget_triggers_drop():
    msg = UserMessage(text="hello", session_id="session-1")
    ident = IdentityPacket(
        user_id="user-1",
        company_id="comp-1",
        mode="enterprise",
        time=datetime.now(tz=timezone.utc),
    )
    
    # We will simulate a very long active_task that exceeds budget to prove dropping works
    # ~2500 tokens
    long_task = {"data": "X" * 10000}
    
    with patch("agent.modules.context_loader._fetch_session_history", return_value=[{"role": "user", "text": "foo"}]), \
         patch("agent.modules.context_loader._fetch_memory_snippets", return_value=["mem1"]), \
         patch("agent.modules.context_loader._fetch_active_tasks", return_value=[long_task]):
        
        # 1000 budget is smaller than the active task size
        pkg = assemble_context(msg, ident, token_budget=1000)
        
        # Company context, history, and memory fit within 1000 tokens
        # The long task takes 2500 tokens, so it will be dropped.
        assert pkg.company_context == {"company_id": "comp-1"}
        assert len(pkg.session_history) == 1
        assert len(pkg.memory_snippets) == 1
        assert len(pkg.active_tasks) == 0  # DROPPED due to budget


def test_default_token_budget_is_8192(monkeypatch):
    monkeypatch.delenv("MEMORY_RETRIEVAL_BUDGET_TOKENS", raising=False)
    import agent.modules.context_loader as cl
    importlib.reload(cl)
    assert cl.DEFAULT_TOKEN_BUDGET == 8192


def test_token_budget_reads_from_env(monkeypatch):
    monkeypatch.setenv("MEMORY_RETRIEVAL_BUDGET_TOKENS", "2000")
    import agent.modules.context_loader as cl
    importlib.reload(cl)
    assert cl.DEFAULT_TOKEN_BUDGET == 2000
