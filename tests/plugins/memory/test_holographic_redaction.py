from __future__ import annotations

import json

import pytest

from agent.redact import redact_sensitive_text
from plugins.memory.holographic import HolographicMemoryProvider


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    p = HolographicMemoryProvider(
        config={
            "db_path": str(tmp_path / "memory_store.db"),
            "default_trust": 0.5,
            "min_trust_threshold": 0.0,
            "auto_extract": False,
        }
    )
    p.initialize("session-1")
    return p


@pytest.mark.parametrize(
    "text",
    [
        "password=FAKE_SECRET_123",
        "admin_password = FAKE_SECRET_123",
        "auth: FAKE_TOKEN_123",
        "Authorization: Bearer FAKE_TOKEN_123",
    ],
)
def test_redact_sensitive_text_covers_freeform_assignments(text):
    redacted = redact_sensitive_text(text)

    assert "FAKE_SECRET_123" not in redacted
    assert "FAKE_TOKEN_123" not in redacted
    assert "[REDACTED]" in redacted or "***" in redacted


@pytest.mark.parametrize(
    ("action", "result_key", "expect_plain"),
    [
        ("list", "facts", True),
        ("search", "results", False),
    ],
)
def test_fact_store_output_redacts_sensitive_content(provider, action, result_key, expect_plain):
    secret = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX"
    plain = "Jordan likes concise docs."
    provider._store.add_fact(f"Project alpha uses key {secret}", category="project")
    provider._store.add_fact(plain, category="user_pref")

    args = {"action": action, "limit": 10}
    if action == "search":
        args["query"] = "Project"

    raw = provider.handle_tool_call("fact_store", args)
    payload = json.loads(raw)

    assert secret not in raw
    assert payload["count"] >= 1
    assert result_key in payload

    entries = payload[result_key]
    if expect_plain:
        assert any(item.get("content") == plain for item in entries)

    secret_entry = next(item for item in entries if item.get("category") == "project")
    assert secret not in secret_entry["content"]
    assert secret_entry["fact_id"]
    assert secret_entry["category"] == "project"
    assert "trust_score" in secret_entry
    assert "created_at" in secret_entry
    assert "updated_at" in secret_entry


def test_prefetch_redacts_sensitive_content(provider):
    secret = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX"
    provider._store.add_fact(f"Project alpha uses key {secret}", category="project")
    provider._store.add_fact("Jordan likes concise docs.", category="user_pref")

    output = provider.prefetch("Project")

    assert secret not in output
    assert "Project alpha uses key" in output
    assert "## Holographic Memory" in output


def test_fact_store_output_redacts_freeform_assignment_content(provider):
    provider._store.add_fact(
        "Open WebUI admin credentials: password=FAKE_SECRET_123 admin_password: FAKE_SECRET_123 auth=FAKE_TOKEN_123 Authorization: Bearer FAKE_TOKEN_123",
        category="project",
    )

    raw_list = provider.handle_tool_call("fact_store", {"action": "list", "limit": 10})
    raw_search = provider.handle_tool_call("fact_store", {"action": "search", "query": "Open WebUI", "limit": 10})
    prefetch = provider.prefetch("Open WebUI")

    assert "FAKE_SECRET_123" not in raw_list
    assert "FAKE_TOKEN_123" not in raw_list
    assert "FAKE_SECRET_123" not in raw_search
    assert "FAKE_TOKEN_123" not in raw_search
    assert "FAKE_SECRET_123" not in prefetch
    assert "FAKE_TOKEN_123" not in prefetch
    assert "password=[REDACTED]" in raw_list or "password: [REDACTED]" in raw_list
    assert "admin_password: [REDACTED]" in raw_list or "admin_password=[REDACTED]" in raw_list
    assert "Authorization: Bearer" in raw_list
