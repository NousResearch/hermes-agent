"""Credential recovery for named custom providers flattened by a task-level base_url (#34651).

The desktop GUI persists ``auxiliary.<task> = {provider: custom:<name>, model, base_url}`` (#65254).
``_resolve_task_provider_model`` flattens ``custom:<name>`` + task ``base_url`` to bare ``custom`` before
the named branch can read the entry's key, so the credential chain degraded to OPENAI_API_KEY →
main-runtime-same-host → ``no-key-required`` — a 401 on every aux call once the main model moved off
that host. ``_resolve_custom_branch`` must now lift the configured entry's own key from the URL
(``find_custom_provider_entry``), and the sibling gaps close with it: the named branch honoring an
explicit per-task api_key (#96232), and the vision fallthrough forwarding the resolved key.
"""

import pytest

import hermes_cli.runtime_provider as rp
import agent.auxiliary_client as aux


def _entry_config():
    return {
        "custom_providers": [
            {
                "name": "Local (127.0.0.1:8888)",
                "base_url": "http://127.0.0.1:8888/v1",
                "api_key": "sk-entry-key",
                "model": "qwen-test",
            }
        ]
    }


def test_find_custom_provider_entry_lifts_full_entry(monkeypatch):
    """URL → full entry (not just the identity slug), with the credential intact."""
    monkeypatch.setattr(rp, "load_config", lambda: _entry_config())
    entry = rp.find_custom_provider_entry("http://127.0.0.1:8888/v1")
    assert entry is not None
    assert entry["api_key"] == "sk-entry-key"
    assert entry["base_url"] == "http://127.0.0.1:8888/v1"


def test_find_custom_provider_entry_no_match_returns_none(monkeypatch):
    monkeypatch.setattr(rp, "load_config", lambda: _entry_config())
    assert rp.find_custom_provider_entry("https://elsewhere.example/v1") is None
    assert rp.find_custom_provider_entry("") is None
    assert rp.find_custom_provider_entry(None) is None


def test_find_custom_provider_entry_matches_providers_dict(monkeypatch):
    config = {"providers": {"local": {"api": "http://127.0.0.1:8000/v1", "api_key": "sk-keyed"}}}
    monkeypatch.setattr(rp, "load_config", lambda: config)
    entry = rp.find_custom_provider_entry("http://127.0.0.1:8000/v1")
    assert entry is not None
    assert entry["api_key"] == "sk-keyed"


def test_custom_branch_uses_entry_key_for_task_base_url(monkeypatch):
    """The #34651 shape end-to-end: named provider flattened to bare custom by the task base_url
    still authenticates with the configured entry's key, not the no-key-required placeholder."""
    monkeypatch.setattr(aux, "_scoped_key_env", lambda *_a, **_k: "")
    monkeypatch.setattr(aux, "_read_main_api_key_if_same_host", lambda *_a, **_k: "")
    monkeypatch.setattr(rp, "load_config", lambda: _entry_config())

    req = aux._ResolveRequest(
        provider="custom",
        original_provider="custom:local-(127.0.0.1:8888)",
        model="qwen-test",
        async_mode=False,
        raw_codex=False,
        explicit_base_url="http://127.0.0.1:8888/v1",
        explicit_api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=True,
        task="vision",
    )
    client, final_model = aux._resolve_custom_branch(req)
    assert client is not None
    assert final_model == "qwen-test"
    assert getattr(client, "api_key", "") == "sk-entry-key"


def test_custom_branch_explicit_key_outranks_entry(monkeypatch):
    """An explicit per-task api_key (auxiliary.<task>.api_key) wins over the entry's stored key."""
    monkeypatch.setattr(aux, "_scoped_key_env", lambda *_a, **_k: "")
    monkeypatch.setattr(rp, "load_config", lambda: _entry_config())

    req = aux._ResolveRequest(
        provider="custom",
        original_provider="custom:local-(127.0.0.1:8888)",
        model="qwen-test",
        async_mode=False,
        raw_codex=False,
        explicit_base_url="http://127.0.0.1:8888/v1",
        explicit_api_key="sk-explicit-per-task",
        api_mode=None,
        main_runtime=None,
        is_vision=True,
        task="vision",
    )
    client, _ = aux._resolve_custom_branch(req)
    assert getattr(client, "api_key", "") == "sk-explicit-per-task"


def test_custom_branch_without_matching_entry_keeps_placeholder(monkeypatch):
    """Unchanged fallback: an explicit base_url with NO configured entry still degrades to the
    placeholder (local servers without auth) — the new lookup must not invent credentials."""
    monkeypatch.setattr(aux, "_scoped_key_env", lambda *_a, **_k: "")
    monkeypatch.setattr(aux, "_read_main_api_key_if_same_host", lambda *_a, **_k: "")
    monkeypatch.setattr(rp, "load_config", lambda: {"custom_providers": []})

    req = aux._ResolveRequest(
        provider="custom",
        original_provider="custom",
        model="m",
        async_mode=False,
        raw_codex=False,
        explicit_base_url="http://10.0.0.5:9000/v1",
        explicit_api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=False,
        task="vision",
    )
    client, _ = aux._resolve_custom_branch(req)
    assert getattr(client, "api_key", "") == "no-key-required"


def test_named_branch_honors_explicit_api_key(monkeypatch):
    """#96232: the named branch must not drop an explicit per-task api_key when the entry's own
    api_key field is empty (credentials deliberately stored elsewhere)."""
    config = {
        "custom_providers": [
            {"name": "entry-no-key", "base_url": "https://api.example.test/v1", "model": "m"}
        ]
    }
    monkeypatch.setattr(rp, "load_config", lambda: config)

    req = aux._ResolveRequest(
        provider="entry-no-key",
        original_provider="custom:entry-no-key",
        model="m",
        async_mode=False,
        raw_codex=False,
        explicit_base_url=None,
        explicit_api_key="sk-per-task-96232",
        api_mode=None,
        main_runtime=None,
        is_vision=True,
        task="vision",
    )
    result = aux._resolve_named_custom_branch(req)
    assert result is not None
    client = result[0]
    assert getattr(client, "api_key", "") == "sk-per-task-96232"
