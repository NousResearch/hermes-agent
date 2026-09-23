"""Regression tests for lazy consumers importing from stale cached modules.

The scheduled cron lane constructs a fresh agent inside a long-lived gateway.
These tests model the field failure directly: a foundational module remains in
``sys.modules`` but lacks a symbol added by newer consumer code on disk.
"""

from __future__ import annotations

import importlib
import logging
import sys
from types import SimpleNamespace

import pytest


def test_primary_client_ignores_stale_auxiliary_router(monkeypatch):
    from agent import agent_runtime_helpers, auxiliary_client

    # Model a gateway that cached auxiliary_client before the Codex header
    # helper was added. The fresh runtime helper must use the leaf module rather
    # than asking this stale module object for the new export.
    monkeypatch.delattr(auxiliary_client, "_apply_required_codex_headers")

    captured: dict = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    from agent import process_bootstrap

    monkeypatch.setattr(process_bootstrap, "OpenAI", fake_openai)
    monkeypatch.setattr(
        agent_runtime_helpers,
        "_ra",
        lambda: SimpleNamespace(logger=logging.getLogger(__name__)),
    )
    agent = SimpleNamespace(
        provider="openai-codex",
        _build_keepalive_http_client=lambda *_args, **_kwargs: None,
        _client_log_context=lambda: "test",
    )

    agent_runtime_helpers.create_openai_client(
        agent,
        {
            "api_key": "token",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
        reason="test",
        shared=False,
    )

    assert captured["default_headers"]["originator"] == "hermes-agent"


def test_docker_import_ignores_stale_base_environment(monkeypatch):
    from tools.environments import base
    from tools.environments.path_utils import sanitize_task_id_for_path

    # base.py no longer defines the sanitizer at all (it lives in the leaf module); a Docker
    # module imported later by tool discovery must bind the helper from that leaf, never
    # from base's namespace.
    assert "sanitize_task_id_for_path" not in vars(base)
    previous = sys.modules.pop("tools.environments.docker", None)
    try:
        docker = importlib.import_module("tools.environments.docker")
        assert docker._sandbox_dir_name is sanitize_task_id_for_path
        assert docker._sandbox_dir_name("session:cron:job") == sanitize_task_id_for_path(
            "session:cron:job"
        )
    finally:
        sys.modules.pop("tools.environments.docker", None)
        if previous is not None:
            sys.modules["tools.environments.docker"] = previous


def test_chat_completion_helpers_binds_shim_from_leaf():
    """``chat_completion_helpers`` binds the router-timeout-shim predicates from the
    leaf module, never from ``agent.transports.chat_completions``.

    The predicates used to live in ``chat_completions`` ~110 lines below that heavy
    module's own top-level imports, and ``chat_completion_helpers`` imported them eagerly
    at module load (``from agent.transports.chat_completions import ...``). A long-lived
    gateway that holds a stale ``chat_completions`` in ``sys.modules`` (or observes it
    before the symbol is bound) then fails that eager import with
    ``ImportError: cannot import name 'is_router_timeout_shim'``. The predicates now live
    in the leaf ``agent.transports.router_timeout_shim``, which has no heavy imports, so
    the consumer binds them atomically.
    """
    from agent import chat_completion_helpers as cch
    from agent.transports import router_timeout_shim

    assert cch.is_router_timeout_shim is router_timeout_shim.is_router_timeout_shim
    assert cch.router_timeout_shim_may_follow is router_timeout_shim.router_timeout_shim_may_follow


def test_chat_completions_no_longer_reexports_shim():
    """The heavy transport no longer re-exports the predicates.

    Internal paths are not API (root AGENTS.md: "No re-export shims for internal moves"),
    so the only import path for the predicates is the defining leaf module. Importing them
    from ``agent.transports.chat_completions`` must raise ``ImportError``.
    """
    with pytest.raises(ImportError):
        from agent.transports.chat_completions import is_router_timeout_shim  # noqa: F401


def test_auxiliary_client_binds_shim_from_leaf(monkeypatch):
    """``auxiliary_client._validate_llm_response`` binds the predicate from the leaf at
    call time (lazy import), not from a re-export on the heavy transport.

    A monkeypatch on the leaf's ``is_router_timeout_shim`` is observed by the lazy import
    only if the function imports from the leaf; the old re-export bound the object at
    import time and would not be intercepted.
    """
    from agent import auxiliary_client
    from agent.transports import router_timeout_shim

    seen: list = []
    monkeypatch.setattr(
        router_timeout_shim,
        "is_router_timeout_shim",
        lambda response: seen.append(response) or False,
    )

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))],
        usage=SimpleNamespace(completion_tokens=1),
        model="m",
    )
    assert auxiliary_client._validate_llm_response(response, "title") is response
    assert seen == [response]
