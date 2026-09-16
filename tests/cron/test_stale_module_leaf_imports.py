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
