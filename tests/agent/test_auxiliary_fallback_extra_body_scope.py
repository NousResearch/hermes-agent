"""Tests for #138392 — ``auxiliary.<task>.extra_body`` must not ride onto a foreign fallback.

Task-level ``extra_body`` describes the primary endpoint's wire dialect (e.g.
``chat_template_kwargs`` for a local vLLM/llama.cpp server). Forwarded unchanged to a
``fallback_chain`` entry that resolves to a different provider/base_url it 400s the very
fallback that exists to cover the primary being down (Anthropic Messages rejects
``chat_template_kwargs`` as an extra input). The provider-level ``extra_body`` path already
resolves per destination (``_with_custom_endpoint_extra_body``); these tests pin the same
scoping for the task-level body on the fallback wire: keys are dropped when the candidate
resolves elsewhere, caller keys still travel, and a fallback entry may re-declare its own
``extra_body``.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.auxiliary_client import _call_fallback_candidate_sync


_LOCAL_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
_LOCAL_BASE = "http://127.0.0.1:18000/v1"


def _patch_task_config(chain):
    return patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value={"fallback_chain": chain},
    )


def _fake_client(seen):
    class _FakeCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    return SimpleNamespace(
        base_url="https://fallback.example.invalid/v1",
        chat=SimpleNamespace(completions=_FakeCompletions()),
    )


def _call(seen, chain, *, effective_extra_body, task_extra_body, label,
          primary_provider, primary_base_url):
    with _patch_task_config(chain):
        return _call_fallback_candidate_sync(
            _fake_client(seen), "fallback-model", label,
            task="explain_command", messages=[{"role": "user", "content": "hi"}],
            temperature=None, max_tokens=None, tools=None,
            effective_timeout=30.0, effective_extra_body=effective_extra_body,
            reasoning_config=None, task_extra_body=task_extra_body,
            primary_provider=primary_provider, primary_base_url=primary_base_url,
        )


def test_foreign_provider_fallback_drops_task_extra_body():
    """The issue's shape: primary is a local server, fallback is another vendor — the
    server-specific keys must not reach the fallback request body."""
    seen = {}
    resp = _call(
        seen, [{"provider": "custom:other-vendor"}],
        effective_extra_body=dict(_LOCAL_BODY), task_extra_body=dict(_LOCAL_BODY),
        label="fallback_chain[0](custom:other-vendor)",
        primary_provider="custom:mylocal", primary_base_url=_LOCAL_BASE,
    )
    assert resp is not None
    # An empty routed body is omitted from the wire kwargs entirely.
    assert not seen.get("extra_body")


def test_same_endpoint_fallback_keeps_task_extra_body():
    """A fallback that resolves to the same provider + base_url as the primary still
    carries the task-level body: it speaks the same dialect."""
    seen = {}
    resp = _call(
        seen, [{"provider": "custom:mylocal", "base_url": _LOCAL_BASE}],
        effective_extra_body=dict(_LOCAL_BODY), task_extra_body=dict(_LOCAL_BODY),
        label="fallback_chain[0](custom:mylocal)",
        primary_provider="custom:mylocal", primary_base_url=_LOCAL_BASE,
    )
    assert resp is not None
    assert seen.get("extra_body") == _LOCAL_BODY


def test_same_provider_other_base_url_drops_task_extra_body():
    """Same provider name resolving to a different base_url is still a foreign endpoint:
    the body was written for the primary's server, not this one."""
    seen = {}
    resp = _call(
        seen, [{"provider": "custom:mylocal"}],  # no entry base_url -> fake client's base
        effective_extra_body=dict(_LOCAL_BODY), task_extra_body=dict(_LOCAL_BODY),
        label="fallback_chain[0](custom:mylocal)",
        primary_provider="custom:mylocal", primary_base_url=_LOCAL_BASE,
    )
    assert resp is not None
    # An empty routed body is omitted from the wire kwargs entirely.
    assert not seen.get("extra_body")


def test_fallback_entry_extra_body_overrides():
    """A fallback entry may re-declare its own ``extra_body``; it is applied on top of
    the stripped body for that destination."""
    seen = {}
    resp = _call(
        seen, [{"provider": "custom:other-vendor", "extra_body": {"anthropic_beta": "tools-2026"}}],
        effective_extra_body=dict(_LOCAL_BODY), task_extra_body=dict(_LOCAL_BODY),
        label="fallback_chain[0](custom:other-vendor)",
        primary_provider="custom:mylocal", primary_base_url=_LOCAL_BASE,
    )
    assert resp is not None
    assert seen.get("extra_body") == {"anthropic_beta": "tools-2026"}


def test_caller_keys_survive_the_strip():
    """Caller-passed keys travel regardless of the primary endpoint (explicit caller
    intent beats task-level scoping), including a caller override of a task-level key."""
    seen = {}
    body = {"caller_key": "keep", "chat_template_kwargs": {"caller": True}}
    task_body = {"chat_template_kwargs": {"task": True}}
    resp = _call(
        seen, [{"provider": "custom:other-vendor"}],
        effective_extra_body=body, task_extra_body=task_body,
        label="fallback_chain[0](custom:other-vendor)",
        primary_provider="custom:mylocal", primary_base_url=_LOCAL_BASE,
    )
    assert resp is not None
    assert seen.get("extra_body") == {"caller_key": "keep", "chat_template_kwargs": {"caller": True}}


def test_no_task_body_snapshot_keeps_current_behaviour():
    """Callers that do not pass the new scoping inputs (older callers, tests) keep the
    exact previous behaviour — including ``reasoning`` folded from ``reasoning_effort``,
    which is provider-generic and must never be dropped by endpoint scoping."""
    seen = {}
    body = {"reasoning": {"effort": "none"}, "chat_template_kwargs": {"enable_thinking": False}}
    with _patch_task_config([{"provider": "custom:other-vendor"}]):
        resp = _call_fallback_candidate_sync(
            _fake_client(seen), "fallback-model", "fallback_chain[0](custom:other-vendor)",
            task="explain_command", messages=[{"role": "user", "content": "hi"}],
            temperature=None, max_tokens=None, tools=None,
            effective_timeout=30.0, effective_extra_body=body, reasoning_config=None,
        )
    assert resp is not None
    assert seen.get("extra_body") == body
