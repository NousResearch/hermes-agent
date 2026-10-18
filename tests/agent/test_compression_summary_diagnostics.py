"""Compression route attribution regressions for #72636/#72637."""

from __future__ import annotations

import json
import logging
import time
from unittest.mock import patch

import pytest

from agent.context_compressor_attribution import SummaryRoute
from tests.agent.compression_route_helpers import (
    _MAIN_BASE_URL, _MAIN_MODEL, _MAIN_PROVIDER, _AUX_PROVIDER, _AUX_MODEL, _AUX_BASE_URL,
)


def _build_summary_compressor():
    """Real ContextCompressor initialized against the MAIN runtime (production shape:
    summary_model empty — agent_init passes summary_model_override=None)."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        from agent.context_compressor import ContextCompressor

        return ContextCompressor(
            model=_MAIN_MODEL, base_url=_MAIN_BASE_URL, provider=_MAIN_PROVIDER,
            quiet_mode=True, protect_first_n=2, protect_last_n=2,
            abort_on_summary_failure=False,
        )


def _fake_summary_response(content: str, finish_reason: str = "stop") -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def _aux_route_writing_call_llm(response_factory):
    """Fake call_llm that records the route it 'selected' into route_info — exactly
    what the real call_llm does via _record_route_info before each physical attempt."""
    def _fake(*_args, **kwargs):
        route_info = kwargs.get("route_info")
        if route_info is not None:
            route_info["provider"] = _AUX_PROVIDER
            route_info["model"] = _AUX_MODEL
        kwargs["route_callback"](_AUX_PROVIDER, _AUX_MODEL, _AUX_BASE_URL)
        return response_factory()
    return _fake


def test_empty_content_error_names_aux_route_not_main() -> None:
    """The empty-content RuntimeError (proxy 200 with blank body) must name the
    aux route the summary call actually used, never the main model (#113582)."""
    comp = _build_summary_compressor()

    fake = _aux_route_writing_call_llm(lambda: _fake_summary_response("   "))
    with patch("agent.context_compressor.call_llm", side_effect=fake):
        with pytest.raises(RuntimeError, match="returned empty content") as excinfo:
            comp._call_summary_llm("summarize this", time.monotonic())

    msg = str(excinfo.value)
    assert f"provider={_AUX_PROVIDER}" in msg, f"aux provider missing from: {msg}"
    assert f"model={_AUX_MODEL}" in msg, f"aux model missing from: {msg}"
    assert _MAIN_MODEL not in msg, f"main model leaked into aux failure: {msg}"


def test_truncated_summary_error_names_aux_route() -> None:
    """The length-truncated (PARTIAL summary) RuntimeError names the aux route too —
    same ``where`` string, same misattribution class (#72637 review / #113582)."""
    from agent.context_compressor import _TRUNCATED_SUMMARY_MARKER

    comp = _build_summary_compressor()

    fake = _aux_route_writing_call_llm(
        lambda: _fake_summary_response("A partial summary that hit", finish_reason="length")
    )
    with patch("agent.context_compressor.call_llm", side_effect=fake):
        with pytest.raises(RuntimeError, match="truncated") as excinfo:
            comp._call_summary_llm("summarize this", time.monotonic())

    msg = str(excinfo.value)
    assert _TRUNCATED_SUMMARY_MARKER in msg
    assert f"provider={_AUX_PROVIDER}" in msg, f"aux provider missing from: {msg}"
    assert f"model={_AUX_MODEL}" in msg, f"aux model missing from: {msg}"
    assert _MAIN_MODEL not in msg, f"main model leaked into aux failure: {msg}"


def test_summary_failure_non_json_log_names_aux_route(caplog) -> None:
    """The non-JSON-response logger.error in _on_summary_failure must name the
    recorded aux identity instead of the permanently-empty summary_model, which
    logged the MAIN provider and "(main)" for an auxiliary failure."""
    comp = _build_summary_compressor()
    # Post-dispatch failure state, as _call_summary_llm leaves it: the wire
    # identity was recorded by the route_callback before the call failed.
    comp._last_aux_route = SummaryRoute(_AUX_PROVIDER, _AUX_MODEL, _AUX_BASE_URL, dispatched=True)
    # Skip the one-shot main-model retry arm; this test pins the log identity.
    comp._summary_model_fallen_back = True

    err = json.JSONDecodeError("Expecting value", "doc", 0)
    with caplog.at_level(logging.ERROR, logger="agent.context_compressor"):
        comp._on_summary_failure(err, [{"role": "user", "content": "x"}], None, "")

    rendered = "\n".join(r.getMessage() for r in caplog.records)
    assert "non-JSON response" in rendered, f"expected log line not captured: {caplog.records}"
    assert _AUX_PROVIDER in rendered, f"aux provider missing from log: {rendered}"
    assert _AUX_MODEL in rendered, f"aux model missing from log: {rendered}"
    assert _AUX_BASE_URL in rendered, f"aux endpoint missing from log: {rendered}"
    assert "(main)" not in rendered, f"phantom '(main)' summary model in log: {rendered}"
    assert _MAIN_PROVIDER not in rendered, f"main provider misattributed in log: {rendered}"


def test_config_aux_route_gets_one_shot_main_retry(caplog) -> None:
    """KeyArgo's second observation: the one-shot main-model retry gate used to
    require a truthy ``self.summary_model`` — permanently empty in production —
    so a config-driven auxiliary.compression route could never get the retry and
    dropped straight to cooldown/abort. Current main routes the gate through
    ``_last_aux_resolved_model`` (#116472); this pins that the retry FIRES for
    a resolved aux route distinct from the main model."""
    comp = _build_summary_compressor()
    assert comp.summary_model == ""  # production shape: no static summary model

    dispatches: list[dict] = []

    def _fake(*_args, **kwargs):
        dispatches.append(dict(kwargs))
        route_info = kwargs.get("route_info")
        if route_info is not None:
            route_info["provider"] = _AUX_PROVIDER
            route_info["model"] = _AUX_MODEL
        if len(dispatches) == 1:
            # Generic transient failure on the aux route (not auth: that path
            # is terminal and must not consume the retry).
            raise RuntimeError("500 Internal Server Error from aux route")
        return _fake_summary_response("Completed: the task ran. Active State: done.")

    comp._clear_compression_failure_cooldown()
    with patch("agent.context_compressor.call_llm", side_effect=_fake), caplog.at_level(
        logging.WARNING, logger="agent.context_compressor"
    ):
        summary = comp._generate_summary(
            [{"role": "user", "content": "please do the thing"},
             {"role": "assistant", "content": "doing it"}],
            focus_topic=None, memory_context="",
        )

    # First attempt failed on the aux route; the gate must have fired the
    # one-shot main-model fallback and retried (second dispatch), and the
    # fallback must name the FAILED aux model — not "" / "(auto)".
    assert len(dispatches) == 2, f"one-shot main-model retry did not fire: {len(dispatches)} dispatches"
    assert summary, f"retry did not produce a summary: {summary!r}"
    assert comp._last_aux_model_failure_model == _AUX_MODEL, (
        f"fallback did not name the failed aux model: {comp._last_aux_model_failure_model!r}"
    )
    rendered = "\n".join(r.getMessage() for r in caplog.records)
    assert _AUX_MODEL in rendered, f"fallback warning does not name the aux model: {rendered}"
