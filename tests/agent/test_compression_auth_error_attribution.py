"""Regression for #72636: when auxiliary.compression fails with an auth
error (HTTP 401/403), the conversation-loop error display sites must
identify the *compression* model's provider/model/endpoint, not the main
model's.

Previously both error display sites (retry buffer and terminal abort)
used ``agent.provider`` / ``agent.base_url`` / ``agent.model`` — which
are always the main model — even when the actual failure came from the
compression auxiliary. Users saw the working main model reported as the
broken endpoint and chased the wrong fix.

The fix extracts a small helper in ``agent.conversation_loop``,
``_maybe_emit_compression_auth_hint(agent, *, force_vprint=False)``,
called from both error sites. Tests import the real helper — not a
copy — so the helper and the inline call sites cannot drift.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.conversation_loop import _maybe_emit_compression_auth_hint


def _make_compressor(
    *,
    auth_failure: bool,
    provider: str = "auto",
    summary_model: str = "",
    base_url: str = "",
) -> MagicMock:
    """Stand-in for ContextCompressor carrying only the fields the helper
    reads. Real compressor construction drags in network calls and
    config side effects we don't need here.
    """
    comp = MagicMock()
    comp._last_summary_auth_failure = auth_failure
    comp.provider = provider
    comp.summary_model = summary_model
    comp.base_url = base_url
    return comp


def _make_agent(
    *,
    main_provider: str = "custom",
    main_base_url: str = "https://api.siliconflow.cn/v1",
    main_model: str = "deepseek-ai/DeepSeek-V4-Pro",
    compressor: MagicMock | None = None,
    log_prefix: str = "",
) -> SimpleNamespace:
    """Minimal stand-in for AIAgent exposing the attributes the helper
    reads. We don't instantiate a real AIAgent — it drags in the session
    DB, credential pool, and globals unrelated to this regression.
    """
    captured: list[str] = []

    def _capture(*args, **kwargs) -> None:
        if args:
            captured.append(str(args[0]))

    agent = SimpleNamespace(
        provider=main_provider,
        base_url=main_base_url,
        model=main_model,
        log_prefix=log_prefix,
        context_compressor=compressor,
        _buffer_vprint=_capture,
        _vprint=_capture,
    )
    agent._captured = captured  # type: ignore[attr-defined]
    return agent


# ── Positive case ──────────────────────────────────────────────────────


def test_compression_auth_failure_emits_compression_model_lines() -> None:
    """The auxiliary compression model is the one that 401'd. The
    emitted lines must point at the compression provider/endpoint, NOT
    the main model.
    """
    agent = _make_agent(
        compressor=_make_compressor(
            auth_failure=True,
            provider="Oneai.17usoft.com",
            summary_model="deepseek-v4-flash",
            base_url="https://oneai.17usoft.com/v1",
        ),
    )

    _maybe_emit_compression_auth_hint(agent)

    rendered = "\n".join(agent._captured)  # type: ignore[attr-defined]
    # Compression model must appear.
    assert "Oneai.17usoft.com" in rendered, f"compression provider missing: {agent._captured}"
    assert "deepseek-v4-flash" in rendered, f"compression model missing: {agent._captured}"
    assert "oneai.17usoft.com" in rendered, f"compression endpoint missing: {agent._captured}"
    # Main model must NOT appear — that was the bug.
    assert "siliconflow.cn" not in rendered, (
        f"main endpoint leaked into compression auth block: {agent._captured}"
    )
    assert "deepseek-ai/DeepSeek-V4-Pro" not in rendered, (
        f"main model leaked into compression auth block: {agent._captured}"
    )


# ── Negative cases (helper must stay silent) ───────────────────────────


def test_no_lines_when_compression_succeeded() -> None:
    """When the compressor has no pending auth failure, the helper must
    be silent — otherwise every successful compression would spam a
    phantom auth-error banner.
    """
    agent = _make_agent(
        compressor=_make_compressor(
            auth_failure=False,
            provider="Oneai.17usoft.com",
            summary_model="deepseek-v4-flash",
            base_url="https://oneai.17usoft.com/v1",
        ),
    )

    _maybe_emit_compression_auth_hint(agent)

    assert agent._captured == []  # type: ignore[attr-defined]


def test_no_lines_when_compressor_missing() -> None:
    """Defensive: if context_compressor is absent (compression disabled
    or pre-init), the helper must not raise — silently skip.
    """
    agent = _make_agent(compressor=None)

    _maybe_emit_compression_auth_hint(agent)

    assert agent._captured == []  # type: ignore[attr-defined]


def test_no_lines_when_only_main_model_failed_auth() -> None:
    """Main model 401 with compression healthy → no compression block.
    Verifies the gate is *compressor-driven*, not main-model-driven —
    otherwise we'd flag the wrong source.
    """
    agent = _make_agent(
        main_provider="anthropic",
        main_base_url="https://api.anthropic.com",
        main_model="claude-sonnet-4.5",
        compressor=_make_compressor(auth_failure=False),
    )

    _maybe_emit_compression_auth_hint(agent)

    assert agent._captured == []  # type: ignore[attr-defined]


# ── Display site variants ──────────────────────────────────────────────


def test_force_vprint_routes_through_vprint_for_terminal_abort_site() -> None:
    """The terminal-abort display site uses ``_vprint(..., force=True)``,
    the retry-buffer site uses ``_buffer_vprint``. Both must produce the
    same lines; the helper accepts ``force_vprint`` to pick the channel.
    """
    agent = _make_agent(
        compressor=_make_compressor(
            auth_failure=True,
            provider="Oneai.17usoft.com",
            summary_model="deepseek-v4-flash",
            base_url="https://oneai.17usoft.com/v1",
        ),
        log_prefix="[agent] ",
    )

    _maybe_emit_compression_auth_hint(agent, force_vprint=True)

    rendered = "\n".join(agent._captured)  # type: ignore[attr-defined]
    assert "Oneai.17usoft.com" in rendered
    # Terminal site uses log_prefix; retry site does not. Sanity-check
    # the prefix actually got applied (not a hard assertion — log_prefix
    # formatting may evolve — but it must be present in at least one line).
    assert any("[agent]" in line for line in agent._captured), (  # type: ignore[attr-defined]
        f"log_prefix not applied in force_vprint mode: {agent._captured}"
    )


# ── Robustness to unexpected compressor shapes ─────────────────────────


@pytest.mark.parametrize(
    "missing_attr",
    ["provider", "summary_model", "base_url"],
)
def test_missing_compressor_attribute_does_not_raise(missing_attr: str) -> None:
    """A misbehaving compressor missing one of the fields the helper
    reads must not crash the error path. ``getattr(..., default)`` covers
    this in the implementation; the test pins the behavior.
    """
    compressor = _make_compressor(
        auth_failure=True,
        provider="Oneai.17usoft.com",
        summary_model="deepseek-v4-flash",
        base_url="https://oneai.17usoft.com/v1",
    )
    # Delete the attribute so getattr hits its default branch.
    delattr(compressor, missing_attr)
    agent = _make_agent(compressor=compressor)

    _maybe_emit_compression_auth_hint(agent)  # must not raise

    # At least one line still got emitted (the "auth failed" header).
    assert len(agent._captured) >= 1  # type: ignore[attr-defined]