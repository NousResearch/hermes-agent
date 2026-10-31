"""Compression route attribution regressions for #72636/#72637."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


from hermes_state import SessionDB
from tests.agent.compression_route_helpers import (
    _MAIN_BASE_URL, _MAIN_MODEL, _MAIN_PROVIDER,
)


def _build_real_compressor_agent(
    db: SessionDB,
    session_id: str,
    *,
    wire_provider: str,
    wire_model: str,
    wire_base_url: str,
):
    """Build an agent with a REAL ContextCompressor whose call_llm is patched
    to invoke route_callback with the wire identity, then raise a 401."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url=_MAIN_BASE_URL,
            model=_MAIN_MODEL,
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    # Replace the MagicMock compressor that AIAgent installs with a real one.
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor(
            model=_MAIN_MODEL,
            base_url=_MAIN_BASE_URL,
            provider=_MAIN_PROVIDER,
            quiet_mode=True,
            protect_first_n=2,
            protect_last_n=2,
            abort_on_summary_failure=False,
        )
    agent.context_compressor = comp

    # Patch call_llm so it (1) invokes route_callback with the wire identity
    # (mirroring what real call_llm does after building the client) and
    # (2) raises a 401 that _is_summary_access_or_quota_error classifies as
    # an auth failure → _last_attempt_failure_class = "auth".
    _Stub401 = type("Stub401", (Exception,), {"status_code": 401})

    def _fake_call_llm(*args, **kwargs):
        _cb = kwargs.get("route_callback")
        if _cb is not None:
            _cb(wire_provider, wire_model, wire_base_url)
        raise _Stub401("401 Client Error: Unauthorized")

    # Force past the cooldown so compress() actually calls _generate_summary.
    comp._clear_compression_failure_cooldown()
    return agent, "agent.context_compressor.call_llm", _fake_call_llm


def test_real_compressor_route_callback_identity_reaches_diagnostic(
    tmp_path: Path,
) -> None:
    """End-to-end with a REAL ContextCompressor: call_llm's route_callback
    writes the wire identity, the 401 abort surfaces it in the diagnostic.

    This proves the identity supplied by call_llm flows through to
    _emit_compression_auth_hint, not the pre-resolution guess from
    _resolve_task_provider_model. Real call_llm routing is covered separately.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_REAL"
    db.create_session(sid, source="cli")

    agent, _patch_target, _fake = _build_real_compressor_agent(
        db, sid,
        wire_provider="api.deepseek.com",
        wire_model="deepseek-chat",
        wire_base_url="https://api.deepseek.com",
    )
    emitted: list[str] = []
    agent._emit_warning = lambda message: emitted.append(message)

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    with patch(_patch_target, side_effect=_fake):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    rendered = "\n".join(emitted)
    # The wire identity written by route_callback MUST surface...
    assert "api.deepseek.com" in rendered, (
        f"wire provider from route_callback missing: {emitted}"
    )
    assert "deepseek-chat" in rendered, (
        f"wire model from route_callback missing: {emitted}"
    )
    assert "https://api.deepseek.com" in rendered, (
        f"wire endpoint from route_callback missing: {emitted}"
    )
    # ...and the MAIN model must NOT leak (route_callback overrode it).
    assert "siliconflow.cn" not in rendered, (
        f"main endpoint leaked despite route_callback: {emitted}"
    )


def test_real_compressor_strips_query_from_base_url(tmp_path: Path) -> None:
    """route_callback's base_url is query-stripped, so credentials some
    proxies carry as ?key=... are not leaked into the user-facing diagnostic."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_QUERY"
    db.create_session(sid, source="cli")

    _SECRET = "sk-super-secret-token-12345"
    agent, _patch_target, _fake = _build_real_compressor_agent(
        db, sid,
        wire_provider="custom",
        wire_model="some-aux",
        # Wire base_url carries a credential in the query string — as a real
        # client built from a config base_url like this would.
        wire_base_url=f"https://proxy.example.com/v1?key={_SECRET}",
    )
    emitted: list[str] = []
    agent._emit_warning = lambda message: emitted.append(message)

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    with patch(_patch_target, side_effect=_fake):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    rendered = "\n".join(emitted)
    # The proxy host is reported (sanitized)...
    assert "proxy.example.com" in rendered, (
        f"sanitized endpoint missing: {emitted}"
    )
    # ...but the query-string credential MUST NOT leak.
    assert _SECRET not in rendered, (
        f"query-string credential leaked into diagnostic: {emitted}"
    )
    assert "key=" not in rendered, (
        f"query string leaked into diagnostic: {emitted}"
    )


def test_real_compressor_no_provider_does_not_emit_aux_diagnostic(
    tmp_path: Path,
) -> None:
    """When call_llm raises 'no provider configured' BEFORE building a client,
    route_callback never fires and compression falls back to the static marker
    (no abort, no sticky auth flag). The aux-identity diagnostic therefore
    does NOT fire — there is no wire route to attribute, and the generic
    fallback warning already tells the user what happened."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_NOPROV"
    db.create_session(sid, source="cli")

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url=_MAIN_BASE_URL,
            model=_MAIN_MODEL,
            quiet_mode=True,
            session_db=db,
            session_id=sid,
            skip_context_files=True,
            skip_memory=True,
        )
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor(
            model=_MAIN_MODEL, base_url=_MAIN_BASE_URL, provider=_MAIN_PROVIDER,
            quiet_mode=True, protect_first_n=2, protect_last_n=2,
            abort_on_summary_failure=False,
        )
    agent.context_compressor = comp
    comp._clear_compression_failure_cooldown()

    # call_llm raises "no provider configured" without calling route_callback.
    def _fake_no_provider(*args, **kwargs):
        raise RuntimeError("No LLM provider configured for task=compression")

    emitted: list[str] = []
    agent._emit_warning = lambda message: emitted.append(message)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    with patch("agent.context_compressor.call_llm", side_effect=_fake_no_provider):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    rendered = "\n".join(emitted)
    # No client was built → no abort → no aux-identity diagnostic. The static
    # fallback warning fires instead (the no-provider case is already covered
    # by the generic warning, not the per-route diagnostic).
    assert "Compression auxiliary endpoint" not in rendered, (
        f"aux diagnostic fired on no-provider (no route to attribute): {emitted}"
    )
    # The broken 'Provider: auto / Endpoint: (empty)' shape that the
    # pre-resolution approach produced must NOT appear either.
    assert "Provider: auto" not in rendered, (
        f"phantom 'Provider: auto' on no-provider: {emitted}"
    )


def test_real_compressor_explicit_provider_missing_key_reports_pre_dispatch(
    tmp_path: Path,
) -> None:
    """auxiliary.compression names an explicit provider whose API key is
    missing → call_llm raises BEFORE any client exists, so route_callback
    never fires. The diagnostic must report the CONFIGURED auxiliary
    identity as a pre-dispatch failure — it must NOT substitute the healthy
    main endpoint and must NOT claim auxiliary.compression is unconfigured
    (#72636 review, defect 2)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_NOKEY"
    db.create_session(sid, source="cli")

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url=_MAIN_BASE_URL,
            model=_MAIN_MODEL,
            quiet_mode=True,
            session_db=db,
            session_id=sid,
            skip_context_files=True,
            skip_memory=True,
        )
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor(
            model=_MAIN_MODEL, base_url=_MAIN_BASE_URL, provider=_MAIN_PROVIDER,
            quiet_mode=True, protect_first_n=2, protect_last_n=2,
            abort_on_summary_failure=False,
        )
    agent.context_compressor = comp
    comp._clear_compression_failure_cooldown()

    # call_llm dies before dispatch — route_callback never fires.
    def _fake_missing_key(*args, **kwargs):
        kwargs["configured_route_callback"]("openrouter", "or-aux-model", "https://openrouter.ai/api/v1")
        raise RuntimeError(
            "Provider 'openrouter' is set in config.yaml but no API key was found"
        )

    emitted: list[str] = []
    agent._emit_warning = lambda message: emitted.append(message)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    with (
        patch("agent.context_compressor.call_llm", side_effect=_fake_missing_key),
    ):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    rendered = "\n".join(emitted)
    # The CONFIGURED auxiliary identity is reported...
    assert "openrouter" in rendered, (
        f"configured aux provider missing from pre-dispatch diagnostic: {emitted}"
    )
    assert "or-aux-model" in rendered, (
        f"configured aux model missing from pre-dispatch diagnostic: {emitted}"
    )
    assert "https://openrouter.ai/api/v1" in rendered, (
        f"configured aux endpoint missing from pre-dispatch diagnostic: {emitted}"
    )
    # ...flagged as pre-dispatch...
    assert "no request was dispatched" in rendered, (
        f"pre-dispatch note missing: {emitted}"
    )
    # ...and the two false claims the old fallback produced are gone: the
    # healthy MAIN endpoint must not appear as the failed route...
    assert _MAIN_BASE_URL not in rendered, (
        f"main endpoint substituted on pre-dispatch failure: {emitted}"
    )
    # ...and the diagnostic must not claim auxiliary.compression is unset.
    assert "not configured" not in rendered, (
        f"false 'not configured' note on explicit-provider failure: {emitted}"
    )
