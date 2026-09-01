"""End-to-end regression for #72636: when ``auxiliary.compression``
fails, the compression-abort diagnostic must (a) point at the
*compression* model's provider/model/endpoint (not the main model's),
(b) classify the CURRENT attempt's failure mode correctly (a 401 then
a forced retry that fails with a 500 must NOT be mis-attributed as an
auth failure), and (c) actually reach messaging gateways rather than
be swallowed by the noise filter.

Three failure modes are guarded:

1. **Ordering + identity** — the diagnostic renders from the
   compression-abort branch in ``compress_context`` (after the
   compressor has recorded the failure), reads the resolved auxiliary
   identity (``_last_aux_call_*``) rather than the main-model fields on
   the compressor, and runs after the current attempt completed.

2. **Per-attempt classification** — the gate is the per-attempt
   ``_last_attempt_failure_class`` (reset on every ``_generate_summary``
   entry), NOT the sticky ``_last_summary_auth_failure`` (which persists
   across compress() calls to protect the cooldown guard). Without this,
   a 401 followed by a forced retry that fails with a 500 would surface
   a stale "auth failure" verdict.

3. **Gateway visibility** — the diagnostic wording is chosen to pass
   ``_TELEGRAM_NOISY_STATUS_RE`` so it reaches Telegram/Discord/Slack,
   not just local/CLI. The generic "Compression aborted" warning is
   already visible; the companion identity block must be too.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB

# The MAIN model the agent is running against. Healthy, NOT the source of
# the failure. The compressor is initialized against this runtime, so its
# provider / summary_model / base_url / model fields carry these values.
_MAIN_BASE_URL = "https://api.siliconflow.cn/v1"
_MAIN_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
_MAIN_PROVIDER = "custom"

# Identity of the failing AUXILIARY compression model — deliberately
# distinct from the main model (DeepSeek official API vs the main model's
# SiliconFlow endpoint), so assertions can verify attribution.
_AUX_PROVIDER = "api.deepseek.com"
_AUX_MODEL = "deepseek-chat"
_AUX_BASE_URL = "https://api.deepseek.com"


def _build_agent_with_compressor(
    db: SessionDB,
    session_id: str,
    *,
    failure_class: str | None,  # "auth" | "network" | "other" | None
    compression_count: int = 0,
    aux_provider: str = _AUX_PROVIDER,
    aux_model: str = _AUX_MODEL,
    aux_base_url: str = _AUX_BASE_URL,
    aux_config_provider: str = "",
    aux_config_model: str = "",
    aux_config_base_url: str = "",
):
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

    compressor = MagicMock()

    def _noop_compress(messages, **_kwargs):
        # Mirror what ContextCompressor.compress does after a failed
        # auxiliary summary call: the identity was already resolved and
        # recorded on the instance, then the call failed and the compressor
        # aborted, preserving the transcript unchanged. compress_context's
        # abort branch fires because the returned list equals the input.
        compressor._last_compress_aborted = True
        compressor._last_summary_error = "simulated summary failure"
        compressor._last_attempt_failure_class = failure_class
        compressor._last_aux_call_provider = aux_provider
        compressor._last_aux_call_model = aux_model
        compressor._last_aux_call_base_url = aux_base_url
        return list(messages)

    compressor.compress.side_effect = _noop_compress
    compressor.compression_count = compression_count
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_attempt_failure_class = None
    # IMPORTANT: these carry the MAIN model's identity, mirroring production
    # (the compressor is initialized against the main runtime). They are
    # distinct from the auxiliary identity above. A diagnostic that reads
    # these instead of _last_aux_call_* misattributes the failure to the
    # main model — exactly the bug this regression guards against.
    compressor.provider = _MAIN_PROVIDER
    compressor.summary_model = ""  # empty in production (agent_init passes None)
    compressor.base_url = _MAIN_BASE_URL
    compressor.model = _MAIN_MODEL
    # Pre-populate the resolved auxiliary identity as the real compressor
    # would at construction (empty until _generate_summary resolves it).
    compressor._last_aux_call_provider = ""
    compressor._last_aux_call_model = ""
    compressor._last_aux_call_base_url = ""
    # Config-layer auxiliary identity (empty until _generate_summary resolves
    # an explicit auxiliary.compression provider). Set explicitly because a
    # MagicMock auto-attribute would be a truthy Mock and leak into the
    # pre-dispatch branch of the diagnostic.
    compressor._last_aux_config_provider = aux_config_provider
    compressor._last_aux_config_model = aux_config_model
    compressor._last_aux_config_base_url = aux_config_base_url
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    return agent, compressor


def _run_compression(agent) -> list[str]:
    """Drive the real ``_compress_context`` → ``compress_context`` path and
    capture every user-visible warning emitted along the way."""
    emitted: list[str] = []
    agent._emit_warning = lambda message: emitted.append(message)

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    agent._compress_context(messages, "sys", approx_tokens=120_000)
    return emitted


# ── Failure-class attribution (the fix) ───────────────────────────────


def test_auth_failure_abort_surfaces_compression_identity(tmp_path: Path) -> None:
    """Compression aborted on an auxiliary auth failure → the user must see
    the *compression* provider/model/endpoint, never the main model's."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_AUTH"
    db.create_session(sid, source="cli")

    agent, _ = _build_agent_with_compressor(db, sid, failure_class="auth")
    emitted = _run_compression(agent)

    rendered = "\n".join(emitted)

    # Compression auxiliary identity MUST appear in the abort diagnostics.
    assert _AUX_PROVIDER in rendered, (
        f"compression provider missing from abort output: {emitted}"
    )
    assert _AUX_MODEL in rendered, (
        f"compression model missing from abort output: {emitted}"
    )
    assert _AUX_BASE_URL in rendered, (
        f"compression endpoint missing from abort output: {emitted}"
    )
    # The MAIN model is healthy and must NOT be named as the failing endpoint.
    assert "siliconflow.cn" not in rendered, (
        f"main endpoint leaked into compression diagnostic: {emitted}"
    )
    assert _MAIN_MODEL not in rendered, (
        f"main model leaked into compression diagnostic: {emitted}"
    )


def test_network_failure_abort_uses_network_guidance(tmp_path: Path) -> None:
    """A network/connection abort surfaces the compression identity with the
    transient-error guidance, not the auth credential guidance."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_NET"
    db.create_session(sid, source="cli")

    agent, _ = _build_agent_with_compressor(db, sid, failure_class="network")
    emitted = _run_compression(agent)
    rendered = "\n".join(emitted)

    assert _AUX_BASE_URL in rendered, f"compression endpoint missing: {emitted}"
    assert "transient" in rendered.lower(), (
        f"network guidance missing on network abort: {emitted}"
    )
    # Must NOT show the auth credential guidance for a network failure.
    assert "credential" not in rendered.lower(), (
        f"auth guidance leaked into network abort: {emitted}"
    )


def test_diagnostic_silent_when_no_failure_class(tmp_path: Path) -> None:
    """When compression ABORTS but no failure was classified (e.g. the
    compressor aborted for a non-summary reason), the identity diagnostic
    must stay silent — only the generic "Compression aborted" warning fires.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_NOCLASS"
    db.create_session(sid, source="cli")

    agent, _ = _build_agent_with_compressor(db, sid, failure_class=None)
    emitted = _run_compression(agent)
    rendered = "\n".join(emitted)

    # Generic abort warning is expected...
    assert any("Compression aborted" in m for m in emitted), (
        f"generic abort warning missing: {emitted}"
    )
    # ...but the compression-identity block must NOT appear.
    assert _AUX_PROVIDER not in rendered, (
        f"compression identity leaked when no failure was classified: {emitted}"
    )


def test_successful_compression_emits_no_diagnostic(tmp_path: Path) -> None:
    """When compression succeeds, no diagnostic fires."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_OK"
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

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 0
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_attempt_failure_class = None
    compressor.provider = _MAIN_PROVIDER
    compressor.summary_model = ""
    compressor.base_url = _MAIN_BASE_URL
    compressor.model = _MAIN_MODEL
    compressor._last_aux_call_provider = _AUX_PROVIDER
    compressor._last_aux_call_model = _AUX_MODEL
    compressor._last_aux_call_base_url = _AUX_BASE_URL
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor

    emitted = _run_compression(agent)
    rendered = "\n".join(emitted)

    assert _AUX_PROVIDER not in rendered, (
        f"compression identity surfaced on a successful compression: {emitted}"
    )


# ── Per-attempt classification: the 401 → 500 sequence ────────────────


def test_stale_auth_flag_does_not_poison_subsequent_non_auth_abort(
    tmp_path: Path,
) -> None:
    """Regression for the sticky-flag bug: a 401 that sets
    ``_last_summary_auth_failure`` must NOT cause a later, unrelated abort
    (e.g. a 500 on a forced retry) to be mis-attributed as an auth failure.

    The gate is the per-attempt ``_last_attempt_failure_class``, reset on
    every ``_generate_summary`` entry — not the sticky
    ``_last_summary_auth_failure`` (which intentionally persists across
    compress() calls to protect the cooldown guard).

    This test simulates the two-attempt sequence directly against the
    helper's contract: the SECOND abort carries a non-auth
    ``_last_attempt_failure_class`` even though the sticky auth flag from
    the first attempt is still set on the compressor.
    """
    from agent.conversation_compression import _emit_compression_auth_hint

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_SEQ"
    db.create_session(sid, source="cli")

    agent, comp = _build_agent_with_compressor(db, sid, failure_class="other")
    # Simulate the aftermath of a PRIOR 401: the sticky auth flag is still
    # True (compress() never clears it — only a successful summary does).
    comp._last_summary_auth_failure = True
    # But the CURRENT attempt failed with a 500 (transient/other), so the
    # per-attempt class must win.
    comp._last_attempt_failure_class = "other"

    emitted: list[str] = []
    agent._emit_warning = lambda message: emitted.append(message)
    _emit_compression_auth_hint(agent)

    rendered = "\n".join(emitted)
    # The diagnostic DID fire (the current attempt failed)...
    assert any("Compression auxiliary endpoint" in m for m in emitted), (
        f"diagnostic missing for current 'other' attempt: {emitted}"
    )
    # ...but it must NOT show auth credential guidance — the current
    # attempt is not an auth failure, regardless of the stale sticky flag.
    assert "credential" not in rendered.lower(), (
        f"stale auth flag mis-attributed a 500 as an auth failure: {emitted}"
    )
    assert "see agent.log" in rendered.lower(), (
        f"'other' guidance missing on non-auth abort: {emitted}"
    )


# ── Gateway visibility (the diagnostic must not be swallowed) ─────────


def test_diagnostic_survives_gateway_noise_filter() -> None:
    """The diagnostic wording must pass ``_TELEGRAM_NOISY_STATUS_RE`` so it
    reaches Telegram/Discord/Slack, not just local/CLI. The generic
    "Compression aborted" warning already passes; this pins that the
    companion identity block does too — across the failure classes.
    """
    from gateway.platforms.base import Platform
    from gateway.run import _prepare_gateway_status_message

    from agent.conversation_compression import _emit_compression_auth_hint

    # Build a minimal stand-in agent whose _emit_warning captures the
    # message, then route that message through the real gateway filter.
    for failure_class in ("auth", "network", "other"):
        comp = MagicMock()
        comp._last_attempt_failure_class = failure_class
        comp._last_aux_call_provider = _AUX_PROVIDER
        comp._last_aux_call_model = _AUX_MODEL
        comp._last_aux_call_base_url = _AUX_BASE_URL
        comp.provider = _MAIN_PROVIDER
        comp.model = _MAIN_MODEL
        comp.base_url = _MAIN_BASE_URL
        agent = MagicMock()
        agent.context_compressor = comp
        captured: list[str] = []
        agent._emit_warning = lambda message, _c=captured: _c.append(message)

        _emit_compression_auth_hint(agent)
        assert captured, f"no diagnostic emitted for class={failure_class}"

        for platform in (Platform.TELEGRAM, "discord", "slack"):
            result = _prepare_gateway_status_message(platform, "warn", captured[0])
            assert result is not None, (
                f"diagnostic swallowed by gateway filter "
                f"(class={failure_class}, platform={platform}): {captured[0]}"
            )


# ── Identity fallback when auxiliary is not separately configured ──────


def test_aux_unset_falls_back_to_main_model_with_note(tmp_path: Path) -> None:
    """When ``auxiliary.compression`` is unset / "auto", the summary call
    runs against the main model — no separate auxiliary identity was
    resolved. The diagnostic falls back to the main-model identity and
    says so explicitly, rather than inventing a phantom auxiliary endpoint."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_AUTO"
    db.create_session(sid, source="cli")

    agent, _ = _build_agent_with_compressor(
        db, sid, failure_class="auth", aux_provider="", aux_model="", aux_base_url=""
    )
    emitted = _run_compression(agent)

    rendered = "\n".join(emitted)
    # Falls back to the main-model identity (since aux is unset)...
    assert _MAIN_BASE_URL in rendered, (
        f"main endpoint missing from fallback diagnostic: {emitted}"
    )
    # ...and explicitly flags that auxiliary.compression is not configured.
    assert "not configured" in rendered, (
        f"missing 'not configured' note on aux-unset fallback: {emitted}"
    )


def test_aux_unset_fallback_strips_query_from_main_base_url(tmp_path: Path) -> None:
    """The aux-unset fallback reads compressor.base_url (the MAIN model's
    URL) raw — no producer strips it on this path. When the main endpoint
    carries a proxy credential as ?key=..., the sink must sanitize it before
    the user-facing warning: route_callback never fired (pre-dispatch abort)
    and auxiliary.compression is unset, so nothing else stands between the
    raw URL and Telegram/Discord/Slack (#72636 review, defect 2)."""
    _SECRET = "sk-super-secret-token-12345"
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_SINK_SANITIZE"
    db.create_session(sid, source="cli")

    agent, compressor = _build_agent_with_compressor(
        db, sid, failure_class="auth", aux_provider="", aux_model="", aux_base_url=""
    )
    # The main endpoint carries a credential in the query string — as real
    # proxy deployments do.
    compressor.base_url = f"https://proxy.example.com/v1?key={_SECRET}"

    emitted = _run_compression(agent)
    rendered = "\n".join(emitted)

    # Prove we exercised the third fallback (not another branch)...
    assert "not configured" in rendered, (
        f"expected the aux-unset fallback note, got: {emitted}"
    )
    # ...the sanitized endpoint is still reported...
    assert "https://proxy.example.com/v1" in rendered, (
        f"sanitized main endpoint missing from diagnostic: {emitted}"
    )
    # ...but the query-string credential MUST NOT leak.
    assert _SECRET not in rendered, (
        f"query-string credential leaked into diagnostic: {emitted}"
    )
    assert "key=" not in rendered, (
        f"query string leaked into diagnostic: {emitted}"
    )


# ── Robustness ─────────────────────────────────────────────────────────


def test_missing_compressor_identity_does_not_crash_abort(tmp_path: Path) -> None:
    """A misbehaving compressor that aborts but is missing the resolved
    auxiliary identity must not crash the abort path — the diagnostic falls
    back to the main-model identity."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_PARTIAL"
    db.create_session(sid, source="cli")

    agent, _ = _build_agent_with_compressor(
        db,
        sid,
        failure_class="auth",
        aux_provider="",
        aux_model="",
        aux_base_url="",
    )
    emitted = _run_compression(agent)

    assert any("Compression aborted" in m for m in emitted), (
        f"generic abort warning missing on partial-compressor abort: {emitted}"
    )
    assert any("Compression auxiliary endpoint" in m for m in emitted), (
        f"diagnostic missing on partial-compressor abort: {emitted}"
    )


# ── Real call_llm routing: observe every physical wire attempt ─────────


def test_real_call_llm_reports_concrete_auto_route_and_strips_query() -> None:
    """The real call_llm orchestration must resolve ``auto`` to the client
    backend and sanitize its endpoint before publishing the wire attempt."""
    from agent.auxiliary_client import call_llm

    secret = "sk-route-secret"
    client = MagicMock()
    client.base_url = f"https://api.minimax.chat/v1?key={secret}"
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    routes: list[tuple[str, str | None, str]] = []

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "minimax/minimax-m2.7", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "minimax/minimax-m2.7"),
        ),
    ):
        call_llm(
            task="compression",
            messages=[{"role": "user", "content": "summarize"}],
            route_callback=lambda *route: routes.append(route),
        )

    assert routes == [(
        "minimax",
        "minimax/minimax-m2.7",
        "https://api.minimax.chat/v1",
    )]
    assert secret not in repr(routes)


def test_real_call_llm_updates_route_when_fallback_fails() -> None:
    """If a fallback becomes the terminal failing request, the last observed
    identity must describe that fallback rather than the stale primary."""
    from agent.auxiliary_client import call_llm

    auth_error = type("Auth401", (Exception,), {"status_code": 401})("expired key")
    primary = MagicMock()
    primary.base_url = "https://api.minimax.chat/v1"
    primary.chat.completions.create.side_effect = auth_error

    fallback = MagicMock()
    fallback.base_url = "https://openrouter.ai/api/v1"
    fallback.chat.completions.create.side_effect = ValueError(
        "fallback malformed response"
    )
    routes: list[tuple[str, str | None, str]] = []

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "minimax/minimax-m2.7", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary, "minimax/minimax-m2.7"),
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_payment_fallback",
            return_value=(fallback, "fallback-model", "openrouter"),
        ),
    ):
        with pytest.raises(ValueError, match="fallback malformed response"):
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
                route_callback=lambda *route: routes.append(route),
            )

    assert routes == [
        ("minimax", "minimax/minimax-m2.7", "https://api.minimax.chat/v1"),
        ("openrouter", "fallback-model", "https://openrouter.ai/api/v1"),
    ]


# ── Real ContextCompressor: verify callback state reaches the abort ───
#
# These tests do NOT mock compressor.compress. They use a real
# ContextCompressor and patch its imported call_llm reference so the callback
# state can be checked independently of routing. The tests above exercise the
# real auxiliary_client.call_llm auto/fallback orchestration; these tests pin
# that the resulting state flows through to the abort diagnostic.


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


def test_real_call_llm_quarantined_fallback_auth_propagates_last_attempt_error() -> None:
    """Primary rate-limit → fallback candidate with an unrefreshable 401 →
    no remaining candidate. The error that propagates must be the FALLBACK's
    terminal auth error, not the primary's rate-limit: the route snapshot
    already identifies the fallback, so re-raising the primary would pair one
    attempt's endpoint with a different attempt's failure class (#72636
    review, defect 1)."""
    from agent.auxiliary_client import call_llm

    primary_err = type("Rate429", (Exception,), {"status_code": 429})(
        "429 Too Many Requests: rate limit exceeded"
    )
    fallback_err = type("Auth401", (Exception,), {"status_code": 401})(
        "expired fallback key"
    )
    primary = MagicMock()
    primary.base_url = "https://api.minimax.chat/v1"
    primary.chat.completions.create.side_effect = primary_err

    fallback = MagicMock()
    fallback.base_url = "https://openrouter.ai/api/v1"
    fallback.chat.completions.create.side_effect = fallback_err
    routes: list[tuple[str, str | None, str]] = []

    # Discovery order for an auto primary: configured chain (None) → main
    # chain (None) → payment fallback (the 401-looping candidate). After the
    # candidate is quarantined, the chain is walked once more — this time it
    # must yield nothing, so the chain exhausts.
    payment_walks = [
        (fallback, "fallback-model", "openrouter"),
        (None, None, ""),
    ]

    def _payment_fallback(*_args, **_kwargs):
        return payment_walks.pop(0) if payment_walks else (None, None, "")

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "minimax/minimax-m2.7", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary, "minimax/minimax-m2.7"),
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_payment_fallback",
            side_effect=_payment_fallback,
        ),
    ):
        with pytest.raises(Exception) as excinfo:
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
                route_callback=lambda *route: routes.append(route),
            )

    # The terminal route snapshot is the fallback...
    assert routes == [
        ("minimax", "minimax/minimax-m2.7", "https://api.minimax.chat/v1"),
        ("openrouter", "fallback-model", "https://openrouter.ai/api/v1"),
    ]
    # ...so the propagated error must be the fallback's own 401, chained to
    # the primary origin — never the primary's 429 re-raised bare.
    assert excinfo.value is fallback_err, (
        f"propagated error is not the quarantined fallback 401: {excinfo.value!r}"
    )
    assert excinfo.value.__cause__ is primary_err


def test_real_call_llm_quarantined_fallback_keeps_primary_error_for_other_tasks() -> None:
    """Same wire scenario as the compression test above, but for an auxiliary
    task that did NOT opt into route attribution (no route_callback, task is
    not compression — e.g. web_extract or title generation): call_llm's
    exception contract must be unchanged from before this PR. The swallowed
    fallback 401 stays swallowed and the caller observes the PRIMARY error
    (429), which upper layers key retry/backoff decisions on (#72636
    review, defect 1 scoping)."""
    from agent.auxiliary_client import call_llm

    primary_err = type("Rate429", (Exception,), {"status_code": 429})(
        "429 Too Many Requests: rate limit exceeded"
    )
    fallback_err = type("Auth401", (Exception,), {"status_code": 401})(
        "expired fallback key"
    )
    primary = MagicMock()
    primary.base_url = "https://api.minimax.chat/v1"
    primary.chat.completions.create.side_effect = primary_err

    fallback = MagicMock()
    fallback.base_url = "https://openrouter.ai/api/v1"
    fallback.chat.completions.create.side_effect = fallback_err

    payment_walks = [
        (fallback, "fallback-model", "openrouter"),
        (None, None, ""),
    ]

    def _payment_fallback(*_args, **_kwargs):
        return payment_walks.pop(0) if payment_walks else (None, None, "")

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "minimax/minimax-m2.7", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary, "minimax/minimax-m2.7"),
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_payment_fallback",
            side_effect=_payment_fallback,
        ),
    ):
        with pytest.raises(Exception) as excinfo:
            call_llm(
                task="web_extract",
                messages=[{"role": "user", "content": "describe"}],
            )

    # Pre-PR contract: the caller sees the primary 429, not the fallback's
    # quarantined 401 — retry/backoff classification must not flip to
    # "credential broken" for tasks outside the attribution path.
    assert excinfo.value is primary_err, (
        f"non-compression task received the quarantined fallback error "
        f"instead of the primary: {excinfo.value!r}"
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
        raise RuntimeError(
            "Provider 'openrouter' is set in config.yaml but no API key was found"
        )

    emitted: list[str] = []
    agent._emit_warning = lambda message: emitted.append(message)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    with (
        patch("agent.context_compressor.call_llm", side_effect=_fake_missing_key),
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("openrouter", "or-aux-model", "https://openrouter.ai/api/v1", None, None),
        ),
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
