"""Compression route attribution regressions for #72636/#72637."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch


from hermes_state import SessionDB
from agent.context_compressor_attribution import SummaryRoute
from tests.agent.compression_route_helpers import (
    _MAIN_BASE_URL, _MAIN_MODEL, _MAIN_PROVIDER, _AUX_PROVIDER, _AUX_MODEL, _AUX_BASE_URL,
)


def _build_agent_with_compressor(
    db: SessionDB,
    session_id: str,
    *,
    failure_class: str | None,  # "auth" | "network" | "other" | None
    compression_count: int = 0,
    aux_provider: str = _AUX_PROVIDER,
    aux_model: str = _AUX_MODEL,
    aux_base_url: str = _AUX_BASE_URL,
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
        compressor._last_aux_route = (
            SummaryRoute(aux_provider, aux_model, aux_base_url, dispatched=True)
            if aux_provider or aux_model or aux_base_url else None
        )
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
    # these instead of _last_aux_route misattributes the failure to the
    # main model — exactly the bug this regression guards against.
    compressor.provider = _MAIN_PROVIDER
    compressor.summary_model = ""  # empty in production (agent_init passes None)
    compressor.base_url = _MAIN_BASE_URL
    compressor.model = _MAIN_MODEL
    # Pre-populate the resolved auxiliary identity as the real compressor
    # would at construction (empty until _generate_summary resolves it).
    compressor._last_aux_route = None
    # Config-layer auxiliary identity (empty until _generate_summary resolves
    # an explicit auxiliary.compression provider). Set explicitly because a
    # MagicMock auto-attribute would be a truthy Mock and leak into the
    # pre-dispatch branch of the diagnostic.
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
    compressor._last_aux_route = SummaryRoute(_AUX_PROVIDER, _AUX_MODEL, _AUX_BASE_URL, dispatched=True)
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor

    emitted = _run_compression(agent)
    rendered = "\n".join(emitted)

    assert _AUX_PROVIDER not in rendered, (
        f"compression identity surfaced on a successful compression: {emitted}"
    )


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
