"""End-to-end regression for #72636: when ``auxiliary.compression`` fails
with an auth/quota error (HTTP 401/403), the compression-abort warning
must point the user at the *compression* model's provider/model/endpoint,
NOT the main model's.

Two failure modes are guarded here, both reported by the original
reviewer / follow-up:

1. **Ordering** — the diagnostic must render from the centralized
   compression-abort branch in ``compress_context`` (after the
   compressor has set ``_last_summary_auth_failure``), not from a
   main-model API-error display site in the conversation loop (where
   the flag is still false for a fresh 401/403 and goes stale until
   the next successful summary).

2. **Identity attribution** — the compressor's ``provider`` /
   ``summary_model`` / ``base_url`` fields carry the *main* model's
   identity (the compressor is initialized against the main runtime).
   The identity actually used on the wire for the summary call is
   resolved from ``auxiliary.compression`` config and recorded by the
   compressor as ``_last_aux_call_provider`` / ``_last_aux_call_model``
   / ``_last_aux_call_base_url``. The diagnostic must read those, not
   the main-model fields — otherwise it points the user at the wrong
   endpoint whenever the compression auxiliary is configured
   separately.

These tests drive the real ``agent._compress_context`` →
``compress_context`` → abort path with a mock compressor whose
``compress()`` mirrors what the real ``ContextCompressor.compress``
does when the auxiliary summary call 401s: abort, preserve the
transcript unchanged, and record both the failure flag and the
resolved auxiliary identity on itself. The main-model identity on the
compressor is deliberately distinct from the auxiliary identity, so a
diagnostic that reads the wrong fields is caught.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    auth_failure: bool,
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
        # Mirror what ContextCompressor.compress does when the auxiliary
        # summary call 401s: the identity was already resolved from
        # auxiliary.compression config and recorded on the instance
        # (_last_aux_call_*), then the call failed and the compressor
        # aborts, preserving the transcript unchanged with the auth-failure
        # flag set. compress_context's abort branch fires because the
        # returned list is left equal to the input.
        compressor._last_compress_aborted = True
        compressor._last_summary_error = "401 Client Error: Unauthorized"
        if auth_failure:
            compressor._last_summary_auth_failure = True
        # The resolved auxiliary identity — recorded BEFORE the call_llm
        # that 401'd, so it is present at abort time.
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
    compressor._last_summary_auth_failure = bool(auth_failure)
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


# ── Auth-failure attribution (the fix) ─────────────────────────────────


def test_aux_auth_failure_abort_surfaces_compression_identity(tmp_path: Path) -> None:
    """Compression aborted on an auxiliary 401 → the user must see the
    *compression* provider/model/endpoint, never the main model's.

    This is the end-to-end ordering the fix targets: overflow → compress →
    auxiliary auth failure → abort → diagnostic. It also pins identity
    attribution: the compressor's main-model fields (provider / model /
    base_url) are the SiliconFlow main runtime, but the diagnostic must
    surface the DeepSeek auxiliary identity actually used on the wire.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_AUTH"
    db.create_session(sid, source="cli")

    agent, _ = _build_agent_with_compressor(db, sid, auth_failure=True)
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
    # The MAIN model is healthy and must NOT be named as the failing endpoint —
    # that misattribution is exactly the bug this regression guards against.
    assert "siliconflow.cn" not in rendered, (
        f"main endpoint leaked into compression auth diagnostic: {emitted}"
    )
    assert _MAIN_MODEL not in rendered, (
        f"main model leaked into compression auth diagnostic: {emitted}"
    )


def test_aux_auth_diagnostic_silent_on_non_auth_abort(tmp_path: Path) -> None:
    """When compression ABORTS but NOT on an auth failure (e.g. transient
    network, summary-abort), the compression-identity diagnostic must stay
    silent — only the generic "Compression aborted" warning fires.

    Pins that the gate is ``_last_summary_auth_failure``, not
    ``_last_compress_aborted``.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_NOAUTH"
    db.create_session(sid, source="cli")

    agent, _ = _build_agent_with_compressor(db, sid, auth_failure=False)
    emitted = _run_compression(agent)

    rendered = "\n".join(emitted)
    # Generic abort warning is expected...
    assert any("Compression aborted" in m for m in emitted), (
        f"generic abort warning missing: {emitted}"
    )
    # ...but the compression-identity block must NOT appear.
    assert _AUX_PROVIDER not in rendered, (
        f"compression identity leaked on a non-auth abort: {emitted}"
    )
    assert _AUX_MODEL not in rendered, (
        f"compression model leaked on a non-auth abort: {emitted}"
    )


def test_successful_compression_emits_no_auth_diagnostic(tmp_path: Path) -> None:
    """When compression succeeds, no auth diagnostic fires — otherwise every
    healthy compression would spam a phantom auth-error banner."""
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

    # A *successful* compressor: returns a compressed transcript, no abort.
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
    compressor._last_summary_auth_failure = False
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


# ── Identity fallback when auxiliary is not separately configured ──────


def test_aux_unset_falls_back_to_main_model_with_note(tmp_path: Path) -> None:
    """When ``auxiliary.compression`` is unset / "auto", the summary call
    runs against the main model — no separate auxiliary identity was
    resolved. The diagnostic must NOT invent a phantom endpoint; it falls
    back to the main-model identity and says so explicitly, so the user
    is not sent chasing a non-existent auxiliary config."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_AUTO"
    db.create_session(sid, source="cli")

    # No auxiliary identity resolved — all _last_aux_call_* empty.
    agent, _ = _build_agent_with_compressor(
        db, sid, auth_failure=True, aux_provider="", aux_model="", aux_base_url=""
    )
    emitted = _run_compression(agent)

    rendered = "\n".join(emitted)
    # Falls back to the main-model identity (since aux is unset)...
    assert _MAIN_BASE_URL in rendered, (
        f"main endpoint missing from fallback diagnostic: {emitted}"
    )
    # ...and explicitly flags that auxiliary.compression is not configured,
    # so the user knows the main model is the one to fix.
    assert "not configured" in rendered, (
        f"missing 'not configured' note on aux-unset fallback: {emitted}"
    )


# ── Robustness ─────────────────────────────────────────────────────────


def test_missing_compressor_identity_does_not_crash_abort(tmp_path: Path) -> None:
    """A misbehaving compressor that aborts on auth failure but is missing
    the resolved auxiliary identity must not crash the abort path — the
    generic "Compression aborted" warning must still reach the user, and
    the diagnostic falls back to the main-model identity."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_72636_PARTIAL"
    db.create_session(sid, source="cli")

    agent, _ = _build_agent_with_compressor(
        db,
        sid,
        auth_failure=True,
        aux_provider="",
        aux_model="",
        aux_base_url="",
    )
    emitted = _run_compression(agent)

    # Did not raise, and the generic abort warning still fired...
    assert any("Compression aborted" in m for m in emitted), (
        f"generic abort warning missing on partial-compressor abort: {emitted}"
    )
    # ...and the diagnostic still emitted, falling back to the main-model
    # identity rather than crashing.
    assert any("auth/permission" in m for m in emitted), (
        f"compression auth diagnostic missing on partial-compressor abort: {emitted}"
    )
