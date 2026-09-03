"""Shared output-reservation policy — coherence owning tests (remediation pass).

This file pins the canonical output-reservation semantics corrected from the
upstream read-only audit, and the #90877 clamp-before-gate composition. It
does NOT implement #90877/#70242; it only pins the contract this branch must
hold so a future merge of those PRs is a clean superset.

Canonical precedence (highest → lowest), one shared policy in
``agent.model_metadata.resolve_output_reservation``:

  1. Final provider-bound request cap (``max_tokens`` /
     ``max_completion_tokens`` / ``max_output_tokens`` / Bedrock
     ``inferenceConfig.maxTokens`` / ``maxOutputTokens``). A legitimate
     clamp/ephemeral/continuation value is AUTHORITATIVE over a larger
     provider default (default 65536 + final cap 8192 → reserve 8192).
  2. Explicit invocation/user output cap (``explicit_cap``) — surfaces that
     budget BEFORE the final request is built.
  3. Provider/profile-derived implicit cap (``provider`` via the registry).
     Omitted cap + registered profile 65536 → reserve 65536.
  4. ``DEFAULT_OUTPUT_RESERVATION`` (4096) — ONLY when nothing more
     authoritative is known (unknown provider, no resolvable cap). Never 0.

Every budgeting surface derives the SAME allowance from this policy:
  * compressor preflight budgeting (ContextCompressor._resolve_output_reservation)
  * terminal FinalContextBudget (build_final_context_budget)
  * MoA reference trim/gate + aggregator gate (_resolve_moa_output_reserve)
  * provider-switch refresh (the resolver re-derives the implicit cap)

The #90877 composition under test (E):
    request construction/clamping → relay/final transformations
    → terminal gate → physical provider call
  The terminal gate judges the FINAL payload. A legitimate clamp that reduces
  max_tokens so ``input + final_cap <= effective_context`` must ALLOW dispatch.
  The gate still refuses when the input alone is at/over the effective window,
  or the final request still exceeds the budget after all transformations.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from agent.model_metadata import (
    DEFAULT_OUTPUT_RESERVATION,
    ContextCeilingExceeded,
    build_final_context_budget,
    enforce_final_context_budget,
    resolve_output_reservation,
    is_output_cap_error,
    parse_available_output_tokens_from_error,
)

# A registered provider whose implicit output cap is a known value (65536).
# Used for the "provider-derived implicit reservation" and "final-cap
# precedence" tests. Registered in-process via the providers registry so the
# test is self-contained (no plugin loading order dependency).
_FAKE_PROVIDER = "fake-65k"
_FAKE_IMPLICIT_CAP = 65_536


@pytest.fixture
def fake_65k_provider(monkeypatch):
    """Register a fake provider with a known implicit output cap (65536)."""
    import providers
    from providers import ProviderProfile, register_provider

    saved_registry = dict(getattr(providers, "_REGISTRY", {}))
    monkeypatch.setattr(providers, "_REGISTRY", {}, raising=False)
    monkeypatch.setattr(providers, "_ALIASES", {}, raising=False)
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None, raising=False)
    monkeypatch.setattr(providers, "_discovered", True, raising=False)
    register_provider(
        ProviderProfile(
            name=_FAKE_PROVIDER,
            api_mode="chat_completions",
            base_url="https://fake.example/v1",
            default_max_tokens=_FAKE_IMPLICIT_CAP,
        )
    )
    yield _FAKE_PROVIDER
    monkeypatch.setattr(providers, "_REGISTRY", saved_registry, raising=False)


def _msgs(n_tokens_target: int = 4_000) -> list:
    """A messages list whose rough estimate is near n_tokens_target.

    estimate_messages_tokens_rough uses ~4 chars/token for ASCII, so this is
    an approximation; tests that assert exact budget totals read the
    ``FinalContextBudget.input_tokens_estimate`` field rather than assuming a
    specific count, so the estimator ratio cannot make them flaky.
    """
    return [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "x" * (n_tokens_target * 4)},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# A. Provider-derived implicit reservation
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderDerivedImplicitReservation:
    """A registered provider's implicit cap (65536) with an omitted request
    cap must drive the reservation on EVERY budgeting surface."""

    def test_resolver_uses_provider_implicit_cap(self, fake_65k_provider):
        # Omitted cap + registered profile → provider's implicit cap.
        assert resolve_output_reservation({}, provider=_FAKE_PROVIDER) == 65_536
        assert resolve_output_reservation(None, provider=_FAKE_PROVIDER) == 65_536
        # A final cap still wins over the provider default (see D).
        assert resolve_output_reservation(
            {"max_tokens": 8192}, provider=_FAKE_PROVIDER
        ) == 8192

    def test_terminal_gate_uses_provider_implicit_cap(self, fake_65k_provider):
        # No output cap in the final request → the gate reserves the provider's
        # implicit 65536, not 4096.
        budget = build_final_context_budget(
            {"model": "m", "messages": _msgs(1_000)},
            provider=_FAKE_PROVIDER,
            model="m",
        )
        assert budget.output_reservation == 65_536

    def test_moa_budgeting_uses_same_policy(self, fake_65k_provider):
        from agent.moa_loop import _resolve_moa_output_reserve

        # MoA reference/aggregator reservation agrees with the shared policy:
        # omitted explicit cap → provider's implicit 65536 (not 4096, not 0).
        assert _resolve_moa_output_reserve(None, provider=_FAKE_PROVIDER) == 65_536
        # An explicit cap still wins (MoA's existing explicit-cap contract).
        assert _resolve_moa_output_reserve(
            2048, provider=_FAKE_PROVIDER
        ) == 2048

    def test_compressor_budgeting_uses_provider_reservation(self, fake_65k_provider):
        """The compressor preflight budget derives its output reservation from
        the SAME shared policy: provider implicit cap (65536) when there is no
        explicit cap — coherent with the terminal gate and the wire."""
        from agent.context_compressor import ContextCompressor

        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=200_000,
        ):
            comp = ContextCompressor(
                "fake-65k/model", provider=_FAKE_PROVIDER, quiet_mode=True
            )
            # Direct resolver: explicit max_tokens is None → provider cap.
            assert comp._resolve_output_reservation() == 65_536
            # The threshold budget reflects that reservation:
            # (200_000 - 65_536) * 0.50, not (200_000 - 0) * 0.50.
            _ = comp.context_length
            expected = int((200_000 - 65_536) * 0.50)
            # Apply the small-context floor guard: 200K < 512K → 75% floor.
            expected = int((200_000 - 65_536) * 0.75)
            assert comp.threshold_tokens == expected


# ─────────────────────────────────────────────────────────────────────────────
# B. Unknown provider fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownProviderFallback:
    """No explicit cap + no resolvable provider default → finite 4096, never 0."""

    def test_unknown_provider_falls_back_to_4096(self):
        assert resolve_output_reservation({}, provider="not-a-provider") == 4096
        assert resolve_output_reservation(None, provider=None) == 4096
        assert resolve_output_reservation({}) == 4096
        assert resolve_output_reservation({}, provider="") == 4096

    def test_terminal_gate_unknown_provider_4096(self):
        budget = build_final_context_budget(
            {"model": "m", "messages": _msgs(1_000)}, provider="nope"
        )
        assert budget.output_reservation == DEFAULT_OUTPUT_RESERVATION == 4096

    def test_compressor_unknown_provider_never_zero(self):
        from agent.context_compressor import ContextCompressor

        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=200_000,
        ):
            comp = ContextCompressor(
                "test/model", provider="unknown-prov", quiet_mode=True
            )
            # The audit's flagged "compressor may reserve 0/None" is fixed:
            # the reservation is the finite 4096 default, never 0.
            assert comp._resolve_output_reservation() == 4096
            assert comp._resolve_output_reservation() > 0


# ─────────────────────────────────────────────────────────────────────────────
# C. Provider switch — auto-derived changes, explicit stays stable
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderSwitchReservation:
    """A provider switch re-derives the AUTO-IMPLICIT reservation, but an
    explicit user/invocation cap stays explicit and stable."""

    def test_auto_derived_changes_with_provider(self, fake_65k_provider):
        # Auto-derived (no explicit cap): follows the provider's implicit cap.
        assert (
            resolve_output_reservation({}, provider=_FAKE_PROVIDER)
            != resolve_output_reservation({}, provider="unknown")
        )
        assert resolve_output_reservation({}, provider=_FAKE_PROVIDER) == 65_536
        assert resolve_output_reservation({}, provider="unknown") == 4096

    def test_explicit_cap_stable_across_provider_switch(self, fake_65k_provider):
        # An explicit user/invocation cap is authoritative and must NOT change
        # when the provider (and its implicit default) changes.
        explicit = 3_000
        assert (
            resolve_output_reservation(None, explicit_cap=explicit, provider=_FAKE_PROVIDER)
            == explicit
        )
        assert (
            resolve_output_reservation(None, explicit_cap=explicit, provider="other")
            == explicit
        )
        # A final request cap is likewise stable and beats the provider default.
        assert resolve_output_reservation(
            {"max_tokens": 3_000}, provider=_FAKE_PROVIDER
        ) == 3_000


# ─────────────────────────────────────────────────────────────────────────────
# D. Final-cap precedence (clamped cap beats larger provider default)
# ─────────────────────────────────────────────────────────────────────────────

class TestFinalCapPrecedence:
    """provider default 65536 + finalized request cap 8192 → reserve 8192."""

    def test_final_cap_beats_provider_default(self, fake_65k_provider):
        assert (
            resolve_output_reservation(
                {"max_tokens": 8192}, provider=_FAKE_PROVIDER
            )
            == 8192
        )
        assert (
            resolve_output_reservation(
                {"max_completion_tokens": 8192}, provider=_FAKE_PROVIDER
            )
            == 8192
        )
        assert (
            resolve_output_reservation(
                {"max_output_tokens": 8192}, provider=_FAKE_PROVIDER
            )
            == 8192
        )
        # Bedrock Converse nests the cap under inferenceConfig.maxTokens.
        assert (
            resolve_output_reservation(
                {"inferenceConfig": {"maxTokens": 8192}}, provider="bedrock"
            )
            == 8192
        )

    def test_terminal_gate_reserves_final_clamped_cap(self, fake_65k_provider):
        """The terminal gate must reserve 8192 (the clamped final cap), NOT the
        65536 provider default — the exact user-given example."""
        budget = build_final_context_budget(
            {"model": "m", "messages": _msgs(1_000), "max_tokens": 8192},
            provider=_FAKE_PROVIDER,
            model="m",
        )
        assert budget.output_reservation == 8192
        # ...and the total is input + 8192 (not input + 65536).
        assert budget.total == (
            budget.input_tokens_estimate + 8192
        )

    def test_no_cap_with_profile_reserves_profile_cap(self, fake_65k_provider):
        """Converse of D: no max_tokens in final request + profile implicit
        65536 → terminal reservation must be 65536, not 4096."""
        budget = build_final_context_budget(
            {"model": "m", "messages": _msgs(1_000)},
            provider=_FAKE_PROVIDER,
            model="m",
        )
        assert budget.output_reservation == 65_536


# ─────────────────────────────────────────────────────────────────────────────
# E. Clamp-before-gate composition (#90877 semantics)
# ─────────────────────────────────────────────────────────────────────────────

class TestClampBeforeGateComposition:
    """The terminal gate judges the FINAL payload. A legitimate clamp that
    reduces max_tokens so ``input + final_cap <= effective`` must ALLOW; a
    request whose INPUT alone is at/over the effective window must REFUSE."""

    def _allow(self, input_tokens: int, final_cap: int, effective: int):
        """Assert the gate allows when input + final_cap <= effective."""
        # Build a budget with the exact input estimate and the FINAL clamped
        # cap, then enforce against the effective limit.
        budget = build_final_context_budget(
            {"messages": _msgs(max(1, input_tokens // 4)), "max_tokens": final_cap}
        )
        # Pin the input estimate to the exact target so the arithmetic is exact
        # (the rough estimator is approximate; the resolver arithmetic is not).
        object.__setattr__(
            budget, "input_tokens_estimate", input_tokens
        )
        object.__setattr__(
            budget, "total",
            input_tokens + budget.system_tokens_estimate
            + budget.tool_tokens_estimate + budget.output_reservation,
        )
        assert budget.output_reservation == final_cap
        try:
            enforce_final_context_budget(budget, ceiling=effective, reason="test")
            allowed = True
        except ContextCeilingExceeded:
            allowed = False
        return allowed

    def test_clamped_request_within_budget_is_allowed(self):
        """Original/default output allowance would overflow, but the FINAL
        clamped cap makes input + final_cap <= effective → gate ALLOWS."""
        # Effective window 100_000. Pre-clamp the output allowance was, say,
        # 200_000 (would overflow). A legitimate clamp sets it to 8_000.
        # input = 80_000 → 80_000 + 8_000 = 88_000 <= 100_000 → ALLOW.
        assert self._allow(80_000, 8_000, 100_000) is True

    def test_input_alone_at_or_over_window_is_refused(self):
        """input >= effective window → terminal refusal regardless of cap."""
        # input = 100_000, effective = 100_000, any cap → total > limit.
        assert self._allow(100_000, 1_000, 100_000) is False
        # input = 150_000 > 100_000 window → refuse.
        assert self._allow(150_000, 1_000, 100_000) is False

    def test_final_still_over_budget_after_transform_is_refused(self):
        """After all transformations, if input + final_cap > effective → refuse.

        input = 95_000, final_cap = 8_000 → 103_000 > 100_000 → REFUSE.
        (Contrast with the allow case: 80_000 + 8_000 = 88_000 <= 100_000.)
        """
        assert self._allow(95_000, 8_000, 100_000) is False

    def test_no_test_requires_rejecting_a_legitimate_clamp(self, fake_65k_provider):
        """Pinning the audit's semantic point: a request is NOT rejected merely
        because its PRE-CLAMP form would have exceeded the window — only its
        FINAL (clamped) form is judged. With the final cap, it fits; the gate
        must allow it."""
        # Effective 100_000. Final (clamped) cap 8_000, input 80_000 → fits.
        # Even though a hypothetical un-clamped 200_000 allowance would NOT fit,
        # the gate sees only the FINAL 8_000 and must allow.
        budget = build_final_context_budget(
            {"messages": _msgs(20_000), "max_tokens": 8_000},
            provider=_FAKE_PROVIDER, model="m",
        )
        object.__setattr__(budget, "input_tokens_estimate", 80_000)
        object.__setattr__(
            budget, "total",
            80_000 + budget.system_tokens_estimate
            + budget.tool_tokens_estimate + budget.output_reservation,
        )
        assert budget.output_reservation == 8_000  # clamped, not 65536
        # 80_000 + 8_000 = 88_000 <= 100_000 → no raise.
        enforce_final_context_budget(budget, ceiling=100_000, reason="clamp")


# ─────────────────────────────────────────────────────────────────────────────
# F. #92211 local-vs-native distinction (current equivalent contract)
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalVsNativeDistinction:
    """A LOCAL ContextCeilingExceeded is a refusal, NOT an API error: it must
    not be eligible for output-cap recovery and must not trigger a second
    provider call. A GENUINE native provider output-cap 400 (after the local
    gate allowed the request) remains eligible for the existing reduced-cap
    recovery path."""

    def test_local_refusal_is_not_an_output_cap_recovery_candidate(self):
        # A local refusal is not a provider error string at all — the recovery
        # predicates operate on provider error text, so a local refusal can
        # never be misrouted into reduced-cap recovery.
        exc = ContextCeilingExceeded(120_000, 100_000, reason="local")
        # The exception is NOT an httpx/requests error and carries no provider
        # status — it is a pure local refusal.
        assert isinstance(exc, Exception)
        err_text = str(exc)
        # The output-cap recovery classifier must not treat it as a recoverable
        # provider output-cap 400.
        assert is_output_cap_error(err_text) is False
        # (It has no parseable "available output tokens" number either.)
        assert parse_available_output_tokens_from_error(err_text) is None

    def test_genuine_native_output_cap_400_is_recoverable(self):
        # A genuine native provider 400 that the LOCAL gate did NOT raise (it
        # allowed the request) — the provider itself says the output cap is too
        # large — must remain eligible for the existing reduced-cap recovery.
        native_err = (
            "400 error code - 'Range of max_tokens should be [1, 65536], "
            "but got 81920 (request_id: abc123)'"
        )
        assert is_output_cap_error(native_err) is True
        # ...and the recoverable number is extractable (65536) so the reduced
        # cap can be applied.
        assert parse_available_output_tokens_from_error(native_err) == 65_536

    def test_local_refusal_and_native_400_are_distinct(self):
        # The two are categorically different: local refusal is terminal by
        # type; native 400 is a provider signal eligible for recovery.
        local = ContextCeilingExceeded(120_000, 100_000, reason="local")
        # A genuine native output-cap 400 (a phrasing the reduced-cap recovery
        # path recognises and can extract a bounded cap from).
        native = "Range of max_tokens should be [1, 65536], but got 81920"
        # Local refusal: NOT an output-cap error, no recoverable cap.
        assert not is_output_cap_error(str(local))
        assert parse_available_output_tokens_from_error(str(local)) is None
        # Native 400: IS an output-cap error with a recoverable cap (65536).
        assert is_output_cap_error(native)
        assert parse_available_output_tokens_from_error(native) == 65_536


# ─────────────────────────────────────────────────────────────────────────────
# Item 3 — Auxiliary refusal terminality (owning test)
# ─────────────────────────────────────────────────────────────────────────────

class TestAuxiliaryRefusalTerminality:
    """A LOCAL auxiliary ContextCeilingExceeded is terminal by TYPE: it must
    NOT trigger a same-provider transient retry, a credential refresh, or a
    fallback candidate — and it must result in ZERO provider calls.

    This proves the ``except ContextCeilingExceeded: raise`` guards are
    effective, independent of whether the retry predicate happens to match the
    exception today.
    """

    def test_local_refusal_is_terminal_no_retry_no_refresh_no_fallback(
        self, monkeypatch
    ):
        from types import SimpleNamespace
        from agent import auxiliary_client as ac
        from agent.model_metadata import ContextCeilingExceeded

        # ── Spy client that records every physical provider call ──────────
        class _SpyCompletions:
            def __init__(self, calls):
                self._calls = calls

            def create(self, **kwargs):
                self._calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=[]),
                        finish_reason="stop",
                    )],
                    usage=None, model="fake-model",
                )

        class _SpyChat:
            def __init__(self, calls):
                self.completions = _SpyCompletions(calls)

        class _SpyClient:
            def __init__(self):
                self.create_calls: list = []
                self.chat = _SpyChat(self.create_calls)
                self.base_url = "https://fake.example/v1"

        spy = _SpyClient()

        # ── Record if credential refresh / fallback ever fires ────────────
        refresh_calls: list = []

        def _no_refresh_provider(*a, **k):
            refresh_calls.append(("refresh_provider", a, k))
            return False

        def _no_nous_refresh(*a, **k):
            refresh_calls.append(("nous_refresh", a, k))
            return (None, None)

        def _no_fallback_candidate_sync(*a, **k):
            refresh_calls.append(("fallback_sync", a, k))
            raise RuntimeError("fallback candidate called — MUST NOT happen")

        def _no_fallback_candidate_async(*a, **k):
            refresh_calls.append(("fallback_async", a, k))
            raise RuntimeError("fallback candidate called — MUST NOT happen")

        # effective_context_length is imported from model_metadata inside
        # _call_llm_impl, so patch it there (not on the aux module). A tiny
        # ceiling (1000) makes the FINAL payload's budget exceed it → the
        # terminal gate refuses BEFORE any provider I/O.
        with patch.object(ac, "_get_cached_client", return_value=(spy, "fake-model")), \
             patch("agent.model_metadata.effective_context_length", return_value=1000), \
             patch.object(ac, "_refresh_provider_credentials", side_effect=_no_refresh_provider, creating=True), \
             patch.object(ac, "_refresh_nous_auxiliary_client", side_effect=_no_nous_refresh, creating=True), \
             patch.object(ac, "_call_fallback_candidate_sync", side_effect=_no_fallback_candidate_sync, creating=True), \
             patch.object(ac, "_call_fallback_candidate_async", side_effect=_no_fallback_candidate_async, creating=True):
            with pytest.raises(ContextCeilingExceeded):
                ac.call_llm(
                    task="compression",
                    provider="fake-65k",
                    model="fake-model",
                    base_url="https://fake.example/v1",
                    api_key="test-key",
                    messages=[
                        {"role": "system", "content": "test"},
                        # Large enough that input + reservation > 1000 ceiling.
                        {"role": "user", "content": "x" * 20_000},
                    ],
                    max_tokens=8192,
                )

        # ── Assertions: the refusal was terminal by type ───────────────────
        # 1. ZERO physical provider calls (no transient retry, no second call).
        assert len(spy.create_calls) == 0, (
            f"Physical provider .create was called {len(spy.create_calls)} times; "
            "a local ContextCeilingExceeded must result in ZERO provider I/O."
        )
        # 2. NO credential refresh (provider, Nous, or pool rotation).
        assert len(refresh_calls) == 0, (
            f"Credential refresh / fallback fired {len(refresh_calls)} times: "
            f"{refresh_calls}; a local refusal must not trigger any of these."
        )
