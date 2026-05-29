## Summary

Tahap 4 Controlled Runtime Route Selection Hook selesai: provider gateway sekarang mengobservasi kandidat route berikutnya setelah error classification, tanpa mengubah model/client/runtime routing.

## Changes Made

- **provider_gateway/runtime.py**: menambahkan import policy foundation untuk observasi route.
  - OLD:
    ```python
    from provider_gateway.config import GatewayConfig, load_gateway_config
    from provider_gateway.usage_tracker import ProviderUsageRecord, ProviderUsageTracker
    ```
  - NEW:
    ```python
    from provider_gateway.config import GatewayConfig, load_gateway_config
    from provider_gateway.policy import (
        ProviderRouteCandidate,
        build_gateway_policy,
        should_consider_gateway_fallback,
    )
    from provider_gateway.usage_tracker import ProviderUsageRecord, ProviderUsageTracker
    ```
  - Context: runtime hook perlu memakai policy Stage 3, bukan membuat logic fallback baru.

- **provider_gateway/runtime.py**: menambahkan dry-run route observation helper.
  - OLD:
    ```python
    def record_provider_error_usage(
        agent: Any,
        error: BaseException,
        *,
        latency_seconds: float,
    ) -> bool:
        """Record an OpenAI-compatible provider request error if enabled."""
    ```
  - NEW:
    ```python
    def observe_gateway_route_selection(
        agent: Any,
        reason: Any,
    ) -> ProviderRouteCandidate | None:
        """Record the next gateway route candidate without mutating runtime routing."""
        config = _get_gateway_config(agent)
        if not config.enabled or not should_consider_gateway_fallback(reason):
            return None

        policy = build_gateway_policy(agent, config)
        candidate = policy.next_after(
            _agent_str(agent, "provider", ""),
            _agent_str(agent, "model", ""),
            base_url=getattr(agent, "base_url", None),
        )
        if candidate is None:
            return None

        try:
            setattr(agent, "_provider_gateway_last_route_candidate", candidate)
        except Exception:
            pass
        logger.debug(
            "Provider gateway observed next route candidate: provider=%s model=%s source=%s",
            candidate.provider,
            candidate.model,
            candidate.source,
        )
        return candidate
    ```
  - Context: Stage 4 harus memberi controlled runtime hook, tetapi belum boleh mengganti route secara nyata.

- **agent/conversation_loop.py**: memanggil route observation setelah structured error classification.
  - OLD:
    ```python
    logger.debug(
        "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
        classified.reason.value, classified.status_code,
        classified.retryable, classified.should_compress,
        classified.should_rotate_credential, classified.should_fallback,
    )

    if (
        classified.reason == FailoverReason.billing
    ```
  - NEW:
    ```python
    logger.debug(
        "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
        classified.reason.value, classified.status_code,
        classified.retryable, classified.should_compress,
        classified.should_rotate_credential, classified.should_fallback,
    )
    try:
        from provider_gateway.runtime import observe_gateway_route_selection

        observe_gateway_route_selection(agent, classified.reason)
    except Exception as _provider_gateway_exc:
        logger.debug(
            "Provider gateway route observation failed: %s",
            _provider_gateway_exc,
        )

    if (
        classified.reason == FailoverReason.billing
    ```
  - Context: hook ditempatkan setelah `classify_api_error()` supaya memakai `FailoverReason` existing dan sebelum recovery branches, namun tetap non-blocking.

- **tests/provider_gateway/test_runtime.py**: menambahkan import untuk test observation.
  - OLD:
    ```python
    from __future__ import annotations

    from types import SimpleNamespace

    import pytest

    from agent.chat_completion_helpers import interruptible_api_call
    from provider_gateway.config import GatewayConfig
    from provider_gateway.runtime import (
        record_provider_error_usage,
        record_provider_response_usage,
    )
    ```
  - NEW:
    ```python
    from __future__ import annotations

    import inspect
    from types import SimpleNamespace

    import pytest

    from agent import conversation_loop
    from agent.chat_completion_helpers import interruptible_api_call
    from agent.error_classifier import FailoverReason
    from provider_gateway.config import GatewayConfig
    from provider_gateway.policy import ProviderRouteCandidate
    from provider_gateway.runtime import (
        observe_gateway_route_selection,
        record_provider_error_usage,
        record_provider_response_usage,
    )
    ```
  - Context: tests perlu memverifikasi helper runtime dan hook di conversation loop.

- **tests/provider_gateway/test_runtime.py**: menambahkan disabled no-op observation test.
  - OLD:
    ```python
    # no route observation test existed
    ```
  - NEW:
    ```python
    def test_observe_route_selection_disabled_is_noop() -> None:
        agent = SimpleNamespace(
            _provider_gateway_config=GatewayConfig(enabled=False),
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
            _fallback_chain=[{"provider": "openai", "model": "gpt-4o"}],
        )

        candidate = observe_gateway_route_selection(agent, FailoverReason.rate_limit)

        assert candidate is None
        assert not hasattr(agent, "_provider_gateway_last_route_candidate")
    ```
  - Context: memastikan hook tidak membuat state observability ketika gateway disabled.

- **tests/provider_gateway/test_runtime.py**: menambahkan no-mutation observation test.
  - OLD:
    ```python
    # no no-mutation route observation test existed
    ```
  - NEW:
    ```python
    def test_observe_route_selection_records_candidate_without_mutating_route() -> None:
        agent = SimpleNamespace(
            _provider_gateway_config=GatewayConfig(
                enabled=True,
                fallback_models=["openai/gpt-5.4"],
            ),
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
            _fallback_chain=[{"provider": "openai", "model": "gpt-4o"}],
        )

        candidate = observe_gateway_route_selection(agent, FailoverReason.rate_limit)

        assert candidate == ProviderRouteCandidate(
            provider="openrouter",
            model="openai/gpt-5.4",
            source="provider_gateway.routing.fallback_models",
            base_url="https://openrouter.ai/api/v1",
        )
        assert agent.provider == "openrouter"
        assert agent.model == "anthropic/claude-sonnet-4.6"
        assert agent._provider_gateway_last_route_candidate == candidate
    ```
  - Context: observasi boleh menyimpan kandidat, tetapi tidak boleh mengganti route aktif.

- **tests/provider_gateway/test_runtime.py**: menambahkan test reason gate dan hook conversation loop.
  - OLD:
    ```python
    # no conversation loop hook assertion existed
    ```
  - NEW:
    ```python
    def test_observe_route_selection_ignores_non_fallback_reason() -> None:
        agent = SimpleNamespace(
            _provider_gateway_config=GatewayConfig(
                enabled=True,
                fallback_models=["openai/gpt-5.4"],
            ),
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
            _fallback_chain=[],
        )

        candidate = observe_gateway_route_selection(
            agent,
            FailoverReason.content_policy_blocked,
        )

        assert candidate is None
        assert not hasattr(agent, "_provider_gateway_last_route_candidate")


    def test_conversation_loop_invokes_gateway_route_observation_after_classification() -> None:
        source = inspect.getsource(conversation_loop.run_conversation)

        assert "observe_gateway_route_selection" in source
        assert "classified.reason" in source
    ```
  - Context: hook harus memakai classifier reason existing dan tetap menghormati reason gate.

## Technical Details

- Stage 4 masih tidak mengaktifkan runtime route switching.
- `observe_gateway_route_selection()` hanya:
  - membaca `GatewayConfig`
  - memakai `should_consider_gateway_fallback()`
  - membangun `ProviderGatewayPolicy`
  - menghitung `policy.next_after(current provider/model/base_url)`
  - menyimpan kandidat ke `agent._provider_gateway_last_route_candidate`
- Hook di `conversation_loop.py` dibungkus `try/except` agar observability tidak mengganggu retry/fallback existing.
- Tidak ada perubahan pada `_try_activate_fallback()`, credential pool recovery, client construction, atau fallback index.
- Ini memberi dasar aman untuk Stage 5 jika nanti route switching ingin diaktifkan dengan config gate tambahan.

## Results

- Gateway disabled tetap no-op dan tidak membuat state kandidat.
- Gateway enabled bisa mengobservasi kandidat route berikutnya berdasarkan policy Stage 3.
- Route aktif (`agent.provider`, `agent.model`) tidak berubah saat observasi.
- Non-fallback reason seperti `content_policy_blocked` diabaikan.
- Conversation loop sekarang memiliki hook observasi setelah error classification.

Verification:

```text
uv run --extra dev python -m pytest tests/provider_gateway/test_runtime.py -q
9 passed in 0.24s

uv run --extra dev python -m pytest tests/provider_gateway -q
21 passed in 0.49s

uv run --extra dev python -m pytest tests/run_agent/test_provider_fallback.py -q
22 passed in 11.21s

uv run --extra dev python -m pytest tests/run_agent/test_provider_parity.py -q
89 passed in 23.20s

uv run --extra dev python -m pytest tests/providers/test_provider_profiles.py -q
43 passed in 0.22s

uv run --extra dev python -m pytest tests/hermes_cli/test_config.py::TestLoadConfigDefaults -q
2 passed in 0.09s

uv run --extra dev python -m ruff check provider_gateway tests/provider_gateway agent/chat_completion_helpers.py agent/conversation_loop.py agent/agent_init.py
All checks passed!

git diff --check
passed
```

Working tree note:
- `.gitignore` still has an unrelated/pre-existing modification adding `.plan_improvement/`; it was not reverted.
- `.plan_improvement/` is ignored, so this report is present on disk but outside normal `git diff`.

## What to Do Next / Things to Consider

- Recommended next step: stop after 4 implementation stages unless leader wants Stage 5.
- If continuing, Stage 5 should avoid direct route switching unless a stronger config gate is added.
- Safer Stage 5 options:
  - CLI/status command to inspect provider gateway policy and last observed route candidate.
  - usage summary command backed by `ProviderUsageTracker.summarize_by_provider()`.
  - docs/config examples for `provider_gateway`.
- If actual switching is requested later, extract shared route activation from `_try_activate_fallback()` instead of duplicating credential/client setup.
