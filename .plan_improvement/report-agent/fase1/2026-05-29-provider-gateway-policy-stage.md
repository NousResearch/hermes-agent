## Summary

Tahap 3 Routing/Fallback Policy Foundation selesai: provider gateway sekarang punya policy object untuk menyusun kandidat route dari primary model, config gateway, dan fallback chain Hermes tanpa mengubah runtime routing.

## Changes Made

- **provider_gateway/policy.py**: menambahkan route candidate dataclass.
  - OLD:
    ```python
    # file did not exist
    ```
  - NEW:
    ```python
    @dataclass(frozen=True)
    class ProviderRouteCandidate:
        """One possible provider/model route."""

        provider: str
        model: str
        source: str
        base_url: str | None = None
        key_env: str | None = None
        api_key: str | None = None
    ```
  - Context: route provider perlu direpresentasikan secara eksplisit sebelum routing runtime disentuh.

- **provider_gateway/policy.py**: menambahkan policy object dan `next_after()`.
  - OLD:
    ```python
    # no route policy object existed
    ```
  - NEW:
    ```python
    @dataclass(frozen=True)
    class ProviderGatewayPolicy:
        """Ordered route policy assembled from gateway config and Hermes fallback state."""

        enabled: bool
        routing_strategy: str
        candidates: tuple[ProviderRouteCandidate, ...]

        def next_after(
            self,
            provider: str,
            model: str,
            *,
            base_url: str | None = None,
        ) -> ProviderRouteCandidate | None:
            """Return the next candidate after the matching current route."""
            current_key = _candidate_key(
                provider=provider,
                model=model,
                base_url=base_url,
                ignore_base_url=base_url is None,
            )
    ```
  - Context: Stage 3 hanya menentukan kandidat berikutnya; belum melakukan switch client/model.

- **provider_gateway/policy.py**: menambahkan builder dari agent state.
  - OLD:
    ```python
    # provider gateway could not inspect existing Hermes fallback state
    ```
  - NEW:
    ```python
    def build_gateway_policy(
        agent: Any,
        config: GatewayConfig | None = None,
    ) -> ProviderGatewayPolicy:
        """Build an ordered route policy without mutating the agent."""
        config = config if isinstance(config, GatewayConfig) else load_gateway_config()
        candidates: list[ProviderRouteCandidate] = []
        seen: set[tuple[str, str, str | None]] = set()

        primary_provider = _clean_text(getattr(agent, "provider", None))
        primary_model = _clean_text(getattr(agent, "model", None))
        primary_base_url = _clean_base_url(getattr(agent, "base_url", None))
    ```
  - Context: policy harus memanfaatkan struktur Hermes yang sudah ada, bukan membuat sistem fallback kedua.

- **provider_gateway/policy.py**: menambahkan urutan kandidat gateway enabled.
  - OLD:
    ```python
    # no provider gateway candidate ordering existed
    ```
  - NEW:
    ```python
    if config.enabled:
        for model in config.fallback_models:
            _append_candidate(
                candidates,
                seen,
                ProviderRouteCandidate(
                    provider=primary_provider,
                    model=_clean_text(model),
                    source="provider_gateway.routing.fallback_models",
                    base_url=primary_base_url,
                ),
            )

        for entry in getattr(agent, "_fallback_chain", []) or []:
            if not isinstance(entry, dict):
                continue
            fallback_provider = _clean_text(entry.get("provider"))
            fallback_base_url = _clean_base_url(entry.get("base_url"))
            if not fallback_base_url and fallback_provider == primary_provider:
                fallback_base_url = primary_base_url
    ```
  - Context: urutan route menjadi primary -> gateway fallback models -> existing `_fallback_chain`.

- **provider_gateway/policy.py**: menambahkan reason gate untuk fallback gateway.
  - OLD:
    ```python
    # no provider gateway reason gate existed
    ```
  - NEW:
    ```python
    _GATEWAY_FALLBACK_REASONS = {
        FailoverReason.billing,
        FailoverReason.rate_limit,
        FailoverReason.overloaded,
        FailoverReason.server_error,
        FailoverReason.timeout,
        FailoverReason.model_not_found,
        FailoverReason.provider_policy_blocked,
        FailoverReason.unknown,
    }


    def should_consider_gateway_fallback(reason: FailoverReason | None) -> bool:
        """Return whether a failure reason should try another route."""
        if reason is None:
            return True
        return reason in _GATEWAY_FALLBACK_REASONS
    ```
  - Context: tidak semua error boleh diarahkan ke route lain. Content policy block dan auth permanent tidak dianggap kandidat gateway fallback.

- **provider_gateway/__init__.py**: mengekspor policy API.
  - OLD:
    ```python
    from provider_gateway.config import GatewayConfig, load_gateway_config
    from provider_gateway.usage_tracker import ProviderUsageRecord, ProviderUsageTracker

    __all__ = [
        "GatewayConfig",
        "ProviderUsageRecord",
        "ProviderUsageTracker",
        "load_gateway_config",
    ]
    ```
  - NEW:
    ```python
    from provider_gateway.config import GatewayConfig, load_gateway_config
    from provider_gateway.policy import (
        ProviderGatewayPolicy,
        ProviderRouteCandidate,
        build_gateway_policy,
        should_consider_gateway_fallback,
    )
    from provider_gateway.usage_tracker import ProviderUsageRecord, ProviderUsageTracker

    __all__ = [
        "GatewayConfig",
        "ProviderGatewayPolicy",
        "ProviderRouteCandidate",
        "ProviderUsageRecord",
        "ProviderUsageTracker",
        "build_gateway_policy",
        "load_gateway_config",
        "should_consider_gateway_fallback",
    ]
    ```
  - Context: policy foundation menjadi bagian resmi dari package provider gateway.

- **tests/provider_gateway/test_policy.py**: menambahkan coverage untuk disabled no-op policy.
  - OLD:
    ```python
    # file did not exist
    ```
  - NEW:
    ```python
    def test_policy_keeps_disabled_gateway_as_primary_only() -> None:
        agent = SimpleNamespace(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
            _fallback_chain=[
                {"provider": "openai", "model": "gpt-4o"},
            ],
        )

        policy = build_gateway_policy(agent, GatewayConfig(enabled=False))

        assert policy.enabled is False
        assert policy.routing_strategy == "round-robin"
    ```
  - Context: gateway disabled tidak boleh mulai mempertimbangkan fallback chain.

- **tests/provider_gateway/test_policy.py**: menambahkan coverage ordering dan dedup kandidat.
  - OLD:
    ```python
    # no policy ordering tests existed
    ```
  - NEW:
    ```python
    def test_policy_orders_primary_gateway_models_then_existing_fallback_chain() -> None:
        agent = SimpleNamespace(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
            _fallback_chain=[
                {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "base_url": "https://api.openai.com/v1",
                    "key_env": "OPENAI_API_KEY",
                },
                {"provider": "zai", "model": "glm-4.7"},
            ],
        )
        config = GatewayConfig(
            enabled=True,
            routing_strategy="lowest-cost",
            fallback_models=["openai/gpt-5.4", "anthropic/claude-haiku-4.5"],
        )
    ```
  - Context: test mengunci kontrak bahwa policy memanfaatkan config gateway dan `_fallback_chain` existing secara berurutan.

## Technical Details

- Stage 3 tidak memodifikasi `agent._try_activate_fallback()` dan tidak mengganti client/model runtime.
- Policy hanya membaca:
  - `agent.provider`
  - `agent.model`
  - `agent.base_url`
  - `agent._fallback_chain`
  - `GatewayConfig`
- Saat gateway disabled, policy hanya berisi primary candidate. Ini menjaga default-off semantics tetap ketat.
- Saat gateway enabled, urutan kandidat adalah:
  - primary current route
  - `provider_gateway.routing.fallback_models` pada provider/base_url primary
  - existing `_fallback_chain` Hermes
- Dedup memakai tuple `(provider, model, base_url)` setelah normalisasi provider dan trailing slash base URL.
- Jika fallback chain memiliki provider sama dengan primary dan tidak menyebut `base_url`, policy menganggap base URL-nya sama dengan primary agar kandidat tidak terduplikasi.
- Reason gate sengaja konservatif: provider/server/rate/billing/model failures bisa fallback; content policy dan auth permanent tidak.
- Trade-off: `routing_strategy` saat ini disimpan di policy tetapi belum dieksekusi. Ini disengaja supaya tahap ini menjadi foundation aman sebelum runtime routing berubah.

## Results

- Provider gateway sekarang punya policy object yang bisa diuji tanpa network dan tanpa side effect.
- Existing Hermes fallback chain bisa dibaca sebagai kandidat route tanpa menyalin logic `_try_activate_fallback()`.
- Gateway disabled tetap primary-only.
- Gateway enabled bisa menyusun kandidat ordered dan memilih kandidat berikutnya dengan `next_after()`.
- Reason gate fallback sudah tersedia untuk tahap runtime berikutnya.

Verification:

```text
uv run --extra dev python -m pytest tests/provider_gateway/test_policy.py -q
5 passed in 0.07s

uv run --extra dev python -m pytest tests/provider_gateway -q
17 passed in 0.46s

uv run --extra dev python -m pytest tests/run_agent/test_provider_fallback.py -q
22 passed in 10.67s

uv run --extra dev python -m pytest tests/hermes_cli/test_config.py::TestLoadConfigDefaults -q
2 passed in 0.11s

uv run --extra dev python -m pytest tests/providers/test_provider_profiles.py -q
43 passed in 0.20s

uv run --extra dev python -m pytest tests/run_agent/test_provider_parity.py -q
89 passed in 22.16s

uv run --extra dev python -m ruff check provider_gateway tests/provider_gateway agent/chat_completion_helpers.py agent/agent_init.py
All checks passed!

git diff --check
passed
```

Working tree note:
- `.gitignore` still has an unrelated/pre-existing modification adding `.plan_improvement/`; it was not reverted.
- `.plan_improvement/` is ignored, so this report is present on disk but outside normal `git diff`.

## What to Do Next / Things to Consider

- Recommended Stage 4: Controlled Runtime Route Selection Hook.
- Keep it config-gated by `provider_gateway.enabled`.
- Use `build_gateway_policy()` and `should_consider_gateway_fallback()` instead of creating a second fallback implementation.
- First runtime hook should probably be observational or dry-run logging before actually switching route.
- If actual switching is added, reuse `_try_activate_fallback()` or extract shared route-activation code instead of duplicating credential/client setup.
- Later optional LiteLLM backend should remain behind `provider_gateway.backend = "litellm"` and should not add LiteLLM to the default dependency path.
