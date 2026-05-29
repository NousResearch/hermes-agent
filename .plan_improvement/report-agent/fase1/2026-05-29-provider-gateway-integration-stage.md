## Summary

Tahap 2 Provider Gateway Integration Points selesai: usage tracking sekarang tersambung ke jalur non-streaming `chat_completions` secara config-gated dan tetap no-op saat gateway disabled.

## Changes Made

- **.plan_improvement/report-agent/2026-05-29-provider-gateway-foundation-stage.md**: report Stage 1 dijadikan report gabungan untuk foundation config dan usage tracker.
  - OLD:
    ```md
    Tahap 1 dari rencana peningkatan provider gateway selesai sebagai foundation yang masih default-off dan belum mengubah runtime routing Hermes.
    ```
  - NEW:
    ```md
    Tahap 1 dari rencana peningkatan provider gateway selesai sebagai foundation yang masih default-off dan belum mengubah runtime routing Hermes. Report ini menjadi report gabungan untuk foundation config dan usage tracker, serta menggantikan report slice kecil `2026-05-29-provider-gateway-config-slice.md`.
    ```
  - Context: leader meminta dua report kecil sebelumnya digabung supaya cadence laporan mengikuti arahan user terbaru.

- **.plan_improvement/report-agent/2026-05-29-provider-gateway-config-slice.md**: report slice kecil dihapus karena sudah terserap ke report Stage 1.
  - OLD:
    ```md
    ## Summary
    Provider Gateway config foundation selesai...
    ```
  - NEW:
    ```md
    Removed; content merged into 2026-05-29-provider-gateway-foundation-stage.md.
    ```
  - Context: menjaga `report-agent` tetap per tahap, bukan per micro-slice.

- **provider_gateway/runtime.py**: menambahkan runtime hook untuk mencatat success/error provider usage.
  - OLD:
    ```python
    # file did not exist
    ```
  - NEW:
    ```python
    def record_provider_response_usage(
        agent: Any,
        response: Any,
        *,
        latency_seconds: float,
    ) -> bool:
        """Record a successful OpenAI-compatible provider response if enabled."""
        config = _get_gateway_config(agent)
        if not _should_track_usage(agent, config):
            return False

        raw_usage = getattr(response, "usage", None)
        usage = normalize_usage(
            raw_usage,
            provider=getattr(agent, "provider", None),
            api_mode=getattr(agent, "api_mode", None),
        )
        estimated_cost = _estimate_cost_usd(agent, usage, config)
    ```
  - Context: Stage 1 baru menyediakan config dan storage; Stage 2 butuh seam runtime yang bisa dipanggil dari request path.

- **provider_gateway/runtime.py**: menambahkan error usage hook.
  - OLD:
    ```python
    # no error tracking helper existed
    ```
  - NEW:
    ```python
    def record_provider_error_usage(
        agent: Any,
        error: BaseException,
        *,
        latency_seconds: float,
    ) -> bool:
        """Record an OpenAI-compatible provider request error if enabled."""
        config = _get_gateway_config(agent)
        if not _should_track_usage(agent, config):
            return False

        return _record_usage(
            agent,
            ProviderUsageRecord(
                provider=_agent_str(agent, "provider", "unknown"),
                model=_agent_str(agent, "model", "unknown"),
                api_mode=_agent_str(agent, "api_mode", "chat_completions"),
                latency_ms=round(max(0.0, latency_seconds) * 1000.0, 2),
                status="error",
                session_id=getattr(agent, "session_id", None),
                error_type=type(error).__name__,
            ),
        )
    ```
  - Context: usage tracker perlu menghitung error path, bukan hanya response sukses.

- **provider_gateway/runtime.py**: menambahkan gating defensif.
  - OLD:
    ```python
    # no runtime gating existed
    ```
  - NEW:
    ```python
    def _should_track_usage(agent: Any, config: GatewayConfig) -> bool:
        return (
            config.enabled
            and config.track_usage
            and _agent_str(agent, "api_mode", "") == "chat_completions"
        )
    ```
  - Context: integrasi harus tetap default-off dan dibatasi ke OpenAI-compatible chat completions dulu.

- **agent/chat_completion_helpers.py**: memasang success tracking setelah request non-streaming `chat.completions.create`.
  - OLD:
    ```python
    result["response"] = request_client.chat.completions.create(**api_kwargs)
    ```
  - NEW:
    ```python
    result["response"] = request_client.chat.completions.create(**api_kwargs)
    try:
        from provider_gateway.runtime import record_provider_response_usage

        record_provider_response_usage(
            agent,
            result["response"],
            latency_seconds=time.monotonic() - provider_gateway_started_at,
        )
    except Exception as gateway_exc:
        logger.debug(
            "Provider gateway response tracking failed: %s",
            gateway_exc,
        )
    ```
  - Context: hook ditempatkan pada seam request yang sudah punya raw response dan tetap tidak mengganggu chat jika tracker gagal.

- **agent/chat_completion_helpers.py**: memasang error tracking di exception path non-streaming.
  - OLD:
    ```python
    except Exception as e:
        result["error"] = e
    ```
  - NEW:
    ```python
    except Exception as e:
        if agent.api_mode == "chat_completions":
            try:
                from provider_gateway.runtime import record_provider_error_usage

                record_provider_error_usage(
                    agent,
                    e,
                    latency_seconds=time.monotonic() - provider_gateway_started_at,
                )
            except Exception as gateway_exc:
                logger.debug(
                    "Provider gateway error tracking failed: %s",
                    gateway_exc,
                )
        result["error"] = e
    ```
  - Context: error tracking tidak boleh mengubah retry/fallback behavior existing.

- **agent/agent_init.py**: memuat config gateway sekali saat agent init dan menyiapkan tracker lazy slot.
  - OLD:
    ```python
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "none"
    ```
  - NEW:
    ```python
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "none"
    try:
        from provider_gateway.config import load_gateway_config

        agent._provider_gateway_config = load_gateway_config()
    except Exception as _provider_gateway_exc:
        logger.debug(
            "Provider gateway config initialization failed: %s",
            _provider_gateway_exc,
        )
        agent._provider_gateway_config = None
    agent._provider_usage_tracker = None
    ```
  - Context: menghindari reload config di setiap request untuk agent normal; tracker tetap dibuat lazy saat benar-benar enabled.

- **tests/provider_gateway/test_runtime.py**: menambahkan test runtime dan integration seam.
  - OLD:
    ```python
    # file did not exist
    ```
  - NEW:
    ```python
    def test_response_usage_disabled_is_noop() -> None:
        tracker = CapturingTracker()
        agent = SimpleNamespace(
            _provider_gateway_config=GatewayConfig(enabled=False),
            _provider_usage_tracker=tracker,
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            api_mode="chat_completions",
        )

        recorded = record_provider_response_usage(
            agent,
            SimpleNamespace(usage=_response_usage()),
            latency_seconds=1.25,
        )

        assert recorded is False
        assert tracker.records == []
    ```
  - Context: membuktikan default-off/no-op behavior.

- **tests/provider_gateway/test_runtime.py**: menambahkan test success/error pada `interruptible_api_call`.
  - OLD:
    ```python
    # no provider_gateway integration tests existed
    ```
  - NEW:
    ```python
    def test_interruptible_api_call_records_success_when_gateway_enabled() -> None:
        tracker = CapturingTracker()
        response = SimpleNamespace(usage=_response_usage(), choices=[SimpleNamespace()])
        agent = FakeAgent(response, tracker)

        result = interruptible_api_call(agent, {"model": agent.model, "messages": []})

        assert result is response
        assert len(tracker.records) == 1
        assert tracker.records[0].status == "success"
        assert tracker.records[0].total_tokens == 20
    ```
  - Context: test mengunci bahwa hook benar-benar dipanggil dari jalur request non-streaming, bukan hanya helper standalone.

## Technical Details

- Integrasi dipasang di `interruptible_api_call()` karena titik ini punya:
  - raw response `chat.completions.create()`
  - exception aktual dari provider
  - latency per physical request memakai `time.monotonic()`
  - worker-local client lifecycle yang sudah mapan
- Hook dibatasi ke `api_mode == "chat_completions"` untuk Stage 2. Streaming, Codex Responses, Anthropic native, dan Bedrock tidak diubah.
- Runtime helper menggunakan `normalize_usage()` dari `agent.usage_pricing` agar token bucket konsisten dengan session accounting yang sudah ada.
- Cost estimate memakai `estimate_usage_cost()` hanya jika `provider_gateway.track_cost` true. Jika pricing tidak tersedia atau estimator gagal, nilai cost jatuh ke `0.0` dan chat tetap berjalan.
- Tracking failure ditelan dan hanya masuk debug log. Prinsipnya observability tidak boleh memutus percakapan model.
- `agent._provider_gateway_config` dimuat saat init untuk agent normal; fake/test agent masih bisa mengoverride langsung.
- `ProviderUsageTracker` tetap lazy, sehingga DB `provider_usage.db` tidak dibuat saat gateway disabled.
- Trade-off: Stage 2 mencatat success/error per non-streaming physical request. Retry/fallback lanjutan akan terekam sebagai beberapa request jika gateway enabled. Ini cocok untuk observability provider, tapi bukan summary final turn.

## Results

- Gateway disabled terbukti no-op.
- Gateway enabled bisa mencatat response success dari raw usage OpenAI-compatible.
- Gateway enabled bisa mencatat error type dan latency saat provider request melempar exception.
- Hook integration di `interruptible_api_call()` terbukti mencatat success/error tanpa mengubah return/raise behavior.
- Report kecil Stage 1 sudah digabung dan report slice config kecil dihapus.

Verification:

```text
uv run --extra dev python -m pytest tests/provider_gateway/test_runtime.py -q
5 passed in 0.19s

uv run --extra dev python -m pytest tests/provider_gateway -q
12 passed in 0.47s

uv run --extra dev python -m pytest tests/hermes_cli/test_config.py::TestLoadConfigDefaults -q
2 passed in 0.11s

uv run --extra dev python -m pytest tests/providers/test_provider_profiles.py -q
43 passed in 0.24s

uv run --extra dev python -m pytest tests/run_agent/test_provider_parity.py -q
89 passed in 21.46s

uv run --extra dev python -m ruff check provider_gateway tests/provider_gateway agent/chat_completion_helpers.py agent/agent_init.py
All checks passed!

git diff --check
passed
```

Working tree note:
- `.gitignore` still has an unrelated/pre-existing modification adding `.plan_improvement/`; it was not reverted.
- `.plan_improvement/` is ignored, so these stage reports are present on disk but outside normal `git diff`.

## What to Do Next / Things to Consider

- Recommended Stage 3: Routing/Fallback Policy Foundation.
- Use existing Hermes semantics first:
  - `fallback_model`
  - `_fallback_chain`
  - credential pool / failover behavior
  - existing `FailoverReason` classification
- Avoid replacing `ProviderProfile` with a new adapter hierarchy; Fase 0 showed current provider profiles are declarative config, not transport adapters.
- Add tests for policy selection without changing runtime routing yet.
- After policy foundation, consider a controlled routing hook that remains disabled unless `provider_gateway.enabled` is true.
- Later Stage 4 can add optional LiteLLM backend behind `provider_gateway.backend = "litellm"` without adding LiteLLM to default install.
