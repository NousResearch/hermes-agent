## Summary

Evaluasi menyeluruh atas batch Provider Gateway selesai: implementasi 5 tahap konsisten dengan prinsip default-off, observability-first, dan belum mengubah actual provider routing.

## Changes Made

- **.plan_improvement/report-agent/2026-05-29-provider-gateway-implementation-evaluation.md**: menambahkan report evaluasi menyeluruh.
  - OLD:
    ```md
    # file did not exist
    ```
  - NEW:
    ```md
    ## Summary
    Evaluasi menyeluruh atas batch Provider Gateway selesai...

    ## Technical Details
    - Scope implementation reviewed
    - Verification evidence reviewed
    - Risks and intentional limitations documented
    ```
  - Context: leader meminta evaluasi detail atas seluruh hasil implementasi, bukan hanya laporan per-stage.

- **hermes_cli/config.py**: dievaluasi sebagai config gate utama.
  - OLD:
    ```python
    DEFAULT_CONFIG = {
        "model": "",
        "providers": {},
        "fallback_providers": [],
    ```
  - NEW:
    ```python
    DEFAULT_CONFIG = {
        "model": "",
        "providers": {},
        "provider_gateway": {
            "enabled": False,
            "backend": "native",
            "track_usage": True,
            "track_cost": True,
            "routing": {
                "strategy": "round-robin",
                "fallback_models": [],
            },
        },
        "fallback_providers": [],
    ```
  - Context: config default-off berhasil menjaga behavior existing tetap tidak berubah sampai user eksplisit enable.

- **provider_gateway/config.py**: dievaluasi sebagai parser config lokal.
  - OLD:
    ```python
    # provider_gateway package did not exist
    ```
  - NEW:
    ```python
    @dataclass(frozen=True)
    class GatewayConfig:
        """Runtime configuration for provider gateway features."""

        enabled: bool = False
        backend: str = "native"
        track_usage: bool = True
        track_cost: bool = True
        routing_strategy: str = "round-robin"
        fallback_models: list[str] = field(default_factory=list)
    ```
  - Context: parser terisolasi dan mudah dites; tidak perlu menyentuh provider profiles atau transport adapters.

- **provider_gateway/usage_tracker.py**: dievaluasi sebagai storage usage lokal.
  - OLD:
    ```python
    # no provider gateway usage storage existed
    ```
  - NEW:
    ```python
    class ProviderUsageTracker:
        """Persist and summarize provider gateway usage records."""

        def __init__(self, db_path: str | Path | None = None) -> None:
            self.db_path = Path(db_path) if db_path is not None else get_hermes_home() / "provider_usage.db"
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
    ```
  - Context: storage memakai `get_hermes_home()` sehingga profile-aware dan tidak memakai hardcoded `~/.hermes`.

- **agent/chat_completion_helpers.py**: dievaluasi sebagai integration point usage tracking.
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
  - Context: hook berada pada seam non-streaming OpenAI-compatible request; failure tracking tidak boleh memutus chat.

- **provider_gateway/policy.py**: dievaluasi sebagai foundation route policy.
  - OLD:
    ```python
    # no provider gateway routing policy existed
    ```
  - NEW:
    ```python
    @dataclass(frozen=True)
    class ProviderGatewayPolicy:
        """Ordered route policy assembled from gateway config and Hermes fallback state."""

        enabled: bool
        routing_strategy: str
        candidates: tuple[ProviderRouteCandidate, ...]
    ```
  - Context: policy membaca state existing (`provider`, `model`, `base_url`, `_fallback_chain`) tanpa membuat mekanisme fallback kedua.

- **agent/conversation_loop.py**: dievaluasi sebagai controlled runtime observation hook.
  - OLD:
    ```python
    logger.debug(
        "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
        classified.reason.value, classified.status_code,
        classified.retryable, classified.should_compress,
        classified.should_rotate_credential, classified.should_fallback,
    )
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
    ```
  - Context: observation hook memakai `FailoverReason` existing dan tidak melakukan route switching.

- **cli.py**: dievaluasi sebagai status surface.
  - OLD:
    ```python
    if account_lines:
        print()
        for line in account_lines:
            print(line)

    if self.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    ```
  - NEW:
    ```python
    if account_lines:
        print()
        for line in account_lines:
            print(line)

    try:
        from provider_gateway.status import (
            build_gateway_status,
            format_gateway_status_lines,
        )

        gateway_lines = format_gateway_status_lines(build_gateway_status(agent))
    except Exception:
        gateway_lines = []
    if gateway_lines:
        print()
        for line in gateway_lines:
            print(f"  {line}")

    if self.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    ```
  - Context: `/usage` menjadi visibility surface tanpa menambah command baru dan tanpa mengaktifkan switching.

## Technical Details

### Scope yang dievaluasi

- Existing tracked files touched:
  - `agent/agent_init.py`
  - `agent/chat_completion_helpers.py`
  - `agent/conversation_loop.py`
  - `cli-config.yaml.example`
  - `cli.py`
  - `hermes_cli/config.py`
  - `pyproject.toml`
- New package:
  - `provider_gateway/__init__.py`
  - `provider_gateway/config.py`
  - `provider_gateway/usage_tracker.py`
  - `provider_gateway/runtime.py`
  - `provider_gateway/policy.py`
  - `provider_gateway/status.py`
- New tests:
  - `tests/provider_gateway/test_config.py`
  - `tests/provider_gateway/test_usage_tracker.py`
  - `tests/provider_gateway/test_runtime.py`
  - `tests/provider_gateway/test_policy.py`
  - `tests/provider_gateway/test_status.py`

### Evaluation findings

1. **No blocking issue found in verified scope.**
   - Config defaults are off.
   - Runtime hooks are defensive.
   - Existing fallback behavior is not replaced.
   - Tests cover disabled no-op behavior, enabled usage tracking, policy construction, route observation, and status formatting.

2. **Architecture direction is coherent with live Hermes code.**
   - Implementation does not introduce a new provider adapter hierarchy.
   - It respects the Fase 0 finding that `ProviderProfile` is declarative.
   - It uses existing `normalize_usage()`, `estimate_usage_cost()`, `FailoverReason`, and `_fallback_chain` semantics.

3. **Runtime blast radius is intentionally limited.**
   - Usage tracking is only attached to non-streaming `chat_completions`.
   - Route policy is observational only.
   - No actual route switching is added.
   - No credential pool/client rebuild logic is duplicated.

4. **Failure mode is intentionally fail-open for chat.**
   - If tracking, status formatting, config load, cost estimate, or route observation fails, the model call continues.
   - Failures are debug-logged rather than surfaced as user-facing errors.

5. **Storage design is acceptable for foundation stage.**
   - SQLite writes use a fresh connection per `record_usage()`.
   - DB path is profile-aware via `get_hermes_home()`.
   - Summary query is deterministic and grouped by provider.
   - Schema has indexes for provider/time and session.

### Intentional limitations

- Actual gateway route switching is not implemented.
- Streaming `chat_completions` usage is not tracked by provider gateway yet.
- `codex_responses`, `anthropic_messages`, and `bedrock_converse` usage are not tracked by provider gateway yet.
- **[x] SQLite schema has no explicit migration/version table yet.** *(TERATASI: Versi skema `SCHEMA_VERSION = 1` dan tabel `provider_usage_schema_version` telah ditambahkan di sesi ini)*
- `/usage` status surface is compact; it is not a full analytics UI.
- `provider_gateway.routing_strategy` is parsed and exposed but not executed.

### Risks to monitor

- If actual switching is added later, duplicating `_try_activate_fallback()` would be risky. It currently handles API mode, client rebuild, provider headers, credential pool contamination, context compressor, prompt caching, timeout, and fallback index.
- **[x] If gateway is enabled in high-concurrency gateway deployments, SQLite write contention should be monitored.** *(TERATASI: Ketahanan konkurensi ditingkatkan secara signifikan melalui mode WAL, `synchronous = NORMAL`, dan `busy_timeout = 5000` via helper `_connect()`)*
- If users expect streaming usage in `/usage`, provider gateway summary may undercount until streaming tracking is added.
- `.gitignore` has an external/pre-existing `.plan_improvement/` change. I did not revert it.

## Results

### Measurable implementation result

- Provider gateway package added: 6 files, 760 lines.
- Provider gateway tests added: 5 files, 721 lines.
- Stage reports written:
  - `.plan_improvement/report-agent/2026-05-29-fase-0-live-code-audit.md`
  - `.plan_improvement/report-agent/2026-05-29-provider-gateway-foundation-stage.md`
  - `.plan_improvement/report-agent/2026-05-29-provider-gateway-integration-stage.md`
  - `.plan_improvement/report-agent/2026-05-29-provider-gateway-policy-stage.md`
  - `.plan_improvement/report-agent/2026-05-29-provider-gateway-runtime-observation-stage.md`
  - `.plan_improvement/report-agent/2026-05-29-provider-gateway-status-stage.md`

### Final verification

```text
uv run --extra dev python -m pytest tests/provider_gateway -q
24 passed in 0.44s
```

```text
uv run --extra dev python -m pytest tests/run_agent/test_provider_fallback.py tests/run_agent/test_provider_parity.py tests/providers/test_provider_profiles.py tests/hermes_cli/test_config.py::TestLoadConfigDefaults -q
156 passed in 31.82s
```

```text
uv run --extra dev python -m ruff check provider_gateway tests/provider_gateway agent/chat_completion_helpers.py agent/conversation_loop.py agent/agent_init.py cli.py
All checks passed!
```

```text
git diff --check
passed
```

### Evaluation conclusion

Implementation is acceptable as a foundation batch. It adds useful provider gateway configuration, usage tracking, policy modeling, route observation, and status output while keeping the most dangerous part, actual route switching, out of scope.

## What to Do Next / Things to Consider

1. **Do a focused design review before actual route switching.**
   - Review `_try_activate_fallback()`.
   - Extract shared activation logic if needed.
   - Do not duplicate credential/client/context/prompt-cache handling.

2. **Add a separate active-switching gate.**
   - Keep `provider_gateway.enabled` as foundation enablement.
   - Add a separate `provider_gateway.routing.mode: observe | active` or equivalent before route switching.

3. **Add streaming tracking only after deciding token source.**
   - Existing streaming path collects usage chunks.
   - Provider gateway should reuse that normalized usage path, not create a second parser.

4. **Consider schema versioning before broad use.**
   - Add `schema_version` or migrations if the usage DB will be long-lived.

5. **Docs should be updated after API is stable.**
   - `cli-config.yaml.example` exists, but user docs should wait until route switching semantics are final.

6. **Optional hardening.**
   - Add tests for quoted string booleans in `GatewayConfig.from_dict()` if config inputs may come from non-YAML sources.
   - Add a lightweight status test for `/usage` output if CLI command tests have stable capture helpers.
