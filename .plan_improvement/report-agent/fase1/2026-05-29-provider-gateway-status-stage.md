## Summary

Tahap 5 Provider Gateway Status Surface selesai: `/usage` sekarang dapat menampilkan ringkasan provider gateway secara compact tanpa mengaktifkan route switching.

## Changes Made

- **provider_gateway/status.py**: menambahkan helper status snapshot.
  - OLD:
    ```python
    # file did not exist
    ```
  - NEW:
    ```python
    def build_gateway_status(agent: Any) -> dict[str, Any]:
        """Build a compact provider gateway status snapshot for CLI/status surfaces."""
        config = getattr(agent, "_provider_gateway_config", None)
        if not isinstance(config, GatewayConfig):
            config = GatewayConfig()

        policy = build_gateway_policy(agent, config)
        tracker = getattr(agent, "_provider_usage_tracker", None)
        usage_summary = []
        if tracker is not None and hasattr(tracker, "summarize_by_provider"):
            try:
                usage_summary = tracker.summarize_by_provider()
            except Exception:
                usage_summary = []
    ```
  - Context: Stage 5 perlu visibility ke config/policy/usage tanpa menyentuh runtime switching.

- **provider_gateway/status.py**: menambahkan formatter compact untuk terminal/CLI.
  - OLD:
    ```python
    # no provider gateway terminal formatter existed
    ```
  - NEW:
    ```python
    def format_gateway_status_lines(status: dict[str, Any]) -> list[str]:
        """Format a status snapshot as compact terminal lines."""
        usage_summary = status.get("usage_summary") or []
        if not status.get("enabled") and not usage_summary:
            return []

        enabled_label = "enabled" if status.get("enabled") else "disabled"
        lines = [
            (
                "Provider Gateway: "
                f"{enabled_label} "
                f"(backend={status.get('backend', 'native')}, "
                f"strategy={status.get('routing_strategy', 'round-robin')}, "
                f"candidates={int(status.get('candidate_count') or 0)})"
            ),
            (
                "Tracking: "
                f"usage={_on_off(status.get('track_usage'))}, "
                f"cost={_on_off(status.get('track_cost'))}"
            ),
        ]
    ```
  - Context: output harus ringkas agar `/usage` tetap dapat discan.

- **provider_gateway/status.py**: menampilkan kandidat terakhir dan usage summary bila ada.
  - OLD:
    ```python
    # no provider gateway observed route or provider summary output existed
    ```
  - NEW:
    ```python
    candidate = status.get("last_observed_candidate")
    if isinstance(candidate, dict) and candidate.get("provider") and candidate.get("model"):
        lines.append(
            "Next observed: "
            f"{candidate['provider']}/{candidate['model']} "
            f"via {candidate.get('source') or 'unknown'}"
        )

    for row in usage_summary:
        if not isinstance(row, dict):
            continue
        provider = row.get("provider") or "unknown"
        request_count = int(row.get("request_count") or 0)
        success_count = int(row.get("success_count") or 0)
        error_count = int(row.get("error_count") or 0)
        total_tokens = int(row.get("total_tokens") or 0)
        cost = float(row.get("estimated_cost_usd") or 0.0)
        avg_latency = float(row.get("avg_latency_ms") or 0.0)
    ```
  - Context: user/operator bisa melihat route candidate dan usage tanpa membuka SQLite manual.

- **provider_gateway/__init__.py**: mengekspor status helpers.
  - OLD:
    ```python
    from provider_gateway.policy import (
        ProviderGatewayPolicy,
        ProviderRouteCandidate,
        build_gateway_policy,
        should_consider_gateway_fallback,
    )
    from provider_gateway.usage_tracker import ProviderUsageRecord, ProviderUsageTracker
    ```
  - NEW:
    ```python
    from provider_gateway.policy import (
        ProviderGatewayPolicy,
        ProviderRouteCandidate,
        build_gateway_policy,
        should_consider_gateway_fallback,
    )
    from provider_gateway.status import build_gateway_status, format_gateway_status_lines
    from provider_gateway.usage_tracker import ProviderUsageRecord, ProviderUsageTracker
    ```
  - Context: status surface menjadi public package API seperti config/policy/runtime data.

- **cli.py**: menambahkan provider gateway section ke `/usage`.
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
  - Context: `/usage` sudah menjadi tempat token/cost/account visibility; provider gateway status cocok ditambahkan di sana tanpa slash command baru.

- **tests/provider_gateway/test_status.py**: menambahkan test status disabled quiet.
  - OLD:
    ```python
    # file did not exist
    ```
  - NEW:
    ```python
    def test_status_disabled_without_usage_is_quiet() -> None:
        agent = SimpleNamespace(
            _provider_gateway_config=GatewayConfig(enabled=False),
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
            _fallback_chain=[],
            _provider_usage_tracker=None,
        )

        status = build_gateway_status(agent)

        assert status["enabled"] is False
        assert status["usage_summary"] == []
        assert format_gateway_status_lines(status) == []
    ```
  - Context: provider gateway disabled tidak boleh menambah noise di `/usage`.

- **tests/provider_gateway/test_status.py**: menambahkan test status enabled dengan policy, last candidate, dan usage summary.
  - OLD:
    ```python
    # no provider gateway status tests existed
    ```
  - NEW:
    ```python
    def test_status_reports_policy_and_last_observed_candidate() -> None:
        last_candidate = ProviderRouteCandidate(
            provider="openrouter",
            model="openai/gpt-5.4",
            source="provider_gateway.routing.fallback_models",
            base_url="https://openrouter.ai/api/v1",
        )
        agent = SimpleNamespace(
            _provider_gateway_config=GatewayConfig(
                enabled=True,
                backend="native",
                track_usage=True,
                track_cost=False,
                routing_strategy="lowest-cost",
                fallback_models=["openai/gpt-5.4"],
            ),
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
            _fallback_chain=[{"provider": "openai", "model": "gpt-4o"}],
            _provider_gateway_last_route_candidate=last_candidate,
            _provider_usage_tracker=FakeTracker(),
        )
    ```
  - Context: memastikan status helper bisa merangkum semua data yang dibuat Stage 1-4.

## Technical Details

- Stage 5 tidak menambah command baru; ia memperluas `/usage`, karena command itu sudah menampilkan token/cost/account usage.
- `build_gateway_status()` tidak membuat `ProviderUsageTracker()` baru. Ia hanya membaca tracker yang sudah ada di agent agar `/usage` tidak membuat DB secara tidak sengaja saat gateway disabled.
- `format_gateway_status_lines()` sengaja mengembalikan list kosong jika gateway disabled dan tidak ada usage summary, supaya default-off tetap tidak berisik.
- Status output mencakup:
  - enabled/backend/routing strategy/candidate count
  - tracking usage/cost on/off
  - last observed route candidate dari Stage 4
  - provider usage summary dari Stage 1 tracker
- Semua error status formatting ditelan di `cli.py`, konsisten dengan prinsip observability tidak boleh merusak CLI.

## Results

- Operator bisa melihat provider gateway state lewat `/usage` ketika gateway aktif atau ada usage summary.
- Tidak ada actual route switching yang ditambahkan.
- Default-off tetap tenang: tidak ada output provider gateway jika disabled dan tidak ada usage.
- Status helper menghubungkan hasil Stage 1-4 menjadi surface yang dapat dibaca.

Verification:

```text
uv run --extra dev python -m pytest tests/provider_gateway/test_status.py -q
3 passed in 0.07s

uv run --extra dev python -m pytest tests/provider_gateway -q
24 passed in 0.50s

uv run --extra dev python -m pytest tests/run_agent/test_provider_fallback.py -q
22 passed in 11.50s

uv run --extra dev python -m pytest tests/run_agent/test_provider_parity.py -q
89 passed in 23.48s

uv run --extra dev python -m pytest tests/providers/test_provider_profiles.py -q
43 passed in 0.22s

uv run --extra dev python -m pytest tests/hermes_cli/test_config.py::TestLoadConfigDefaults -q
2 passed in 0.09s

uv run --extra dev python -m ruff check provider_gateway tests/provider_gateway agent/chat_completion_helpers.py agent/conversation_loop.py agent/agent_init.py cli.py
All checks passed!

git diff --check
passed
```

Working tree note:
- `.gitignore` still has an unrelated/pre-existing modification adding `.plan_improvement/`; it was not reverted.
- `.plan_improvement/` is ignored, so this report is present on disk but outside normal `git diff`.

## What to Do Next / Things to Consider

- Batch 5 tahap sudah selesai: audit, foundation, integration, policy, runtime observation/status.
- Sebelum actual route switching, lakukan review khusus atas `_try_activate_fallback()` dan putuskan apakah perlu ekstrak shared route activation helper.
- Jika ingin lanjut setelah review, route switching harus punya config gate eksplisit terpisah dari `provider_gateway.enabled`, misalnya dry-run vs active.
- Tambahkan docs pengguna untuk `provider_gateway` setelah API final.
