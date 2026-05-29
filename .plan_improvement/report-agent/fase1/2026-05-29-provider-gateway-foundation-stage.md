## Summary

Tahap 1 dari rencana peningkatan provider gateway selesai sebagai foundation yang masih default-off dan belum mengubah runtime routing Hermes. Report ini menjadi report gabungan untuk foundation config dan usage tracker, serta menggantikan report slice kecil `2026-05-29-provider-gateway-config-slice.md`.

Scope tahap ini:
- Menambahkan konfigurasi `provider_gateway` default-off.
- Menambahkan package `provider_gateway` yang import-safe.
- Menambahkan parser config kecil untuk gateway.
- Menambahkan SQLite usage tracker lokal untuk request provider.
- Menambahkan test RED/GREEN untuk config dan usage tracker.
- Menyesuaikan cadence laporan sesuai arahan user terbaru: laporan GUI/MCP digabung per 3-5 tahap besar, bukan tiap slice kecil.

Status: selesai untuk foundation tahap 1, siap lanjut ke tahap 2.

## Changes Made

1. `hermes_cli/config.py`

OLD:
```python
DEFAULT_CONFIG = {
    "model": "",
    "providers": {},
    "fallback_providers": [],
```

NEW:
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

OLD:
```python
_KNOWN_ROOT_KEYS = {
    "_config_version", "model", "providers", "fallback_model",
```

NEW:
```python
_KNOWN_ROOT_KEYS = {
    "_config_version", "model", "providers", "provider_gateway", "fallback_model",
```

2. `cli-config.yaml.example`

OLD:
```yaml
# =============================================================================
# OpenRouter Response Caching (only applies when using OpenRouter)
# =============================================================================
```

NEW:
```yaml
# =============================================================================
# Provider Gateway (optional, default OFF)
# =============================================================================
# Foundation for future multi-provider routing, usage tracking, and cost
# tracking across providers. Leaving this disabled preserves the current Hermes
# provider behavior.
#
# provider_gateway:
#   enabled: false
#   backend: native              # native | litellm
#   track_usage: true
#   track_cost: true
#   routing:
#     strategy: round-robin      # round-robin | lowest-cost | lowest-latency
#     fallback_models: []

# =============================================================================
# OpenRouter Response Caching (only applies when using OpenRouter)
# =============================================================================
```

3. `pyproject.toml`

OLD:
```toml
include = ["agent", "agent.*", "tools", "tools.*", "hermes_cli", "gateway", "gateway.*", "tui_gateway", "tui_gateway.*", "cron", "acp_adapter", "plugins", "plugins.*", "providers", "providers.*"]
```

NEW:
```toml
include = ["agent", "agent.*", "tools", "tools.*", "hermes_cli", "gateway", "gateway.*", "tui_gateway", "tui_gateway.*", "cron", "acp_adapter", "plugins", "plugins.*", "providers", "providers.*", "provider_gateway", "provider_gateway.*"]
```

4. `provider_gateway/__init__.py`

NEW:
```python
"""Provider gateway foundation.

This package holds opt-in multi-provider routing helpers. Importing it must not
change Hermes runtime behavior; integration points should stay config-gated.
"""

from provider_gateway.config import GatewayConfig, load_gateway_config
from provider_gateway.usage_tracker import ProviderUsageRecord, ProviderUsageTracker
```

5. `provider_gateway/config.py`

NEW:
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

NEW:
```python
def load_gateway_config(root_config: Mapping[str, Any] | None = None) -> GatewayConfig:
    """Load provider gateway config from a full Hermes config mapping."""
    if root_config is None:
        try:
            from hermes_cli.config import load_config

            root_config = load_config()
        except Exception as exc:
            logger.debug("Could not load Hermes config for provider gateway: %s", exc)
            root_config = {}
```

6. `provider_gateway/usage_tracker.py`

NEW:
```python
@dataclass(frozen=True)
class ProviderUsageRecord:
    """One provider request outcome."""

    provider: str
    model: str
    api_mode: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    status: str = "success"
    session_id: str | None = None
    error_type: str | None = None
    created_at: float | None = None
```

NEW:
```python
class ProviderUsageTracker:
    """Persist and summarize provider gateway usage records."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else get_hermes_home() / "provider_usage.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
```

NEW:
```python
def summarize_by_provider(self) -> list[dict[str, Any]]:
    """Return aggregate request, token, cost, and latency totals by provider."""
    ...
    SELECT
        provider,
        COUNT(*) AS request_count,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count,
        SUM(CASE WHEN status = 'success' THEN 0 ELSE 1 END) AS error_count,
        COALESCE(SUM(total_tokens), 0) AS total_tokens,
        COALESCE(SUM(estimated_cost_usd), 0.0) AS estimated_cost_usd,
        COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms
```

7. `tests/provider_gateway/test_config.py`

NEW:
```python
def test_default_config_exposes_provider_gateway_default_off() -> None:
    provider_gateway = DEFAULT_CONFIG["provider_gateway"]

    assert provider_gateway["enabled"] is False
    assert provider_gateway["backend"] == "native"
    assert provider_gateway["track_usage"] is True
    assert provider_gateway["track_cost"] is True
    assert provider_gateway["routing"]["strategy"] == "round-robin"
    assert provider_gateway["routing"]["fallback_models"] == []
```

8. `tests/provider_gateway/test_usage_tracker.py`

NEW:
```python
def test_usage_tracker_default_path_uses_hermes_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    tracker = ProviderUsageTracker()

    assert tracker.db_path == tmp_path / "provider_usage.db"
```

NEW:
```python
assert tracker.summarize_by_provider() == [
    {
        "provider": "openrouter",
        "request_count": 2,
        "success_count": 2,
        "error_count": 0,
        "total_tokens": 170,
        "estimated_cost_usd": 0.0133,
        "avg_latency_ms": 575.0,
    }
]
```

## Technical Details

- `provider_gateway` remains opt-in. `enabled` defaults to `False`, preserving current Hermes provider behavior.
- `GatewayConfig.from_dict()` validates backend and routing strategy against small allowlists:
  - backend: `native`, `litellm`
  - routing strategy: `round-robin`, `lowest-cost`, `lowest-latency`
- `load_gateway_config()` avoids import-time config coupling by importing `hermes_cli.config.load_config` only inside the function.
- `ProviderUsageTracker` writes to `get_hermes_home() / "provider_usage.db"` by default, matching Hermes profile-aware storage rules.
- Usage schema stores request metadata, tokens, cache tokens, reasoning tokens, estimated cost, latency, status, and error type.
- Summary API currently groups by provider and returns deterministic scalar dicts for request count, success/error count, total tokens, estimated cost, and average latency.
- Package discovery was updated so `provider_gateway` is included in setuptools packaging.
- TDD evidence:
  - Config test first failed before `provider_gateway` existed.
  - Usage tracker test first failed with `ModuleNotFoundError: No module named 'provider_gateway.usage_tracker'`.
  - Both areas now pass.
- Reporting cadence updated:
  - OLD: report to leader after each small implementation slice.
  - NEW: aggregate work into about 3-5 larger stages and report once per stage.

## Results

Verification commands run:

```bash
uv run --extra dev python -m pytest tests/provider_gateway/test_usage_tracker.py -q
```

Result:
```text
3 passed in 0.20s
```

```bash
uv run --extra dev python -m pytest tests/provider_gateway/test_config.py -q
```

Result:
```text
4 passed in 0.07s
```

```bash
uv run --extra dev python -m pytest tests/hermes_cli/test_config.py::TestLoadConfigDefaults -q
```

Result:
```text
2 passed in 0.10s
```

```bash
uv run --extra dev python -m pytest tests/providers/test_provider_profiles.py -q
```

Result:
```text
43 passed in 0.21s
```

```bash
uv run --extra dev python -m pytest tests/run_agent/test_provider_parity.py -q
```

Result:
```text
89 passed in 22.88s
```

```bash
git diff --check
```

Result:
```text
passed
```

Working tree note:
- `.gitignore` is modified with `.plan_improvement/`, but that change was pre-existing or external to this implementation slice and was not reverted.
- `.plan_improvement/` is ignored, so report-agent files are present on disk but not shown by normal `git diff`.

## What to Do Next / Things to Consider

Recommended next stage: Provider Gateway Integration Points.

Suggested stage 2 scope:
- Add an integration seam near `agent/chat_completion_helpers.py` or `agent/transports/chat_completions.py` that can record usage only when `provider_gateway.enabled` and `track_usage` are true.
- Keep current provider behavior unchanged when gateway is disabled.
- Capture response token usage from OpenAI-compatible responses without changing routing yet.
- Add tests proving disabled gateway is a no-op.
- Add tests proving enabled gateway records one request success and one error path.

Suggested later stages:
- Stage 3: fallback/routing policy abstraction using existing `fallback_model`, `_fallback_chain`, and credential pool semantics.
- Stage 4: optional LiteLLM backend adapter behind `provider_gateway.backend = "litellm"`, without making LiteLLM required for default install.
- Stage 5: CLI/status/reporting command or dashboard surface for provider usage summaries.
