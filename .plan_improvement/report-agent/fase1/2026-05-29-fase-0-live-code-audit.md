# Fase 0 Live Code Audit - Provider Gateway Improvement

Tanggal: 2026-05-29
Project: hermes_agent
Status: selesai untuk audit awal, belum ada implementasi fitur.

## Summary

Audit Fase 0 mengonfirmasi bahwa Hermes sudah memiliki arsitektur provider deklaratif berbasis `ProviderProfile`, transport terpisah di `agent/transports`, fallback/credential recovery dasar, session-level usage/cost tracking, dan test coverage yang cukup besar. Implementasi multi-provider berikutnya harus memperluas sistem yang ada, bukan membuat adapter provider paralel.

## Scope Audit

File dan area yang dibaca:

- `providers/base.py`
- `providers/__init__.py`
- `plugins/model-providers/openrouter/__init__.py`
- `agent/agent_init.py`
- `run_agent.py`
- `agent/agent_runtime_helpers.py`
- `agent/chat_completion_helpers.py`
- `agent/transports/chat_completions.py`
- `agent/conversation_loop.py`
- `model_tools.py`
- `toolsets.py`
- `tools/registry.py`
- `hermes_state.py`
- `cli-config.yaml.example`
- `hermes_cli/config.py`
- `pyproject.toml`
- `tests/`

## Key Findings

### 1. Provider architecture is declarative, not adapter-owned transport

`providers/base.py` explicitly states that provider profiles describe behavior and do not own client construction, credential rotation, or streaming.

Current shape:

```python
@dataclass
class ProviderProfile:
    name: str
    api_mode: str = "chat_completions"
    aliases: tuple = ()
    display_name: str = ""
    description: str = ""
    signup_url: str = ""
    env_vars: tuple = ()
    base_url: str = ""
    models_url: str = ""
    auth_type: str = "api_key"
    supports_health_check: bool = True
    fallback_models: tuple = ()
    hostname: str = ""
    default_headers: dict[str, str] = field(default_factory=dict)
    fixed_temperature: Any = None
    default_max_tokens: int | None = None
    default_aux_model: str = ""
```

Provider hooks already exist:

```python
def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return messages

def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
    return {}

def build_api_kwargs_extras(
    self,
    *,
    reasoning_config: dict | None = None,
    **context: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return {}, {}
```

Decision:

- Do not add a new `ProviderAdapter` ABC with `complete()` or `stream()`.
- Add routing/capability metadata to `ProviderProfile` only if live code proves it is needed.
- Keep provider transport in the existing AIAgent/transport pipeline.

### 2. Provider plugin discovery is already lazy and override-friendly

`providers/__init__.py` discovers provider profiles from:

- bundled plugins: `plugins/model-providers/<name>/`
- user plugins: `$HERMES_HOME/plugins/model-providers/<name>/`
- legacy single-file modules under `providers/*.py`

Current registration shape:

```python
def register_provider(profile: ProviderProfile) -> None:
    _REGISTRY[profile.name] = profile
    for alias in profile.aliases:
        _ALIASES[alias] = profile.name
```

There are 28 bundled model-provider plugin init files. OpenRouter is representative: it subclasses `ProviderProfile`, overrides catalog fetching and request extras, then registers the profile.

```python
class OpenRouterProfile(ProviderProfile):
    def fetch_models(self, *, api_key: str | None = None, timeout: float = 8.0) -> list[str] | None:
        ...

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        ...

openrouter = OpenRouterProfile(
    name="openrouter",
    aliases=("or",),
    env_vars=("OPENROUTER_API_KEY",),
    base_url="https://openrouter.ai/api/v1",
    models_url="https://openrouter.ai/api/v1/models",
)

register_provider(openrouter)
```

Decision:

- A new provider should be a `ProviderProfile` plugin unless it is a fundamentally new transport.
- LiteLLM integration should probably be a backend/router path, not a replacement for provider plugin discovery.

### 3. LLM callsite is split across helpers and transports

`run_agent.py` is now mostly a facade. `AIAgent.__init__` delegates to `agent.agent_init.init_agent`, and `run_conversation` delegates to `agent.conversation_loop.run_conversation`.

Client construction is centralized in `agent/agent_runtime_helpers.py`:

```python
def create_openai_client(agent, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
    ...
    client = _ra().OpenAI(**client_kwargs)
    ...
    return client
```

Non-streaming chat completion calls happen in `agent/chat_completion_helpers.py`:

```python
request_client = _set_request_client(
    agent._create_request_openai_client(
        reason="chat_completion_request",
        api_kwargs=api_kwargs,
    )
)
result["response"] = request_client.chat.completions.create(**api_kwargs)
```

Streaming chat completion calls happen in the same helper:

```python
request_client = _set_request_client(
    agent._create_request_openai_client(
        reason="chat_completion_stream_request",
        api_kwargs=stream_kwargs,
    )
)
stream = request_client.chat.completions.create(**stream_kwargs)
```

Request kwargs are built through transport classes. For known providers, `agent/chat_completion_helpers.py` passes `provider_profile` into `agent/transports/chat_completions.py`.

```python
return _ct.build_kwargs(
    model=agent.model,
    messages=api_messages,
    tools=tools_for_api,
    base_url=agent.base_url,
    timeout=agent._resolved_api_call_timeout(),
    max_tokens=agent.max_tokens,
    provider_profile=_profile,
    ...
)
```

`agent/transports/chat_completions.py` then delegates provider quirks to profile hooks:

```python
if _profile:
    return self._build_kwargs_from_profile(
        _profile, model, sanitized, tools, params
    )
```

Decision:

- The safest integration point for request shaping is `agent/transports/chat_completions.py`.
- The safest integration point for backend selection is near `agent/chat_completion_helpers.py` or client construction helpers, guarded by config.
- Avoid editing the huge `run_agent.py` body directly unless a facade method must be added.

### 4. `model_tools.py` is tool orchestration, not model routing

`model_tools.py` imports tool modules, filters schemas by toolset, sanitizes schemas, and dispatches tool calls through `tools.registry`.

Important current behavior:

```python
discover_builtin_tools()

try:
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
except Exception as e:
    logger.debug("Plugin discovery failed: %s", e)
```

And:

```python
def handle_function_call(...):
    ...
    result = registry.dispatch(
        function_name, function_args,
        task_id=task_id,
        user_task=user_task,
    )
```

Decision:

- Do not put provider routing, LiteLLM selection, or usage tracking in `model_tools.py`.
- Only touch `model_tools.py` if a new agent-facing diagnostic tool or status command needs tool schema exposure.

### 5. State DB already stores session-level usage and cost

`hermes_state.py` contains `sessions`, `messages`, `state_meta`, and FTS tables/triggers.

Session table already has cost/usage columns:

```sql
input_tokens INTEGER DEFAULT 0,
output_tokens INTEGER DEFAULT 0,
cache_read_tokens INTEGER DEFAULT 0,
cache_write_tokens INTEGER DEFAULT 0,
reasoning_tokens INTEGER DEFAULT 0,
billing_provider TEXT,
billing_base_url TEXT,
billing_mode TEXT,
estimated_cost_usd REAL,
actual_cost_usd REAL,
cost_status TEXT,
cost_source TEXT,
pricing_version TEXT,
```

Decision:

- For per-request provider gateway analytics, a separate DB like `provider_usage.db` is still reasonable to avoid coupling hot routing data to the session history DB.
- Do not duplicate session aggregate fields unless the new DB has a clear per-request/per-provider purpose.

### 6. Usage and cost tracking already exist in the conversation loop

`agent/conversation_loop.py` normalizes usage and estimates cost after responses:

```python
canonical_usage = normalize_usage(
    response.usage,
    provider=agent.provider,
    api_mode=agent.api_mode,
)
...
cost_result = estimate_usage_cost(
    agent.provider,
    agent.model,
    canonical_usage,
    ...
)
```

Decision:

- New usage tracking should hook after `normalize_usage()` and `estimate_usage_cost()` rather than recalculate separately from raw provider response data.
- LiteLLM cost estimation can complement existing `agent.usage_pricing`, but should not silently replace it without parity tests.

### 7. Fallback and credential recovery already exist

Existing fallback surfaces include:

- `fallback_model` config validation in `hermes_cli/config.py`
- `agent.chat_completion_helpers.try_activate_fallback`
- `agent.agent_runtime_helpers.recover_with_credential_pool`
- rate-limit and billing classification in `agent.conversation_loop`

The config validator already knows top-level `fallback_model`:

```python
"_config_version", "model", "providers", "fallback_model",
"fallback_providers", "credential_pool_strategies", "toolsets",
```

Decision:

- Weighted routing should be added as an extension of existing fallback and credential pool behavior.
- Do not build a disconnected fallback engine that ignores `_fallback_chain`, `_credential_pool`, and `FailoverReason`.

### 8. Config already has provider routing and timeout sections

`cli-config.yaml.example` already documents:

- `model.provider`
- `model.base_url`
- per-provider timeouts under `providers:`
- OpenRouter-specific `provider_routing:`
- `model_aliases:`
- `fallback_model` in `hermes_cli/config.py`

Decision:

- A future `provider_gateway:` section must be opt-in and should not duplicate existing `provider_routing:` semantics unless it is deliberately broader than OpenRouter.
- Config migration must be explicit if new keys replace old keys.

### 9. Dependency policy needs an explicit decision before adding LiteLLM

The plan document suggests:

```toml
gateway = ["litellm>=1.40.0"]
```

But live `pyproject.toml` currently uses exact pins for core and optional dependencies, with comments explaining the supply-chain rationale:

```toml
dependencies = [
  "openai==2.24.0",
  ...
]

[project.optional-dependencies]
anthropic = ["anthropic==0.86.0"]
web = ["fastapi==0.133.1", "uvicorn[standard]==0.41.0"]
```

Decision:

- Do not add LiteLLM until the pinning convention is resolved for this repo.
- If following the current live file style, LiteLLM should be an exact-pinned optional extra, with `uv lock` regenerated.
- If following the AGENTS dependency policy text instead, use a bounded range. This conflict should be resolved before editing `pyproject.toml`.

### 10. Test infrastructure is large enough for focused TDD

Current test inventory from filesystem:

- 1309 Python files under `tests/`
- 1261 `test*.py` files
- 66 root-level test files
- notable directories:
  - `tests/providers`: 5 test files
  - `tests/run_agent`: 100 test files
  - `tests/agent`: 134 direct test files plus LSP tests
  - `tests/hermes_cli`: 270 test files
  - `tests/gateway`: 270 test files
  - `tests/tools`: 231 test files

`pyproject.toml` configures pytest:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: marks tests requiring external services (API keys, Modal, etc.)",
    "real_concurrent_gate: opt out of the autouse stub that disables _detect_concurrent_hermes_instances",
]
addopts = "-m 'not integration' --timeout=30 --timeout-method=signal"
```

Decision:

- Start new implementation with focused tests in existing domains:
  - provider profile metadata: `tests/providers/`
  - kwargs/backend path: `tests/run_agent/` or `tests/agent/`
  - usage/cost persistence: `tests/agent/` and possibly a new `tests/provider_gateway/`
  - config parsing: `tests/hermes_cli/`

## Answer to Fase 0 Questions

### Q1: Where exactly is the LLM API call made?

Primary OpenAI-compatible calls are made in `agent/chat_completion_helpers.py`:

- non-streaming: `request_client.chat.completions.create(**api_kwargs)`
- streaming: `request_client.chat.completions.create(**stream_kwargs)`

The OpenAI SDK client is constructed in `agent/agent_runtime_helpers.py` via `OpenAI(**client_kwargs)`.

Native alternatives already exist:

- `anthropic_messages` uses `_anthropic_messages_create`
- `bedrock_converse` uses boto3 `client.converse`
- `codex_responses` uses the Codex Responses transport
- Gemini and Copilot can wrap their own clients behind `.chat.completions.create`

### Q2: How is `ProviderProfile` used by `AIAgent`?

`agent.agent_init.init_agent` resolves provider and `api_mode`, warms transport cache, and configures client kwargs. During request building, `agent.chat_completion_helpers.build_api_kwargs` calls `get_provider_profile(agent.provider)`. If a profile exists, it passes the profile into `agent.transports.chat_completions.ChatCompletionsTransport.build_kwargs`, which calls profile hooks for messages, extra body, and top-level kwargs.

### Q3: What SQLite schema already exists in `hermes_state.py`?

The main state DB schema includes:

- `schema_version`
- `sessions`
- `messages`
- `state_meta`
- indexes for sessions/messages
- FTS5 `messages_fts`
- FTS5 trigram `messages_fts_trigram`
- triggers to keep FTS tables synchronized with `messages`

`sessions` already stores aggregate usage and cost fields.

### Q4: Can `hermes_state.py` be extended, or should gateway use a separate DB?

It can technically be extended because `hermes_state.py` reconciles columns from `SCHEMA_SQL`, but for provider gateway telemetry a separate DB is safer initially. Reasons:

- high-volume provider telemetry has different query patterns than session history
- router health/circuit breaker state should not contend with message persistence
- the plan already wants `provider_usage.db`
- existing `sessions` should remain source of truth for per-session aggregate usage

## Recommended Execution Plan

### Next task: Create final implementation plan reconciled with live code

Create `.plan_improvement/report-agent/2026-05-29-provider-gateway-implementation-plan.md` with tasks that reflect the actual codebase:

1. Add `provider_gateway/` skeleton with no runtime integration.
2. Add config parser for opt-in gateway settings, reading from existing config loaders.
3. Add SQLite `provider_usage.db` for per-request telemetry only.
4. Add circuit breaker module that can be used independently and tested without network.
5. Add routing policy module that consumes existing provider profiles and fallback config.
6. Add LiteLLM backend as optional only after dependency pin policy is resolved.
7. Wire usage/cost hooks after `normalize_usage()` and `estimate_usage_cost()`.
8. Wire backend/routing only behind `provider_gateway.enabled: true`.
9. Add `hermes status providers` or equivalent, avoiding large `cli.py` edits where possible.

### Implementation guardrails

- Keep all new features opt-in.
- Keep logs at DEBUG unless user explicitly asks for status output.
- Do not touch `trajectory_compressor.py`.
- Do not reimplement the provider system as `ProviderAdapter`.
- Do not put routing code in `model_tools.py`.
- Do not add LiteLLM as a core dependency.
- Do not edit `run_agent.py` unless a thin forwarding method is truly needed.

## Suggested Focused Test Commands

For future implementation tasks:

```bash
python -m pytest tests/providers -q
python -m pytest tests/run_agent/test_provider_parity.py -q
python -m pytest tests/run_agent/test_fallback_credential_isolation.py -q
python -m pytest tests/agent/test_usage_pricing.py -q
python -m pytest tests/hermes_cli/test_config.py -q
```

Use `scripts/run_tests.sh` only after the first implementation slice is stable.

## Risks to Track

1. Dependency pinning mismatch: plan says bounded LiteLLM range, live pyproject uses exact pins.
2. Existing fallback behavior is already complex; new routing must not bypass credential-pool recovery.
3. Streaming and non-streaming call paths both need parity if backend selection is introduced.
4. Native providers and OpenAI-compatible wrappers share `.chat.completions.create` shape but not identical semantics.
5. Per-request telemetry can become high-write-volume; keep it out of `hermes_state.py` until proven necessary.

## Status

- [x] Provider architecture audited.
- [x] Provider plugin discovery audited.
- [x] OpenRouter provider plugin audited.
- [x] Agent init and `api_mode` selection audited.
- [x] LLM callsite audited.
- [x] Tool orchestration audited.
- [x] State schema audited.
- [x] Config and dependency surfaces audited.
- [x] Test infrastructure inventoried.
- [ ] Implementation plan reconciled with live code.
- [ ] First implementation slice.
