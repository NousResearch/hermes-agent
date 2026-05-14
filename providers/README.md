# providers/

Registry and ABC for every inference provider Hermes knows about.

Each provider is declared once as a `ProviderProfile`. Every other layer —
auth resolution, transport kwargs, model listing, runtime routing — reads from
these profiles instead of maintaining its own parallel data.

---

## Layout

```
providers/
├── base.py         ProviderProfile dataclass + OMIT_TEMPERATURE sentinel
├── __init__.py     Registry: register_provider(), get_provider_profile(), list_providers()
└── README.md       This file
```

The **profiles themselves** live as plugins under
`plugins/model-providers/<name>/` (bundled in this repo) and
`$HERMES_HOME/plugins/model-providers/<name>/` (per-user overrides). The
registry in `providers/__init__.py` lazily discovers them the first time any
consumer calls `get_provider_profile()` or `list_providers()`. See
`plugins/model-providers/README.md` for the plugin contract and examples.

---

## How it wires in

The registry is populated on first access. After that, every downstream
layer reads from it:

- `hermes_cli/auth.py` extends `PROVIDER_REGISTRY` with every api-key
  profile it sees (skipping `copilot`, `kimi-coding`, `kimi-coding-cn`,
  `zai`, `openrouter`, `custom` — those need bespoke token resolution).
- `hermes_cli/models.py` extends `CANONICAL_PROVIDERS` and calls
  `profile.fetch_models()` inside `provider_model_ids()`.
- `hermes_cli/doctor.py` adds a `/models` health check for each
  `auth_type="api_key"` profile.
- `hermes_cli/config.py` injects every `env_var` into
  `OPTIONAL_ENV_VARS` so the setup wizard knows about it.
- `hermes_cli/runtime_provider.py` reads `profile.api_mode` as a fallback
  when URL detection finds nothing.
- `agent/model_metadata.py` maps hostname → provider via
  `profile.get_hostname()`.
- `agent/auxiliary_client.py` reads `profile.default_aux_model` first
  before falling back to the legacy hardcoded dict.
- `agent/transports/chat_completions.py::_build_kwargs_from_profile()`
  invokes `profile.prepare_messages()`, `profile.build_extra_body()`,
  and `profile.build_api_kwargs_extras()` on every call.
- `run_agent.py` passes `provider_profile=<ProviderProfile>` so the
  transport takes the profile path instead of the legacy flag path.

---

## Adding a provider

See `plugins/model-providers/README.md` — drop a new directory there (or
under `$HERMES_HOME/plugins/model-providers/` for a private plugin).

---

## Hooks you can override on `ProviderProfile`

| Hook | Purpose |
|------|---------|
| `get_hostname()` | URL-based detection — default derives from `base_url`. |
| `prepare_messages(msgs)` | Provider-specific message preprocessing (Qwen normalises to list-of-parts, injects `cache_control`). |
| `build_extra_body(**ctx)` | Provider-specific `extra_body` (OpenRouter provider prefs, Gemini `thinking_config`). |
| `build_api_kwargs_extras(**ctx)` | `(extra_body_additions, top_level_kwargs)` — Kimi puts reasoning_effort top-level, Qwen splits `enable_thinking`/`thinking_budget`. |
| `fetch_models(*, api_key)` | Live catalog fetch — default hits `{models_url or base_url}/models` with Bearer auth. Override for no-REST providers (Bedrock), OAuth catalogs (Anthropic), or public catalogs (OpenRouter). |

---

## Configuration fields

Full reference in `providers/base.py` dataclass definition.

---

## Overriding the Anthropic OAuth client profile

`hermes auth add anthropic --type oauth` runs a PKCE flow against
Hermes's own registered OAuth application. The env vars below let you
point that flow at a different OAuth client without forking the code.
Set any of them and leave the rest unset to keep the upstream defaults:

| Env var | Default |
|---|---|
| `HERMES_ANTHROPIC_OAUTH_CLIENT_ID`      | Hermes's registered client id |
| `HERMES_ANTHROPIC_OAUTH_AUTHORIZE_URL`  | `https://claude.ai/oauth/authorize` |
| `HERMES_ANTHROPIC_OAUTH_TOKEN_URL`      | `https://console.anthropic.com/v1/oauth/token` |
| `HERMES_ANTHROPIC_OAUTH_REDIRECT_URI`   | `https://console.anthropic.com/oauth/code/callback` |
| `HERMES_ANTHROPIC_OAUTH_SCOPES`         | `org:create_api_key user:profile user:inference` |

The resolver is `agent.anthropic_adapter.get_anthropic_oauth_client_profile()`
— it re-reads the environment on every call so overrides set in `.env`
or the shell take effect without restarting hermes. Both the PKCE
login flow and the refresh path go through it. The refresh path also
keeps the upstream endpoints as fallbacks after the configured one so
refresh tokens minted elsewhere keep working.

### When this is appropriate

This override is intended **only** for:

- **Development and testing** against a local or staging OAuth server you
  control (e.g. running PKCE end-to-end without hitting production).
- **Enterprise deployments** where you operate the OAuth bridge yourself —
  a self-hosted Anthropic-compatible proxy, an internal SSO gateway, or
  any other OAuth-2.0 server you administer.
- **Anthropic-approved integrations** that have their own registered
  client identifier from Anthropic.

Do **not** point this at an OAuth client you do not own or operate.
That is impersonation, violates the client owner's terms of service,
and almost certainly violates Anthropic's acceptable use policy.
Hermes does not validate or whitelist client identifiers — the
responsibility is entirely yours.
