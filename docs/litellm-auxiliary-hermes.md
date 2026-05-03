# LiteLLM auxiliary configuration (Hermes)

This note summarizes documentation and example updates for routing **Hermes auxiliary tasks** (title generation, compression, vision, etc.) through the **same LiteLLM Proxy** as the main agent while using **different `model` names** (e.g. cheaper models).

## Purpose

- Auxiliary tasks use `config.yaml` → `auxiliary:` and resolve **separately** from the main `model:` section unless you set `provider: main` or duplicate `base_url` / `api_key`.
- LiteLLM Proxy commonly enforces **Virtual Keys** (`sk-…`) on client `Authorization: Bearer` headers. Placeholder or non-`sk-` tokens cause 401 errors on side tasks even when the main turn appears healthy.

## Configuration summary

| Goal | Approach |
|------|----------|
| Same Proxy + same credentials as main | `provider: main` under `auxiliary.<task>`; set only `model` to a LiteLLM `model_name`. |
| Same Proxy + explicit credentials per task | `base_url: http://HOST:4000/v1`, `api_key: sk-…` (Virtual Key), `model: …`. |

Apply to tasks you care about (e.g. `title_generation`, `compression`, `vision`, `web_extract`).

## Difference from main model

- **Main model**: resolved via `model.provider` / `resolve_runtime_provider` in the gateway and passed into `AIAgent`.
- **Auxiliary**: resolved in `agent/auxiliary_client.py` from `auxiliary.*` keys in `config.yaml` (plus auto-detection when `provider: auto`).

## Operational notes

- **Messaging (e.g. Feishu)**: Restart the **gateway** after changing `config.yaml` so auxiliary routing and env reload pick up changes.
- **Virtual Key vs Master Key**: Prefer **issued Virtual Keys** for app traffic; Master Key is primarily for LiteLLM admin — see your LiteLLM deployment policy.

## Related files

| File | Change |
|------|--------|
| [website/docs/integrations/providers.md](../website/docs/integrations/providers.md) | LiteLLM section: custom anchors, **Auxiliary models (same proxy, cheaper models)** with YAML examples and auth notes. |
| [website/docs/user-guide/configuration.md](../website/docs/user-guide/configuration.md) | **Auxiliary Models**: **LiteLLM Proxy** short subsection linking to providers doc. |
| [cli-config.yaml.example](../cli-config.yaml.example) | LiteLLM comment block (Virtual Key / Master Key); commented `auxiliary` example for `title_generation` / `compression`. |
