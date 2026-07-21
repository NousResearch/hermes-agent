# Zai-router

Zai-router is the project home for Hermes Agent's separate **Z.ai Indirect** provider. It routes `glm-5.2` through Z.ai's Claude Code-compatible Anthropic Messages endpoint and keeps the existing OpenAI-compatible `zai` provider unchanged.

## Runtime identity

| Field | Value |
|---|---|
| Provider ID | `zai-indirect` |
| Display name | Z.ai Indirect |
| Model | `glm-5.2` |
| API mode | `anthropic_messages` |
| Endpoint | `https://api.z.ai/api/anthropic` |
| Credential | `ZAI_INDIRECT_API_KEY` |

## Repository layout

This directory contains the portable project documentation. Runtime files intentionally remain in Hermes Agent's standard locations so bundled plugin discovery and transport routing work:

- `plugins/model-providers/zai-indirect/` — provider registration.
- `agent/anthropic_adapter.py` — endpoint-specific Anthropic request adapter.
- `agent/agent_init.py`, `agent/agent_runtime_helpers.py`, and `agent/chat_completion_helpers.py` — initialisation, switching, and fallback routing.
- `hermes_cli/providers.py` — provider API-mode resolution.
- `tests/agent/test_zai_indirect_anthropic.py` and `tests/plugins/model_providers/test_zai_indirect_profile.py` — regression coverage.
- `website/docs/integrations/zai-indirect.md` — user-facing documentation.

Moving those files into this directory would prevent Hermes from loading them. See [EXTRACTION.md](EXTRACTION.md) for a future standalone-repository plan.

## Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) — routing and design boundaries.
- [INSTALLATION.md](INSTALLATION.md) — Windows application instructions.
- [VERIFICATION.md](VERIFICATION.md) — tests and live checks.
- [EXTRACTION.md](EXTRACTION.md) — path to a future independent repository.

## Security

Never place API keys, captured request dumps, `.env` files, or debug logs in this directory or in Git. Credentials belong in the active Hermes profile's `.env` or credential store.
