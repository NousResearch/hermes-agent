# Hermes Antigravity & Gateway Context

Context and domain model for Google Antigravity integration, background cron execution, and gateway messaging integrations in the Hermes Agent fork.

## Language

**Antigravity Runtime**:
A delegated external-process execution engine communicating with the `agy` CLI over NDJSON `stream-json`, where model inference, sandboxing, and tool execution are delegated to the Antigravity process.
_Avoid_: OpenAI-compatible provider, generic LLM client

**Session Model Route**:
The immutable tuple `(model, provider, base_url, api_mode)` representing the exact LLM or runtime route a session was created on, persisted in `sessions.model_config`.
_Avoid_: Default model, ambient route

**Cron Session**:
A headless, non-interactive execution of a scheduled job in background without an active user terminal or TTY.
_Avoid_: Interactive session, user turn

**Messaging Session**:
A multi-turn interactive session bound to a communication platform (Slack, WhatsApp, Telegram, Discord) with persistent message history and platform-specific formatting.
_Avoid_: CLI REPL, cron tick

## Relationships

- A **Cron Session** executes using a **Session Model Route** and may dispatch reports to a **Messaging Session**
- An **Antigravity Runtime** session maintains an external `conversation_id` mapped to the Hermes session
- A **Messaging Session** can override its **Session Model Route** per session or per turn (defined in [ADR 0002](file:///home/rafalmuraro/.hermes/hermes-agent/docs/adr/0002-gateway-antigravity-integration.md))
- An **Antigravity Runtime** runs as an autonomous engine preserving system authentication credentials in headless and chat sessions (defined in [ADR 0003](file:///home/rafalmuraro/.hermes/hermes-agent/docs/adr/0003-antigravity-credential-inheritance-and-tool-model.md))

## Example dialogue

> **User (in Slack):** `/model agy/gemini-3.8-flash-high`
> **Gateway:** "✓ Model switched to gemini-3.8-flash-high (Google Antigravity)."
> **User:** "Faça um resumo dos commits recentes e execute os testes."
> **Gateway (live update):** "[Antigravity tool running: run_command] `git log -n 5`"
> **Gateway (final response):** "Aqui está o resumo dos 5 commits e o relatório dos testes executados com sucesso..."

## Flagged ambiguities

- "o modelo que eu estava usando aparece no provedor chatgpt" - resolved: `AIAgent._session_init_model_config` omitted `provider` and `api_mode`, causing session restore on resume to fall back to the global `config.yaml` provider (`openai-codex`). Resolved in [ADR 0001](file:///home/rafalmuraro/.hermes/hermes-agent/docs/adr/0001-hybrid-antigravity-session-continuity.md).
- "o agente não tem memória recente" - resolved: switching providers or dropping `_antigravity.conversation_id` causes `agy` to spawn a detached conversation without previous turn context. Solved via hybrid fail-soft transcript injection in [ADR 0001](file:///home/rafalmuraro/.hermes/hermes-agent/docs/adr/0001-hybrid-antigravity-session-continuity.md).
- "modelos do antigravity não aparecem no /model do Slack/WhatsApp" - resolved: `list_picker_providers` excluded `google-antigravity`. Resolved in [ADR 0002](file:///home/rafalmuraro/.hermes/hermes-agent/docs/adr/0002-gateway-antigravity-integration.md).
- "agy reported terminal status ERROR: authentication failed or timed out" - resolved: `hermes_subprocess_env(inherit_credentials=False)` stripped Google credentials in headless cron/gateway runs. Resolved in [ADR 0003](file:///home/rafalmuraro/.hermes/hermes-agent/docs/adr/0003-antigravity-credential-inheritance-and-tool-model.md).
