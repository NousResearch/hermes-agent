# 1. Estratégia Híbrida de Continuidade de Sessão do Antigravity

## Status
Accepted

## Contexto
O Antigravity é um runtime delegado que gerencia sua própria sessão e sandbox no processo `agy` via `conversation_id`. Anteriormente, sessões criadas ou executadas com Antigravity não gravavam o `provider` e o `api_mode` no `model_config` do banco SQLite (`sessions.model_config`). Ao retomar a sessão (`hermes --resume` / `-c` / Desktop / Gateway), o Hermes preservava apenas o nome do modelo (ex: `gemini-3.8-flash-high`) e caía no provedor padrão global do `config.yaml` (`openai-codex` / ChatGPT), que rejeitava o modelo e quebrava o encadeamento de memória.

## Decisão
Adotamos uma abordagem híbrida de continuidade em duas camadas:
1. **Persistência Estrita de Rota:** `AIAgent._session_init_model_config` e `update_session_model` gravam sempre o par completo `(provider: "google-antigravity", api_mode: "antigravity_runtime")` e o `antigravity_conversation_id` no SQLite. Na restauração, o Hermes retoma a sessão passando `--conversation <conversation_id>` ao `agy`.
2. **Re-hidratação Fail-soft de Contexto:** Caso o `conversation_id` seja perdido, expire ou haja troca transitória de modelo, o Hermes não inicia o `agy` vazio: reconstrói as mensagens recentes do transcript salvo no SQLite e as injeta no prefill do primeiro turno do novo `agy`, prevenindo amnésia do agente.

## Consequências
- Sessões iniciadas ou manipuladas pelo Cron e Gateway nunca mais farão fallback acidental para provedores globais como o ChatGPT.
- O agente retém a memória recente mesmo após reinicialização do Gateway ou do `agy`.
