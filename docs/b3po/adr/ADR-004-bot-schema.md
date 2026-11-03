# ADR-004 — schema do Bot v1

**Status:** aceito (F4) · **Data:** 2026-11-02

## Contexto

No código, "bot" é um **profile** do Hermes com metadados visuais (`BotMeta`). Faltam três campos
aprovados: `schema_version`, `mascot_id` e `permission_policy`.

## Decisão

Três blocos versionados:

```
identity:      schema_version, id, name, description, mascot_id?, instructions/persona, labels
runtime_config: engine=hermes, provider, model, permission_policy (obrigatória, default restritivo),
                workspace_ref?, tools, skills, mcp_refs?, limites
links_and_state: conversas, sessões, rotinas, memórias (referências), estado transitório de UI
```

- `engine` tem **um único valor válido no v1**: `hermes`.
- `permission_policy` ausente = schema inválido.
- Referências podem ser `null` até auditoria; capacidade não confirmada **não** gera promessa de UI.
- Mascote é identidade visual; nunca executor.
- Transcript, memória canônica e sessão continuam no Hermes.

## Consequências

- Migração versionada obrigatória (fixtures + rollback) antes de qualquer dado real.
- O core Hermes não conhece regra visual.
- Adicionar segundo runtime exigiria novo ADR + mudança de enum, não um campo livre.
