# BOT-SCHEMA-AUDIT — o que o código realmente tem (F4)

Base: `apps/desktop/src/plugins/hermes-bots/` (202 arquivos) + `hermes_state*`. Método: leitura de tipos e
do editor de profile. **Nada aqui foi validado rodando o app** — é auditoria de código.

## 1. O que existe

`BotMeta` (`types.ts:64`) — metadados persistidos por bot:

| Campo | Tipo | Uso |
|---|---|---|
| `title` | string | nome exibido |
| `description` | string | descrição |
| `sectionId` / `sectionName` | string, null | organização em seções (persistida no bot) |
| `groups` / `group` | string[] / string | agrupamento |
| `color` | string | cor do avatar |
| `image` | data URL / null | avatar customizado (enviado via `set_asset`) |
| `imageKind` | `photo` \| `shape` | origem do avatar |
| `shape` | string | forma do avatar gerado |
| `custom` | boolean | avatar personalizado |
| `hidden` | boolean | oculto na lista |
| `pinned` | boolean | fixado |
| `screenAutoOpen` | boolean | abre a aba Screen ao dirigir o desktop |
| `created` | number | criação |

Editor avançado (`profile-config.tsx`): **skills, toolsets, servidores MCP, modelo e SOUL** por profile.

**Bot = profile do Hermes.** `data.ts` resolve bot → `profile` (141 arquivos citam `profile`); a
persistência real é `profile.yaml`, sincronizado como o resto das configurações.

## 2. Mapeamento para o schema aprovado (F4)

| Bloco F4 | Campo aprovado | Existe hoje? | Observação |
|---|---|---|---|
| identity | `schema_version` | **não** | criar no v1 |
| identity | `id` | sim (nome do profile) | id estável = nome do profile |
| identity | `name` | `title` | renomear só na camada B-3PO |
| identity | `description` | `description` | direto |
| identity | `mascot_id` | **não** (há `shape`/`image`) | adicionar como campo opcional |
| identity | `instructions/persona` | SOUL por profile (`soul.ts`) | referência, não cópia |
| identity | `labels` | `groups`/`sectionId` | mapear |
| runtime_config | `engine` | implícito (Hermes) | fixar `hermes` no v1 |
| runtime_config | `provider` / `model` | sim (`model-picker`) | direto |
| runtime_config | `permission_policy` | **não** | obrigatório no v1 |
| runtime_config | `tools` / `skills` / `mcp_refs` | sim (`toolset`, `skills-hub`, `mcp-setup`) | opcionais |
| runtime_config | `workspace_ref` | **não confirmado** | permanece `null` |
| links_and_state | sessões/conversas | sim (`canonical-chat`) | referência |
| links_and_state | rotinas | sim (`cron.tsx`) | referência |
| links_and_state | memória | profile-level | referência, nunca cópia |

## 3. Lacunas para o v1

1. **`schema_version` + `mascot_id` + `permission_policy` não existem** — são o núcleo do que F4 aprovou.
2. **Migração versionada** não existe para BotMeta; há `data.bot-meta-migration.test.ts`, sinal de que já
   houve migração — precisa ser lida antes de adicionar campos.
3. `workspace_ref` e memória: **não confirmados** como atributo por bot → entram `null`.

## 4. Veredito

O schema real é **compatível** com F4 (nada contradiz), mas incompleto: faltam três campos obrigatórios e
a migração v1. Implementação viável sem tocar o core Hermes — fica no plugin/camada B-3PO.
