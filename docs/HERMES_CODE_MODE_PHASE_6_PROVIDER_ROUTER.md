# Hermes Code Mode — Fase 6: ProviderRouter

## 1. Resumo

Esta fase implementa o ProviderRouter — roteamento inteligente de modelos por CodeSession. O ProviderRouter permite:

- Selecionar modelos por tipo de tarefa (fast/strong/cheap/reviewer/planner)
- Criar presets de modelo por sessao
- Rastrear custo por sessao de codigo
- Atualizar provider/model diretamente
- Fallback para presets padrao quando nao ha overrides de sessao

## 2. Arquivos alterados

| Arquivo | Motivo |
|---------|--------|
| `hermes_state.py` | SCHEMA_VERSION atualizado para 13; migration v13; tabelas code_session_model_presets e code_session_cost_entries adicionadas ao SCHEMA_SQL; classe ProviderRouterDB criada |
| `hermes_cli/code/provider_router.py` | Novo — ProviderRouter com selecao de modelo, presets e rastreamento de custo |
| `hermes_cli/web_server.py` | 8 endpoints REST para operacoes de provider routing e custo |
| `tests/hermes_cli/test_provider_router.py` | Novo — 39 testes unitarios e de integracao |
| `tests/test_artifacts.py` | Atualizado schema version 12 -> 13 |
| `tests/test_hermes_state.py` | Atualizado schema version 12 -> 13 |
| `tests/hermes_cli/test_workspace_service.py` | Atualizado schema version 12 -> 13 |
| `tests/hermes_cli/test_code_session_service.py` | Atualizado schema version 12 -> 13 |

## 3. Banco/migration

### Schema version

`SCHEMA_VERSION = 13`

### Tabela `code_session_model_presets`

```sql
CREATE TABLE IF NOT EXISTS code_session_model_presets (
  id TEXT PRIMARY KEY,
  code_session_id TEXT NOT NULL,
  name TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  metadata_json TEXT DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(code_session_id) REFERENCES code_sessions(id)
);
```

### Indices

- `idx_code_session_model_presets_session_id`
- `idx_code_session_model_presets_name` (code_session_id, name)

### Tabela `code_session_cost_entries`

```sql
CREATE TABLE IF NOT EXISTS code_session_cost_entries (
  id TEXT PRIMARY KEY,
  code_session_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  task_type TEXT,
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  cache_read_tokens INTEGER DEFAULT 0,
  cache_write_tokens INTEGER DEFAULT 0,
  cost_usd REAL DEFAULT 0.0,
  metadata_json TEXT DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(code_session_id) REFERENCES code_sessions(id)
);
```

### Indices

- `idx_code_session_cost_entries_session_id`
- `idx_code_session_cost_entries_provider`
- `idx_code_session_cost_entries_model`
- `idx_code_session_cost_entries_created_at` (DESC)

A migration v13 e executada automaticamente em `SessionDB._init_schema()`. As tabelas tambem sao criadas no `_init_schema()` de `ProviderRouterDB` para garantir compatibilidade standalone.

## 4. Servico criado

### ProviderRouter (`hermes_cli/code/provider_router.py`)

Responsabilidades:
- Selecionar provider/model baseado em tipo de tarefa (task_type)
- Criar, listar, atualizar e deletar presets de modelo por sessao
- Rastrear custo de tokens e USD por iteracao
- Calcular resumo de custo por sessao (total, por provider, por modelo)
- Atualizar provider/model diretamente no CodeSession
- Registrar timeline events na CodeSession
- Emitir eventos WebSocket via realtime_hub

#### Presets de modelo

Presets sao associados a um CodeSession e identificados por nome. O ProviderRouter define 5 tipos de tarefa padrao:

| task_type | Model Padrao | Uso |
|-----------|-------------|-----|
| `fast` | anthropic/claude-haiku-4.5 | Iteracoes rapidas, queries simples |
| `strong` | anthropic/claude-opus-4.6 | Raciocinio complexo, arquitetura |
| `cheap` | anthropic/claude-haiku-4.5 | Minimizar custo |
| `reviewer` | anthropic/claude-sonnet-4.6 | Code review |
| `planner` | anthropic/claude-sonnet-4.6 | Planejamento de tarefas |

Quando um preset e criado para um CodeSession, ele sobrescreve o padrao. O metodo `create_preset` faz upsert — se ja existe um preset com o mesmo nome, ele e atualizado.

#### Selecao de modelo

O metodo `select_model(code_session_id, task_type)`:
1. Verifica se a sessao existe
2. Busca preset de sessao pelo task_type
3. Se nao encontrado, usa o preset padrao
4. Atualiza provider/model no CodeSession
5. Registra timeline event
6. Retorna resultado com source (`session_preset` ou `default`)

#### Rastreamento de custo

O metodo `track_cost()` registra uma entrada de custo por iteracao. O `get_session_cost_summary()` retorna:
- entry_count, total_*_tokens, total_cost_usd
- by_provider: breakdown por provider (count, cost, tokens)
- by_model: breakdown por modelo

## 5. Endpoints REST

| Metodo | Rota | Descricao |
|--------|------|-----------|
| GET | `/api/code/sessions/{id}/model` | Obter provider/model atual da sessao |
| POST | `/api/code/sessions/{id}/model/select` | Selecionar modelo por task_type |
| PUT | `/api/code/sessions/{id}/model` | Atualizar provider/model diretamente |
| GET | `/api/code/sessions/{id}/presets` | Listar presets (com defaults) |
| POST | `/api/code/sessions/{id}/presets` | Criar preset de modelo |
| DELETE | `/api/code/sessions/{id}/presets/{preset_id}` | Deletar preset |
| POST | `/api/code/sessions/{id}/cost` | Registrar entrada de custo |
| GET | `/api/code/sessions/{id}/cost` | Obter resumo de custo |
| GET | `/api/code/sessions/{id}/cost/entries` | Listar entradas de custo |

### Eventos WebSocket

- `provider.model_selected` — modelo selecionado por task_type
- `provider.model_updated` — provider/model atualizado diretamente

### Timeline events

- `provider.model_selected` — modelo selecionado por task_type
- `provider.model_updated` — provider/model atualizado
- `provider.preset_created` — preset criado ou atualizado

## 6. Testes

Total: **39 testes** (`tests/hermes_cli/test_provider_router.py`)

### ProviderRouterDB (14 testes)

- create_preset, get_preset_by_name, list_presets
- update_preset, delete_preset (including nonexistent)
- add_cost_entry, get_cost_summary (including by_provider breakdown)
- list_cost_entries (with limit), empty_cost_summary

### ProviderRouter service (17 testes)

- select_model (default, session_preset, updates_session, invalid_task_type, nonexistent_session)
- get_session_model (exists, nonexistent)
- update_session_model
- create_preset (valid, invalid_name, upsert)
- list_presets, delete_preset
- track_cost, get_session_cost_summary, list_cost_entries
- get_presets_summary (mixed session + defaults)

### HTTP endpoints (8 testes)

- get_session_model, get_session_model_not_found
- select_session_model, select_session_model_invalid_type
- update_session_model
- create_list_presets, delete_preset
- track_cost, get_cost_summary, list_cost_entries

## 7. Compatibilidade

Confirmado funcionamento de:

- Fase 1 artifacts (create_artifact, get_artifacts_by_session)
- Fase 2 workspaces (WorkspaceDB, endpoints)
- Fase 3 code sessions (CodeSessionDB, endpoints, timeline)
- Fase 4 commands (CodeCommandDB, endpoints)
- Fase 5 git (GitService, endpoints, snapshots)
- ProviderRouter (select_model, presets, cost tracking)
- Schema migration v6 -> v13

## 8. Proximos passos

Possiveis melhorias para fases futuras:

- Integrar ProviderRouter no loop do agent (run_agent.py) para selecao automatica de modelo por tarefa
- Adicionar estimativa de custo usando `get_model_capabilities` para modelos sem pricing fixo
- Implementar fallback entre providers (se um provider falha, tentar o proximo)
- Adicionar limites de custo por sessao (orcamento maximo)
- Sincronizar presets com a configuracao global do Hermes
