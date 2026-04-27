# Hermes Code Mode — Auditoria Fase 0

## 1. Resumo executivo

O Hermes Agent é um agente Python com arquitetura modular baseado em tools (OpenAI tool-calling format), sessões SQLite com FTS5, e um sistema de skills baseado em arquivos Markdown. O runtime principal é o `AIAgent` em `run_agent.py` (10.871 linhas), com loop principal síncrono que chama modelos via API e executa tools registradas em `tools/registry.py`.

**Pronto para Code Mode?** Não integralmente. A infraestrutura de sessão, tools de arquivo e patches já existe, mas faltam: ArtifactRegistry dedicado, GitService, CodeSessionService, provider routing por workspace, e skills de desenvolvimento. A área mais segura para iniciar é consertar/validar `get_artifacts_by_session` e o endpoint `/api/sessions/{session_id}/artifacts`.

---

## 2. Runtime principal identificado

- **Linguagem:** Python 3 (100%)
- **Arquitetura:** Síncrona para o agent principal; Assíncrona (asyncio) para gateway e web server
- **Principais arquivos:**
  - `run_agent.py` — AIAgent class, loop principal (10.871 linhas)
  - `model_tools.py` — Tool orchestration, `_discover_tools()`, `handle_function_call()`
  - `tools/registry.py` — ToolRegistry singleton (central de registro de tools)
  - `toolsets.py` — Definições de toolsets (`_HERMES_CORE_TOOLS` list)
  - `gateway/run.py` — Gateway runner (9.411 linhas), adapters de plataforma
  - `hermes_cli/web_server.py` — FastAPI server, WebSocket (4.885 linhas), HermesWeb
  - `cli.py` — CLI interativo (447KB)
  - `hermes_state.py` — SessionDB SQLite com FTS5 (2.003 linhas)

---

## 3. Entrada do agente e loop principal

**Classe:** `AIAgent` em `run_agent.py:526`

**Método principal:** `run_conversation()` em `run_agent.py:7745`

**Loop em `run_agent.py:8067:**
```python
while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) or self._budget_grace_call:
    api_call_count += 1
    # ... build messages, call API ...
    # _execute_tool_calls() em 10057
    # context compression check em 10134
```

**Fluxo resumido:**
1. `run_conversation()` recebe mensagem do usuário
2. Prepara `api_messages` (system prompt + history)
3. Loop: chama API (`_call_model_api()`) → recebe `assistant_message`
4. Se `assistant_message.tool_calls`: executa `_execute_tool_calls()` (sequential ou concurrent)
5. Append tool results como `tool` messages
6. Check context compression (`should_compress`)
7. Repetir até `finish_reason` != `tool_calls` ou `max_iterations`
8. Retorna `final_response`

**Tool invocation:** `_invoke_tool()` em `run_agent.py:6891` delega para registry (`tools/registry.py:212 dispatch()`) ou handlers especiais (todo, memory, session_search).

---

## 4. Tools

**Pasta:** `tools/`

**Registro:** cada arquivo em `tools/` chama `registry.register()` no import. O `ToolRegistry` em `tools/registry.py` é o ponto central.

**Formatos de tool:** OpenAI function-calling format (schemas com `name`, `description`, `parameters`).

**Ferramentas de arquivo/shell/patch/git mais relevantes para Code Mode:**

| Arquivo | Tools | Descrição |
|---------|-------|-----------|
| `tools/file_tools.py` | `read_file`, `write_file`, `patch`, `search_files` | Operações de arquivo de alto nível |
| `tools/file_operations.py` | `ShellFileOperations` | Abstração sobre backends (local, docker, ssh, modal, daytona) |
| `tools/patch_parser.py` | `parse_v4a_patch`, `apply_v4a_operations` | Parser do formato V4A (codex/cline) |
| `tools/terminal_tool.py` | `terminal`, `process` | Execução de comandos shell |
| `tools/delegate_tool.py` | `delegate_task` | Subagentes com ferramentas isoladas |
| `tools/approval.py` | Dangerous command detection | Aprovações humanas |

**Como uma tool é chamada:**
1. LLM retorna `tool_calls` na response
2. `_execute_tool_calls()` decide sequential vs concurrent
3. `_invoke_tool()` ou registry `dispatch()` executa
4. Resultado JSON string é appendado como `tool` message

**Limitações atuais para Code Mode:**
- `write_file` não calcula diff (apenas conteúdo novo)
- `patch` retorna dict com `files_modified`, `diff`, mas o diff é o raw patch text
- Não há `read_file` por diff (sem contexto de git)

---

## 5. Skills

**Pasta:** `skills/` — 25+ categorias (software-development, productivity, mlops, etc.)

**Formato:** cada skill é um diretório com `SKILL.md` (Markdown com frontmatter).

**Exemplo de skill:** `skills/software-development/writing-plans/SKILL.md`
- 296 linhas de Markdown estruturado
- Frontmatter com name, description, version, author, license, metadata
- Conteúdo: guidelines, templates, exemplos de código

**Carregamento:**
- `agent/skill_utils.py` — parsing de frontmatter, matching de platform
- `agent/skill_commands.py` — injeção no system prompt (não no history, para preservar prompt caching)
- `skills_tool.py` — tools: `skills_list`, `skill_view`, `skill_manage`

**Skills existentes que podem ser reaproveitadas:**
- `software-development/subagent-driven-development` — delegação para subagentes
- `software-development/test-driven-development` — workflow TDD
- `software-development/writing-plans` — planejamento de implementação
- `software-development/systematic-debugging` — debugging metódico

**O que NÃO existe ainda para Code Mode:**
- Skills de refatoração automática
- Skills de git workflow (commit, branch, PR)
- Skills de code review
- Skills de LSP/integration

---

## 6. Sessões e memória

**SessionDB** em `hermes_state.py`:
- SQLite com WAL mode
- Tables: `sessions`, `messages`, `approvals`
- FTS5 virtual table `messages_fts` para busca full-text
- Schema version: 7

**Sessions table** (`hermes_state.py:52`):
```
id, source, user_id, model, model_config, system_prompt,
parent_session_id, started_at, ended_at, end_reason,
message_count, tool_call_count, input_tokens, output_tokens,
cache_read_tokens, cache_write_tokens, reasoning_tokens,
billing_provider, billing_base_url, billing_mode,
estimated_cost_usd, actual_cost_usd, cost_status,
cost_source, pricing_version, title
```

**Messages table** (`hermes_state.py:82`):
```
id, session_id, role, content, tool_call_id, tool_calls,
tool_name, timestamp, token_count, finish_reason,
reasoning, reasoning_details, codex_reasoning_items
```

**MemoryManager** em `agent/memory_manager.py`:
- Orquestra BuiltinMemoryProvider + 1 plugin provider externo
- Prefetch no pré-LLM call
- Sync no pós-LLM call

**ContextCompressor** em `agent/context_compressor.py`:
- Summarization dos middle turns
- Projeta head + tail messages
- Usa modelo auxiliar (barato) para summarization
- Token budget tail protection

**Como isso serve ao futuro CodeSessionService:**
- SessionDB pode ser extendida com tabelas `code_sessions`, `artifacts`, `commands`
- MemoryManager pode ser expandido com provider de workspace context
- ContextCompressor já existe e funciona — precisa apenas de trigger mais frequente para Code Mode

---

## 7. Banco e persistência

**Tipo:** SQLite com WAL mode (journal_mode=WAL)

**Arquivo:** `~/.hermes/state.db` (ou `HERMES_HOME/state.db` para profiles)

**Tabelas atuais:**
- `schema_version` — tracking de migrations
- `sessions` — metadata de sessão
- `messages` — histórico de mensagens (role, content, tool_calls, etc.)
- `messages_fts` — FTS5 full-text search
- `approvals` — aprovações pendentes/resolvidas
- `chat_runs` — run registry para HermesWeb (em `hermes_cli/run_registry.py`)
- `run_steps` — steps por run (em `hermes_cli/run_registry.py`)

**Onde adicionar tabelas futuras (Code Mode):**
- `code_sessions` — sessões de code mode (herança de sessions, com workspace_path, git_branch, etc.)
- `artifacts` — artifacts dedicados (não dependentes de parsear tool call results)
- `commands` — log de comandos executados (terminal_tool output)

**BUG CONHECIDO (documentado em `hermes_state.py:1353`):**
`get_artifacts_by_session()` extrai artifacts de tool messages raw (role='tool', tool_name IN ('patch', 'write_file')). Isso é frágil porque:
- Depende do formato do content string (JSON parse)
- Não calcula diff para `write_file` (diff="")
- Aditions/deletions só para patch
- Não persiste artifacts de forma independente

---

## 8. HTTP API e WebSocket

**Servidor HTTP:** `hermes_cli/web_server.py` (FastAPI/Uvicorn)

**Porta padrão:** 9119

**Endpoints principais:**

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/ws` | WebSocket | Realtime events, token auth via `_SESSION_TOKEN` HMAC |
| `/api/chat` | POST | HermesWeb: cria chat run async |
| `/api/chat/runs/{run_id}/steps` | GET | Steps do chat run |
| `/api/agents` | GET | Lista sessões/agents |
| `/api/sessions/{session_id}/artifacts` | GET |Artifacts por sessão |
| `/api/sessions/{session_id}/messages` | GET |Mensagens da sessão |
| `/api/sessions/{session_id}` | DELETE |Deleta sessão |
| `/api/chat/history` | GET |Histórico do HermesWeb |

**WebSocket (`/ws`):**
- Auth: `_require_ws_token()` — valida `authorization` header ou `token` query param contra `_SESSION_TOKEN` (HMAC)
- `_REALTIME_HUB` — fanout in-process para events
- Tipos de evento: `hello`, `heartbeat`, `pong`, `info`, `chat_started`, `chat_completed`, `chat_failed`, `chat_timeout`
- Keep-alive: ping/pong a cada 15s timeout

**Autenticação:**
- `_SESSION_TOKEN` gerado random em cada startup do server (32 bytes url-safe)
- Injetado no SPA HTML
- HermesWeb usa `token` query param ou `authorization` header

**Como HermesWeb se conecta:**
1. Carrega página do FastAPI (FileResponse do `web_dist/`)
2. Extrai `_SESSION_TOKEN` do HTML
3. Conecta WebSocket em `/ws?token=<token>`
4. POST `/api/chat` com content
5. Recebe eventos via WebSocket + polling em `/api/chat/runs/{run_id}/steps`

**Gap para Code Mode:**
- Não há endpoint para criar "code session" (workspace + git branch)
- Não há endpoint para listar artifacts por sessão com filtros
- Não há endpoint para emitir `artifact.created` event via WebSocket

---

## 9. Artifacts/diffs

**O que já existe:**

1. **`get_artifacts_by_session()`** em `hermes_state.py:1353`:
   - Extrai de `messages` onde `role='tool'` e `tool_name IN ('patch', 'write_file')`
   - Para `patch`: extrai `files_modified`, `files_created`, `files_deleted`, `diff`
   - Para `write_file`: path, status='added', diff=""
   - Retorna lista de dicts: `{id, tool_call_id, tool_name, path, status, diff, additions, deletions, timestamp}`

2. **V4A Patch Parser** em `tools/patch_parser.py`:
   - Parser completo para formato V4A (codex/cline)
   - `parse_v4a_patch()` — parsing
   - `apply_v4a_operations()` — aplicação via `ShellFileOperations`
   - Suporta: ADD, UPDATE, DELETE, MOVE

3. **`patch_tool`** em `tools/file_tools.py:565`:
   - Mode `replace`: old_string → new_string
   - Mode `patch`: V4A patch format
   - Validação de paths sensíveis

4. **`write_file_tool`** em `tools/file_tools.py:541`:
   - Sem diff (apenas conteúdo)
   - Sem tracking de versão

**Bugs/limitações:**

1. **`get_artifacts_by_session` não persiste artifacts de forma independente** — derivados de tool call results on-the-fly. Se o conteúdo do message for alterado, artifacts mudam.

2. **Diff para `write_file` é sempre vazio** — não há compare, não há stored diff.

3. **O diff de `patch` é o raw patch text** — não é um diff unix estruturado. `additions`/`deletions` são contabilizados incorretamente (conta linhas `+`/`-` que são de header, não de conteúdo).

4. **Não há `artifact.created` WebSocket event** — HermesWeb não é notificado quando uma tool modifica arquivos.

**O que não existe:**
- `ArtifactRegistry` dedicado
- Versionamento de artifacts
- Diff para reads (git diff do working tree)
- Patch do working tree (git diff --cached)

---

## 10. Providers/modelos

**Arquivos principais:**
- `agent/model_metadata.py` — context lengths, token estimation, provider prefix stripping
- `hermes_cli/models.py` — catálogo de modelos por provider
- `hermes_cli/auth.py` — resolução de credenciais por provider

**Providers suportados:**
- `openrouter`, `anthropic`, `openai`, `google`, `deepseek`, `qwen`, `github`, `mistral`, `moonshot`, `kimi`, `z-ai`, `custom`, `local` (Ollama)
- Provider prefix no model string: `"anthropic/claude-opus-4.6"`

**Seleção de modelo:**
- Config: `~/.hermes/config.yaml` → `model`
- Env: `ANTHROPIC_MODEL`, `OPENAI_MODEL`, etc.
- Por sessão: `AIAgent(model="...")`
- Provider ordering: `providers_order`, `providers_ignored`, `providers_allowed` (OpenRouter-specific)

**Limitação para Code Mode:**
- Cada `AIAgent` instance tem um model fixo (`self.model`)
- `model_switch` via `/model` command muda globalmente (reescreve config)
- **Não há model por sessão** — uma code session não pode usar modelo diferente da sessão principal
- `model_metadata.py:24` lista provider prefixes, mas não há routing dinâmico baseado em task type

---

## 11. Gaps para virar agente tipo OpenCode

### 11.1 CodeWorkspaceService
- **Não existe.** Workspace path por sessão não é tracked.
- Existe `_get_file_ops()` em `file_tools.py` que usa `task_id`, mas não há workspace isolado.
- Terminal environments (`tools/environments/`) suportam backends (local, docker, ssh, modal, daytona, singularity), mas sem workspace conceito.

### 11.2 CodeSessionService
- **Não existe.** SessionDB só tem `sessions` genérico.
- Necessário: `code_sessions` table com `workspace_path`, `git_branch`, `git_status`, `initial_commit`.

### 11.3 CommandRunner seguro
- **Parcialmente existe.** `terminal_tool.py` executa shell commands.
- `approval.py` detecta dangerous commands e pede aprovação humana.
- **Falta:** sandbox mais forte (container isolado, syscalls filter), timeout real por command, command history persistente.

### 11.4 ArtifactRegistry
- **Não existe como serviço dedicado.**
- `get_artifacts_by_session()` existe mas é frágil (parse on-the-fly).
- Necessário: tabela `artifacts` independente com versionamento.

### 11.5 GitService
- **Não existe.** Terminal tool executa `git` command, mas não há wrapper.
- Funcionalidades faltando: `git status` parse, `git diff` structured, `git branch` list, `git log`, `git stash`.

### 11.6 ProviderRouter
- **Não existe.** Todo routing é estático (config ou env var).
- Para Code Mode: modelo barato para coding tasks simples, modelo forte para raciocínio complexo.

### 11.7 LSP/Code Intelligence
- **Não existe.** Hermes não tem LSP client.
- Faltando: goto definition, find references, hover, completions.

### 11.8 MCPBridge
- **`mcp_tool.py` existe (90.256 bytes)** — MCP client para ferramentas MCP.
- Mas não é orientado a coding tools (LSP, file watching, etc.).

### 11.9 Skills de desenvolvimento
- **Parcialmente existe.** Há `software-development/` skills.
- Faltando: `auto-refactor`, `git-workflow`, `code-review`, `debug-with-breakpoints`.

### 11.10 Integração HermesWeb
- **Existe infraestrutura básica** (`/ws`, `/api/chat`, run registry).
- **Falta:** `code-session.create` endpoint, `artifact.created` event, `terminal.output` streaming via WebSocket.

---

## 12. Arquivos candidatos para a Fase 1

### Alta prioridade

1. **`hermes_state.py:1353`** — `get_artifacts_by_session()`: validar/corrigir parsing de diffs, adicionar `artifact.created` event emission, considerar criar tabela `artifacts` dedicada.

2. **`hermes_cli/web_server.py:3506`** — `/api/sessions/{session_id}/artifacts`: adicionar WebSocket notification após artifact criado, validar que HermesWeb recebe o evento.

3. **`tools/patch_parser.py`** — O diff counting em `get_artifacts_by_session` está incorreto (conta linhas de header do patch como adições/remoções). Validar e corrigir.

4. **`agent/prompt_builder.py:164`** — Prompt guidance para artifacts existe mas vago. Adicionarguidance mais específico para Code Mode.

### Média prioridade

5. **`run_agent.py`** — Explorar onde injetar code session context (workspace, git branch) no system prompt.

6. **`tools/terminal_tool.py`** — Command timeout/checkpointing: verificar se há timeout real por command (vs iteration budget que é por LLM call).

7. **`model_tools.py`** — Adicionar tools de coding (git diff, git status) como extensões do toolset `file`.

8. **`agent/context_compressor.py`** — Verificar se trigger threshold é apropriado para code sessions (sessões de código podem precisar comprimir mais frequentemente).

9. **`hermes_cli/models.py`** — Avaliar se existe modelo adequado para Code Mode (provavelmente um modelo de coding dedicado como `claude-code` ou `qwen3-coder`).

### Baixa prioridade

10. **`tools/delegate_tool.py`** — Avaliar se delegate_task pode ser usado para Code Mode parallel task execution.

11. **`skills/software-development/`** — Desenvolver skills de coding (git-workflow, refactor, code-review) como extensão.

12. **`hermes_cli/run_registry.py`** — Avaliar se run_steps pode rastrear artifacts criados durante o run.

---

## 13. Riscos técnicos

1. **Duplicação de arquitetura:** Code Mode pode acabar sendo um "segundo agent" com structs paralelas (CodeSession vs Session, ArtifactRegistry vs get_artifacts_by_session). Precisamos unificar, não duplicar.

2. **Inconsistência entre docs e código real:** `AGENTS.md` diz que AIAgent é o centro, mas `gateway/run.py` (9.411 linhas) é enorme e tem muita lógica de negócio que não está documentada.

3. **Tasks presas:** `terminal_tool.py` executa commands em background, mas não há timeout por command — apenas iteration budget geral. Commands bloqueados podem travar o agent.

4. **Comandos sem timeout:** O `terminal` tool não impõe timeout por command. O iteration budget é por LLM call, não por command executado.

5. **Artifacts sem persistência independente:** `get_artifacts_by_session()` depende de parsear mensagens — se o message store for limpo/comprimido, artifacts se perdem.

6. **Provider/model não rastreado por workspace:** Cada AIAgent instance tem um model fixo. Code sessions não podem usar modelos diferentes.

7. **Ausência de LSP:** Não há code intelligence (goto, find refs, hover). O agent opera purely por file reads/writes.

8. **Endpoint quebrado ou mal documentado:** O `/api/chat` do HermesWeb não está claro se funciona para code sessions ou só para chat sessions genéricas.

9. **`_SESSION_TOKEN` por processo:** O token é regenerado a cada web server start — HermesWeb precisa fazer reload para pegar novo token.

10. **Patch diff counting bug:** `get_artifacts_by_session()` conta `+` e `-` lines incluindo patch headers (`--- a/`, `+++ b/`), superestimando adições/deleções.

---

## 14. Recomendação para a Fase 1

**Confirmada: a Fase 1 deve focar em artifacts/diffs.**

Baseado no código real, a sequência recomendada é:

### Passo 1 — Validar e corrigir `get_artifacts_by_session()`
- O diff counting está incorreto (conta headers do patch como adições)
- `write_file` nunca tem diff (precisamos de git diff para novos arquivos?)

### Passo 2 — Criar endpoint de artifact events
- Adicionar emissão de `artifact.created` via `_REALTIME_HUB` quando `patch` ou `write_file` completam
- Isso permite que HermesWeb receba notificação em tempo real

### Passo 3 — Persistir artifacts de forma independente
- Criar tabela `artifacts` no SQLite (não depender de parse de tool messages)
- Adicionar colunas: `session_id`, `tool_call_id`, `tool_name`, `path`, `status`, `diff`, `additions`, `deletions`, `content_hash`, `created_at`

### Passo 4 — Validar endpoint `/api/sessions/{session_id}/artifacts`
- Garantir que HermesWeb consegue consumir artifacts corretamente
- Testar com session real (criar artifacts via patch/write_file, consultar via API)

**Não fazer na Fase 1:**
- Não implementar CodeSessionService completo
- Não adicionar GitService
- Não criar ProviderRouter
- Não mexer no loop do agent

**Por quê?** Porque artifacts/diffs é a coisa mais concreta que o HermesWeb precisa para visualizar mudanças de código. É o ponto de integração mais seguro e visível. Corrigir isso primeiro dá ROI imediato e não arrisca quebrar o agent loop.

---

## 15. Checklist final

- [x] Runtime principal identificado
- [x] Loop do agente identificado
- [x] Tools mapeadas
- [x] Skills mapeadas
- [x] Sessões mapeadas
- [x] Banco mapeado
- [x] API HTTP mapeada
- [x] WebSocket mapeado
- [x] Artifacts/diffs mapeados
- [x] Providers/modelos mapeados
- [x] Gaps identificados
- [x] Arquivos para Fase 1 listados
