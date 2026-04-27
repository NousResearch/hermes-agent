# Phase 8 — Multi-Agent Coding Flow

## Objetivo

Criar uma camada de orquestração de fluxo de codificação multi-agente rastreável e segura, integrada ao Hermes Code Mode. O objetivo é permitir que tarefas de código passem por papéis claros com persistência de estado e integração com todos os serviços existentes.

---

## Arquitetura do Fluxo

```
create_flow()
    │
    ▼
Orchestrator  → plano determinístico baseado no workspace
    │
    ▼
Coder         → coleta contexto (git status + diagnostics iniciais)
    │
    ▼
Tester        → executa comandos de validação via CommandRunner
    │           (safe → executa; needs_approval → pausa; blocked → erro)
    ▼
Reviewer      → analisa diff, diagnostics, riscos
    │           → cria approval se necessário
    ▼
completed / waiting_approval / failed
```

---

## Papéis dos Agentes

| Role | Responsabilidade |
|------|-----------------|
| `orchestrator` | Entende a tarefa, detecta stack, cria plano com steps e comandos de teste |
| `coder` | Coleta contexto do workspace (git status, diagnostics iniciais), registra intenção |
| `tester` | Executa comandos classificados como `safe` via CommandRunner; pausa em `needs_approval` |
| `reviewer` | Analisa diff do GitService, diagnostics finais, riscos do plano; cria approval se necessário |
| `researcher` | Reservado para futuras fases (coleta de documentação/contexto externo) |

### Presets por role

| Role | Preset padrão |
|------|--------------|
| orchestrator | `planner` |
| coder | `strong` |
| tester | `fast` |
| reviewer | `reviewer` |
| researcher | `fast` |

---

## Estados do Flow

```
created
  │
  ▼
planning          (Orchestrator rodando)
  │
  ▼
coding            (Coder coletando contexto)
  │
  ▼
running_tests     (Tester executando comandos)
  │
  ├──────────────► waiting_approval   (comando needs_approval detectado)
  │                      │
  │                      └─ resume_flow() ──► reviewing
  ▼
reviewing         (Reviewer analisando diff/diagnostics)
  │
  ├──────────────► waiting_approval   (diff/erros requerem aprovação humana)
  │
  └──────────────► completed
```

Estados terminais: `completed`, `failed`, `cancelled`.

---

## Tabelas Criadas (SCHEMA_VERSION 15)

### `code_agent_flows`

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `id` | TEXT PK | UUID do flow |
| `code_session_id` | TEXT NOT NULL | CodeSession vinculada |
| `workspace_id` | TEXT NOT NULL | Workspace alvo |
| `task_id` | TEXT | Task Hermes vinculada (opcional) |
| `title` | TEXT | Título do flow |
| `description` | TEXT | Descrição da tarefa |
| `status` | TEXT | Estado atual do flow |
| `current_role` | TEXT | Role executando agora |
| `provider` | TEXT | Provider LLM selecionado |
| `model` | TEXT | Modelo selecionado |
| `preset` | TEXT | Preset inicial |
| `plan_json` | TEXT | Plano do Orchestrator (JSON) |
| `review_json` | TEXT | Review do Reviewer (JSON) |
| `approval_id` | TEXT | ID da approval pendente |
| `error` | TEXT | Erro se status=failed |
| `created_at` | TEXT | ISO timestamp |
| `updated_at` | TEXT | ISO timestamp |
| `completed_at` | TEXT | ISO timestamp (terminal) |

### `code_agent_flow_steps`

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `id` | TEXT PK | UUID do step |
| `flow_id` | TEXT NOT NULL | Flow pai |
| `role` | TEXT | Role do step |
| `name` | TEXT | Nome descritivo |
| `status` | TEXT | `pending/running/completed/failed/skipped` |
| `input_json` | TEXT | Input do step (JSON) |
| `output_json` | TEXT | Output do step (JSON) |
| `error` | TEXT | Erro se falhou |
| `started_at` | TEXT | ISO timestamp |
| `completed_at` | TEXT | ISO timestamp |
| `created_at` | TEXT | ISO timestamp |

---

## Endpoints REST

| Método | Path | Descrição |
|--------|------|-----------|
| `GET` | `/api/code/agent-flows` | Lista flows (filtros: code_session_id, workspace_id, status, limit) |
| `POST` | `/api/code/agent-flows` | Cria novo flow |
| `GET` | `/api/code/agent-flows/:flow_id` | Consulta flow com steps |
| `POST` | `/api/code/agent-flows/:flow_id/run` | Executa flow |
| `POST` | `/api/code/agent-flows/:flow_id/cancel` | Cancela flow |
| `POST` | `/api/code/agent-flows/:flow_id/resume` | Retoma flow em waiting_approval |
| `GET` | `/api/code/sessions/:code_session_id/agent-flows` | Lista flows de uma session |

### Payload de criação

```json
{
  "workspace_id": "...",
  "code_session_id": "...",
  "task_id": "...",
  "title": "Fix chat navigation bug",
  "description": "Clicking back button causes state loss in chat view",
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "preset": "planner"
}
```

### Resposta padrão

```json
{
  "flow": {
    "id": "...",
    "code_session_id": "...",
    "workspace_id": "...",
    "status": "created",
    "current_role": null,
    "plan": {},
    "review": null,
    "steps": [],
    "approval_id": null,
    "created_at": "...",
    "updated_at": "..."
  }
}
```

---

## Eventos WebSocket

Eventos emitidos via `_REALTIME_HUB.broadcast`:

| Evento | Quando |
|--------|--------|
| `code_flow.created` | Flow criado |
| `code_flow.updated` | Flow executado ou retomado |
| `code_flow.cancelled` | Flow cancelado |

Eventos no timeline da CodeSession (`code_session_events`):

| Evento | Quando |
|--------|--------|
| `agent.started` | Flow criado |
| `agent.status_changed` | Mudança de role/status |
| `agent.waiting_approval` | Pausa para aprovação |
| `agent.completed` | Flow concluído |
| `agent.failed` | Falha inesperada |

---

## Segurança

### Classificação de comandos (CommandRunner)

| Classificação | Comportamento |
|--------------|---------------|
| `safe` | Executa automaticamente |
| `needs_approval` | Cria `Approval`, pausa flow em `waiting_approval` |
| `blocked` | Step marcado `failed`, erro claro registrado, flow continua |

### Comandos nunca executados automaticamente

- `npm install`, `pnpm install`, `yarn install`, `bun install`
- `git push`, `git commit`, `git checkout`
- `sudo`, `rm -rf`, `git reset --hard`, `git clean -fd`
- `docker compose down -v`
- qualquer outro classificado como `needs_approval` ou `blocked`

### Aprovações criadas automaticamente

- Comando `needs_approval` detectado pelo Tester
- Reviewer detecta: erros de diagnóstico, riscos no plano, arquivos modificados

Tipo de approval: `command` (Tester) ou `code_review` (Reviewer).

---

## Integrações

### CodeSession
- Flow vinculado por `code_session_id`
- Herda `provider` e `model` da session se não especificados
- Eventos registrados na timeline da session

### Workspace
- Validado na criação do flow
- Stack detectada usada para gerar comandos de teste no plano

### CommandRunner
- Tester delega execução de comandos
- Classificação de segurança respeitada em 100% dos casos

### GitService
- Coder coleta git status + arquivos modificados
- Reviewer coleta diff e stat para incluir no review

### ProviderRouter
- `provider`, `model` e `preset` guardados no flow
- Mapeamento role→preset definido para uso futuro com LLMs reais

### CodeIntelligence/LSP
- Diagnostics coletados no início (Coder) e fim (Reviewer)
- Erros novos detectados pelo Reviewer disparam `request_changes`

### Approvals
- Criadas via `ApprovalDB.create_approval()`
- `approval_id` salvo no flow
- `resume_flow()` retoma após aprovação externa

---

## Como Testar

```bash
# Testes Phase 8
uv run pytest tests/hermes_cli/test_multi_agent_coding.py -v

# Suíte crítica completa
uv run pytest \
  tests/hermes_cli/test_multi_agent_coding.py \
  tests/hermes_cli/test_lsp_service.py \
  tests/hermes_cli/test_provider_router.py \
  tests/hermes_cli/test_git_service.py \
  tests/hermes_cli/test_command_runner.py \
  tests/hermes_cli/test_code_session_service.py \
  tests/hermes_cli/test_workspace_service.py \
  tests/test_artifacts.py \
  tests/test_hermes_state.py \
  -v
```

---

## Limitações Conhecidas

1. **LLM não integrado**: O Orchestrator usa heurísticas determinísticas baseadas no stack detectado. A chamada real ao LLM via ProviderRouter é estrutura preparada para Fase 9+.
2. **Coder não edita código**: O Coder coleta contexto mas não aplica patches automaticamente. Fase 9 introduzirá edição via ferramentas seguras.
3. **Researcher não implementado**: Role `researcher` está definido mas sem steps concretos nesta fase.
4. **run_flow é síncrono**: Para flows longos, uma versão assíncrona em background task será adicionada nas próximas fases.
5. **HermesWeb**: Sem UI nova nesta fase. Endpoints documentados para integração futura na Fase 10.

---

## Próximos Passos (Fase 9+)

- Integrar ProviderRouter para chamar LLM real no Orchestrator e Reviewer
- Coder com capacidade de aplicar patches via file_tools seguras
- Researcher com coleta de documentação e contexto externo
- run_flow assíncrono com background tasks e polling WebSocket
- UI no HermesWeb: visualização de flows dentro da Code Session View
