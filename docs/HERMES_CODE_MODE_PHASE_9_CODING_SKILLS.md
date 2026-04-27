# Phase 9 — CodingSkills / Skills Customizadas de Desenvolvimento

## Objetivo

Criar rotinas estruturadas de codificação sobre o Hermes Code Mode — previsíveis, rastreáveis e seguras — que reaproveitam todos os serviços já implementados: CodeSession, CommandRunner, GitService, CodeIntelligence, MultiAgentCodingService e Approvals.

---

## Arquitetura das Skills

```
CodingSkillsService
    │
    ├─ create_run()     → SkillRunDB (code_skill_runs)
    ├─ run_skill()      → dispatch para implementação da skill
    ├─ cancel_run()     → terminal
    └─ resume_run()     → re-executa a partir de waiting_approval

Cada skill:
    ├─ coleta contexto (git status, diagnostics)
    ├─ executa ações seguras (safe only via CommandRunner)
    ├─ cria Approval para needs_approval/code_review
    └─ produz summary + output estruturado
```

---

## Skills Disponíveis

| Skill | Título | Objetivo |
|-------|--------|----------|
| `fix_build` | Fix Build | Detecta e relata erros de build/typecheck/lint; executa comandos safe |
| `review_diff` | Review Diff | Revisa git diff, detecta riscos, cria approval |
| `stabilize_hanging_task` | Stabilize Hanging Task | Detecta e estabiliza sessions/flows/commands presos |
| `fix_runtime_error` | Fix Runtime Error | Analisa stack trace, localiza arquivos, sugere plano de correção |
| `implement_feature` | Implement Feature | Cria MultiAgentCodingFlow para implementar uma feature |
| `refactor_react_page` | Refactor React Page | Prepara plano de refatoração respeitando o design system do HermesWeb |
| `benchmark_provider` | Benchmark Provider | Compara providers/modelos em dry_run (sem alterar arquivos) |

---

## Fluxo de cada Skill

### `fix_build`

1. Coleta diagnostics antes (CodeIntelligenceService)
2. Detecta comandos do stack (npm run typecheck/lint/build, go vet/test, pytest)
3. Classifica cada comando:
   - `safe` → executa via CommandRunner
   - `needs_approval` → cria Approval + pausa em `waiting_approval`
   - `blocked` → step falha com erro claro, continua
4. Se `auto_fix=True` e erros encontrados → cria MultiAgentCodingFlow
5. Coleta diagnostics depois
6. Produz summary com contagem de erros

### `review_diff`

1. Coleta git status + diff (GitService)
2. Coleta diagnostics atuais (CodeIntelligenceService)
3. Detecta riscos:
   - secrets/credentials no diff
   - dependências modificadas
   - migrations SQL
   - diff grande (>20 arquivos)
   - erros de diagnóstico
4. Gera review com `decision: approve | request_changes | blocked`
5. Cria Approval `code_review` se houver arquivos modificados

### `stabilize_hanging_task`

1. Busca sessions em estado ativo (running, planning, coding, etc.)
2. Busca agent flows em estado ativo
3. Busca comandos com status `running`
4. Para cada comando:
   - `safe` → cancela via CommandRunner
   - outros → cria Approval para cancelamento
5. Gera relatório do que foi estabilizado

### `fix_runtime_error`

1. Extrai referências de arquivos do stack trace via regex
2. Adiciona `file_hint` se fornecido
3. Coleta diagnostics do workspace
4. Produz plano de investigação e correção
5. Registra `diagnostics_summary`

### `implement_feature`

1. Valida `code_session_id` (obrigatório)
2. Cria MultiAgentCodingFlow com description da feature
3. Executa o flow (Orchestrator → Coder → Tester → Reviewer)
4. Propaga status do flow (completed / waiting_approval)

### `refactor_react_page`

1. Coleta diagnostics antes
2. Constrói plano de refatoração respeitando princípios HermesWeb:
   - Interface calma, sem animações decorativas
   - Dark mode primeiro
   - Hierarquia clara
   - Baixa carga cognitiva
3. Cria Approval `code_review` — refatoração sempre requer aprovação humana

### `benchmark_provider`

1. `dry_run=True` por padrão — nenhum arquivo é modificado
2. Para cada provider/model: cria flow + executa Orchestrator (planning only)
3. Mede tempo e qualidade do plano
4. Produz ranking por qualidade
5. `dry_run=False` → cria Approval antes de qualquer execução real

---

## Tabela Criada (SCHEMA_VERSION 16)

### `code_skill_runs`

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `id` | TEXT PK | UUID do run |
| `skill_name` | TEXT NOT NULL | Nome da skill |
| `workspace_id` | TEXT NOT NULL | Workspace alvo |
| `code_session_id` | TEXT | CodeSession vinculada (opcional) |
| `task_id` | TEXT | Task vinculada (opcional) |
| `agent_flow_id` | TEXT | Flow criado pela skill (se aplicável) |
| `status` | TEXT | `created/running/waiting_approval/completed/failed/cancelled` |
| `input_json` | TEXT | Input da skill (JSON) |
| `output_json` | TEXT | Output estruturado (JSON) |
| `summary` | TEXT | Texto de resumo final |
| `diagnostics_before_json` | TEXT | Diagnostics antes da execução |
| `diagnostics_after_json` | TEXT | Diagnostics depois da execução |
| `commands_json` | TEXT | Comandos executados (JSON array) |
| `artifacts_json` | TEXT | Artifacts vinculados (JSON array) |
| `approval_id` | TEXT | ID da approval pendente |
| `error` | TEXT | Erro se status=failed |
| `created_at` | TEXT | ISO timestamp |
| `updated_at` | TEXT | ISO timestamp |
| `completed_at` | TEXT | ISO timestamp (terminal) |

---

## Endpoints REST

| Método | Path | Descrição |
|--------|------|-----------|
| `GET` | `/api/code/skills` | Lista skills disponíveis |
| `GET` | `/api/code/skill-runs` | Lista runs (filtros: workspace_id, code_session_id, skill_name, status, limit) |
| `POST` | `/api/code/skill-runs` | Cria novo run |
| `GET` | `/api/code/skill-runs/:run_id` | Consulta run |
| `POST` | `/api/code/skill-runs/:run_id/run` | Executa run |
| `POST` | `/api/code/skill-runs/:run_id/cancel` | Cancela run |
| `POST` | `/api/code/skill-runs/:run_id/resume` | Retoma run em waiting_approval |
| `GET` | `/api/code/sessions/:code_session_id/skill-runs` | Lista runs de uma session |
| `POST` | `/api/code/skills/:skill_name/run` | Atalho: cria + executa em um request |

### Payload de criação

```json
{
  "skill_name": "fix_build",
  "workspace_id": "...",
  "code_session_id": "...",
  "task_id": "...",
  "input": {
    "commands": ["npm run typecheck", "npm run build"],
    "auto_fix": false
  }
}
```

---

## Eventos/Timeline

Eventos WebSocket (via `_REALTIME_HUB.broadcast`):

| Evento | Quando |
|--------|--------|
| `skill.started` | Run criado via POST |
| `skill.updated` | Após run_skill ou resume |
| `skill.cancelled` | Após cancel_run |

Eventos na timeline da CodeSession (`code_session_events`):

| Evento | Quando |
|--------|--------|
| `skill.started` | create_run (se code_session_id) |
| `skill.updated` | run_skill start |
| `skill.waiting_approval` | Pause por approval |
| `skill.completed` | Skill terminada com sucesso |
| `skill.failed` | Falha inesperada |
| `skill.cancelled` | Cancelamento |

---

## Integrações

| Serviço | Uso |
|---------|-----|
| **CodeSession** | Vinculação + eventos na timeline |
| **Workspace** | Validação + detecção de stack para comandos |
| **CommandRunner** | classify_command + create_command + run_command_sync + cancel_command |
| **GitService** | get_status + get_diff em review_diff e stabilize |
| **ProviderRouter** | provider/model/preset armazenados e passados para flows |
| **CodeIntelligence/LSP** | run_diagnostics antes e depois em fix_build, review_diff, refactor |
| **MultiAgentCodingFlow** | implement_feature + fix_build(auto_fix) + benchmark_provider |
| **Approvals** | needs_approval, code_review, dry_run=False |

---

## Segurança

| Classificação | Comportamento |
|--------------|---------------|
| `safe` | Executa automaticamente via CommandRunner |
| `needs_approval` | Cria Approval + flow entra em `waiting_approval` |
| `blocked` | Entry marcado `skipped=True`, erro claro, execução continua |

Nunca executado automaticamente:
- `npm/pnpm/yarn/bun install`
- `git push`, `git commit`, `git reset --hard`, `git clean -fd`
- `sudo`, `rm -rf`, `docker compose down -v`
- `go get`
- Qualquer outro `needs_approval` ou `blocked`

---

## Como Testar

```bash
# Testes Phase 9
uv run pytest tests/hermes_cli/test_coding_skills.py -v

# Suíte crítica completa (502 testes)
uv run pytest \
  tests/hermes_cli/test_coding_skills.py \
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

1. **LLM não integrado**: Skills usam heurísticas determinísticas; chamadas LLM via ProviderRouter ficam para Fase 10+.
2. **Coder sem edição automática**: Skills preparam planos mas não aplicam patches. Edição automática fica para Fase 10.
3. **run_skill é síncrono**: Skills longas podem travar a request. Background tasks são deixados para Fase 10.
4. **benchmark sem LLM real**: dry_run mede apenas tempo de planejamento; métricas de qualidade LLM ficam para Fase 10.
5. **HermesWeb**: Endpoints documentados; UI de skill runs fica para Fase 10.

---

## Próximos Passos (Fase 10+)

- Integrar ProviderRouter para chamar LLM real em cada skill
- Implementar edição de código segura (patch via tools aprovadas)
- Background tasks para run_skill assíncrono com polling WebSocket
- UI no HermesWeb: painel de skill runs dentro de Code Session View
- Métricas de benchmark com qualidade LLM real
