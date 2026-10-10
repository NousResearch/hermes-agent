# ESTADO_ORCHESTRATOR

> Este arquivo é o cérebro da sessão do Orchestrator. Ele persiste o progresso entre interações e permite resumir trabalho se a sessão cair ou o contexto esgotar.
>
> **Regra**: O Orchestrator deve ler este arquivo ao iniciar e escrever nele ao final de cada fase.

---

## Sessão

- **iniciado_em**: `YYYY-MM-DD HH:MM:SS`
- **fase_atual**: `Fase 3` | `Fase 4` | `Fase 5`
- **repositorio**: `<nome-do-repo>`

---

## GAPs Identificados (Fase 3)

| # | ID | Dimensão | Severidade | Descritivo | Tier Risco | Status |
|---|----|----------|------------|------------|------------|--------|
| 1 | `GAP-001` | Segurança | P1 | ... | T3 Bloqueante | 🔴 open |
| 2 | `GAP-002` | Arquitetura | P2 | ... | T2 Batchável | 🟡 queued |
| 3 | `GAP-003` | Lint | P4 | ... | T1 Auto | 🟢 done |

---

## Tarefas (Fase 4 — Fila DAG)

### Convenção de ID

- `TASK-<NNN>` — identificador unico.
- `depends_on: ["TASK-<MMM>", ...]` — lista de IDs das tarefas predecessoras.
- `issue_ref` — numero da Issue no GitHub (#123).
- `spec_ref` — `.specs/SPEC-{YYYYMMDD}-{slug}.md` vinculado.

### Tarefas Pendentes

```yaml
- id: TASK-001
  desc: "Extract email validation into validators/email.ts"
  skill: /execute-specs
  gap_ref: GAP-002
  issue_ref: "#101"
  spec_ref: ".specs/SPEC-20260908-email-validation.md"
  depends_on: []
  status: ready

- id: TASK-002
  desc: "Create tests for validators/email.ts"
  skill: /qa-analyst
  gap_ref: GAP-002
  issue_ref: "#102"
  spec_ref: ".specs/SPEC-20260908-email-validation.md"
  depends_on: [TASK-001]
  status: blocked

- id: TASK-003
  desc: "Update auth architecture decision"
  skill: /write-specs
  gap_ref: GAP-002
  issue_ref: "#103"
  spec_ref: ".specs/SPEC-20260908-auth-architecture.md"
  depends_on: []
  status: ready
```

### Tarefas Concluídas

```yaml
- id: TASK-000
  desc: "Example of completed task"
  skill: /create-agent-harness
  gap_ref: GAP-000
  issue_ref: "#100"
  spec_ref: ".specs/SPEC-20260908-harness.md"
  depends_on: []
  status: done
  concluido_em: "YYYY-MM-DD HH:MM:SS"
```

---

## Checkpoints de Sanidade (Fase 4.5)

| # | Após Tarefa | Data | Reavaliação Necessária? | Ação Tomada |
|---|-------------|------|------------------------|------------|
| 1 | TASK-003 | `...` | Não | Continuar fila |
| 2 | TASK-006 | `...` | Sim — contexto mudou | Re-executar Fase 3 parcial |

---

## Log de Ações Auto-Aplicáveis (Tier T1)

| # | Data | GAP | Ação | Resultado |
|---|------|-----|------|-----------|
| 1 | `...` | GAP-003 | Fix typo em README.md | `git diff` limpo |
| 2 | `...` | GAP-004 | Aplicar prettier em `src/` | Nenhum erro de lint |

---

## Batch de Execução (Tier T2)

> Tarefas batcháveis aguardando aprovação em lote ao final do ciclo.

| # | GAP | Tarefa | Skill | Status |
|---|-----|--------|-------|--------|
| 1 | GAP-002 | Create unit tests for `auth.ts` | /execute-specs | 🟡 pending_approval |
