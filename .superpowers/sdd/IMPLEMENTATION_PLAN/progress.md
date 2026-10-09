# SDD ledger — plan: /Users/talesauxagr/.openclaw/workspace/agents-workspaces/tales-aux/shared/4_outputs/hermes-zeca-v0215-regression-recovery-20261009/IMPLEMENTATION_PLAN.md

## Pre-flight

| Tasks | Interface compartilhada | Resultado |
|---|---|---|
| 1 ↔ 2 | init/runtime model state versus recovery state | Sem conflito; retry consome o agente já inicializado e não deve conhecer token policy. |
| 1 ↔ 3 | limites/context compressor versus persistência/compactação | Acoplamento real; Task 3 deve testar o estado efetivo produzido por Task 1 sem reimplementar policy. |
| 1 ↔ 4 | config sintética e catálogo de modelos | Task 4 valida ativação; não altera contratos internos de Task 1. |
| 2 ↔ 4 | override de recovery por perfil | Task 2 produz a chave/semântica; Task 4 apenas prepara o delta Zeca. |
| 3 ↔ 4 | DB temporário versus matriz funcional | Sem escrita no DB live; Task 4 consome apenas os vereditos. |
| 1–4 ↔ 5 | commits, evidências e pacote de promoção | Task 5 não cria comportamento novo; revisa e promove somente o que passou nos gates anteriores. |

- Task 1: coerente; testes preservados definem o contrato e o alvo novo define as APIs de integração.
- Task 2: coerente; configuração por perfil evita alterar o default global sem evidência.
- Task 3: coerente; patches só para bugs reproduzidos em DB temporário.
- Task 4: coerente; delta live permanece candidato até promoção.
- Task 5: coerente; qualquer side effect live continua separado.

Ruling: tratar a falha remota Codex como causa externa e corrigir somente amplificadores locais — evita um patch especulativo de transporte; se estiver errado, o Zeca continuará falhando mesmo com provider saudável e a revisão final detectará isso.

Ruling: `gpt-6.1-sol` não será adicionado ao fallback estático sem reprodução do seletor — o cache real e sessões já provam disponibilidade; se estiver errado, cache frio ainda poderá omitir o modelo até um teste reproduzir o ramo.

## Task 1 — route-scoped token budgets

- Status: DONE_WITH_CONCERNS; contrato focal e adjacente verde, suíte global com falhas ambientais/baseline fora do escopo.
- RED preservado: 103 passaram e 30 falharam (133 total), todas as falhas no runtime ainda não integrado.
- GREEN final focal: 134 passaram; focal + adjacentes: 593 passaram e 4 skips Windows-only.
- Ruling: integrar por `TokenBudgetRuntimeMixin`, após init, mantendo os helpers 0.21.5 como autoridade de switch/fallback/restore.
- Ruling: toda mutação de rota é transacional; falha explícita ou exceção restaura o snapshot completo, inclusive campos adicionados em 0.21.5.
- Ruling: argumentos opcionais `capabilities`/`reset_at` só são encaminhados quando presentes, preservando compatibilidade de callers e mocks legados.
- Ruling: o teste adversarial foi tornado autônomo para funcionar no runner oficial por arquivo, sem adicionar `tests/__init__.py`.
- Evidência detalhada: `.superpowers/sdd/IMPLEMENTATION_PLAN/task-1-report.md`.
- `GLOBAL_SUITE_BASELINE_RED`: suíte oficial completa em 970,2 s — 48.180 passaram, 239 falharam e 772 skips em 4.716 arquivos; 5 arquivos flaky passaram no retry. Falhas observadas são ambientais/baseline fora da Task 1.
