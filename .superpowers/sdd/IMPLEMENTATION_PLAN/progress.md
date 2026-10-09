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
- Task 1 review round 0: spec FAIL; 1 Critical + 4 Important abertos — account/route reapply, init transaction, policy-off fast path, DB/notification commit ordering e request one-shot rollback.
- Task 1: fix round 1/5 (5 addressed, 0 open; commits d81b7af..87606fa).
- Task 1: complete (commits f97608f..87606fa, review clean).

## Task 2 — retry amplification

- Existing mechanism verified: `agent.auto_recovery_cycles=0` disables the post-exhaustion ladder; tracked suite `test_turn_recovery_autorecover.py` passed 5/5.
- Ruling: no production-code patch — the upstream ladder is intentional and already profile-configurable; changing the global default would widen scope. Zeca requires a profile override only.
- Live apply was rejected by the independent gate despite the broad recovery request; classified `POLICY_LIMITED` until Tales explicitly authorizes the exact profile mutations.
- Task 2: complete as candidate (0 code commits; config promotion pending policy gate).

## Task 3 — continuity, compaction and accounting

- Commit `a9d42556db` adicionou quatro contratos de caracterização; nenhum código de produção mudou.
- A/C/D/B passaram; não houve duplicação ativa/display, apenas histórico arquivado preservado.
- Controller rerun no ambiente Hermes `.venv`: 119 passed em 28.14s; os 9 fails do runner do implementador eram ausência ambiental de `psutil`.
- Revisão independente em curso.
- Task 3 review round 0: 2 Important abertos no teste B — display completeness não provada e check de IDs tautológico; relatório desatualizado.
- Task 3: fix round 1/5 (4 addressed, 0 open; commits a9d4255..708aadc).
- Task 3: complete (commits 87606fa..708aadc, review clean; 119 passed).

## Task 4 — configuration and functional matrix

- 24/24 groups mapped; baseline target tests: 48,180 global pass with 239 known baseline/environment failures, and 1,106 pass / 136 fail / 4 skip in the selected matrix.
- 132 selected failures belong to parked customizations (102 MCP proposal + 30 token integration); token integration is now fixed in branch, MCP remains pre-existing/out of update scope.
- GPT-6.1 disappearance NOT_REPRODUCED: live cache and current sessions contain/use `gpt-6.1-sol`.
- Live registry proves `/plan` and `/blueprint` belong to core while skills remain available through `/skill plan` and `/skill blueprint`; intentional behavior.
- Config deltas were validated in sandbox; the live gate rejected the exact mutations. Recorded as POLICY_LIMITED, no bypass.
- Task 4: complete as evidence/candidate (0 code commits; live promotion pending policy gate).

## Final review and architectural freeze

- Round 0: NOT READY; 0 Critical, 6 Important, 2 Minor.
- Final fix wave commit: `5298a9f2af`; fresh controller verification: 971 passed, 4 skipped; Ruff, compat pointers and diff-check green.
- Scoped re-review: NOT READY. Four of six Important findings were resolved; complete compressor rollback and credential-revert persistence on legitimate `False` remain open. The credential log can consequently still report a success later rolled back.
- After three correction rounds in the same transaction boundary, classify as POSSIBLE_ARCHITECTURAL_FAILURE. Freeze patching and promotion; next work must redesign prepare/commit ownership for compressor and credential-pool transitions before implementation.
- Task 1 fix round 1: cinco achados resolvidos com TDD (8 RED → 8 GREEN); auto-revisão adicional provou/corrigiu que a conta B não herda baseline promovida da conta A. Verificação final: 601 passaram, 0 falharam, 4 skips Windows-only; Ruff/compat/diff-check verdes.
- `GLOBAL_SUITE_BASELINE_RED`: suíte oficial completa em 970,2 s — 48.180 passaram, 239 falharam e 772 skips em 4.716 arquivos; 5 arquivos flaky passaram no retry. Falhas observadas são ambientais/baseline fora da Task 1.
