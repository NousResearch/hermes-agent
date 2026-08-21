# Revisão do Plano — 2026-08-10

**Revisor:** code-reviewer agent  
**Veredicto:** Aprovado com ressalvas → **correções aplicadas nos docs**

---

## Achados críticos (corrigidos)

| # | Achado | Correção |
|---|---|---|
| 1 | Smoke `/api/model/library` sem auth | Fluxo 3 documenta token obrigatório |
| 2 | Claim desktop usa model/library | PATCHES_LOCAIS + Fluxo 3: consumer = Hermes One externo |
| 3 | Glob inválido em test-slice | CATALOGO + checklist: diretórios explícitos |
| 4 | run_tests.sh requer bash no Windows | D-006 + CATALOGO: Git Bash / MinGit |

## Achados médios (corrigidos)

| # | Achado | Correção |
|---|---|---|
| 5 | `e2e_filter=model` inexistente | hermes-desktop-smoke usa `spec` path |
| 6 | Commit final sem passo | Fluxo 1 passo 10 + aprovação |
| 7 | `apply` vs `doctor --fix` | Contrato usa `fix: bool` |
| 8 | Sem teste model library | Checklist: smoke manual até teste existir |
| 9 | Gateway Windows | hermes-gateway-ops menciona gateway-service |

## Claims verificadas ✅

- Patches HERMES_ONE e OpenRouter prune existem no código
- HERMES_HOME Windows = `%LOCALAPPDATA%\hermes`
- Entry points e versão 0.19.0 corretos
- Restrições AGENTS.md refletidas na spec

## Pendente (não bloqueia patch-guard)

- Confirmar D-003..D-006
- Implementar skill `hermes-patch-guard`
- Testes HTTP automatizados para model library (follow-up)
- `.git/info/exclude` para `harness/` (D-007)
