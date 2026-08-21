# Catálogo de Skills — Hermes Agent Harness

Contratos resumidos. Implementação em `~/.cursor/skills/hermes-*/SKILL.md` + scripts em `harness/scripts/`.

| id | script / skill | status |
|---|---|---|
| `hermes-patch-guard` | `harness/scripts/hermes_patch_guard.py` + `~/.cursor/skills/hermes-patch-guard/` | ✅ v0.1.0 |
| `hermes-test-slice` | `harness/scripts/hermes_test_slice.py` + skill | ✅ v0.1.0 (`.venv` + `--extra dev`) |
| `hermes-gateway-ops` | `harness/scripts/hermes_gateway_ops.py` + skill | ✅ v0.1.0 |
| `hermes-credential-audit` | `harness/scripts/hermes_credential_audit.py` + skill | ✅ v0.1.0 |

---

## hermes-patch-guard

| Campo | Valor |
|---|---|
| **id** | `hermes-patch-guard` |
| **versão** | 0.1.0 |
| **objetivo** | Verificar integridade dos patches Hermes One pós-merge |
| **entradas** | `{ "repo_path": string }` |
| **saídas** | `{ "ok": bool, "checks": [{ "name", "status", "evidence" }] }` |
| **ferramentas** | Grep, Read, Shell (git diff) |
| **pré-condição** | Repo git válido |
| **pós-condição** | Relatório sem segredos |
| **validação** | Todos checks `HERMES_ONE_*` e OpenRouter prune presentes |
| **sucesso** | `ok: true` ou lista de regressões |
| **aprovação** | Não |
| **rollback** | N/A (somente leitura) |
| **máx tentativas** | 3 |

---

## hermes-sync-upstream

| Campo | Valor |
|---|---|
| **id** | `hermes-sync-upstream` |
| **versão** | 0.1.0 |
| **objetivo** | Fetch/merge `origin/main` com checkpoint em conflitos |
| **entradas** | `{ "strategy": "merge" \| "rebase" }` |
| **saídas** | `{ "merged": bool, "conflicts": string[] }` |
| **ferramentas** | Shell (git), hermes-patch-guard |
| **pré-condição** | Working tree commitável ou stash |
| **pós-condição** | Se merged, dispara hermes-test-slice |
| **validação** | git status clean ou conflicts listados |
| **sucesso** | Merge clean + patch-guard ok |
| **aprovação** | **Sim** (merge/rebase) |
| **rollback** | `git merge --abort` / `git rebase --abort` |
| **máx tentativas** | 1 por estratégia; trocar estratégia = nova aprovação |

---

## hermes-test-slice

| Campo | Valor |
|---|---|
| **id** | `hermes-test-slice` |
| **versão** | 0.1.0 |
| **objetivo** | Rodar testes CI-parity no slice afetado |
| **entradas** | `{ "paths": string[] }` — diretórios ou arquivos `.py` explícitos (ex. `["tests/agent/"]`) — **sem globs** |
| **saídas** | `{ "passed": bool, "output_summary": string }` |
| **ferramentas** | Shell (`scripts/run_tests.sh` via Git Bash / MinGit no Windows) |
| **pré-condição** | venv ativo ou run_tests.sh resolve venv; **bash disponível** (D-006) |
| **pós-condição** | Exit code 0 |
| **validação** | Wrapper exit 0 |
| **sucesso** | Todos paths verdes |
| **aprovação** | Não |
| **rollback** | N/A |
| **máx tentativas** | 3 |

---

## hermes-doctor-fix

| Campo | Valor |
|---|---|
| **id** | `hermes-doctor-fix` |
| **versão** | 0.1.0 |
| **objetivo** | Executar `hermes doctor`, propor fixes config |
| **entradas** | `{ "fix": bool }` — mapeia para `hermes doctor --fix` |
| **saídas** | `{ "issues": [], "fixes_applied": [] }` |
| **ferramentas** | Shell, Read (config.yaml) |
| **pré-condição** | `hermes` no PATH |
| **pós-condição** | Se apply, backup config antes |
| **validação** | doctor exit 0 ou issues explicadas |
| **sucesso** | Zero issues críticos |
| **aprovação** | **Sim** se `fix: true` |
| **rollback** | Restaurar config backup |
| **máx tentativas** | 2 |

---

## hermes-gateway-ops

| Campo | Valor |
|---|---|
| **id** | `hermes-gateway-ops` |
| **versão** | 0.1.0 |
| **objetivo** | Status, logs, restart gateway |
| **entradas** | `{ "action": "status" \| "logs" \| "restart", "follow": bool }` |
| **saídas** | `{ "status": string, "log_tail": string }` |
| **ferramentas** | Shell, Read (logs) |
| **pré-condição** | HERMES_HOME conhecido; Windows: gateway em `%LOCALAPPDATA%\hermes\gateway-service\` (scheduled task / `.cmd`) |
| **pós-condição** | Logs sem tokens |
| **validação** | Gateway responde / process alive |
| **sucesso** | action completada |
| **aprovação** | **Sim** para restart |
| **rollback** | Restart inverso (stop/start manual) |
| **máx tentativas** | 2 |

---

## hermes-credential-audit

| Campo | Valor |
|---|---|
| **id** | `hermes-credential-audit` |
| **versão** | 0.1.0 |
| **objetivo** | Auditar pool sem expor tokens |
| **entradas** | `{ "provider": string? }` |
| **saídas** | `{ "entries": [{ "provider", "source", "has_token": bool }] }` |
| **ferramentas** | Read (auth.json metadata only), Python one-liner seguro |
| **pré-condição** | auth.json existe |
| **pós-condição** | Output sem valores de token |
| **validação** | Nenhum campo `token`/`refresh_token` no output |
| **sucesso** | Inventário completo |
| **aprovação** | Não |
| **rollback** | N/A |
| **máx tentativas** | 3 |

---

## hermes-cron-audit

| Campo | Valor |
|---|---|
| **id** | `hermes-cron-audit` |
| **versão** | 0.1.0 |
| **objetivo** | Listar jobs cron, flag paused/overdue |
| **entradas** | `{}` |
| **saídas** | `{ "jobs": [{ "id", "schedule", "paused", "last_run" }] }` |
| **ferramentas** | Shell (`hermes cron list`) |
| **aprovação** | Não (list); **Sim** (pause/run) |
| **máx tentativas** | 2 |

---

## hermes-desktop-smoke

| Campo | Valor |
|---|---|
| **id** | `hermes-desktop-smoke` |
| **versão** | 0.1.0 |
| **objetivo** | Build + subset e2e desktop |
| **entradas** | `{ "spec": string? }` — path relativo em `apps/desktop/e2e/` (ex. `"settings.spec.ts"`) |
| **saídas** | `{ "build_ok": bool, "e2e_ok": bool }` |
| **ferramentas** | Shell (`npm run test:e2e` ou `npx playwright test e2e/<spec>`) |
| **aprovação** | Não |
| **máx tentativas** | 2 |

---

## hermes-deploy-swarm

| Campo | Valor |
|---|---|
| **id** | `hermes-deploy-swarm` |
| **versão** | 0.1.0 |
| **objetivo** | Deploy/atualizar stack hermes na VPS Swarm |
| **entradas** | `{ "host", "stack", "image_tag" }` |
| **saídas** | `{ "deployed": bool, "services": [] }` |
| **ferramentas** | Shell (ssh), MCP se disponível |
| **aprovação** | **Sim — produção** |
| **rollback** | Rollback stack anterior |
| **máx tentativas** | 1 |

---

## Ordem de implementação sugerida

1. `hermes-patch-guard` (leitura, zero risco)
2. `hermes-test-slice`
3. `hermes-sync-upstream` (depende 1+2)
4. ~~`hermes-gateway-ops`~~ ✅
5. ~~`hermes-credential-audit`~~ ✅
6. Demais conforme confirmação D-003..D-006
