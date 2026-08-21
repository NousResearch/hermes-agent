# Processos — Hermes Agent

Processos operacionais e candidatos à automação via harness.

---

## 1. Instalação e setup

| Passo | Comando | Notas |
|---|---|---|
| Install Windows | `iex (irm https://hermes-agent.nousresearch.com/install.ps1)` | MinGit bundled |
| Setup wizard | `hermes setup` | Config + `.env` secrets |
| Doctor | `hermes doctor` | Diagnóstico config/deps |
| Tools UI | `hermes tools` | Enable/disable toolsets, MCP |

**HERMES_HOME Windows:** `%LOCALAPPDATA%\hermes\` (não `~/.hermes`).

---

## 2. Desenvolvimento local

| Passo | Comando |
|---|---|
| Ativar venv | `.venv\Scripts\Activate.ps1` |
| Testes (obrigatório) | `scripts/run_tests.sh [tests/path]` |
| Testes JS | `npm run check` ou `npm test -w ui-tui` |
| TUI dev | `cd ui-tui; npm run dev` |
| Desktop dev | `cd apps/desktop; npm run dev` |

**Regra:** nunca `pytest` direto — CI parity via wrapper.

---

## 3. Sync upstream + patches locais

**Gatilho:** merge de `main` NousResearch

1. `git fetch origin main`
2. Inventariar blocos `HERMES_ONE_*` e edits manuais (ver `PATCHES_LOCAIS.md`)
3. Merge/rebase — resolver conflitos em `web_server.py`, `credential_pool.py`
4. `scripts/run_tests.sh` nos paths tocados
5. Smoke: `hermes doctor`, desktop model picker, gateway auth

**Checkpoint humano:** qualquer conflito em patches Hermes One.

---

## 4. Gateway (mensageria)

| Passo | Comando / path |
|---|---|
| Start | `hermes gateway` ou serviço em `%LOCALAPPDATA%\hermes\gateway-service\` |
| Logs | `hermes logs --follow` ou `logs/gateway.log` |
| Config | `config.yaml` → seção `gateway:` |
| Platforms | Telegram, Discord, Slack, WhatsApp, Signal, Matrix… |

**Invariantes:**

- Dois guards de mensagem (adapter + runner) — comandos de controle bypass ambos
- Background terminal notifications via `display.background_process_notifications`

---

## 5. Desktop app

| Passo | Detalhe |
|---|---|
| Spawn backend | `hermes serve` (headless) |
| Transport | JSON-RPC/WS via `@hermes/shared` |
| Slash commands | Curados em `apps/desktop/src/lib/desktop-slash-commands.ts` |
| E2E | Playwright — `.github/workflows/e2e-desktop.yml` |

**Patch Hermes One:** `/api/model/library` para shortcuts SSH/remote model picker.

---

## 6. Cron (jobs agendados)

| CLI | `hermes cron list|add|edit|pause|resume|run|remove` |
| Tool | `cronjob` (agente) |
| Store | `%LOCALAPPDATA%\hermes\cron\` |
| Lock | `{HERMES_HOME}/cron/.tick.lock` (profile-aware via `get_hermes_home()`) |

Cron sessions: `skip_memory=True` por default; interrupt hard 3 min.

---

## 7. Credential pool

| Item | Detalhe |
|---|---|
| Store | `auth.json` em HERMES_HOME |
| Loader | `agent/credential_pool.py` |
| Patch local | Prune entries `env:OPENROUTER_API_KEY` quando var ausente |

**Incidente referenciado:** INCIDENTE-AUTH-JSON-REWRITE — entries zumbi sem prune.

---

## 8. Deploy (inferido — confirmar D-006)

Possível target: VPS Swarm `72.60.249.139` (stack `hermes-*`).  
**Regra CLAUDE.md:** EasyPanel só via UI; Swarm sem EasyPanel.

Deploy **somente com aprovação explícita**.

---

## Candidatos à automação (prioridade provisória)

| Prioridade | Processo | Skill proposta |
|---|---|---|
| Alta | Sync upstream + patch guard | `hermes-sync-upstream`, `hermes-patch-guard` |
| Alta | Gateway ops + logs | `hermes-gateway-ops` |
| Alta | Test slice pré-merge | `hermes-test-slice` |
| Média | Credential audit (sem expor secrets) | `hermes-credential-audit` |
| Média | Cron audit | `hermes-cron-audit` |
| Baixa | Deploy Swarm | `hermes-deploy-swarm` (aprovação prod) |
