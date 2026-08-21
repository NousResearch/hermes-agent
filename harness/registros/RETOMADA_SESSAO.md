# Retomada de Sessão — Harness Hermes Agent

**Salvo em:** 2026-08-10 17:33 (sync origin/main +47 commits)  
**Objetivo:** reiniciar Cursor sem perder contexto.

---

## Prompt para colar na nova janela

```
Retomar harness Hermes Agent — ler harness/registros/RETOMADA_SESSAO.md e executar próximo passo.
python "$env:LOCALAPPDATA\hermes\hermes-agent\harness\scripts\hermes_patch_guard.py"
```

CLI no PATH (`hermes`) já aponta para `venv\Scripts\hermes.exe`. Fallback:

```powershell
& "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\hermes.exe" gateway status
```

---

## Feito nesta sessão (sync 17:33)

1. `git fetch origin main` — tip `b614f70361`
2. patch-guard baseline ✅
3. `git merge origin/main` — ort, sem conflito (+47 commits)
4. patch-guard pós-merge ✅ (Hermes One + OpenRouter prune)
5. `hermes doctor` ✅ exit 0 — 2 avisos setup (API keys opcionais)
6. Docs STATUS / RETOMADA / LOG atualizados

---

## Estado atual

| Item | Valor |
|---|---|
| Branch | `local/harness` @ `2027ea6279` (**0** atrás / **13** à frente de origin/main) |
| Upstream | `origin/main` @ `b614f70361` |
| Pacote | pyproject **0.20.0** |
| Config | v34 · `minimax-oauth` / `MiniMax-M3` |
| SQLite prod | **3.53.1** |
| Gateway | (não reiniciado nesta sync — restart só com `--confirm`) |
| Cron | fleet anterior intacta |

### Cron fleet

| Job | Model | Nota |
|---|---|---|
| relatorio-repos-18h | MiniMax-M3 | ok |
| inbox-listar-dande | MiniMax-M3 | pausado |
| cliente-perfil-dande | MiniMax-M3 | ok |
| avaliacao-agente-dande | MiniMax-M3 | ✅ revalidado 12:06 (`ok`) |
| v1-inbox-snapshot-30min | gpt-5.5 | ok |
| cerebro-faxina | gpt-5.5 | ok |

---

## Pendências

| # | Item | Prioridade |
|---|---|---|
| 1 | Commit merge + docs sync (humano) | Média |
| 2 | (Opcional) PR upstream OpenRouter prune | Baixa |
| 3 | (Opcional) hook `register_api_mount` p/ plugin puro | Baixa |
| 4 | ~~Sync origin/main +47~~ ✅ | — |
| 5 | ~~Fix avaliacao-agente-dande~~ ✅ | — |
| 6 | ~~D-004 extração modular~~ ✅ | — |
| 7 | ~~Remover Gemini (title_generation)~~ ✅ | — |

---

## Gemini

Removido em 2026-08-10: chaves `GEMINI_API_KEY`/`GOOGLE_API_KEY` do User env; `auxiliary.title_generation` usa MiniMax-M3.

**Usar:** `gpt-5.5`, `gpt-5.4`, `gpt-5.3-codex`, `gpt-5.4-mini`  
**Não usar:** `gpt-5.2-codex`, `gpt-5.1-codex-max`, `gpt-5.1-codex-mini`

---

## Restrições

- Nunca `git add .`
- `hermes update` pode checkoutar `main` — voltar para `local/harness`
- Restart gateway: `--confirm` no gateway-ops
- `hermes.cmd` já corrigido; se voltar a abrir agent chat, apontar de novo para `hermes.exe`
