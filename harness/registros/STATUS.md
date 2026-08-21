# Status — Harness Hermes Agent

**Última atualização:** 2026-08-10 17:33 (sync origin/main +47)

## Fase atual

| Fase | Status |
|---|---|
| Sync origin/main | ✅ `b614f70361` (0 atrás / 13 à frente) |
| Cron `avaliacao-agente-dande` | ✅ revalidado (`ok`) |
| D-004 model library | ✅ módulo + mount fino |
| OpenRouter prune | ⏸ local (candidato upstream) |
| patch-guard pós-sync | ✅ |

## Repositório

| Campo | Valor |
|---|---|
| Branch | `local/harness` @ `2027ea6279` |
| Upstream tip | `origin/main` @ `b614f70361` |
| Pacote | `pyproject` **0.20.0** |
| Provider | `minimax-oauth` / `MiniMax-M3` |
| Config | v34 (`%LOCALAPPDATA%\hermes\config.yaml`) |
| SQLite prod | 3.53.1 |

## Próximo passo

1. Commit merge + docs desta sync (humano)
2. Opcional: PR upstream OpenRouter prune
3. Opcional: hook `register_api_mount` se precisar plugin puro

## Boot

Ler: **`harness/registros/RETOMADA_SESSAO.md`**
