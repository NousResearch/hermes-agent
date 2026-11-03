# TEST-RESULTS — execução desta rodada

**Commit avaliado:** `b3po-brand @ 21595add` + docs desta rodada
**Ambiente:** Windows 11 Pro · Node v26.12.1 · npm 12.0.2 · Python 3.14.9 (PM)

| # | Teste | Comando | Resultado | Evidência |
|---|---|---|---|---|
| T1 | Gate do repositório | `python scripts/check` | **14 checks, ok** (exit 0) | saída no log da sessão |
| T2 | Gate de marca | `python scripts/brand_check.py` | **14 checagens, 0 falhas** | idem |
| T2b | Gerador idempotente | `python scripts/brand_sync.py --check` | verificado | idem |
| T3 | Typecheck desktop | `npm run typecheck` (apps/desktop) | **exit 0** | renderer + electron + e2e + config |
| T6 | Build desktop | `npm run build` | **exit 0** em 0,5 min | `.dist-build-*/product/electron-main.mjs` 3,0 MB |
| T7 | Empacotamento | `npm run pack` | **exit 0** em 1,3 min | `release/win-unpacked/Hermes.exe` |
| T10 | Turno real do agente | `hermes -z "Responda apenas: B-3PO online"` | **`B-3PO online ✅`**, exit 0, 13,4 s | `c3po-hermes-home/evidence/fase-b-turno.md` |

## Avisos registrados (não escondidos)

1. `[batch-sign] AZURE_SIGN_* not set — payload binaries will be UNSIGNED` — empacotamento Windows **sem
   assinatura** (ADR-007).
2. Build reporta callbacks lentos do plugin Babel (20,6 s em 865 chamadas) — performance de build, não de
   produto; registrado em `PERFORMANCE-REPORT.md` (pendente).
3. `npm ci` trouxe pacotes deprecados (inflight, rimraf@2/3, glob@7) — herdados do upstream; registrar na
   auditoria de dependências.

## Não executado nesta rodada

- `npm run test:ui` e `test:desktop:platforms` (Vitest) — **pendente**; rodar antes do PR final.
- `test:e2e` (Playwright) — exige app em display; não executado.
- `test:desktop:lifecycle` (instalação/atualização/desinstalação) — não executado.
- `npm audit` / SBOM — pendente.
- Android/iOS/Linux builds — impossíveis ou fora de escopo (ver `PLATFORM-MATRIX.md`).
