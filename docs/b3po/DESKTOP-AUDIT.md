# DESKTOP-AUDIT — Electron/React do B-3PO

## Stack real

| Item | Valor |
|---|---|
| Pasta | `apps/desktop` (Electron + React + Vite, TypeScript) |
| Build | `scripts/build.mjs` (renderer + electron main) |
| Empacotamento | `electron-builder.config.cjs` via `scripts/run-electron-builder.mjs` |
| Targets declarados | Windows **msix**; macOS **dmg/zip/mas**; Linux **AppImage, deb, rpm** |
| Testes | Vitest (`test:ui`, `test:desktop:platforms`), Playwright (`test:e2e`), harness de lifecycle (`test:desktop:*`) |
| i18n | 9 locales (`ar, de, en, es, fr, ja, ru, zh, zh-hant`) — **sem `pt`** |
| Plugins | `hermes-bots` (202 arquivos) = módulo de bots |

## Evidência desta rodada

| Passo | Comando | Resultado |
|---|---|---|
| Instalação | `npm ci --no-audit --no-fund` | 1222 pacotes, 1 min, **exit 0** |
| Typecheck | `npm run typecheck` | **exit 0** (renderer, electron, e2e e config) |
| Build | `npm run build` | ver `TEST-RESULTS.md` |

## Riscos identificados

| Risco | Impacto | Estado |
|---|---|---|
| Sem certificado de assinatura Windows | SmartScreen alerta no `.exe` | declarar como não assinado, nunca alegar assinado |
| MSIX exige ferramentas do Windows SDK | build pode falhar por ambiente | validar no primeiro `dist:win` |
| Sem macOS/Xcode | `dist:mac` impossível aqui | **BLOCKED_BY_EXTERNAL_DEPENDENCY** |
| Linux só via Docker/WSL | AppImage/deb não saem nativos no Windows | viável, mas não executado nesta rodada |
| `main` não pode receber nada | qualquer atalho quebra F2-A | commits só em `b3po-brand` |

## O que o marco F5 exige do desktop

Criar/editar bot, `mascot_id` persistido, avatar oficial renderizado, turno real, estado observável, erro,
cancelamento, reinício e migração de config. O código tem infraestrutura para todos (ver
`BOT-SCHEMA-AUDIT.md`), mas **nenhum fluxo foi exercitado pela UI** nesta rodada — permanece
**INFRAESTRUTURA PARCIAL** até haver screenshot + log do fluxo completo.
