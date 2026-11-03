# TEST-PLAN — camadas e o que é executável aqui

## Camadas

| # | Camada | Comando | Executável nesta máquina |
|---|---|---|---|
| T1 | Lint/format Python + gates do repo | `python scripts/check` | **sim** |
| T2 | Gate de marca | `python scripts/brand_check.py` | **sim** |
| T3 | Typecheck do desktop | `npm run typecheck` (apps/desktop) | **sim** |
| T4 | Unit/UI (Vitest) | `npm run test:ui` | **sim** |
| T5 | Electron platform tests | `npm run test:desktop:platforms` | **sim** |
| T6 | Build desktop | `npm run build` | **sim** |
| T7 | Empacotamento | `npm run pack` / `dist:win` | **sim** |
| T8 | E2E Playwright | `npm run test:e2e` | parcial (exige app + display) |
| T9 | Instalação limpa / upgrade / uninstall | `npm run test:desktop:lifecycle` | a verificar |
| T10 | Turno real do agente | `hermes -z` | **sim** (Ollama local) |
| T11 | Migração de schema do Bot | fixtures (a criar na Fase D) | depende de D |
| T12 | Android build | Gradle (não há projeto) | **não** |
| T13 | iOS build | Xcode | **não** (sem macOS) |
| T14 | Linux AppImage/deb | Docker | viável, não executado |
| T15 | npm audit / SBOM | `npm audit`, `npm sbom` | **sim** (pendente) |

## Critérios mínimos de aceite (GOAL §13)

- zero execução duplicada em teste;
- zero capacidade proibida executada;
- zero segredo no log;
- zero perda de configuração em reinicialização;
- erro explícito quando dependência externa falta;
- eventos não suportados declarados.

## Regra

Mocks só valem como teste unitário e devem estar marcados. Integração só conta com execução real
(turno do agente, evento do runtime, arquivo persistido).
