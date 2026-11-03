# EVIDENCE-INDEX — afirmação → prova

| Afirmação | Evidência | Arquivo / comando |
|---|---|---|
| `main` não recebeu commit DZ23 | `git rev-list --count upstream/main..main` → 0 | git |
| Fork 26 commits atrás | `git rev-list --count main..upstream/main` → 26 | git |
| Ambiente provisionado | `setup-hermes.ps1 -RuntimeOnly` exit 0; Python 3.14.9 | log da sessão |
| Turno real do agente | `hermes -z "…"` → `B-3PO online ✅`, 13,4 s | `c3po-hermes-home/evidence/fase-b-turno.md` |
| Provider local sem chave | Ollama 0.40.13, `/v1/chat/completions` respondeu | log da sessão |
| Marca como dado | `brand.json`, `palette.json`, `generated/*` | `brand/` |
| Gate de marca | 14 checagens, 0 falhas | `python scripts/brand_check.py` |
| Gate do repo | 14 checks, ok | `python scripts/check` |
| 20 mascotes 512×512 | contagem + dimensões por IHDR | `brand/avatars/` + `manifest.json` |
| Contraste AA | 20,49 / 15,04 / 10,16 / 10,74 :1 | `BRAND-ASSET-REPORT.md` |
| Dependências instaladas | 1222 pacotes, 1 min, exit 0 | `npm ci` |
| Desktop typecheck | exit 0 | `npm run typecheck` |
| Desktop build | exit 0, 0,5 min | `npm run build` |
| Desktop empacotado | exit 0, 1,3 min; `Hermes.exe` 235,4 MB | `release/win-unpacked/` |
| Artefato não assinado | aviso `AZURE_SIGN_* not set` | log do `pack` |
| Eventos reais do runtime | `turn.start/end/error`, `tool.start/complete` | `tui_gateway/` |
| Aprovação existente | `tools/approval.py` | código |
| Sem macOS/iOS | `xcodebuild`/`xcrun` ausentes | `PLATFORM-MATRIX.md` |
| Sem projeto mobile | caminhos `apps/android|ios` inexistentes | `MOBILE-FEASIBILITY-REPORT.md` |

## O que NÃO tem evidência (e por isso não é afirmado como pronto)

- Fluxo F5 na UI (criar bot → mascote → turno → cancelar → reiniciar).
- Instalação limpa / upgrade / desinstalação.
- Vitest (`test:ui`, `test:desktop:platforms`) e Playwright E2E.
- `npm audit`, SBOM, revisão de licenças de dependências.
- macOS, Linux, Android, iOS.
