# BASELINE-REPORT — B-3PO

**Data:** 2026-11-02 · **Branch:** `b3po-brand` @ `21595add` · **Upstream:** `6ab0de91` (26 commits à frente do fork)
**Método:** leitura + comandos reproduzíveis; nenhuma alteração em `main`.

## 1. Repositório

| Item | Valor |
|---|---|
| Remotes | `origin` = DZ23-LTDA/B-3PO (público), `upstream` = NousResearch/hermes-agent |
| `main` | `f4cf411f` — **0 commits DZ23** (fast-forward only, F2-A respeitado) |
| `b3po-brand` | `21595add` — 2 commits DZ23 (fundação de marca + correções) |
| Worktree | limpo |
| Arquivos rastreados | 22.155 |
| Licença | MIT (Nous Research) + `NOTICE` do fork |

## 2. Ambiente (verificado)

| Ferramenta | Versão | Situação |
|---|---|---|
| Python | 3.14.9 (PM) / 3.12.10 (sistema) | `requires-python >=3.11,<3.15` → OK |
| Node / npm | v26.12.1 / 12.0.2 | `>=26` exigido → OK |
| Dependências do monorepo | `npm ci` → 1222 pacotes, 1 min, exit 0 | OK |
| Provider | Ollama 0.40.13 local (`127.0.0.1:11434`), `qwen3.5:4b` | turno real executado |
| JDK | 21.0.13 | presente (sem projeto Android) |
| Android SDK | `D:\IA-Workspace\Deepseek\tools\android\sdk` | presente; sem projeto mobile |
| Xcode / macOS | ausente | **bloqueia macOS e iOS** |
| Docker / WSL | presentes | viabilizam build Linux por container |

## 3. Classificação dos recursos (regra do GOAL)

| Recurso | Classificação | Evidência |
|---|---|---|
| Runtime Hermes executa turno | **FUNCIONALIDADE VALIDADA** | `hermes -z "…"` → `B-3PO online ✅`, exit 0, 13,4 s |
| Gate de marca | **FUNCIONALIDADE VALIDADA** | `brand_check`: 14/14; `scripts/check`: 14 checks ok |
| Eventos de turno/ferramenta | **CONFIRMADO** (código) | `tui_gateway`: `turn.start/end/error`, `tool.start/complete` |
| Aprovação de ferramenta | **CONFIRMADO** (código) | `tools/approval.py` com notify/resolve/ack |
| Cancelamento | **IMPLEMENTADO MAS NÃO VALIDADO** | `_handle_interrupt` existe; não provado pela UI |
| Módulo de bots do desktop | **INFRAESTRUTURA PARCIAL** | 202 arquivos, nunca executado ponta a ponta |
| Build do desktop | ver `DESKTOP-AUDIT.md` | — |
| Mobile | **BLOCKED_BY_EXTERNAL_DEPENDENCY** | sem projeto; macOS ausente para iOS |

## 4. Bloqueios estruturais

1. **macOS/iOS**: exige hardware Apple + Xcode + certificados — indisponível nesta máquina.
2. **Android**: SDK existe, mas **não há projeto mobile** no repositório; criar cliente é trabalho novo (ver `MOBILE-ARCHITECTURE-DECISION.md`).
3. **Linux**: build viável apenas via Docker/WSL (electron-builder não gera AppImage nativo no Windows).

## 5. Decisão de baseline

Prosseguir com **desktop Windows + núcleo (schema/orquestração)**, que é o marco F5 aprovado, e tratar
mobile/macOS/Linux como frentes separadas com bloqueio registrado — exatamente como o GOAL §18 manda.
