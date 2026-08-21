# Mapa do Repositório — Hermes Agent

**Classificação:** Confirmada (exploração 2026-08-10)  
**Path:** `C:\Users\User\AppData\Local\hermes\hermes-agent`  
**Versão:** 0.19.0 | **Python** ≥3.11,<3.14 | **Node** ≥20

---

## Visão geral

Hermes Agent é um agente pessoal de IA com **mesmo core** em CLI, gateway (~20 plataformas), TUI Ink e desktop Electron. Aprende via memória + skills, delega subagentes, roda cron, terminal real e browser.

**Dois princípios sagrados** (`AGENTS.md`):

1. **Prompt caching** — não mutar contexto/toolsets/system prompt mid-conversation
2. **Core estreito** — capability nas bordas (plugins, skills, MCP)

---

## Estrutura top-level

| Pasta | Propósito |
|---|---|
| `run_agent.py`, `cli.py`, `model_tools.py`, `toolsets.py` | Loop agente, CLI, tools |
| `agent/` | Providers, memória, compressão, credential pool, curator |
| `hermes_cli/` | CLI subcomandos, setup, web server, plugins loader, skins |
| `tools/` | Implementações de tools (terminal, browser, file, MCP…) |
| `gateway/` | Messaging gateway + adapters por plataforma |
| `tui_gateway/` | Backend JSON-RPC Python para TUI |
| `ui-tui/` | Frontend Ink/React (`hermes --tui`) |
| `apps/desktop/` | Electron + React (chat próprio, spawna `hermes serve`) |
| `apps/shared/` | `@hermes/shared` — JSON-RPC/WS |
| `web/` | Dashboard React — embute TUI via PTY |
| `website/` | Docusaurus (docs públicas) |
| `plugins/` | Plugins bundled (memory, providers, platforms…) |
| `skills/`, `optional-skills/` | Skills bundled / opt-in |
| `cron/`, `acp_adapter/` | Scheduler, ACP (IDE) |
| `tests/`, `tests-js/` | Pytest (~17k) + Vitest |
| `scripts/` | `run_tests.sh`, CI, install, release |
| `docs/` | Docs internas (não site público) |
| `harness/` | **Este harness** (local, não upstream) |

---

## Entry points

| Superfície | Comando | Código-chave |
|---|---|---|
| CLI | `hermes` | `hermes_cli/main.py`, `cli.py` |
| TUI | `hermes --tui` | `ui-tui/`, `tui_gateway/server.py` |
| Gateway | `hermes gateway` | `gateway/run.py` |
| Dashboard | `hermes dashboard` | `hermes_cli/web_server.py`, `web/` |
| Backend headless | `hermes serve` | Mesmo server, `HERMES_SERVE_HEADLESS=1` |
| Desktop | App Electron | `apps/desktop/` |
| ACP | `hermes acp` | `acp_adapter/` |
| Agente direto | `hermes-agent` | `run_agent.py` |

Console scripts (`pyproject.toml`):

- `hermes` → `hermes_cli.main:main`
- `hermes-agent` → `run_agent:main`
- `hermes-acp` → `acp_adapter.entry:main`

---

## Config do usuário

| Path | Conteúdo |
|---|---|
| `%LOCALAPPDATA%\hermes\config.yaml` | Settings comportamentais |
| `%LOCALAPPDATA%\hermes\.env` | **Somente segredos** |
| `%LOCALAPPDATA%\hermes\sessions/` | SQLite FTS5 |
| `%LOCALAPPDATA%\hermes\skills/` | Skills do usuário |
| `%LOCALAPPDATA%\hermes\plugins/` | Plugins instalados |
| `%LOCALAPPDATA%\hermes\models.json` | Biblioteca modelos Hermes One (patch local) |
| `%LOCALAPPDATA%\hermes\logs/` | agent.log, errors.log, gateway.log |

Helpers: `get_hermes_home()`, `display_hermes_home()` em `hermes_constants.py`.

Profiles: `hermes -p <name>` → `HERMES_HOME` isolado.

---

## Plugins e skills

**Plugins** (`hermes_cli/plugins.py`): descoberta em repo, `~/.hermes/plugins/`, pip entry points. Hooks: pre/post tool, pre/post LLM, session start/end; register tools e CLI subcommands.

**Skills**: `skills/` (bundled), `optional-skills/` (install explícito), formato `SKILL.md` + scripts. Curator gerencia skills `created_by: agent`.

**Footprint ladder:** extend → CLI+skill → service-gated tool → plugin → MCP → core tool.

---

## CI / testes

- Runner canônico: `scripts/run_tests.sh` (nunca pytest direto)
- Orquestrador: `.github/workflows/ci.yml` → tests, lint, js-tests, docker, e2e-desktop, supply-chain
- Isolamento: temp `HERMES_HOME`, credenciais unset, TZ=UTC

---

## Docs de referência interna

| Doc | Uso |
|---|---|
| `AGENTS.md` | Rubric contribuição + arquitetura (~1400 linhas) |
| `apps/desktop/AGENTS.md` | Regras desktop |
| `docs/` | Design, security, kanban, session-lifecycle |
| `website/docs/` | Docs públicas Docusaurus |

---

## Diagrama de superfícies

```
CLI ──────┐
TUI ──────┼──► AIAgent (run_agent.py) ──► model_tools ──► tools/plugins/MCP
Gateway ──┤
Desktop ──┘         ▲
                    │
Dashboard (PTY) ────┘ embute TUI, não reimplementa chat
```
