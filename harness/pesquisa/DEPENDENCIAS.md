# Dependências — Hermes Agent

**Classificação:** Confirmada (pyproject.toml, package.json, CI)

---

## Python

| Item | Valor |
|---|---|
| Gerenciador | `uv` + `uv.lock` |
| Python | `>=3.11,<3.14` |
| Build | setuptools |
| Versão pkg | 0.19.0 |
| Política pin | `>=floor,<next_major` ou `==exact` CI-only |

### Extras notáveis (lazy install)

Providers, terminal backends (docker, ssh, modal, daytona), browser, voice, etc. — ver `[project.optional-dependencies]` em `pyproject.toml`.

### Deps core (amostra)

openai, httpx, rich, pydantic, prompt_toolkit, fastapi, uvicorn, croniter, psutil, websockets, pyyaml, sqlite (stdlib + FTS5 CJK native opcional)

---

## JavaScript / TypeScript

| Item | Valor |
|---|---|
| Gerenciador | npm workspaces (root `package.json`) |
| Node | `>=20.0.0` (desktop: `^20.19.0 \|\| >=22.12.0`) |

### Workspaces

| Pacote | Stack |
|---|---|
| `ui-tui` | Ink 6, React 19, nanostores, vitest, tsx |
| `apps/desktop` | Electron, Vite, React, `@assistant-ui/react`, Playwright |
| `apps/shared` | JSON-RPC/WS framework-agnostic |
| `web` | Vite, React 19, Tailwind 4, xterm.js |
| `website` | Docusaurus |
| `tests-js` | Vitest cross-workspace |

### Comandos root

```powershell
npm install
npm run check    # propaga lint/typecheck/test workspaces
npm run fix      # autofix JS
```

---

## Infra / runtime

| Componente | Uso |
|---|---|
| SQLite | Sessões FTS5 (`hermes_state.py`) |
| Docker | Terminal backend, CI, deploy |
| Nix | Empacotamento alternativo (`nix/`) |
| systemd | Kanban dispatcher standalone |
| MinGit (Windows) | Git Bash bundled em `%LOCALAPPDATA%\hermes\git` |

---

## CI matrix (`.github/workflows/`)

| Workflow | Gatilho / função |
|---|---|
| `ci.yml` | Orquestrador + classificador de mudanças |
| `tests.yml` | Python 8 slices paralelos |
| `lint.yml` | Ruff / ty |
| `js-tests.yml` | Vitest workspaces |
| `e2e-desktop.yml` | Playwright desktop |
| `docker.yml` | Build imagem |
| `supply-chain-audit.yml`, `osv-scanner.yml`, `uv-lockfile-check.yml` | Segurança deps |
| `docs-site-checks.yml`, `deploy-site.yml` | Docusaurus |

---

## Ambiente de dev local (Windows)

| Recurso | Path / comando |
|---|---|
| Checkout | `%LOCALAPPDATA%\hermes\hermes-agent\` |
| venv | `.venv` ou `venv` (probe em `run_tests.sh`) |
| HERMES_HOME runtime | `%LOCALAPPDATA%\hermes\` |
| Testes | `bash scripts/run_tests.sh [path]` via Git Bash / MinGit (não PowerShell direto) |
| Ativar venv | `.venv\Scripts\Activate.ps1` |

**Limitação confirmada:** PTY dashboard (`/api/pty`) não funciona em Windows nativo — WSL ou desktop app.

---

## Dependências do harness (Cursor)

Skills/MCP disponíveis na sessão que complementam operação:

- `harness-architect`, `mcp-health-check`, `code-review-checklist`
- MCP: Serena, Context7, Supabase, Telegram, Chrome DevTools
- Subagentes: explore, shell, code-reviewer, ci-investigator
