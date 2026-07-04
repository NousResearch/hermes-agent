# Hermes Viewport Computer — Capabilities

## Runtime

- **Conductor dispatcher** — deterministic `SchemeDispatcher` in `hermes_cli/conductor.py`
- **Private client runtime** — bounded sovereign subagent surface (`pc://`)
- **FastAPI bridge** — backend daemon on port `7860`
- **Standalone viewport** — HTML shell on port `5173`

## Schemes

- `c://cc <target>` — change execution context
- `pc://` / `pc://run <name>` — private client address / runtime
- `mcp://` / `mcp://tools` / `mcp://invoke` — tools surface via MCP
- `vscode://` — editor + remote compute control plane
- `reachy://` — robot/operator surface
- `H://` — global agentic domain for Hermes-agent
- `hermes://` — default hermes-agent runtime
- `NOUS://` — Nous Research provider/runtime
- `llc://` — CLI LLC business surface
- `daollc://` — governance/identity
- `+æ://` — intent/permission token surface
- `+æ://media^ffmpeg` — deterministic media/graphics primitive

## Bridge Endpoints

- `POST /conductor` — scheme dispatch
- `GET /mcp/tools` — tool catalog
- `POST /mcp/invoke` — bounded tool execution
- `GET /vscode/status` — control plane health
- `POST /vscode/open` — open local path
- `POST /vscode/uri` — open native URI

## Test Coverage

- `tests/hermes_cli/test_conductor.py` — dispatcher policy / overlap / run_pc
- `tests/hermes_cli` targeted run — 506 passed, 2 windows-only path-completion outliers
