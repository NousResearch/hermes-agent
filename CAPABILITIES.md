# Hermes Viewport Computer — Capabilities

## Runtime

- **Conductor dispatcher** — deterministic `SchemeDispatcher` in `hermes_cli/conductor.py`
- **Private client runtime** — bounded sovereign subagent surface (`pc://`)
- **FastAPI bridge** — backend daemon on port `7860`
- **Standalone viewport** — HTML shell on port `5173`

## Omniverse Surfaces

- `templates/omniverse.html` — NVIDIA Omniverse viewport shell
  - Terminal prompt surfaces as `h://` globally
  - Tab order: terminal, omniverse, monaco, media, entrepreneur
- `omniverse://` — NVIDIA Omniverse mesh surface
- `+æ://identity` — bounded mesh identity (`kind: bounded_private_client_mesh`)
- `pc://mesh/global/<id>` / `pc://mesh/user/<user_id>/<id>` — per-user/global QR-addressable nodes
- `templates/æ.html` — Omniverse-ready sovereign viewport shell
  - Terminal prompt surfaces as `h://` globally
  - Tab order: terminal, monaco, media, entrepreneur, stack, business, apis
  - Monaco via `HermesSurface.dispatch/hydrate`
- `apps/reachy/hermes-monaco.js` — standalone Monaco editor surface with CDN loader
- `h://` / `H://` — global agentic domain, viewport cursor/CLI command prompt

## Media Desktop

- `+æ://media^ffmpeg` — deterministic media/graphics primitive
- Runtime contract: bounded deterministic loop
- Stack: Planner/Auditor + FFmpeg executor + manifest + local HTML/CSS surface
- Supports multimodal media as sovereign private client capability

## Entrepreneur Surface

- `#startabusiness` — formation-ready agent surface
- Doola-backed revenue primitive
- Entrepreneuring as first-class bounded local-first private client surface

## Scheme Registry

- `c://cc <target>` — change execution context
- `H:// <tail>` — global agentic domain
- `pc://` / `pc://run <name>` / `pc://mesh/...` — private client address/runtime/mesh
- `mcp://` / `mcp://tools` / `mcp://invoke` — tools surface via MCP
- `vscode://` — viewport host: VS Code is the runtime surface for the local HTML/CSS/WASM viewport (the `v` in vscode = viewport, the mandate — not Visual Studio). Also reachable as `viewport://` and `hermes viewport open`.
- `reachy://` — robot/operator surface
- `NOUS://` — Nous Research provider/runtime
- `llc://` / `daollc://` — business/governance/identity
- `intent://` / `+æ://` — intent/permission/supremacy surface
- `+æ://media^ffmpeg` — media primitive
- `hermes://` — default runtime
- `h://` / `h://æ^hub` — global cursor/CLI command prompt in HTML viewports

## Bridge Endpoints

- `POST /conductor` — scheme dispatch + QR mesh
- `GET /mcp/tools` — tool catalog
- `POST /mcp/invoke` — bounded tool execution
- `GET /vscode/status` — control plane health
- `POST /vscode/open` — open local path
- `POST /vscode/uri` — open native URI

## Scalar Supremacy

- `_enforce_promote_scalar_supremacy()` in `conductor.py`
- Blocks bare scalar authority tails before dispatchers mutate state
- Enforces: most scalable formal implementation supersedes less scalable alternatives

## Skill

- `skills/æ/SKILL.md` — installable sovereign private client mesh primitive
- Reference docs under `skills/æ/references/`

## Test Coverage

- `tests/hermes_cli/test_conductor.py` — dispatcher policy / overlap / run_pc / scalar supremacy
- targeted run — 11 passed in targeted conductor verification
