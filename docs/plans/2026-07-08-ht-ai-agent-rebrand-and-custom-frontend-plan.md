# HT AI Agent — Rebrand & Custom Frontend Plan

**Status:** Proposed
**Date:** 2026-07-08
**Scope:** Rebrand the user-visible product identity from "Hermes Agent" (Nous Research) to **HT AI Agent**, and build a **new custom web frontend** that talks to the existing agent gateway.

---

## 1. Goals & Non-Goals

### Goals

1. Every user-visible surface (agent self-identification, CLI banner, chat platforms, docs, installers, app names) says **HT AI Agent**.
2. A new, custom-built frontend — owned by this fork, visually independent of the Nous design system — becomes the primary way users chat with the agent.
3. The fork stops pointing at Nous Research infrastructure (update checks, docs URLs, install one-liners).
4. Legal obligations of the MIT license and embedded third-party licenses are met.

### Non-Goals (deliberately deferred — see §8 Phase 6)

- Renaming internal identifiers: `HERMES_*` env vars (329 of them are inter-process protocol), `X-Hermes-*` HTTP headers, Python module names (`hermes_cli`, `hermes_state`, …). Renaming these breaks running installs and third-party API clients for zero user-visible gain. They stay as internal legacy identifiers until a major-version migration.
- Migrating the `~/.hermes` state directory. `get_hermes_home()` in `hermes_constants.py` is the single seam if we ever do; a rename without a data migration orphans user sessions, skills, and memory.
- Feature parity with the existing 19-page management dashboard. The new frontend is chat-first; the existing dashboard remains available at `hermes dashboard` until superseded.

---

## 2. Architecture Facts This Plan Relies On

These were verified against the codebase and are the reason this plan is feasible:

| Fact | Where |
|---|---|
| All four existing frontends (Ink TUI, Electron desktop, web dashboard, iOS) drive the same JSON-RPC 2.0 dispatcher over stdio or WebSocket | `tui_gateway/server.py` (~117 RPC methods), `tui_gateway/transport.py`, `tui_gateway/ws.py` |
| A headless gateway is a first-class CLI mode | `hermes serve --host 127.0.0.1 --port <p>` (`hermes_cli/main.py`) |
| A reusable TypeScript protocol client already exists and is consumed by both the desktop app and web dashboard | `apps/shared/src/json-rpc-gateway.ts` (`JsonRpcGatewayClient`), `websocket-url.ts` |
| The de facto protocol spec is the hand-written type mirror | `ui-tui/src/gatewayTypes.ts` |
| Runtime branding (name, colors, banner art, prompt symbol, welcome/goodbye) is a data-driven skin system pushed to every frontend on connect via `gateway.ready` | `hermes_cli/skin_engine.py` (presets already include "Ares Agent" — the rename path is proven) |
| Agent self-identity has a supported per-install override | `~/.hermes/SOUL.md`; defaults in `agent/prompt_builder.py:126` (`DEFAULT_AGENT_IDENTITY`) |
| An OpenAI-compatible HTTP API, MCP server, and ACP adapter exist as alternative frontends | `gateway/platforms/api_server.py`, `mcp_serve.py`, `acp_adapter/` |

---

## 3. Naming Decisions

| Item | Decision | Rationale |
|---|---|---|
| Product name | **HT AI Agent** | Per request |
| Short display name | **HT** | Sidebar wordmarks, tab titles, response label |
| Response label / prompt | ` HT ` label, keep `❯` prompt symbol | Via skin `branding` dict |
| Brand icon | Pick a replacement glyph/logo for `⚕` (caduceus) | The caduceus is Hermes trade dress; must go. TUI `fromSkin()` currently pins the icon — needs a 1-line code change (Phase 2) |
| CLI command | Add `ht` console script alongside `hermes`; user-facing text says `ht …` | Zero-breakage alias; `hermes` deprecated later |
| New frontend package | `apps/ht-web/` (npm workspace) | Sits beside `apps/desktop`, reuses `apps/shared` |
| Config dir | Keep `~/.hermes` for now (see Non-Goals) | Defer migration |
| PyPI / Docker / repo name | `ht-ai-agent` (when/if published) | Upstream owns `hermes-agent` on PyPI and `nousresearch/hermes-agent` on Docker Hub |

---

## 4. Legal Checklist (do these in Phase 2, they are cheap)

- [ ] `LICENSE`: **retain** `Copyright (c) 2025 Nous Research`; append our own copyright line. MIT requires the notice survive in all copies.
- [ ] Preserve embedded third-party licenses untouched: `plugins/security-guidance/LICENSE` (Apache-2.0, Anthropic), `skills/productivity/powerpoint/LICENSE.txt`, `skills/creative/humanizer/LICENSE`, `plugins/hermes-achievements/LICENSE`, and the Apache-2.0 header in `tests/agent/test_restore_primary_pool_reselect.py`.
- [ ] Replace all trademark material (MIT grants no trademark rights): "Hermes"/"Nous Research" names, caduceus mark (`⚕`/`☤`), `assets/banner.png` (has "HERMES-AGENT" in the pixels), `website/static/img/logo.png` + `nous-logo.png`, `apps/*/public/nous-girl.jpg`, desktop/installer icon sets, and the explicit `"legalTrademarks": "Hermes"` in `apps/desktop/package.json`.
- [ ] `SECURITY.md`: replace `security@nousresearch.com` with our own contact.

---

## 5. Phase 1 — Config-Layer Rebrand (hours, no code)

Quick wins that make the running product say "HT AI Agent" immediately.

1. **Skin**: add builtin preset `ht` in `hermes_cli/skin_engine.py` (or ship `~/.hermes/skins/ht.yaml`):
   - `branding.agent_name: "HT AI Agent"`, `welcome`, `goodbye`, `response_label: " HT "`, `help_header`
   - `banner_logo` / `banner_hero`: new "HT-AI-AGENT" Rich-markup ASCII art (model on the existing `ares` preset)
   - New color palette (pick a non-gold identity)
   - Make `ht` the default skin.
2. **Identity**: rewrite `docker/SOUL.md` and the default soul seed to "You are HT AI Agent…" (full string sweep is Phase 2; this covers fresh Docker installs).
3. **Dashboard theme**: add an "HT" preset via the existing server-side theme system (`web/src/themes/presets.ts` + `_BUILTIN_DASHBOARD_THEMES` in `hermes_cli/web_server.py` — names must stay in sync).

**Acceptance:** fresh `hermes` CLI session shows HT banner/name; agent introduces itself as HT AI Agent in Docker; dashboard offers HT theme.

---

## 6. Phase 2 — Code-Level Rebrand of User-Visible Strings (~2–4 days)

The skin engine doesn't reach everything. File-by-file checklist, grouped by surface:

### 6.1 System-prompt identity (highest priority — the agent says this out loud)

| File | Change |
|---|---|
| `agent/prompt_builder.py:126` | `DEFAULT_AGENT_IDENTITY` → "You are HT AI Agent…" |
| `agent/prompt_builder.py:136` | `HERMES_AGENT_HELP_GUIDANCE` → our docs URL; note it references `skill_view(name='hermes-agent')` — rename the skill dir or keep the internal name and reword |
| `agent/prompt_builder.py:737, 840, 1137` | "Hermes terminal UI / WebUI / desktop GUI" surface hints |
| `hermes_cli/default_soul.py:4` | Duplicate identity text |
| `docker/SOUL.md:1` | Duplicate (done in Phase 1) |
| `scripts/install.sh:1802`, `scripts/install.ps1:2230` | SOUL.md heredocs written on install |
| `agent/system_prompt.py:364,373` | "Active Hermes profile" strings |
| `hermes_cli/doctor.py:1192` | Doctor test prompt |

### 6.2 TUI (`ui-tui/`)

| File | Change |
|---|---|
| `ui-tui/src/theme.ts` | `BRAND` object (name, goodbye), default palettes; **unpin the icon in `fromSkin()`** so skins can override it |
| `ui-tui/src/banner.ts` | Replace `LOGO_ART` / `CADUCEUS_ART` defaults |
| `ui-tui/src/components/branding.tsx` | Remove hardcoded "Nous Research · Messenger of the Digital Gods" taglines and "· Nous Research" model suffix; `hermes update` hint → `ht update` |

### 6.3 Web dashboard (`web/`) — kept as the admin surface

| File | Change |
|---|---|
| `web/index.html`, `web/public/favicon.ico` | Title + favicon |
| `web/src/App.tsx` (~line 578) | Sidebar wordmark |
| `web/src/i18n/*.ts` (18 locales) | `app.brand`, `brandShort`, `footer.org` + Hermes-named strings (mechanical) |
| `web/src/components/SidebarFooter.tsx`, `pages/DocsPage.tsx`, `pages/SystemPage.tsx` | Nous URLs |
| `web/src/index.css`, `web/src/themes/presets.ts` | Default palette + theme names |

### 6.4 Chat platforms & programmatic surfaces

| File | Change |
|---|---|
| `locales/*.yaml` (16 languages × ~13 strings) | "Hermes Commands", "Hermes Gateway Status", `hermes update`/`hermes gateway restart` hints → `ht …` |
| `plugins/platforms/slack/adapter.py:1159,3907` | `/hermes` slash command → `/ht` (+ `hermes_cli/commands.py:1262` manifest generator) |
| `plugins/platforms/discord/adapter.py:4074-4136` | Command descriptions |
| `hermes_cli/telegram_managed_bot.py` | Default bot name, username pattern, setup URL |
| `gateway/run.py:8896,11282` | Pairing/sethome messages |
| `gateway/platforms/api_server.py:1422` | `/v1/models` model id → `ht-ai-agent` |
| `mcp_serve.py:551-557`, `agent/transports/hermes_tools_mcp_server.py:126` | MCP server display names (caution: `hermes-tools` is written into users' `~/.codex/config.toml` — keep key, change display) |
| `acp_registry/agent.json` + `icon.svg`, `acp_adapter/server.py:462,503,886` | ACP editor-registry identity + icon |
| `plugins/kanban/` (dashboard title, systemd unit), `plugins/google_meet/meet_bot.py:452` (guest display name), `plugins/hermes-achievements/` | Misc user-visible plugin branding |
| `cron/scheduler.py:736` | Thread name prefix |

### 6.5 CLI display & distribution

| File | Change |
|---|---|
| `cli.py:3487,13073` + 6 skin-fallback call sites | "Hermes Agent" fallback literals → read from skin, fallback "HT AI Agent" |
| `hermes_cli/main.py` | ~130 `hermes <subcommand>` help/error strings → `ht …`; `OFFICIAL_REPO_URL` + git-remote allowlist (line ~6635) → our repo |
| `hermes_cli/banner.py:123,265,460` | **Update check** → our PyPI name / GitHub releases (until then, disable — otherwise `ht update` installs upstream Hermes) |
| `pyproject.toml` | Add `ht` console script; project name/authors when publishing |
| User-Agent strings | `run_agent.py:287`, `hermes_cli/auth.py:108`, `model_catalog.py:80`, `tools/xai_http.py:72`, `tools/discord_tool.py:103`, `providers/base.py:24`, + plugin adapters → `HTAIAgent/{ver}` |
| `apps/desktop/` (if we keep shipping it) | `package.json` build block (productName, appId, protocol scheme, legalTrademarks), icons, `nous-girl.jpg`/mascot sprites, `index.html`, i18n (~90 strings × 4 locales), `electron/update-remote.cjs` pinned repo |
| `apps/bootstrap-installer/` | `tauri.conf.json`, `Cargo.toml`, icons, manifest, copy |

### 6.6 Docs & repo shell

| File | Change |
|---|---|
| `README.md` | Rewrite for HT AI Agent; delete `README.es/zh-CN/ur-pk.md`, `CONTRIBUTING.es.md`, `SECURITY.es.md` (regenerate later if wanted) |
| `assets/banner.png` | New artwork |
| `website/docusaurus.config.ts` + `static/img/*` + docs prose | Site identity; delete `website/i18n/zh-Hans/` |
| `scripts/install.sh|ps1|cmd`, `setup-hermes.sh` | Repo URLs, echo strings; host our own install one-liner |
| `docker-compose*.yml`, `.github/workflows/docker.yml`, `upload_to_pypi.yml` | Image/package names (publish workflows self-disable in forks via `github.repository ==` guards — safe meanwhile) |
| `.github/ISSUE_TEMPLATE/*`, `PULL_REQUEST_TEMPLATE.md` | Links, "Hermes Version" labels |
| `packaging/homebrew/`, `flake.nix`, `nix/hermes-agent.nix` | Rename or drop channels we won't use |
| `skills/autonomous-ai-agents/hermes-agent/SKILL.md` (the agent's self-manual, 242 refs) + `optional-skills/` self-references | Rewrite name/URLs; keep internal skill id if renaming breaks the prompt pointer |
| `infographic/*/infographic.png`, `hermes-already-has-routines.md` | Delete (marketing artifacts) |

### 6.7 Tests

~1,386 Hermes-mentioning assertion lines across 352 test files. Update the ones asserting **display strings** (`tests/agent/test_prompt_builder.py`, `tests/skills/test_openclaw_migration.py:972`, `tests/acp/test_registry_manifest.py`, `tests/hermes_cli/test_banner.py`). **Caution:** `tests/hermes_cli/test_nous_hermes_non_agentic.py` and similar refer to the *Nous Hermes LLM model family* — those are model names, not product branding; do not rename.

**Acceptance for Phase 2:** `grep -ri "hermes\|nous" --include="*.yaml" locales/ web/src/i18n/` returns no user-visible brand strings; agent self-identifies as HT AI Agent on all platforms; full test suite green.

---

## 7. Phase 3 — New Custom Frontend (`apps/ht-web/`) (~2–3 weeks to MVP)

### 7.1 Architecture

```
┌─────────────────────────┐         WebSocket /api/ws          ┌──────────────────────┐
│  apps/ht-web/           │  JSON-RPC 2.0 (newline-framed)     │  ht serve            │
│  React 19 + Vite + TS   │ ◄────────────────────────────────► │  (tui_gateway)       │
│  Tailwind 4             │   requests → responses             │  ~117 RPC methods    │
│  reuses @hermes/shared  │   + async event frames             │  + agent core        │
└─────────────────────────┘                                    └──────────────────────┘
```

- **Own design system.** No `@nous-research/ui` dependency — plain Tailwind + a small component kit we own. This is what makes the frontend visually independent (the existing web/desktop UIs inherit the Nous look from that package).
- **Protocol client:** reuse `@hermes/shared` (`JsonRpcGatewayClient`, `buildHermesWebSocketUrl`) — framework-agnostic, no runtime deps, already battle-tested by two frontends.
- **Backend:** `ht serve --host 127.0.0.1 --port <p>`; auth via the injected session token / WS ticket mechanism `web_server.py` already implements.
- **Rendering:** render markdown client-side (`react-markdown` + `shiki`, as the desktop app does). Ignore the gateway's pre-rendered, terminal-column-aware Rich-markup fields — request/handle raw text.

### 7.2 Protocol surface for MVP (from `ui-tui/src/gatewayTypes.ts`)

| Direction | Name | Purpose |
|---|---|---|
| call | `session.create` / `session.resume` / `session.list` | Session lifecycle |
| call | `prompt.submit` | Send user message |
| call | `session.interrupt` | Stop generation |
| call | `approval.respond`, `clarify.respond` | Tool-approval / clarification answers |
| event | `gateway.ready` | Connection handshake — carries the **skin payload** (brand name, colors) so the frontend needs no hardcoded branding |
| event | `message.delta` / `message.complete` | Streaming assistant output |
| event | `tool.start` / `tool.*` | Tool-activity feed |
| event | `approval.request` | Render approval prompt |
| event | `status.update`, `skin.changed` | Header status, live re-theme |

**First deliverable of this phase:** a written protocol spec (`docs/ht-web-gateway-protocol.md`) extracted from `gatewayTypes.ts` + `createGatewayEventHandler.ts`, so the frontend isn't built against guesswork. The protocol is currently undocumented; this document becomes the contract test source.

### 7.3 Milestones

| Milestone | Deliverable | Acceptance |
|---|---|---|
| **M1 — Scaffold + connect** | `apps/ht-web` workspace; Vite dev server; connects to `ht serve`, completes `gateway.ready`, applies skin | Handshake renders brand name/colors from skin payload |
| **M2 — Chat MVP** | Streaming chat: prompt box, `message.delta` rendering, markdown + code blocks, interrupt button | Multi-turn conversation with live streaming; interrupt works |
| **M3 — Tool activity + approvals** | Collapsible tool-call feed; approval/clarify modals wired to `approval.respond` | An approval-gated tool run round-trips from UI |
| **M4 — Sessions** | Sidebar: list/resume/create/delete sessions; session titles | Restart browser, resume prior conversation |
| **M5 — Polish + ship** | Slash-command palette, model/status header, light/dark, error/reconnect handling; production build served by `web_server.py` (new route or replacing `web_dist`); `ht gui` launch command | Lighthouse pass; vitest unit tests + Playwright e2e against a live gateway in CI |

### 7.4 Explicitly out of MVP scope

Management pages (models, cron, skills, plugins, MCP, channels — the existing dashboard keeps serving these), voice, the floating pet, billing overlays (Nous Portal-specific), and the PTY-embedded terminal.

---

## 8. Later Phases (optional, priced separately)

- **Phase 4 — Frontend expansion:** port management pages into `ht-web` against the existing ~80 REST endpoints in `web/src/lib/api.ts`; retire the old dashboard.
- **Phase 5 — Distribution identity:** publish `ht-ai-agent` to PyPI, our own Docker Hub namespace, hosted install scripts + docs domain, desktop app re-signing (new bundle IDs mean macOS TCC/mic permissions reset — needs a migration note).
- **Phase 6 — Deep rebrand (backward-compatible):** `HERMES_*` env vars, `~/.hermes` home (needs migration tooling), Python module renames, `X-Hermes-*` headers. ~12,300 occurrences; multi-week. Rather than a breaking big-bang rename, this is done as an **additive backward-compatible migration** (new `HT_*` / `X-HT-*` names with the legacy names kept working). **In progress:** the alias foundation (`ht_compat.py` — env mirroring, `HT_HOME`, header dual-emit/read), entrypoint hardening, and the backward-compatible `~/.hermes → ~/.ht-ai-agent` data-dir migration (atomic move + back-symlink bridge) have landed; the Python module-name shims are the remaining deferred increment. See [`docs/phase-6-internal-identifiers.md`](../phase-6-internal-identifiers.md).
- **Nous Portal decision:** `hermes_cli/auth.py`, `nous_account/subscription/billing.py`, `portal_cli.py` (~1,700 refs) implement Nous Portal login/credits/billing — that's a *provider integration*, not branding. Option A: keep it as one provider among many (it works today, users just don't have to use it). Option B: rip it out and its TUI/dashboard billing surfaces. **Recommendation: Option A for now** — removal is entangled and reversible later.

---

## 9. Risks & Gotchas

1. **Update channel poisoning:** until `banner.py` / `update-remote.cjs` are repointed, `hermes update` reinstalls upstream Hermes over the fork. Do this in the first Phase 2 commit.
2. **Locale drift:** missing i18n keys render raw — every brand-string edit must cover all 16 `locales/*.yaml` and 18 `web/src/i18n/*.ts` files in the same commit.
3. **"Nous Hermes" false positives:** grep-driven renames will hit the Nous Hermes *model family* names in provider/test code — those are external model IDs and must not change.
4. **Skin icon pinning:** `fromSkin()` in `ui-tui/src/theme.ts` pins the `⚕` icon; the caduceus survives a skin-only rebrand until the Phase 2 one-liner lands.
5. **Slack command rename** (`/hermes` → `/ht`) requires users to update their Slack app manifests — release note needed.
6. **Codex config key:** `hermes-tools` MCP server key is written into users' `~/.codex/config.toml`; renaming the key breaks existing configs — change display name only.
7. **Protocol is undocumented:** M1's spec-extraction step de-risks the frontend build; contract tests against `gatewayTypes.ts` keep it honest.

---

## 10. Effort Summary

| Phase | Effort |
|---|---|
| 1 — Config-layer rebrand | Hours |
| 2 — Code-level rebrand | 2–4 days |
| 3 — Custom frontend MVP | 2–3 weeks |
| 4 — Frontend expansion | 3–6 weeks (optional) |
| 5 — Distribution identity | ~1 week + external accounts/domains (optional) |
| 6 — Deep internal rebrand | Multi-week (recommend deferring) |

Recommended execution order: **Phase 1 → Phase 2 (§6.1 + update URLs first) → Phase 3**, with Phases 4–6 re-scoped after the MVP ships.
