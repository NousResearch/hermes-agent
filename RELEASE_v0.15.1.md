# Hermes Agent v0.15.1 (v2026.5.29)

**Release Date:** May 29, 2026
**Since v0.15.0:** 28 commits · 21 merged PRs · hotfix release · 9 contributors

> **The Patch Release.** A same-day hotfix for v0.15.0. Headline fix: the dashboard infinite-reload loop that hit anyone running v0.15.0 in loopback mode (Docker, hosted Hermes, fresh installs). A handful of other v0.15.0 follow-ups go along for the ride — kanban worker SIGTERM, `/model` picker unification, `/yolo` session bypass, the full 19,932-entry skills.sh catalog, `.md` media delivery restoration, gateway probe-stepdown safety, web-URL redaction passthrough, kanban worker vision on referenced images, hindsight observation-default. Docker users get an explicit `--insecure` opt-in env var (no more bind-host inference), MCP server bare-command PATH resolution, and arm64 PR-build cache fixes.

---

## ✨ Highlights

- **Dashboard 401 reload loop fixed** — In loopback mode the dashboard's identity probe (`/api/auth/me`) returns 401 by design, but v0.15.0's stale-token reload guard treated every 401 as a rotated session token and full-page-reloaded to pick up a fresh one. Every successful sibling call cleared the one-shot reload guard, so the page reload-looped forever (Firefox: "Navigated to /sessions" storm; Chrome: React re-render storm). Fix adds an `allowUnauthorized` opt-out to `fetchJSON` that skips only the loopback stale-token reload — 401 still throws so `AuthWidget` swallows it, gated-mode `login_url` redirects are unaffected. Closes [#34206](https://github.com/NousResearch/hermes-agent/issues/34206), [#34202](https://github.com/NousResearch/hermes-agent/issues/34202). ([#30698](https://github.com/NousResearch/hermes-agent/pull/30698) — @austinpickett)

- **Docker dashboard `--insecure` is now an explicit env opt-in, never derived from bind host** — Previously the Docker entrypoint inferred `--insecure` when the dashboard bound to a non-loopback host. That conflated "I want LAN access" with "I want to disable the same-origin guard." The fix splits them: bind host is bind host, and disabling the dashboard's loopback auth requires an explicit `HERMES_DASHBOARD_INSECURE=1`. Existing setups that genuinely wanted insecure binding must now set the env var. ([#34188](https://github.com/NousResearch/hermes-agent/pull/34188), [#34204](https://github.com/NousResearch/hermes-agent/pull/34204) — @benbarclay)

- **MCP bare command resolution under Docker** — MCP servers configured with bare commands (`npx`, `npm`, `node`) now resolve against `/usr/local/bin` so they actually launch inside the Docker image where those binaries live. v0.15.0 left these failing silently in containers when the agent's effective PATH didn't include the Node toolchain location. ([#34186](https://github.com/NousResearch/hermes-agent/pull/34186) — @benbarclay)

- **Skills page sidebar / source pills restored** — A stale `useMemo` dependency in the new dashboard skills page collapsed the source pills and category sidebar to "All" only. Fixed; both surfaces now reflect the live catalog state. ([#34194](https://github.com/NousResearch/hermes-agent/pull/34194))

- **Kanban worker can be killed again** — `SIGTERM` on a kanban worker was being absorbed by an intermediate process and the worker stayed running. Closes [#28181](https://github.com/NousResearch/hermes-agent/issues/28181). ([#34045](https://github.com/NousResearch/hermes-agent/pull/34045))

- **Full skills.sh catalog (858 → 19,932 entries)** — The skills hub page was pulling a partial paginated catalog. The fetch now walks the sitemap, so all 19,932 skills.sh entries surface in the picker instead of just the first 858. ([#34025](https://github.com/NousResearch/hermes-agent/pull/34025))

---

## 🐛 Bug Fixes

### Dashboard / Web

- **`/api/auth/me` 401 no longer triggers reload loop** in loopback mode — ([#30698](https://github.com/NousResearch/hermes-agent/pull/30698) — @austinpickett)
- **Skills page source pills + category sidebar restored** — stale `useMemo` dep ([#34194](https://github.com/NousResearch/hermes-agent/pull/34194))

### Docker

- **`--insecure` is now explicit opt-in via env var**, not derived from bind host ([#34188](https://github.com/NousResearch/hermes-agent/pull/34188) — @benbarclay)
- **Dashboard test suite repaired** to match the insecure-opt-in fix ([#34204](https://github.com/NousResearch/hermes-agent/pull/34204) — @benbarclay)
- **arm64 PR builds skip the GHA cache** to avoid cache-thrash on cross-arch builders ([#33704](https://github.com/NousResearch/hermes-agent/pull/33704) — @BROCCOLO1D)

### MCP

- **Bare `npx`/`npm`/`node` resolve against `/usr/local/bin`** for Docker compatibility ([#34186](https://github.com/NousResearch/hermes-agent/pull/34186) — @benbarclay)

### Kanban

- **Worker SIGTERM actually terminates the process** ([#34045](https://github.com/NousResearch/hermes-agent/pull/34045))
- **Workers receive images referenced in task bodies** for vision-capable models ([#34210](https://github.com/NousResearch/hermes-agent/pull/34210))

### Gateway

- **`.md` files deliver again** — media-delivery validation defaults to denylist-only instead of an overly-narrow allowlist ([#34022](https://github.com/NousResearch/hermes-agent/pull/34022))
- **Probe stepdown safety** — on a context-overflow without an explicit provider context limit, the agent no longer steps down to a smaller model based on an unknown ceiling (salvage of [#33673](https://github.com/NousResearch/hermes-agent/pull/33673)) ([#33826](https://github.com/NousResearch/hermes-agent/pull/33826))

### CLI

- **`/yolo` mid-session enables the per-session bypass** instead of just toggling the env var (which the running agent had already snapshotted) ([#33931](https://github.com/NousResearch/hermes-agent/pull/33931) — @kshitijk4poor)
- **`/model` and `hermes model` show the same list**, plus disk cache for picker startup ([#33867](https://github.com/NousResearch/hermes-agent/pull/33867))

### Skills

- **Full skills.sh catalog via sitemap** — 858 → 19,932 entries ([#34025](https://github.com/NousResearch/hermes-agent/pull/34025))

### Redaction

- **Web URLs pass through unchanged** — the redactor was eating query parameters that looked credential-shaped ([#34029](https://github.com/NousResearch/hermes-agent/pull/34029))

---

## ✨ Small Features

- **Hindsight default narrowed to observation-only** for `recall_types` — tool path is also narrowed ([#34079](https://github.com/NousResearch/hermes-agent/pull/34079) — @nicoloboschi, follow-up [#34091](https://github.com/NousResearch/hermes-agent/pull/4df62d239e38bf8c212a595721c9c01e176f6c3a) — @kshitijk4poor)
- **Memory providers receive completed-turn message context** — salvage of [#28065](https://github.com/NousResearch/hermes-agent/pull/28065) ([#34097](https://github.com/NousResearch/hermes-agent/pull/34097) — @kshitijk4poor, credit to @devwdave)

---

## 📚 Documentation

- **`--no-supervise` / `HERMES_GATEWAY_NO_SUPERVISE` documented** in the reference docs (follow-up to [#33583](https://github.com/NousResearch/hermes-agent/pull/33583)) ([#33751](https://github.com/NousResearch/hermes-agent/pull/33751) — @r266-tech)

---

## 🛠️ Infrastructure

- **Vercel deploy workflow accepts `workflow_dispatch`** so docs deploys can be manually triggered ([#34081](https://github.com/NousResearch/hermes-agent/pull/34081))
- **`@nous-research/ui` bumped to 0.18.2** (Nix `npmDepsHash` also updated to match) ([#34193](https://github.com/NousResearch/hermes-agent/pull/34193) follow-ups — @austinpickett)

---

## 👥 Contributors

### Core
- @teknium1

### Community
- @austinpickett — dashboard 401 reload-loop fix (the headline), `@nous-research/ui` bump, Nix `npmDepsHash` updates
- @benbarclay — Docker `--insecure` opt-in, MCP bare-command resolution, dashboard test repair
- @kshitijk4poor — `/yolo` session bypass, completed-turn memory context salvage, hindsight follow-up docs
- @nicoloboschi — hindsight `recall_types` observation default
- @BROCCOLO1D — arm64 PR build cache fix
- @r266-tech — `--no-supervise` reference docs
- @yangguangjin — probe stepdown safety (salvage of @yanghd's #33673)
- @devwdave — completed-turn memory context (credited via salvage)
- @andrewhosf — co-author

### Issue Reporters (the 401 loop)
- @routesmith ([#34206](https://github.com/NousResearch/hermes-agent/issues/34206))
- @beeaton ([#34202](https://github.com/NousResearch/hermes-agent/issues/34202))

---

**Full Changelog**: [v2026.5.28...v2026.5.29](https://github.com/NousResearch/hermes-agent/compare/v2026.5.28...v2026.5.29)
