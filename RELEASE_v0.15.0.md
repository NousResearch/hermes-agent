# Hermes Agent v0.15.0 (v2026.5.28)

**Release Date:** May 28, 2026
**Since v0.14.0:** 1,302 commits · 747 merged PRs · 1,746 files changed · 282,712 insertions · 36,699 deletions · 560+ issues closed (15 P0, 65 P1, 19 security-tagged) · 321 community contributors (including co-authors)

> **The Velocity Release.** Hermes gets dramatically faster — to start, to run, to ship work, and to grow. The 16,083-line `run_agent.py` collapses to 3,821 (-76%) across 14 cohesive `agent/*` modules. Kanban grew into a real multi-agent platform across 104 PRs — orchestrator auto-decomposition, swarm topology, scheduled tasks, worktree-per-task, per-task model overrides. The cold-start perf wave keeps going: another second shaved off launch, 47% fewer per-conversation function calls, `hermes --version` flipping the head-to-head benchmark against Codex CLI. `session_search` is 4,500× faster and free now. Promptware defense lands against Brainworm-class attacks. Bitwarden Secrets Manager replaces N per-provider API keys with one bootstrap token. Skill bundles let one slash command load a whole workflow. The Ink TUI gets a multi-session orchestrator. Two new image_gen providers (Krea 2 Medium + Large, FAL ported to plugin), the Nous-approved MCP catalog with an interactive picker, an OpenHands orchestration skill, ntfy as the 23rd messaging platform, and a deep xAI integration round (Web Search plugin, xai-oauth `hermes proxy` upstream, retired-May-15 model detection + `hermes migrate xai`, natural TTS speech-tag pauses, base_url leak guard, OpenAI-style execution guidance for Grok). 15 P0 + 65 P1 closures alongside.

---

## ✨ Highlights

- **The Big Refactor — `run_agent.py` is no longer 16,000 lines** — The file at the heart of Hermes — the agent conversation loop — has been reduced from 16,083 lines to 3,821 (-76%), with the extracted code redistributed across 14 cohesive modules under `agent/`. Behavior is unchanged: every extraction keeps a thin forwarder on `AIAgent`, every test patch path still works, every external caller is compatible. The reason you care: future Hermes development moves faster, plugin authors can finally grep the codebase, and the file that took 90 seconds to load in your editor opens in a blink. ([#27248](https://github.com/NousResearch/hermes-agent/pull/27248))

- **Kanban grew into a real multi-agent platform — 104 PRs end to end** — Triage auto-decomposes one task into a tree of sub-tasks. `hermes kanban swarm` creates a full Swarm v1 graph in one command — root, parallel workers, gated verifier, gated synthesizer, shared blackboard. Tasks support per-task model overrides (cheap models for boilerplate, expensive ones for hard sub-tasks), board-level default workdirs, per-task worktree paths and branches, scheduled start times, configurable claim TTL, retry fingerprinting, stale-task detection, respawn guards, and a drag-to-delete trash zone. Workers report through `/workers/active`, `/runs/{id}`, and `/inspect` endpoints. ([#27572](https://github.com/NousResearch/hermes-agent/pull/27572), [#28443](https://github.com/NousResearch/hermes-agent/pull/28443), [#28364](https://github.com/NousResearch/hermes-agent/pull/28364), [#28394](https://github.com/NousResearch/hermes-agent/pull/28394), [#28462](https://github.com/NousResearch/hermes-agent/pull/28462), [#28384](https://github.com/NousResearch/hermes-agent/pull/28384), [#28467](https://github.com/NousResearch/hermes-agent/pull/28467), [#28455](https://github.com/NousResearch/hermes-agent/pull/28455), [#28452](https://github.com/NousResearch/hermes-agent/pull/28452), [#28432](https://github.com/NousResearch/hermes-agent/pull/28432), [#28468](https://github.com/NousResearch/hermes-agent/pull/28468), [#28420](https://github.com/NousResearch/hermes-agent/pull/28420))

- **Cold-start perf wave keeps going — another second saved, 47% fewer per-turn function calls** — Three new optimization rounds: defer `openai._base_client` import (-240ms / -17MB on every CLI invocation), hot-path optimizations cut 47% of per-conversation function calls (399k → 213k for 31-turn chat), defer compression-feasibility check (-170 to -290ms on every agent construction), adaptive subprocess polling (-195ms per tool call, 1+ second per turn). Termux cold start drops from 2.9s to 0.8s. `hermes --version` cold drops 63% (701ms → 258ms), flipping the head-to-head benchmark against Codex CLI from 5/11 wins to 6/11. ([#28864](https://github.com/NousResearch/hermes-agent/pull/28864), [#28866](https://github.com/NousResearch/hermes-agent/pull/28866), [#28957](https://github.com/NousResearch/hermes-agent/pull/28957), [#29006](https://github.com/NousResearch/hermes-agent/pull/29006), [#29419](https://github.com/NousResearch/hermes-agent/pull/29419), [#30121](https://github.com/NousResearch/hermes-agent/pull/30121), [#30609](https://github.com/NousResearch/hermes-agent/pull/30609), [#31968](https://github.com/NousResearch/hermes-agent/pull/31968))

- **`session_search` rebuilt — no LLM, no cost, 4,500× faster** — The old `session_search` was an aux-LLM-powered tool that cost ~$0.30/call and took ~30 seconds to summarize three sessions, sometimes confabulating when the right session wasn't even in the FTS5 hit list. The new shape is one tool with three modes (discovery, scroll, browse) inferred from which args are set — no `mode` parameter, no aux-LLM, no config knob, no companion skill. Discovery is ~20ms instead of ~90s; scroll is ~1ms. Searching your past sessions for context is now free and instant. ([#27590](https://github.com/NousResearch/hermes-agent/pull/27590))

- **Promptware defense — Brainworm-class attacks blocked at three chokepoints** — Inspired by recent Brainworm / Promptware Kill Chain research (Origin HQ, arxiv 2601.09625), Hermes now defends the context window against prompt-injection attacks that try to hijack the agent via tool output, recalled memory, or stored skills. Single source of truth (`tools/threat_patterns.py`) with ~15 new Brainworm/C2 patterns; recalled memory is scanned at load time; tool results get delimiter markers so a malicious file or remote service can't impersonate Hermes' own system content. Paired with a new `security-guidance` plugin that pattern-matches dangerous code writes. ([#32269](https://github.com/NousResearch/hermes-agent/pull/32269), [#33131](https://github.com/NousResearch/hermes-agent/pull/33131), [#9151](https://github.com/NousResearch/hermes-agent/pull/9151))

- **Bitwarden Secrets Manager — one bootstrap token replaces every per-provider API key** — Stop keeping plaintext API keys in `~/.hermes/.env`. Install Bitwarden Secrets Manager (`bws` auto-installs lazily on first use), point Hermes at it with one bootstrap token (`BWS_ACCESS_TOKEN`), and every credential you need comes from Bitwarden at startup. Rotate a key in the Bitwarden web app and the rotation actually takes effect — Bitwarden defaults to source-of-truth so its values overwrite matching env vars on startup. Flip `secrets.bitwarden.override_existing: false` to invert. EU Cloud and self-hosted Bitwarden server URLs supported. Detected credentials are now labeled with their source so you can see at a glance which keys came from Bitwarden vs. the local env. ([#30035](https://github.com/NousResearch/hermes-agent/pull/30035), [#31378](https://github.com/NousResearch/hermes-agent/pull/31378), [#30364](https://github.com/NousResearch/hermes-agent/pull/30364))

- **ntfy as the 23rd messaging platform — push notifications without an account** — ntfy is the self-hostable push-notification service with no signup, no API key, just a topic URL. Hermes now adapts to it as a platform plugin (zero edits to core), so your agent can send you push notifications from any cron job, kanban task completion, or chat `send_message` — to your phone, your watch, your desktop, your homelab. (salvages [#30625](https://github.com/NousResearch/hermes-agent/pull/30625) → originally [#4043](https://github.com/NousResearch/hermes-agent/pull/4043)) ([#30867](https://github.com/NousResearch/hermes-agent/pull/30867))

- **Skill bundles — `/<name>` loads multiple skills at once** — A skill bundle is a named group of skills that loads them all together with one slash command. Set up your "writing day" bundle (humanizer + ideation + obsidian + youtube-content) and `/writing-day` activates all four for the session. Skills Hub now has health checks, a freshness badge, and a watchdog cron. Three new optional skills land: `code-wiki` (Karpathy's LLM-Wiki, persistent indexed dev wiki), `openhands` (delegate to OpenHands for parallel coding agents), and `web-pentest` (OWASP-style web pentest recipes). ([#28373](https://github.com/NousResearch/hermes-agent/pull/28373), [#32345](https://github.com/NousResearch/hermes-agent/pull/32345), [#32240](https://github.com/NousResearch/hermes-agent/pull/32240), [#32261](https://github.com/NousResearch/hermes-agent/pull/32261), [#32265](https://github.com/NousResearch/hermes-agent/pull/32265))

- **TUI session orchestrator — multiple live sessions in one TUI window** — The Ink TUI gained an active-session switcher overlay. List, switch between, refresh, and close multiple live process-local sessions without leaving the TUI; dispatch a new session with a session-scoped model picker. Plus a wave of TUI polish — mouse-tracking DEC mode presets, scrollback preservation across branches and termux, slash-dropdown fixes, x.com link rendering, and CJK / IME input rendering improvements. (salvages [#27642](https://github.com/NousResearch/hermes-agent/pull/27642)) ([#32980](https://github.com/NousResearch/hermes-agent/pull/32980), [#30084](https://github.com/NousResearch/hermes-agent/pull/30084))

- **Two new image_gen providers — Krea 2 Medium + Large, FAL ported to plugin** — Krea joins the image_gen lineup as a built-in plugin: `Krea 2 Medium` ($0.03) and `Krea 2 Large` ($0.06), auto-discovered, selectable via `hermes tools` → Image Generation → Krea. Available through both the native Krea plugin and the FAL.ai catalog. The FAL.ai backend got pulled out of the monolithic image-generation tool into `plugins/image_gen/fal/`, completing the four-way architectural parity already established by web, browser, and video_gen — new image providers are now one file, not a fork. ([#33236](https://github.com/NousResearch/hermes-agent/pull/33236), [#30380](https://github.com/NousResearch/hermes-agent/pull/30380), [#33506](https://github.com/NousResearch/hermes-agent/pull/33506))

- **Nous-approved MCP catalog with interactive picker** — A curated catalog of Nous-vetted MCP servers, mirroring the optional-skills shape. Run `hermes mcp` and you get an interactive picker; install with one keystroke, credentials prompted at install time and written to `~/.hermes/.env`. Ships with the n8n manifest first. Closes the discovery gap that left users hunting GitHub for trusted MCP servers. ([#30870](https://github.com/NousResearch/hermes-agent/pull/30870))

- **OpenHands orchestration skill** — A new optional skill under `optional-skills/autonomous-ai-agents/openhands/` lets the agent delegate coding tasks to the OpenHands CLI alongside `claude-code`, `codex`, and `opencode`. OpenHands is the model-agnostic member of that family — any LiteLLM-supported provider works (OpenAI, Anthropic, OpenRouter, your own), so you can route a sub-task to the cheapest model that can finish it. Drop-in worker for kanban swarms and `/delegate` flows. (closes [#477](https://github.com/NousResearch/hermes-agent/issues/477)) ([#32261](https://github.com/NousResearch/hermes-agent/pull/32261))

- **Deep xAI integration round — Web Search plugin, OAuth proxy upstream, May 15 retirement detection, natural TTS, security hardening** — Six interlocking xAI improvements:
    - **xAI Web Search** lands as a `plugins/web/xai/` provider, slots alongside Brave / Tavily / Exa / SearXNG / DDGS / Firecrawl — reuses your existing Grok OAuth or `XAI_API_KEY` credentials, no new env vars. ([#29042](https://github.com/NousResearch/hermes-agent/pull/29042))
    - **`hermes proxy` gains an xAI upstream** — your local OpenAI-compatible endpoint can now be backed by SuperGrok OAuth, no PKCE-refresh code to write in your client. ([#28356](https://github.com/NousResearch/hermes-agent/pull/28356))
    - **May 15 model retirement detection** — `grok-4`, `grok-4-fast{,-reasoning,-non-reasoning}`, `grok-3`, `grok-code-fast-1`, `grok-imagine-image-pro` etc. are detected in doctor and chat startup, with `hermes migrate xai` to one-shot config migration to the supported model. No more silent 404s after the retirement date. ([#29277](https://github.com/NousResearch/hermes-agent/pull/29277))
    - **Opt-in `auto_speech_tags`** for xAI TTS — inserts light `[pause]` tags between paragraphs and sentences for more natural-sounding voice replies. Default OFF. ([#29376](https://github.com/NousResearch/hermes-agent/pull/29376))
    - **`xai-oauth` `base_url` pinned to `x.ai` origin** — closes a silent credential-leak vector where `XAI_BASE_URL` could repoint OAuth-authenticated inference to an attacker-controlled host. ([#28952](https://github.com/NousResearch/hermes-agent/pull/28952))
    - **OpenAI-style execution guidance applied to Grok models** — Grok and xai-oauth now get the same family-specific execution discipline block GPT/Codex have, so the model stops claiming completion without tool calls and stops suggesting workarounds instead of using existing tools. ([#27797](https://github.com/NousResearch/hermes-agent/pull/27797))
    - Plus `x_search` degraded-results surfacing, tier-gated 403 with API-key fallback, PKCE `code_challenge` round-trip fix, dead-token quarantine on terminal refresh failure, MiniMax-style short-token refresh on per-request, and `WKE=unauthenticated` honor at both classifier sites. ([#29484](https://github.com/NousResearch/hermes-agent/pull/29484), [#28351](https://github.com/NousResearch/hermes-agent/pull/28351), [#27560](https://github.com/NousResearch/hermes-agent/pull/27560), [#28116](https://github.com/NousResearch/hermes-agent/pull/28116), [#30619](https://github.com/NousResearch/hermes-agent/pull/30619), [#30872](https://github.com/NousResearch/hermes-agent/pull/30872))

---

## 🏗️ Core Agent & Architecture

### The Big Refactor — `run_agent.py` 16k → 3.8k

- `run_agent.py` from 16,083 → 3,821 lines (-76%), extracted into 14 cohesive `agent/*` modules. `run_conversation` alone was 3,877 lines before the refactor. Every extraction keeps a thin forwarder on `AIAgent`, every test-patch path is preserved, every external caller stays compatible. ([#27248](https://github.com/NousResearch/hermes-agent/pull/27248))

### Agent loop & conversation

- Auxiliary task layered fallback (primary → chain → main agent → graceful fail) on capacity errors (402/429/connection). (salvages [#26811](https://github.com/NousResearch/hermes-agent/pull/26811) + [#26998](https://github.com/NousResearch/hermes-agent/pull/26998)) ([#27625](https://github.com/NousResearch/hermes-agent/pull/27625))
- Buffer retry/fallback status; surface only on terminal failure (no more noisy "retrying..." spam in mid-run output). ([#33816](https://github.com/NousResearch/hermes-agent/pull/33816))
- Host contract for external context engines — condenses 5 prior PRs into one extension surface. ([#33750](https://github.com/NousResearch/hermes-agent/pull/33750))
- Fallback immediately on provider content-policy blocks. ([#33883](https://github.com/NousResearch/hermes-agent/pull/33883))
- Re-pad `reasoning_content` on cross-provider fallback to require-side providers. (salvage [#33784](https://github.com/NousResearch/hermes-agent/pull/33784)) ([#33795](https://github.com/NousResearch/hermes-agent/pull/33795))
- Per-turn tool-outcome verifier — patch tool gets indent preservation, CRLF preservation, per-file failure escalation. ([#32273](https://github.com/NousResearch/hermes-agent/pull/32273))
- Single-knob native vision for custom-provider models. ([#29679](https://github.com/NousResearch/hermes-agent/pull/29679))
- Background review fork isolated from external memory plugins. ([#27190](https://github.com/NousResearch/hermes-agent/pull/27190))
- Background review inherits parent toolset config for `tools[]` cache parity. ([#29704](https://github.com/NousResearch/hermes-agent/pull/29704))
- Recover from providers returning list-type tool content. ([#30259](https://github.com/NousResearch/hermes-agent/pull/30259))
- Treat partial-stream stub responses as length truncation rather than clean stop. ([#30998](https://github.com/NousResearch/hermes-agent/pull/30998))
- OpenAI execution guidance applied to xAI Grok / xai-oauth. ([#27797](https://github.com/NousResearch/hermes-agent/pull/27797))
- ContextVars propagate to concurrent tool worker threads.
- Preload `jiter` native parser. ([#33692](https://github.com/NousResearch/hermes-agent/pull/33692))
- Expose context engine tools with saved toolsets. (salvage of [#31194](https://github.com/NousResearch/hermes-agent/pull/31194)) ([#33719](https://github.com/NousResearch/hermes-agent/pull/33719))

### Sessions & memory

- `session_search` rebuilt — single-shape (discovery + scroll + browse), no aux-LLM, ~20ms vs. ~90s. ([#27590](https://github.com/NousResearch/hermes-agent/pull/27590))
- Salvage [#29182](https://github.com/NousResearch/hermes-agent/pull/29182) — opt-in JSON snapshot writer for sessions. ([#29278](https://github.com/NousResearch/hermes-agent/pull/29278))
- Persist `platform_message_id` for recall across gateway restarts. ([#29449](https://github.com/NousResearch/hermes-agent/pull/29449))
- Inline memory-context mentions stay visible in conversation. ([#28132](https://github.com/NousResearch/hermes-agent/pull/28132))
- Recalled memory labeled informational, not authoritative. ([#28583](https://github.com/NousResearch/hermes-agent/pull/28583))
- Memory + context-engine tool injection gated on `enabled_toolsets`. ([#30177](https://github.com/NousResearch/hermes-agent/pull/30177))
- Guard against external drift in `MEMORY.md` / `USER.md`. ([#30877](https://github.com/NousResearch/hermes-agent/pull/30877))
- Honcho runtime peer mapping — correctness follow-ups + setup wizard + docs. ([#30077](https://github.com/NousResearch/hermes-agent/pull/30077))
- Periodic memory logging for leak detection. (salvage of [#17667](https://github.com/NousResearch/hermes-agent/pull/17667)) ([#27102](https://github.com/NousResearch/hermes-agent/pull/27102))

### Codex / Responses-API maturation

- TTFB watchdog for stalled Codex Responses streams. ([#32042](https://github.com/NousResearch/hermes-agent/pull/32042))
- Actionable hint when stale-call detector fires on known silent-reject pattern. ([#32016](https://github.com/NousResearch/hermes-agent/pull/32016), [#33133](https://github.com/NousResearch/hermes-agent/pull/33133))
- Drop SDK `responses.stream()` helper; consume events directly. ([#33042](https://github.com/NousResearch/hermes-agent/pull/33042))
- Gracefully recover from `invalid_encrypted_content`. (salvage of [#10144](https://github.com/NousResearch/hermes-agent/pull/10144)) ([#33035](https://github.com/NousResearch/hermes-agent/pull/33035))
- Recover Codex Responses streams with null output. ([#32963](https://github.com/NousResearch/hermes-agent/pull/32963), [#33390](https://github.com/NousResearch/hermes-agent/pull/33390))
- Drop foreign-issuer reasoning and transient `rs_tmp` reasoning replay state. ([#33156](https://github.com/NousResearch/hermes-agent/pull/33156), [#33146](https://github.com/NousResearch/hermes-agent/pull/33146))
- Codex 429 quota classified as rate-limit, not missing credentials. ([#33168](https://github.com/NousResearch/hermes-agent/pull/33168))
- Codex chat path falls back to credential_pool when singleton is empty. ([#33189](https://github.com/NousResearch/hermes-agent/pull/33189))
- Codex re-auth syncs credential_pool. ([#33164](https://github.com/NousResearch/hermes-agent/pull/33164))
- Omit `tools` key when no tools registered. ([#33409](https://github.com/NousResearch/hermes-agent/pull/33409))
- Parse Codex image-generation SSE directly. ([#32933](https://github.com/NousResearch/hermes-agent/pull/32933))

---

## 🎛️ Kanban — Multi-Agent Maturation Wave

### Orchestration & dispatch

- Orchestrator-driven auto-decomposition on triage. ([#27572](https://github.com/NousResearch/hermes-agent/pull/27572))
- Kanban swarm topology helper — `hermes kanban swarm` creates a Swarm v1 graph (root + parallel workers + gated verifier + gated synthesizer + shared blackboard). (salvages [#26791](https://github.com/NousResearch/hermes-agent/pull/26791) by @Niraven) ([#28443](https://github.com/NousResearch/hermes-agent/pull/28443))
- Dispatcher wires review agents from the review column. ([#28449](https://github.com/NousResearch/hermes-agent/pull/28449))
- Stale-detection for running tasks in dispatcher. ([#28452](https://github.com/NousResearch/hermes-agent/pull/28452))
- Respawn guard blocks repeat worker storms. ([#28455](https://github.com/NousResearch/hermes-agent/pull/28455))
- Respawn guard defers `blocker_auth` instead of auto-blocking. ([#28683](https://github.com/NousResearch/hermes-agent/pull/28683))
- Cross-profile cron jobs surface in dashboard. ([#28457](https://github.com/NousResearch/hermes-agent/pull/28457))
- Worker visibility endpoints: `/workers/active`, `/runs/{id}`, `/inspect`. (salvages [#23761](https://github.com/NousResearch/hermes-agent/pull/23761) by @Interstellar-code) ([#28432](https://github.com/NousResearch/hermes-agent/pull/28432))

### Task configuration & scheduling

- Per-task model override. ([#28364](https://github.com/NousResearch/hermes-agent/pull/28364))
- Board-level default workdir. ([#28394](https://github.com/NousResearch/hermes-agent/pull/28394))
- Configurable worktree paths and branches. ([#28462](https://github.com/NousResearch/hermes-agent/pull/28462))
- Scheduled task start times. ([#28384](https://github.com/NousResearch/hermes-agent/pull/28384))
- Scheduled status for delayed follow-ups. ([#28467](https://github.com/NousResearch/hermes-agent/pull/28467))
- Trimmed task comments. ([#28399](https://github.com/NousResearch/hermes-agent/pull/28399))
- Initial-status for human-ops cards. ([#28414](https://github.com/NousResearch/hermes-agent/pull/28414))
- `max_in_progress` config to cap concurrent running tasks. ([#28420](https://github.com/NousResearch/hermes-agent/pull/28420))
- Filter tasks by workflow fields. ([#28454](https://github.com/NousResearch/hermes-agent/pull/28454))
- `--sort` for `hermes kanban list`. ([#28427](https://github.com/NousResearch/hermes-agent/pull/28427))
- Optional `board` parameter on all MCP tools. ([#28444](https://github.com/NousResearch/hermes-agent/pull/28444))
- Stamp originating ACP session_id on tasks. ([#28447](https://github.com/NousResearch/hermes-agent/pull/28447))
- `auto_promote_children` config toggle. ([#28344](https://github.com/NousResearch/hermes-agent/pull/28344))
- `archive --rm` to hard-delete archived tasks. ([#28355](https://github.com/NousResearch/hermes-agent/pull/28355))
- Promote dependents when parent is archived. ([#28372](https://github.com/NousResearch/hermes-agent/pull/28372))
- Promote blocked tasks when parent dependencies complete. ([#28377](https://github.com/NousResearch/hermes-agent/pull/28377))
- Demote ready children when parent is reopened. ([#28382](https://github.com/NousResearch/hermes-agent/pull/28382))
- `promote` verb for manual `todo→ready` recovery + bulk `--ids`. (salvage [#29464](https://github.com/NousResearch/hermes-agent/pull/29464)) ([#31334](https://github.com/NousResearch/hermes-agent/pull/31334))

### Dashboard

- Drag-to-delete trash zone + bulk delete. ([#28468](https://github.com/NousResearch/hermes-agent/pull/28468))
- Surface per-task `model_override` in show + tool output. ([#28442](https://github.com/NousResearch/hermes-agent/pull/28442))
- Cross-profile notification delivery via `kanban.notification_sources`. ([#28395](https://github.com/NousResearch/hermes-agent/pull/28395))
- Scratch-workspace deletion warning for users. ([#30949](https://github.com/NousResearch/hermes-agent/pull/30949))
- Mobile dashboard UX polish. ([#28127](https://github.com/NousResearch/hermes-agent/pull/28127))

### Reliability

- Worker log retention configurable. ([#27867](https://github.com/NousResearch/hermes-agent/pull/27867))
- Configurable claim TTL. ([#28392](https://github.com/NousResearch/hermes-agent/pull/28392))
- Fingerprint crash errors to prevent fleet-wide retry exhaustion. ([#28380](https://github.com/NousResearch/hermes-agent/pull/28380))
- Reset failure counters on `unblock_task`. ([#28379](https://github.com/NousResearch/hermes-agent/pull/28379))
- Detect cycles in `decompose_triage_task` sibling-link pre-validation. ([#28088](https://github.com/NousResearch/hermes-agent/pull/28088))
- Surface unusable triage auxiliary model (auto-decompose aware). ([#27871](https://github.com/NousResearch/hermes-agent/pull/27871))
- Align failure diagnostics with retry limit. ([#27868](https://github.com/NousResearch/hermes-agent/pull/27868))
- Align worker terminal timeout with task runtime. ([#27864](https://github.com/NousResearch/hermes-agent/pull/27864))
- Auto-install bundled skills (kanban-worker) on init. ([#28368](https://github.com/NousResearch/hermes-agent/pull/28368))
- Make legacy task migration idempotent. ([#28397](https://github.com/NousResearch/hermes-agent/pull/28397))
- Serialize DB initialization. ([#28383](https://github.com/NousResearch/hermes-agent/pull/28383))
- Persist worker session metadata on completion. ([#28387](https://github.com/NousResearch/hermes-agent/pull/28387))
- Pass `accept-hooks` to worker chat subprocess. ([#28393](https://github.com/NousResearch/hermes-agent/pull/28393))
- Preserve worker tools with restricted toolsets. ([#28396](https://github.com/NousResearch/hermes-agent/pull/28396))
- Avoid unsafe Windows worker Hermes shim resolution. ([#28398](https://github.com/NousResearch/hermes-agent/pull/28398))
- Sync slash subcommands with live parser. ([#28376](https://github.com/NousResearch/hermes-agent/pull/28376))
- Show scheduled kanban tasks in dashboard. ([#28400](https://github.com/NousResearch/hermes-agent/pull/28400))
- Assign single-task kanban decompositions. ([#28401](https://github.com/NousResearch/hermes-agent/pull/28401))
- Configurable `max_tokens` for kanban specify. ([#28374](https://github.com/NousResearch/hermes-agent/pull/28374))
- Per-job profile support for cron. ([#28124](https://github.com/NousResearch/hermes-agent/pull/28124))
- Codex app-server: include every Kanban-pinned path in `writable_roots`. ([#28435](https://github.com/NousResearch/hermes-agent/pull/28435))
- Cache kanban worker guidance at session init for prompt-cache reuse. ([#28425](https://github.com/NousResearch/hermes-agent/pull/28425))

---

## ⚡ Performance

- `openai._base_client` import deferred — 240ms / 17MB off every CLI cold start. ([#28864](https://github.com/NousResearch/hermes-agent/pull/28864))
- Agent-loop hot-path optimizations — 47% fewer per-conversation function calls (399k → 213k for 31-turn chat). ([#28866](https://github.com/NousResearch/hermes-agent/pull/28866))
- Compression-feasibility check deferred — 170-290ms off every agent construction. ([#28957](https://github.com/NousResearch/hermes-agent/pull/28957))
- Adaptive subprocess poll — ~195ms off every tool call, 1+ second per turn. ([#29006](https://github.com/NousResearch/hermes-agent/pull/29006))
- Termux TUI cold start speedup. ([#29419](https://github.com/NousResearch/hermes-agent/pull/29419))
- Termux non-TUI cold start speedup. (salvage [#29438](https://github.com/NousResearch/hermes-agent/pull/29438)) ([#30121](https://github.com/NousResearch/hermes-agent/pull/30121))
- Termux fast-path version + deferred bare-prompt agent startup. ([#30609](https://github.com/NousResearch/hermes-agent/pull/30609))
- Cut hermes `--version` wall time 63% — flips head-to-head vs Codex CLI. ([#31968](https://github.com/NousResearch/hermes-agent/pull/31968))
- Date-only timestamp + loud gateway-DB roundtrip logging — improves prompt-cache hit rate. ([#27675](https://github.com/NousResearch/hermes-agent/pull/27675))
- Cache kanban worker guidance at session init for prompt-cache reuse. ([#28425](https://github.com/NousResearch/hermes-agent/pull/28425))

---

## 🔧 Tool System

### Tool surface

- `patch`: indent preservation, CRLF preservation, per-file failure escalation. ([#32273](https://github.com/NousResearch/hermes-agent/pull/32273))
- `terminal`: warn at call time when `background=true` runs silently. ([#31289](https://github.com/NousResearch/hermes-agent/pull/31289))
- `terminal`: nudge homebrewed CI pollers at the tool surface. ([#33142](https://github.com/NousResearch/hermes-agent/pull/33142))
- `x_search`: surface degraded results + validate dates. ([#29484](https://github.com/NousResearch/hermes-agent/pull/29484))
- `x_search`: auto-enable toolset when xAI credentials are configured. ([#27376](https://github.com/NousResearch/hermes-agent/pull/27376))
- `computer_use`: route SOM/vision captures via auxiliary.vision. ([#30126](https://github.com/NousResearch/hermes-agent/pull/30126))
- `transcription`: reject symlinked audio inputs. ([#10082](https://github.com/NousResearch/hermes-agent/pull/10082))
- TTS: prevent double `[pause]` in xAI auto speech tags. ([#32237](https://github.com/NousResearch/hermes-agent/pull/32237))
- TTS: preserve native audio outside Telegram voice delivery. ([#28512](https://github.com/NousResearch/hermes-agent/pull/28512))
- TTS: opt-in xAI `auto_speech_tags` speech-tag pauses for natural voice replies. ([#29376](https://github.com/NousResearch/hermes-agent/pull/29376))
- Voice: chunk oversized CLI recordings. ([#30044](https://github.com/NousResearch/hermes-agent/pull/30044))
- Voice: honor `PULSE_SERVER` / `PIPEWIRE_REMOTE` inside Docker. ([#22534](https://github.com/NousResearch/hermes-agent/pull/22534))

### Browser

- All cloud browser providers (Browserbase, Anchor, Camofox, Hyperbrowser, etc.) migrated to image_gen-style plugins. (salvages [#25580](https://github.com/NousResearch/hermes-agent/pull/25580)) ([#27403](https://github.com/NousResearch/hermes-agent/pull/27403))
- Auto-launch Chromium-family browser for CDP. ([#29106](https://github.com/NousResearch/hermes-agent/pull/29106))
- Docker: discover agent-browser Chromium binary at boot. ([#33184](https://github.com/NousResearch/hermes-agent/pull/33184))

### Image generation

- **Krea** provider plugin (Krea 2 Medium + Large). ([#33236](https://github.com/NousResearch/hermes-agent/pull/33236))
- FAL backend ported to `plugins/image_gen/fal`. (salvage [#27966](https://github.com/NousResearch/hermes-agent/pull/27966)) ([#30380](https://github.com/NousResearch/hermes-agent/pull/30380))
- Cache xAI ephemeral URL responses to disk. ([#31759](https://github.com/NousResearch/hermes-agent/pull/31759))

### Web search

- **xAI Web Search** as a provider plugin. ([#29042](https://github.com/NousResearch/hermes-agent/pull/29042))

### MCP

- **Nous-approved MCP catalog** with interactive picker. ([#30870](https://github.com/NousResearch/hermes-agent/pull/30870))
- **TLS client certificate (mTLS) support** for HTTP and SSE MCP servers. ([#33721](https://github.com/NousResearch/hermes-agent/pull/33721))
- Stdin paste-back fallback for headless OAuth flow. ([#32053](https://github.com/NousResearch/hermes-agent/pull/32053))
- `skip` at paste prompt bypasses auth without disabling server. ([#32069](https://github.com/NousResearch/hermes-agent/pull/32069))
- Registry-aware `mcp_` prefix on both ends of round-trip. ([#31700](https://github.com/NousResearch/hermes-agent/pull/31700))

---

## 🧩 Skills Ecosystem

### Skills system

- **Skill bundles** — `/<name>` loads multiple skills. ([#28373](https://github.com/NousResearch/hermes-agent/pull/28373))
- Skills Hub: health checks, freshness badge, and a watchdog cron. ([#32345](https://github.com/NousResearch/hermes-agent/pull/32345))
- Opt-in AST deep diagnostics on skill writes. (salvage of [#30918](https://github.com/NousResearch/hermes-agent/pull/30918)) ([#31198](https://github.com/NousResearch/hermes-agent/pull/31198))
- Bundled/pinned skill protection in background-review prompts. ([#28338](https://github.com/NousResearch/hermes-agent/pull/28338))
- Show user-modified skill names in bundled skill sync summary. ([#28671](https://github.com/NousResearch/hermes-agent/pull/28671))
- Load symlinked skill slash commands. ([#27759](https://github.com/NousResearch/hermes-agent/pull/27759))
- Deduplicate Skills Hub search results by identifier, not name. ([#29490](https://github.com/NousResearch/hermes-agent/pull/29490))

### New skills

- `openhands` — delegate-to-OpenHands orchestration skill (closes [#477](https://github.com/NousResearch/hermes-agent/issues/477)) ([#32261](https://github.com/NousResearch/hermes-agent/pull/32261))
- `code-wiki` — persistent indexed dev wiki (closes [#486](https://github.com/NousResearch/hermes-agent/issues/486)) ([#32240](https://github.com/NousResearch/hermes-agent/pull/32240))
- `web-pentest` — OWASP recipes (closes [#400](https://github.com/NousResearch/hermes-agent/issues/400)) ([#32265](https://github.com/NousResearch/hermes-agent/pull/32265))
- `baoyu-article-illustrator` ([#28287](https://github.com/NousResearch/hermes-agent/pull/28287))

---

## ☁️ Providers

### xAI deep integration

- **xAI Web Search** as a `plugins/web/xai/` provider plugin. ([#29042](https://github.com/NousResearch/hermes-agent/pull/29042))
- **`hermes proxy` xAI upstream** — OpenAI-compatible local proxy backed by xai-oauth. ([#28356](https://github.com/NousResearch/hermes-agent/pull/28356))
- **May 15 model retirement detection + `hermes migrate xai`** for grok-4 / grok-3 / grok-code-fast-1 / grok-imagine-image-pro. ([#29277](https://github.com/NousResearch/hermes-agent/pull/29277))
- **Opt-in `auto_speech_tags`** for natural xAI TTS voice replies. ([#29376](https://github.com/NousResearch/hermes-agent/pull/29376))
- **xai-oauth base_url pinned to x.ai origin** — closes silent credential-leak vector. ([#28952](https://github.com/NousResearch/hermes-agent/pull/28952))
- **OpenAI-style execution guidance** applied to Grok / xai-oauth models. ([#27797](https://github.com/NousResearch/hermes-agent/pull/27797))
- xAI: detect retired May 15 models in doctor/chat startup. ([#29277](https://github.com/NousResearch/hermes-agent/pull/29277))
- xAI: resolve Grok Build context for OAuth. ([#30579](https://github.com/NousResearch/hermes-agent/pull/30579))
- xAI OAuth: tier-gated 403 with API-key fallback. ([#28351](https://github.com/NousResearch/hermes-agent/pull/28351))
- xAI OAuth: PKCE `code_challenge` echo. ([#27560](https://github.com/NousResearch/hermes-agent/pull/27560))
- xAI OAuth: quarantine dead tokens on terminal refresh failure. ([#28116](https://github.com/NousResearch/hermes-agent/pull/28116))
- xAI OAuth: honor `WKE=unauthenticated` disambiguator at both classifier sites. ([#30872](https://github.com/NousResearch/hermes-agent/pull/30872))
- xAI OAuth: accept bare-code manual paste (state=None). (closes [#26923](https://github.com/NousResearch/hermes-agent/issues/26923)) ([#33880](https://github.com/NousResearch/hermes-agent/pull/33880))
- xAI OAuth: fall back to manual paste on loopback timeout. ([#33231](https://github.com/NousResearch/hermes-agent/pull/33231))
- xAI proxy: handle 429 rate-limit responses in proxy retry path. ([#33743](https://github.com/NousResearch/hermes-agent/pull/33743))

### Other providers

- **OpenAI API as a first-class provider** (distinct from Codex runtime). ([#31898](https://github.com/NousResearch/hermes-agent/pull/31898))
- **Microsoft Entra ID** auth for Azure Foundry (with 1M Anthropic-Messages beta preserved on Bearer). (salvages [#27509](https://github.com/NousResearch/hermes-agent/pull/27509), [#27022](https://github.com/NousResearch/hermes-agent/pull/27022)) ([#28101](https://github.com/NousResearch/hermes-agent/pull/28101), [#28084](https://github.com/NousResearch/hermes-agent/pull/28084))
- **OpenRouter** sticky routing — `session_id` passed via `extra_body` so a long-running session keeps landing on the same upstream provider. (@Cybourgeoisie) ([#33939](https://github.com/NousResearch/hermes-agent/pull/33939))
- Nous: JWT token for inference; stop replaying invalid Nous refresh tokens. (@rewbs) ([#27663](https://github.com/NousResearch/hermes-agent/pull/27663))
- Nous Portal: one-shot setup, status CLI, and Nous-included markers. ([#30860](https://github.com/NousResearch/hermes-agent/pull/30860))
- Anthropic adapter: extract 7 helpers from `convert_messages_to_anthropic`. (salvage [#27784](https://github.com/NousResearch/hermes-agent/pull/27784)) ([#30386](https://github.com/NousResearch/hermes-agent/pull/30386))
- Catalog: add `qwen3.7-max` to Alibaba + Alibaba-Coding-Plan model lists. ([#33129](https://github.com/NousResearch/hermes-agent/pull/33129))
- opencode-go: route `qwen3.7-max` via `anthropic_messages`. (@beardthelion) ([#32780](https://github.com/NousResearch/hermes-agent/pull/32780))
- opencode-go: expose Kimi K2 + DeepSeek reasoning controls. ([#30845](https://github.com/NousResearch/hermes-agent/pull/30845))
- Remove Vercel AI Gateway and Vercel Sandbox.
- MiniMax OAuth: refresh short-lived access tokens per request. ([#30619](https://github.com/NousResearch/hermes-agent/pull/30619))
- Codex OAuth: quarantine terminal refresh errors. ([#28118](https://github.com/NousResearch/hermes-agent/pull/28118))
- Codex: drop dead model slugs that HTTP 400 on ChatGPT Pro. ([#33424](https://github.com/NousResearch/hermes-agent/pull/33424))
- Codex: sync `manual:device_code` pool entries on re-auth. ([#33744](https://github.com/NousResearch/hermes-agent/pull/33744))
- MiniMax OAuth: quarantine terminal refresh errors. ([#28119](https://github.com/NousResearch/hermes-agent/pull/28119))

---

## 🔑 Secrets

- **Bitwarden Secrets Manager** integration with lazy `bws` install. ([#30035](https://github.com/NousResearch/hermes-agent/pull/30035))
- Bitwarden: EU Cloud + self-hosted server URL support. ([#31378](https://github.com/NousResearch/hermes-agent/pull/31378))
- Label detected credentials with their source (Bitwarden). ([#30364](https://github.com/NousResearch/hermes-agent/pull/30364))

---

## 📱 Messaging Platforms (Gateway)

### Gateway core

- **Deliverable mode** — agents ship artifacts as native uploads from any platform (Slack/Discord/Telegram/Teams/Email). ([#27813](https://github.com/NousResearch/hermes-agent/pull/27813))
- `hermes send` — pipe any script's output to any messaging platform. (salvage of [#19631](https://github.com/NousResearch/hermes-agent/pull/19631)) ([#27188](https://github.com/NousResearch/hermes-agent/pull/27188))
- Debounce queued text follow-ups during active sessions. (salvage of [#31235](https://github.com/NousResearch/hermes-agent/pull/31235)) ([#31341](https://github.com/NousResearch/hermes-agent/pull/31341))
- Plugin-transformed final_response delivered through streaming gate. ([#31433](https://github.com/NousResearch/hermes-agent/pull/31433))
- Refresh cached agent tools on `/reload-mcp`. ([#32815](https://github.com/NousResearch/hermes-agent/pull/32815))
- Harden kanban + provider cleanup races on long-running workloads. ([#29479](https://github.com/NousResearch/hermes-agent/pull/29479))

### New / reorganized adapters

- **ntfy** — 23rd platform, push notifications, plugin shape, zero core edits. (salvages [#30625](https://github.com/NousResearch/hermes-agent/pull/30625) → [#4043](https://github.com/NousResearch/hermes-agent/pull/4043)) ([#30867](https://github.com/NousResearch/hermes-agent/pull/30867))
- **Discord** adapter migrated to bundled plugin. (salvage of [#24356](https://github.com/NousResearch/hermes-agent/pull/24356)) ([#30591](https://github.com/NousResearch/hermes-agent/pull/30591))
- **Mattermost** adapter migrated to bundled plugin. (salvage of [#30916](https://github.com/NousResearch/hermes-agent/pull/30916)) ([#31748](https://github.com/NousResearch/hermes-agent/pull/31748))

### Telegram

- Edit status messages in place instead of appending. (based on [#30141](https://github.com/NousResearch/hermes-agent/pull/30141) by @qike-ms) ([#30864](https://github.com/NousResearch/hermes-agent/pull/30864))
- Skip-STT audio path + 2GB cap via local Bot API server. ([#28541](https://github.com/NousResearch/hermes-agent/pull/28541))
- Route image documents (.png/.jpg/.webp/.gif) through vision pipeline. ([#28519](https://github.com/NousResearch/hermes-agent/pull/28519))
- Route audio file attachments away from STT pipeline. ([#28478](https://github.com/NousResearch/hermes-agent/pull/28478))
- `disable_topic_auto_rename` gateway flag. ([#28523](https://github.com/NousResearch/hermes-agent/pull/28523))
- `ignore_root_dm` config to drop messages without thread_id. ([#28536](https://github.com/NousResearch/hermes-agent/pull/28536))
- Chat-scoped auth without sender user_id. ([#28525](https://github.com/NousResearch/hermes-agent/pull/28525))
- Fail-closed auth fallback when `TELEGRAM_ALLOWED_USERS` is empty. ([#28494](https://github.com/NousResearch/hermes-agent/pull/28494))
- Roll over tool progress bubbles + scope audio_file_paths. ([#28482](https://github.com/NousResearch/hermes-agent/pull/28482))
- Avoid duplicate text after auto-TTS voice replies. ([#28509](https://github.com/NousResearch/hermes-agent/pull/28509))
- Mark final voice reply notify-worthy so Telegram delivers it audibly. ([#28504](https://github.com/NousResearch/hermes-agent/pull/28504))

### Discord

- Recover Windows voice opus decoding. ([#33182](https://github.com/NousResearch/hermes-agent/pull/33182))
- `allow_any_attachment` config to accept arbitrary file types. ([#27245](https://github.com/NousResearch/hermes-agent/pull/27245))
- Transcribe native voice notes. ([#28993](https://github.com/NousResearch/hermes-agent/pull/28993))
- Define UI view classes after lazy install. ([#28817](https://github.com/NousResearch/hermes-agent/pull/28817))

### Signal / Matrix / Feishu / Slack / WeCom

- Signal: `require_mention` filter for group chats. ([#28574](https://github.com/NousResearch/hermes-agent/pull/28574))
- Matrix: warn on clock-skew silent message drops. ([#27330](https://github.com/NousResearch/hermes-agent/pull/27330))
- Matrix E2EE installs full dep set; plugins respect `is_connected`. ([#31688](https://github.com/NousResearch/hermes-agent/pull/31688))
- Feishu: require webhook auth secret + honor config extras. ([#30746](https://github.com/NousResearch/hermes-agent/pull/30746))
- Feishu: enforce auth and chat binding for approval buttons. ([#30744](https://github.com/NousResearch/hermes-agent/pull/30744))
- Slack: socket recovery + Windows restart dedupe. ([#28873](https://github.com/NousResearch/hermes-agent/pull/28873))
- WeCom: safe-parse untrusted XML. ([#32442](https://github.com/NousResearch/hermes-agent/pull/32442))

### DingTalk / Webhooks / Microsoft Graph

- DingTalk: transcribe native voice notes. ([#28993](https://github.com/NousResearch/hermes-agent/pull/28993))
- Webhook: enforce `INSECURE_NO_AUTH` safety rail on dynamic route reloads. ([#30863](https://github.com/NousResearch/hermes-agent/pull/30863))
- Webhook: restrict default toolset capabilities. ([#30745](https://github.com/NousResearch/hermes-agent/pull/30745))
- Microsoft Graph: harden webhook auth requirements. ([#30169](https://github.com/NousResearch/hermes-agent/pull/30169))

---

## 🖥️ CLI & TUI

### CLI

- `/update` slash command in CLI and TUI. ([#23854](https://github.com/NousResearch/hermes-agent/pull/23854))
- Update auto-rollback when post-pull syntax check fails. ([#28669](https://github.com/NousResearch/hermes-agent/pull/28669))
- `--branch` flag for `hermes update`. (@jquesnelle) ([#29591](https://github.com/NousResearch/hermes-agent/pull/29591))
- `/exit --delete` flag to remove session on quit. (salvage of [#17665](https://github.com/NousResearch/hermes-agent/pull/17665)) ([#27101](https://github.com/NousResearch/hermes-agent/pull/27101))
- `▶ N` indicator in status bar for running `/background` tasks. ([#27175](https://github.com/NousResearch/hermes-agent/pull/27175))
- Live background terminal-process count in status bar. ([#32061](https://github.com/NousResearch/hermes-agent/pull/32061))
- Append session recap to `/status` output. (salvage of [#18587](https://github.com/NousResearch/hermes-agent/pull/18587)) ([#27176](https://github.com/NousResearch/hermes-agent/pull/27176))
- Configurable paste-collapse thresholds (TUI + CLI). (salvage [#29723](https://github.com/NousResearch/hermes-agent/pull/29723)) ([#32087](https://github.com/NousResearch/hermes-agent/pull/32087))
- `/resume` accepts position numbers. ([#31709](https://github.com/NousResearch/hermes-agent/pull/31709))
- Bring tool-call display back — verbose mode, specific failure reasons, todo progress. ([#31293](https://github.com/NousResearch/hermes-agent/pull/31293))
- Validate runtime token refresh in Qwen auth status. ([#31196](https://github.com/NousResearch/hermes-agent/pull/31196))

### TUI

- **TUI session orchestrator** — multiple live sessions in one TUI window. (salvages [#27642](https://github.com/NousResearch/hermes-agent/pull/27642)) ([#32980](https://github.com/NousResearch/hermes-agent/pull/32980))
- `mouse_tracking` DEC mode presets. (salvage of [#26681](https://github.com/NousResearch/hermes-agent/pull/26681) by @OutThisLife) ([#30084](https://github.com/NousResearch/hermes-agent/pull/30084))
- Termux scrollback preservation + touch-friendly defaults. ([#28910](https://github.com/NousResearch/hermes-agent/pull/28910))
- Full assistant text in scrollback (no history truncation). ([#28829](https://github.com/NousResearch/hermes-agent/pull/28829))
- Preserve scrollback when branching sessions. ([#30162](https://github.com/NousResearch/hermes-agent/pull/30162))
- Preserve Python dunder identifiers in markdown. ([#28582](https://github.com/NousResearch/hermes-agent/pull/28582))
- Active profile shown in TUI prompt. ([#28581](https://github.com/NousResearch/hermes-agent/pull/28581))
- Improve Charizard completion menu contrast. ([#28346](https://github.com/NousResearch/hermes-agent/pull/28346))
- Stop slash dropdown chopping last char of `/goal`. ([#31311](https://github.com/NousResearch/hermes-agent/pull/31311))
- Clipboard copy on linux/wayland. ([#29342](https://github.com/NousResearch/hermes-agent/pull/29342))
- Anchor `splitReasoning` unclosed-tag regex; stop eating last paragraph. ([#29426](https://github.com/NousResearch/hermes-agent/pull/29426))
- Surface verbose tool details. ([#30225](https://github.com/NousResearch/hermes-agent/pull/30225))
- Load Linux skills on Termux + salvage @adybag14-cyber's Termux gates. ([#30166](https://github.com/NousResearch/hermes-agent/pull/30166))
- Handle images with codex app-server. ([#31220](https://github.com/NousResearch/hermes-agent/pull/31220))
- Refresh virtual transcript on viewport resize. ([#31077](https://github.com/NousResearch/hermes-agent/pull/31077))
- Ignore late thinking deltas after completion. ([#31055](https://github.com/NousResearch/hermes-agent/pull/31055))
- Commit composer input bursts immediately. ([#31053](https://github.com/NousResearch/hermes-agent/pull/31053))
- Log parent gateway lifecycle exits. ([#31051](https://github.com/NousResearch/hermes-agent/pull/31051))
- Clear TTS env var on voice off + TTS indicator in status bar. ([#30987](https://github.com/NousResearch/hermes-agent/pull/30987))
- Pass `--expose-gc` as node argv instead of NODE_OPTIONS. ([#29998](https://github.com/NousResearch/hermes-agent/pull/29998))
- Align composer cursorLayout with wrap-ansi to kill multiline cursor drift. ([#27489](https://github.com/NousResearch/hermes-agent/pull/27489))
- Harden Terminal.app rendering and color paths. ([#27251](https://github.com/NousResearch/hermes-agent/pull/27251))
- Keep `/goal` verdict out of compact status row. ([#27971](https://github.com/NousResearch/hermes-agent/pull/27971))
- Clamp curses color 8 for 8-color terminals (Docker). ([#30260](https://github.com/NousResearch/hermes-agent/pull/30260))

---

## 🔒 Security & Reliability

### Promptware & memory hardening

- **Promptware defense** — shared threat patterns + memory load-time scan + tool-result delimiters. ([#32269](https://github.com/NousResearch/hermes-agent/pull/32269))
- Expand memory content scanning patterns to parity with skills guard. ([#9151](https://github.com/NousResearch/hermes-agent/pull/9151))
- Harden Skills Guard multi-word prompt patterns. (@YLChen-007) ([#26852](https://github.com/NousResearch/hermes-agent/pull/26852))
- Split cron scanner so skill prose stops false-positiving exfil patterns. ([#32339](https://github.com/NousResearch/hermes-agent/pull/32339))

### File safety

- Protect Hermes control-plane files from prompt injection (`auth.json`, `config.yaml`, `webhook_subscriptions.json`, `mcp-tokens/`). (salvages @PratikRai0101's [#14157](https://github.com/NousResearch/hermes-agent/pull/14157)) ([#30397](https://github.com/NousResearch/hermes-agent/pull/30397))
- Write-deny `<root>/.env` when running under a profile. ([#29687](https://github.com/NousResearch/hermes-agent/pull/29687))
- Defense-in-depth read-deny on credential stores. (salvages [#17659](https://github.com/NousResearch/hermes-agent/pull/17659) + [#8055](https://github.com/NousResearch/hermes-agent/pull/8055)) ([#30721](https://github.com/NousResearch/hermes-agent/pull/30721))
- TTS `output_path` traversal + update ZIP symlink reject. (salvage [#6693](https://github.com/NousResearch/hermes-agent/pull/6693) + [#15881](https://github.com/NousResearch/hermes-agent/pull/15881)) ([#32056](https://github.com/NousResearch/hermes-agent/pull/32056))
- Reject symlinked audio inputs. ([#10082](https://github.com/NousResearch/hermes-agent/pull/10082))

### Credential safety

- Avoid persisting borrowed credential secrets — runtime env-sourced keys no longer leak into `auth.json`. ([#31416](https://github.com/NousResearch/hermes-agent/pull/31416))
- Validate Nous Portal `inference_base_url` against host allowlist. (salvages [#27612](https://github.com/NousResearch/hermes-agent/pull/27612)) ([#30611](https://github.com/NousResearch/hermes-agent/pull/30611))
- Harden API server key placeholder handling. ([#30738](https://github.com/NousResearch/hermes-agent/pull/30738))
- Harden Google Chat OAuth credential persistence. (@Zyrixtrex) ([#24788](https://github.com/NousResearch/hermes-agent/pull/24788))
- xAI OAuth: pin inference `base_url` to x.ai origin. ([#28952](https://github.com/NousResearch/hermes-agent/pull/28952))
- Quarantine dead OAuth tokens on terminal refresh failure (xAI, Codex, MiniMax). ([#28116](https://github.com/NousResearch/hermes-agent/pull/28116), [#28118](https://github.com/NousResearch/hermes-agent/pull/28118), [#28119](https://github.com/NousResearch/hermes-agent/pull/28119))

### Supply-chain

- **On-demand supply-chain audit via OSV.dev** — `hermes audit`. ([#31460](https://github.com/NousResearch/hermes-agent/pull/31460))
- `hermes update` syntax-validates critical files post-pull, auto-rollback on failure. ([#28669](https://github.com/NousResearch/hermes-agent/pull/28669))
- Quarantine `hermes.exe` vs concurrent Windows instance. ([#26677](https://github.com/NousResearch/hermes-agent/pull/26677))

### Other hardening

- Restrict default webhook toolset capabilities. ([#30745](https://github.com/NousResearch/hermes-agent/pull/30745))
- Harden Microsoft Graph webhook auth requirements. ([#30169](https://github.com/NousResearch/hermes-agent/pull/30169))
- Require source CIDR allowlisting for public msgraph webhook binds. ([#33722](https://github.com/NousResearch/hermes-agent/pull/33722))
- Require `API_SERVER_KEY` before dispatching API server work. ([#33232](https://github.com/NousResearch/hermes-agent/pull/33232))
- env_passthrough: apply GHSA-rhgp-j443-p4rf filter to config.yaml path. (@roadhero) ([#27794](https://github.com/NousResearch/hermes-agent/pull/27794))
- Dashboard + WeCom: restrict markdown link schemes; safe-parse untrusted XML. ([#32442](https://github.com/NousResearch/hermes-agent/pull/32442))
- Salvage project-plugin RCE bypass fix from PR [#29311](https://github.com/NousResearch/hermes-agent/pull/29311) (GHSA-5qr3-c538-wm9j). ([#30837](https://github.com/NousResearch/hermes-agent/pull/30837))
- Cross-profile soft guard on file-write tools + system-prompt hint. ([#31290](https://github.com/NousResearch/hermes-agent/pull/31290))
- Reject unsafe tar members in Android psutil compatibility installer. ([#33742](https://github.com/NousResearch/hermes-agent/pull/33742))
- Reject non-regular tar members during tirith auto-install. ([#33786](https://github.com/NousResearch/hermes-agent/pull/33786))

---

## 🪟 Native Windows (Beta Continued)

- Thin desktop installer + first-launch `install.ps1` bootstrap. ([#27822](https://github.com/NousResearch/hermes-agent/pull/27822))
- Complete Windows bootstrap — `dep_ensure` + `install.ps1` + detection. (@alt-glitch) ([#27845](https://github.com/NousResearch/hermes-agent/pull/27845))
- `install.ps1`: strip BOM, `-Commit`/`-Tag` pin params, harden git ops. (@jquesnelle) ([#28169](https://github.com/NousResearch/hermes-agent/pull/28169))
- Consolidate ACP browser bootstrap into `install.{sh,ps1}`. (@alt-glitch) ([#27851](https://github.com/NousResearch/hermes-agent/pull/27851))
- `hermes update` quarantines live `hermes.exe`. ([#26677](https://github.com/NousResearch/hermes-agent/pull/26677))
- Discord voice opus decoding on Windows. ([#33182](https://github.com/NousResearch/hermes-agent/pull/33182))
- Windows Docker Desktop compatible compose file. (@Sunil123135) ([#31031](https://github.com/NousResearch/hermes-agent/pull/31031))

---

## 🖼️ Hermes Desktop GUI

- `hermes gui` launcher — install + build + launch packaged Electron app. (@OutThisLife) ([#30165](https://github.com/NousResearch/hermes-agent/pull/30165))
- Desktop UI lift. ([#27227](https://github.com/NousResearch/hermes-agent/pull/27227))
- `nix` package `.#desktop`. (@ethernet8023) ([#28964](https://github.com/NousResearch/hermes-agent/pull/28964))
- Hardened Slack socket recovery + Windows desktop restart dedupe. ([#28873](https://github.com/NousResearch/hermes-agent/pull/28873))
- Web dashboard: migrate checkboxes to `@nous-research/ui` + design-system polish. (@austinpickett) ([#28814](https://github.com/NousResearch/hermes-agent/pull/28814))
- Web dashboard: collapsible sidebar. (@austinpickett) ([#33421](https://github.com/NousResearch/hermes-agent/pull/33421))
- Dashboard typography & contrast pass. (salvage of [#28832](https://github.com/NousResearch/hermes-agent/pull/28832)) ([#30714](https://github.com/NousResearch/hermes-agent/pull/30714))
- Skills page: lazy-fetch catalog instead of bundling 34MB into JS. ([#33809](https://github.com/NousResearch/hermes-agent/pull/33809))

---

## 🐳 Docker

- **s6-overlay container supervision** — abstract `ServiceManager` protocol (systemd/launchd/Windows/s6 backends), per-profile gateway supervision in-container, container-restart reconciliation, hadolint/shellcheck CI. (salvage of [#30136](https://github.com/NousResearch/hermes-agent/pull/30136), @benbarclay) ([#31760](https://github.com/NousResearch/hermes-agent/pull/31760))
- Auto-redirect `gateway run` to supervised mode inside the s6 image. (@benbarclay) ([#33583](https://github.com/NousResearch/hermes-agent/pull/33583))
- Tee supervised gateway stdout to docker logs. (@benbarclay) ([#33621](https://github.com/NousResearch/hermes-agent/pull/33621))
- Drop `docker exec` to hermes uid before invoking the CLI. (@benbarclay) ([#33628](https://github.com/NousResearch/hermes-agent/pull/33628))
- Align HOME for dashboard and s6 gateway services. (@Dusk1e) ([#33481](https://github.com/NousResearch/hermes-agent/pull/33481))
- Bake build-time git SHA into image so `hermes dump` reports it. (@benbarclay) ([#33655](https://github.com/NousResearch/hermes-agent/pull/33655))
- `hermes update` prints `docker pull` guidance instead of bogus git error. (@benbarclay) ([#33659](https://github.com/NousResearch/hermes-agent/pull/33659))
- Upgrade Node to 22 LTS via multi-stage from `node:22-bookworm-slim`. (@benbarclay) ([#33060](https://github.com/NousResearch/hermes-agent/pull/33060))
- Drop `build-essential` from apt install. (@benbarclay) ([#33028](https://github.com/NousResearch/hermes-agent/pull/33028))
- Propagate env through s6 to cont-init and main CMD. ([#32412](https://github.com/NousResearch/hermes-agent/pull/32412))
- Targeted chown to preserve host file ownership in `HERMES_HOME`. ([#33033](https://github.com/NousResearch/hermes-agent/pull/33033))
- `mkdir HERMES_HOME` as root in stage2 before chown / privilege drop. ([#33078](https://github.com/NousResearch/hermes-agent/pull/33078))
- chown `ui-tui` and `node_modules` on UID remap so TUI esbuild works. ([#33045](https://github.com/NousResearch/hermes-agent/pull/33045))
- Include `anthropic`, `bedrock`, `azure-identity` extras in image. ([#30504](https://github.com/NousResearch/hermes-agent/pull/30504))
- Stop pushing per-commit SHA tags to Docker Hub. ([#29387](https://github.com/NousResearch/hermes-agent/pull/29387))
- Simplify Docker tagging — push both `:main` and `:latest` on main push. ([#33225](https://github.com/NousResearch/hermes-agent/pull/33225))
- Test slicing across GH actions jobs. (@ethernet8023) ([#30575](https://github.com/NousResearch/hermes-agent/pull/30575))
- Discover agent-browser Chromium binary at boot. ([#33184](https://github.com/NousResearch/hermes-agent/pull/33184))

---

## 🌐 API Server

- **Session control API** — `/api/sessions/*` (list/create/read/patch/delete/fork) + SSE-streaming chat. (salvages [#29302](https://github.com/NousResearch/hermes-agent/pull/29302) by @Codename-11 + multimodal followup by @Schwartz10) ([#33134](https://github.com/NousResearch/hermes-agent/pull/33134))
- `GET /v1/skills` and `/v1/toolsets`. ([#33016](https://github.com/NousResearch/hermes-agent/pull/33016))
- Coerce stringified booleans in stream/store/approval payloads. (salvage [#26639](https://github.com/NousResearch/hermes-agent/pull/26639)) ([#27293](https://github.com/NousResearch/hermes-agent/pull/27293))
- Honor `key_env` in auth-failure fallback resolution. ([#30840](https://github.com/NousResearch/hermes-agent/pull/30840))

---

## 🎟️ ACP (VS Code / Zed / JetBrains)

- Session edit auto-approval modes. (salvage of [#27034](https://github.com/NousResearch/hermes-agent/pull/27034)) ([#27862](https://github.com/NousResearch/hermes-agent/pull/27862))
- Enrich Zed permission cards — command in title + `reject_always`. ([#28148](https://github.com/NousResearch/hermes-agent/pull/28148))
- Replay session history before responding to `session/load`. ([#26957](https://github.com/NousResearch/hermes-agent/pull/26957), [#26943](https://github.com/NousResearch/hermes-agent/pull/26943))
- Plugin-transformed final_response delivered through streaming gate. ([#31433](https://github.com/NousResearch/hermes-agent/pull/31433))

---

## 🔌 Plugin Surface

- `register_tts_provider()` plugin hook. (salvage of [#30420](https://github.com/NousResearch/hermes-agent/pull/30420)) ([#31745](https://github.com/NousResearch/hermes-agent/pull/31745))
- `register_transcription_provider()` hook + `stt.providers` command-provider registry. (salvage of [#30493](https://github.com/NousResearch/hermes-agent/pull/30493)) ([#31907](https://github.com/NousResearch/hermes-agent/pull/31907))
- `register_auxiliary_task()` in PluginContext API. (salvage [#29817](https://github.com/NousResearch/hermes-agent/pull/29817)) ([#31177](https://github.com/NousResearch/hermes-agent/pull/31177))
- Bundled `security-guidance` plugin. ([#33131](https://github.com/NousResearch/hermes-agent/pull/33131))
- Discord and Mattermost migrated to bundled plugins. ([#30591](https://github.com/NousResearch/hermes-agent/pull/30591), [#31748](https://github.com/NousResearch/hermes-agent/pull/31748))
- ntfy as platform plugin. ([#30867](https://github.com/NousResearch/hermes-agent/pull/30867))
- Surface category-namespaced plugins in `hermes plugins list`. ([#27187](https://github.com/NousResearch/hermes-agent/pull/27187))
- Plugin discovery failures raised to WARNING level. ([#28318](https://github.com/NousResearch/hermes-agent/pull/28318))
- `hermes_plugins` included in gateway.log component filter. ([#28313](https://github.com/NousResearch/hermes-agent/pull/28313))
- Seed plugin extras before `is_connected` gate. ([#31703](https://github.com/NousResearch/hermes-agent/pull/31703))
- Dashboard: allowlist plugin assets + denylist subprocess-influencing env vars. ([#32277](https://github.com/NousResearch/hermes-agent/pull/32277))

---

## 📦 Distribution & Install

- Install-method stamping + Docker detection. (@alt-glitch) ([#27843](https://github.com/NousResearch/hermes-agent/pull/27843))
- Nix `#messaging` and `#full` package variants. (@alt-glitch) ([#33108](https://github.com/NousResearch/hermes-agent/pull/33108))
- Pre-load messaging gateway deps via `--extra messaging`. (salvage [#26394](https://github.com/NousResearch/hermes-agent/pull/26394)) ([#27558](https://github.com/NousResearch/hermes-agent/pull/27558))
- Avoid piping installer directly into `iex` (Windows). ([#28347](https://github.com/NousResearch/hermes-agent/pull/28347))
- Ship bundled skills in wheel. ([#28421](https://github.com/NousResearch/hermes-agent/pull/28421))
- Ship dashboard plugin assets in wheel. ([#28406](https://github.com/NousResearch/hermes-agent/pull/28406))
- Make Camofox lazy-installed instead of eager. ([#27055](https://github.com/NousResearch/hermes-agent/pull/27055))
- Wire STT lazy-install into transcription_tools.py. ([#30256](https://github.com/NousResearch/hermes-agent/pull/30256))

---

## 🐛 Notable Bug Fixes (highlights only)

- Match bare custom provider by active base URL in `hermes model`. ([#28908](https://github.com/NousResearch/hermes-agent/pull/28908))
- Route `auxiliary.vision.provider=openai` to api.openai.com, skip text-only main. ([#31452](https://github.com/NousResearch/hermes-agent/pull/31452))
- Lint: skip per-file shell linter when LSP will handle the file. ([#29054](https://github.com/NousResearch/hermes-agent/pull/29054))
- Treat empty credential pool entries as unauthenticated in `/model` picker. ([#28312](https://github.com/NousResearch/hermes-agent/pull/28312))
- Reverted within window: Firecrawl integration tag, send_message @username auto-mentions, Telegram quick-command-only menus, Telegram pin-on-turn.

---

## 🧪 Testing

- Disarm lazy-install probe so `_HAS_FASTER_WHISPER` patches work. ([#30334](https://github.com/NousResearch/hermes-agent/pull/30334))
- Cover default board dashboard pin. ([#28361](https://github.com/NousResearch/hermes-agent/pull/28361))
- Cover `_task_dict` `task_age` fallback. ([#28365](https://github.com/NousResearch/hermes-agent/pull/28365))
- Allowlist `tmp_path` for `kanban_notify` artifact delivery tests. ([#30851](https://github.com/NousResearch/hermes-agent/pull/30851), [#30852](https://github.com/NousResearch/hermes-agent/pull/30852))
- Cover null output stream terminal events in Codex. ([#33137](https://github.com/NousResearch/hermes-agent/pull/33137))

---

## 📚 Documentation

- **30-day docs overhaul** — full correctness audit, every PR in the window covered, Nous Portal weave, sidebar reorg. ([#33782](https://github.com/NousResearch/hermes-agent/pull/33782))
- Dedicated Nous Portal integration page and setup guide. ([#31296](https://github.com/NousResearch/hermes-agent/pull/31296))
- Providers: move Nous Portal first, Google Gemini OAuth last. ([#31287](https://github.com/NousResearch/hermes-agent/pull/31287))
- `session_search` rewrite for single-shape tool. ([#27840](https://github.com/NousResearch/hermes-agent/pull/27840))
- Kanban: document failure_limit, max_retries, inline create shortcuts, goals & kanban settings. ([#28357](https://github.com/NousResearch/hermes-agent/pull/28357), [#28358](https://github.com/NousResearch/hermes-agent/pull/28358), [#28359](https://github.com/NousResearch/hermes-agent/pull/28359), [#28360](https://github.com/NousResearch/hermes-agent/pull/28360), [#28362](https://github.com/NousResearch/hermes-agent/pull/28362))
- Kanban Codex lane skill. ([#28430](https://github.com/NousResearch/hermes-agent/pull/28430))
- xAI OAuth: note X Premium+ also unlocks Grok OAuth. ([#29055](https://github.com/NousResearch/hermes-agent/pull/29055))
- Docs site: Docker audio bridge notes, "Installing more tools in the container", xurl auth HOME in Docker.
- Email: clarify gateway vs Himalaya setup. (@helix4u) ([#33634](https://github.com/NousResearch/hermes-agent/pull/33634))
- Auth docs: replace stale `hermes login` references with `hermes auth add`. ([#32859](https://github.com/NousResearch/hermes-agent/pull/32859))

---

## 👥 Contributors

### Core
- @teknium1 (lead)

### Notable salvages & cherry-picks

- **@benbarclay** — s6-overlay container supervision (29 commits salvaged), Node 22 LTS upgrade, build-essential cleanup, `gateway run` auto-redirect in s6, tee supervised stdout to docker logs, `hermes update` Docker guidance, build-time SHA stamping
- **@OutThisLife** — `hermes gui` desktop launcher, `mouse_tracking` DEC mode presets
- **@jquesnelle** — Windows installer hardening, `--branch` flag for `hermes update`, install.ps1 BOM strip / commit-pin
- **@alt-glitch** — Windows `dep_ensure` bootstrap, Nix package variants (`.#messaging`, `.#full`), install-method stamping, ACP browser bootstrap consolidation
- **@austinpickett** — `/update` slash command, dashboard checkboxes → `@nous-research/ui`, mobile dashboard polish, collapsible sidebar
- **@ethernet8023** — Nix `.#desktop` packaging, CI test slicing across GH Actions jobs, TUI clipboard copy fix
- **@kshitijk4poor** — doctor section banner + fail-and-issue helpers extraction, post-tag salvage cluster (curator-fallout, kanban SQLite hardening, install world-readable uv dirs, xAI bare-code paste)
- **@rewbs** — Nous JWT inference switch + refresh-token replay fix
- **@Codename-11** + **@Schwartz10** — session control API (REST + SSE + multimodal followup)
- **@Niraven** — kanban swarm topology helper
- **@Interstellar-code** — kanban worker visibility endpoints
- **@adybag14-cyber** — termux cold-start optimizations (multiple PRs)
- **@qike-ms** — Telegram in-place status edits design
- **@sprmn24** — ntfy adapter
- **@Jaaneek** — xAI Web Search provider plugin
- **@yannsunn** — xAI upstream adapter for `hermes proxy`
- **@Cybourgeoisie** — OpenRouter sticky routing via session_id
- **@memosr** — Nous Portal base_url allowlist validation
- **@Sunil123135** — Windows Docker Desktop compose file
- **@Dusk1e** — Docker HOME alignment for dashboard + s6 gateway services
- **@beardthelion** — opencode-go anthropic_messages routing
- **@YLChen-007** — Skills Guard multi-word prompt patterns
- **@roadhero** — env_passthrough GHSA-rhgp-j443-p4rf filter
- **@Zyrixtrex** — Google Chat OAuth credential persistence hardening
- **@briandevans**, **@tomqiaozc** — defense-in-depth read-deny on credential stores
- **@PratikRai0101** — control-plane file write protection
- **@helix4u**, **@Bartok9**, **@zccyman** — auxiliary fallback ladder components
- **@ms-alan**, **@ticketclosed-wontfix**, **@donovan-yohan** — TUI session orchestrator + follow-ups
- **@daimon-nous[bot]** — cron per-job profile support
- **@bisko** — re-pad `reasoning_content` on cross-provider fallback

### All Contributors

@02356abc, @0xchainer, @0xDevNinja, @0xjackyang, @0xsir0000, @0z1-ghb, @8bit64k, @aaronlab, @AceWattGit,
@ACR27, @adam91holt, @AdamPlatin123, @Ade5954, @AdityaRajeshGadgil, @adybag14-cyber, @AhmetArif0, @ai-hana-ai,
@alaamohanad169-ship-it, @alber70g, @albert748, @alt-glitch, @aqilaziz, @argabor, @asdlem, @austinpickett,
@avifenesh, @awizemann, @B0Tch1, @Bartok9, @BaxBit, @Beandon13, @beardthelion, @benbarclay, @bensargotest-sys,
@binhnt92, @bird, @bisko, @BlackishGreen33, @booker1207, @bradhallett, @briandevans, @Brixyy, @brndnsvr,
@BROCCOLO1D, @btorresgil, @burjorjee, @carltonawong, @Carry00, @chaconne67, @chdlc, @chromalinx, @ChyuWei,
@CipherFrame, @cmullins70, @CNSeniorious000, @codeblackhole1024, @Codename-11, @colin-chang, @counterposition,
@cresslank, @CryptoByz, @cyb0rgk1tty, @Cybourgeoisie, @daizhonggeng, @darvsum, @davidcampbelldc, @deas,
@dgians, @dillweed, @DoGMaTiiC, @donovan-yohan, @draplater, @Drexuxux, @dskwe, @dsr-restyn, @Dusk1e,
@dusterbloom, @duyua9, @egilewski, @el-analista, @eliteworkstation94-ai, @eloklam, @EloquentBrush0x, @emonty,
@emozilla, @erhnysr, @erikengervall, @Erosika, @ether-btc, @ethernet8023, @EvilHumphrey, @fabiosiqueira,
@falasi, @falconexe, @fardoche6, @felix-windsor, @Fewmanism, @ffr31mr, @flamiinngo, @flanny7, @flooryyyy,
@fonhal, @francip, @fujinice, @gianfrancopiana, @glennc, @Glucksberg, @godlin-gh, @Grogger, @guillaumemeyer,
@Gutslabs, @H-Ali13381, @hanzckernel, @haran2001, @hawknewton, @hayka-pacha, @hehehe0803, @helix4u, @HenkDz,
@Hermes, @hermesagent26, @Hinotoi-agent, @hongchen1993, @honor2030, @houenyang-momo, @ht1072, @hueilau,
@iamfoz, @ilonagaja509-glitch, @InB4DevOps, @indigokarasu, @Interstellar-code, @iqdoctor, @iRonin, @Jaaneek,
@JabberELF, @jacevys, @jackey8616, @jackjin1997, @jdelmerico, @jfuenmayor, @Jiahui-Gu, @JimLiu, @joe102084,
@JohnC1009, @jonpol01, @Jpalmer95, @Julientalbot, @justemu, @justincc, @jvinals, @karthikeyann, @kasunvinod,
@kchuang1015, @kenyonxu, @khungate, @kiranvk-2011, @kjames2001, @konsisumer, @kpadilha, @kriscolab,
@krislidimo, @kronexoi, @kshitijk4poor, @kunci115, @Kylejeong2, @kylekahraman, @LaPhilosophie, @leeseoki0,
@lemassykoi, @Lempkey, @LeonJS, @LeonSGP43, @lidge-jun, @LifeJiggy, @liuhao1024, @LizerAIDev, @loicnico96,
@loongfay, @m0n3r0, @malaiwah, @matthewlai, @mavrickdeveloper, @maxmilian, @McClean-Edison, @memosr,
@Mind-Dragon, @momowind, @MoonJuhan, @MoonRay305, @moortekweb-art, @MorAlekss, @ms-alan, @Nami4D,
@nehaaprasaad, @nekwo, @nftpoetrist, @NickLarcombe, @nidhi-singh02, @Niraven, @nnnet, @noctilust, @novax635,
@nthrow, @nv-kasikritc, @nycomar, @OCWC22, @oemtalks, @OmX, @ooovenenoso, @orcool, @oseftg, @outsourc-e,
@OutThisLife, @Paperclip, @PaTTeeL, @pepelax, @phoenixshen, @Pluviobyte, @pnascimento9596, @pochi-gio, @pr7426,
@PratikRai0101, @Prithvi1994, @psionic73, @ptichalouf, @Que0x, @QuenVix, @quocanh261997, @qWaitCrypto, @Qwinty,
@r266-tech, @rak135, @rdasilva1016-ui, @rewbs, @roadhero, @rodrigoeqnit, @RonHillDev, @roycepersonalassistant,
@rudi193-cmd, @RyanRana, @sadiksaifi, @samahn0601, @samggggflynn, @SamuelZ12, @sanghyuk-seo-nexcube,
@Saurav0989, @savanne-kham, @Schrotti77, @Schwartz10, @SerenityTn, @sgtworkman, @sharziki, @shaun0927,
@shellybotmoyer, @shunsuke-hikiyama, @SimbaKingjoe, @SimoKiihamaki, @sir-ad, @Slimydog21, @slowtokki0409,
@Soju06, @someaka, @soynchux, @sprmn24, @Stark-X, @steezkelly, @stepanov1975, @stephenschoettler,
@stevehq26-bot, @steveonjava, @Strontvod, @subtract0, @Sunil123135, @superearn-fisher, @Sylw3ster, @tchanee,
@that-ambuj, @thedavidmurray, @TheOnlyMika, @therahul-yo, @thewillhuang, @ticketclosed-wontfix, @Timur00Kh,
@tomqiaozc, @Tosko4, @Tranquil-Flow, @tw2818, @uzunkuyruk, @vaddisrinivas, @vanthinh6886, @vgocoder,
@victorGPT, @vynxevainglory-ai, @waefrebeorn, @walli, @wangpuv, @wanwan2qq, @wesleysimplicio, @worlldz,
@wpengpeng168, @WuKongAI-CMU, @wuli666, @Wysie, @wysie, @xxxigm, @yannsunn, @YanzhongSu, @YarrowQiao, @ygd58,
@YLChen-007, @yoniebans, @yu-xin-c, @YuanHanzhong, @zapabob, @zccyman, @ziliangpeng, @zwolniony, @Zyrixtrex

---

**Full Changelog**: [v2026.5.16...v2026.5.28](https://github.com/NousResearch/hermes-agent/compare/v2026.5.16...v2026.5.28)
