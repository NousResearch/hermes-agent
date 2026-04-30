# Hermes Agent v0.12.0 (v2026.4.30)

**Release Date:** April 30, 2026
**Since v0.11.0:** 1,096 commits · 550 merged PRs · 1,270 files changed · 217,776 insertions · 213 community contributors (including co-authors)

> The Curator release — Hermes Agent now maintains itself. An autonomous background Curator grades, prunes, and consolidates your skill library on its own schedule. The self-improvement loop that reviews what to save got a substantial upgrade. Four new inference providers, a 18th messaging platform, a 19th via Teams plugin, native Spotify + Google Meet integrations, ComfyUI and TouchDesigner-MCP moved from optional to bundled-by-default, and a ~57% cut to visible TUI cold start.

---

## ✨ Highlights

- **Autonomous Curator** — `hermes curator` runs as a background agent on the gateway's cron ticker (7-day cycle default). It grades your skill library, consolidates related skills, prunes dead ones, and writes per-run reports to `logs/curator/run.json` + `REPORT.md`. Archived skills are classified consolidated-vs-pruned via model + heuristic. Defense-in-depth gates protect bundled/hub skills from mutation. Unified under `auxiliary.curator` — pick the curator's model in `hermes model`, manage it from the dashboard. `hermes curator status` ranks skills by usage (most-used / least-used). ([#17277](https://github.com/NousResearch/hermes-agent/pull/17277), [#17307](https://github.com/NousResearch/hermes-agent/pull/17307), [#17941](https://github.com/NousResearch/hermes-agent/pull/17941), [#17868](https://github.com/NousResearch/hermes-agent/pull/17868), [#18033](https://github.com/NousResearch/hermes-agent/pull/18033))

- **Self-improvement loop — substantially upgraded** — The background review fork (the core of Hermes' self-improvement: after each turn it decides what memories/skills to save or update) is now class-first (rubric-based rather than free-form), active-update biased (prefers the skill the agent just loaded), handles `references/`/`templates/` sub-files, and properly inherits the parent's live runtime (provider, model, credentials actually propagate). Restricted to memory + skills toolsets so it can't sprawl. Memory providers shut down cleanly. Prior-turn tool messages excluded from the summary so the fork sees a clean context. ([#16026](https://github.com/NousResearch/hermes-agent/pull/16026), [#17213](https://github.com/NousResearch/hermes-agent/pull/17213), [#16099](https://github.com/NousResearch/hermes-agent/pull/16099), [#16569](https://github.com/NousResearch/hermes-agent/pull/16569), [#16204](https://github.com/NousResearch/hermes-agent/pull/16204), [#15057](https://github.com/NousResearch/hermes-agent/pull/15057))

- **Skill integrations — major expansion** — **ComfyUI v5** with official CLI + REST + hardware-gated local install, moved from optional to **built-in by default** ([#17610](https://github.com/NousResearch/hermes-agent/pull/17610), [#17631](https://github.com/NousResearch/hermes-agent/pull/17631), [#17734](https://github.com/NousResearch/hermes-agent/pull/17734)). **TouchDesigner-MCP** bundled by default, expanded with GLSL, post-FX, audio, geometry, and 9 new reference docs ([#16753](https://github.com/NousResearch/hermes-agent/pull/16753), [#16624](https://github.com/NousResearch/hermes-agent/pull/16624), [#16768](https://github.com/NousResearch/hermes-agent/pull/16768) — @kshitijk4poor + @SHL0MS). **Humanizer** skill ports a text-cleaner that strips AI-isms ([#16787](https://github.com/NousResearch/hermes-agent/pull/16787)). **claude-design** HTML artifact skill + design-md (Google DESIGN.md spec) + airtable salvage + `skill_manage` edits in `external_dirs` + direct-URL skill install + `/reload-skills` slash command. ([#16358](https://github.com/NousResearch/hermes-agent/pull/16358), [#14876](https://github.com/NousResearch/hermes-agent/pull/14876), [#16291](https://github.com/NousResearch/hermes-agent/pull/16291), [#17512](https://github.com/NousResearch/hermes-agent/pull/17512), [#16323](https://github.com/NousResearch/hermes-agent/pull/16323), [#17744](https://github.com/NousResearch/hermes-agent/pull/17744))

- **LM Studio — first-class provider** — upgraded from a custom-endpoint alias to a full-blown native provider: dedicated auth, `hermes doctor` checks, reasoning transport, live `/models` listing. (Salvage of @kshitijk4poor's #17061.) ([#17102](https://github.com/NousResearch/hermes-agent/pull/17102))

- **Four more new inference providers** — **GMI Cloud** (first-class, salvage of #11955 — @isaachuangGMICLOUD), **Azure AI Foundry** with auto-detection, **MiniMax OAuth** with PKCE browser flow (salvage #15203), **Tencent Tokenhub** (salvage of #16860). ([#16663](https://github.com/NousResearch/hermes-agent/pull/16663), [#15845](https://github.com/NousResearch/hermes-agent/pull/15845), [#17524](https://github.com/NousResearch/hermes-agent/pull/17524), [#16960](https://github.com/NousResearch/hermes-agent/pull/16960))

- **Pluggable gateway platforms + Microsoft Teams** — the gateway is now a plugin host. Drop-in messaging adapters live outside the core, and Microsoft Teams is the first plugin-shipped platform. (Salvage of #17664.) ([#17751](https://github.com/NousResearch/hermes-agent/pull/17751), [#17828](https://github.com/NousResearch/hermes-agent/pull/17828))

- **Tencent 元宝 (Yuanbao) — 18th messaging platform** — native gateway adapter with text + media delivery. ([#16298](https://github.com/NousResearch/hermes-agent/pull/16298), [#17424](https://github.com/NousResearch/hermes-agent/pull/17424))

- **Spotify — native tools + bundled skill + wizard** — 7 tools (play, search, queue, playlists, devices) behind PKCE OAuth, interactive setup wizard, bundled skill, surfacing in `hermes tools`, cron usage documented. ([#15121](https://github.com/NousResearch/hermes-agent/pull/15121), [#15130](https://github.com/NousResearch/hermes-agent/pull/15130), [#15154](https://github.com/NousResearch/hermes-agent/pull/15154), [#15180](https://github.com/NousResearch/hermes-agent/pull/15180))

- **Google Meet plugin** — join calls, transcribe, speak, follow up. Realtime OpenAI transport + Node bot server, full pipeline bundled as a plugin. ([#16364](https://github.com/NousResearch/hermes-agent/pull/16364))

- **`hermes -z` one-shot mode + `hermes update --check`** — non-interactive `hermes -z <prompt>` with `--model`/`--provider`/`HERMES_INFERENCE_MODEL`. `hermes update --check` preflight. Opt-in pre-update HERMES_HOME backup. ([#15702](https://github.com/NousResearch/hermes-agent/pull/15702), [#15704](https://github.com/NousResearch/hermes-agent/pull/15704), [#15841](https://github.com/NousResearch/hermes-agent/pull/15841), [#16539](https://github.com/NousResearch/hermes-agent/pull/16539), [#16566](https://github.com/NousResearch/hermes-agent/pull/16566))

- **Models dashboard tab + in-browser model config** — rich per-model analytics, switch main + auxiliary models from the dashboard. ([#17745](https://github.com/NousResearch/hermes-agent/pull/17745), [#17802](https://github.com/NousResearch/hermes-agent/pull/17802))

- **Remote model catalog manifest** — OpenRouter + Nous Portal model catalogs are now pulled from a remote manifest so new models show up without a release. ([#16033](https://github.com/NousResearch/hermes-agent/pull/16033))

- **Native multimodal image routing** — images now route based on the model's actual vision capability rather than provider defaults. ([#16506](https://github.com/NousResearch/hermes-agent/pull/16506))

- **Gateway media parity** — native multi-image sending across Telegram, Discord, Slack, Mattermost, Email, and Signal; centralized audio routing with FLAC support + Telegram document fallback. ([#17909](https://github.com/NousResearch/hermes-agent/pull/17909), [#17833](https://github.com/NousResearch/hermes-agent/pull/17833))

- **TUI catches up to (and past) the classic CLI** — LaTeX rendering (@austinpickett), `/reload` .env hot-reload, pluggable busy-indicator styles (@OutThisLife, #13610), opt-in auto-resume of last session, expanded light-terminal auto-detection, session delete from `/resume` picker with `d`, modified mouse-wheel line scroll, and a `/mouse` toggle that kills ConPTY's phantom mouse injection (@kevin-ho). ([#17175](https://github.com/NousResearch/hermes-agent/pull/17175), [#17286](https://github.com/NousResearch/hermes-agent/pull/17286), [#17150](https://github.com/NousResearch/hermes-agent/pull/17150), [#17130](https://github.com/NousResearch/hermes-agent/pull/17130), [#17113](https://github.com/NousResearch/hermes-agent/pull/17113), [#17668](https://github.com/NousResearch/hermes-agent/pull/17668), [#17669](https://github.com/NousResearch/hermes-agent/pull/17669), [#15488](https://github.com/NousResearch/hermes-agent/pull/15488))

- **Observability + achievements plugins** — bundled Langfuse observability plugin (salvage #16845) + bundled hermes-achievements plugin that scans full session history. ([#16917](https://github.com/NousResearch/hermes-agent/pull/16917), [#17754](https://github.com/NousResearch/hermes-agent/pull/17754))

- **TTS provider registry + Piper local TTS** — pluggable `tts.providers.<name>` registry; Piper ships as a native local TTS provider. (Closes #8508.) ([#17843](https://github.com/NousResearch/hermes-agent/pull/17843), [#17885](https://github.com/NousResearch/hermes-agent/pull/17885))

- **Vercel Sandbox backend** — Vercel sandboxes as an execute_code/terminal backend (@kshitijk4poor). ([#17445](https://github.com/NousResearch/hermes-agent/pull/17445))

- **Secret redaction off by default** — default flipped to off. Prevents the long-standing patch-corruption incidents where fake secret-shaped substrings mangled tool outputs. Opt in via `redaction.enabled: true` when you need it. ([#16794](https://github.com/NousResearch/hermes-agent/pull/16794))

- **Cold-start performance** — visible TUI cold start cut **~57%** via lazy agent init (@OutThisLife), lazy imports of OpenAI / Anthropic / Firecrawl / account_usage, mtime-cached `load_config()`, memoized `get_tool_definitions()` with TTL-cached `check_fn` results, precompiled dangerous-command patterns. ([#17190](https://github.com/NousResearch/hermes-agent/pull/17190), [#17046](https://github.com/NousResearch/hermes-agent/pull/17046), [#17041](https://github.com/NousResearch/hermes-agent/pull/17041), [#17098](https://github.com/NousResearch/hermes-agent/pull/17098), [#17206](https://github.com/NousResearch/hermes-agent/pull/17206))

- **Configurable prompt cache TTL** — `prompt_caching.cache_ttl` (5m default, 1h opt-in — cost savings for bursty sessions that keep cache warm). Salvage of #12659. ([#15065](https://github.com/NousResearch/hermes-agent/pull/15065))

---

## 🧠 Autonomous Curator & Self-Improvement Loop

### Curator — autonomous skill maintenance
- **`hermes curator` as a background agent** — runs on the gateway's cron ticker, 7-day cycle by default, umbrella-first prompt, inherits parent config, unbounded iterations ([#17277](https://github.com/NousResearch/hermes-agent/pull/17277) — issue #7816)
- **Per-run reports** — `logs/curator/run.json` + `REPORT.md` per cycle ([#17307](https://github.com/NousResearch/hermes-agent/pull/17307))
- **Consolidated vs pruned classification** — archived skills split with model + heuristic ([#17941](https://github.com/NousResearch/hermes-agent/pull/17941))
- **`hermes curator status`** — ranks skills by usage, shows most-used and least-used ([#18033](https://github.com/NousResearch/hermes-agent/pull/18033))
- **Unified under `auxiliary.curator`** — pick the model in `hermes model`, configure from the dashboard ([#17868](https://github.com/NousResearch/hermes-agent/pull/17868))
- **Documentation** — dedicated curator feature page on the docs site ([#17563](https://github.com/NousResearch/hermes-agent/pull/17563))
- Fix: seed defaults on update, create `logs/curator/` directory, defer fire import ([#17927](https://github.com/NousResearch/hermes-agent/pull/17927))
- Fix: scan nested archive subdirs in `restore_skill` (@0xDevNinja) ([#17951](https://github.com/NousResearch/hermes-agent/pull/17951))
- Fix: use actual skill activity in curator status (@y0shua1ee) ([#17953](https://github.com/NousResearch/hermes-agent/pull/17953))
- Fix: `skill_manage` refuses writes on pinned skills; pinning now blocks curator writes ([#17562](https://github.com/NousResearch/hermes-agent/pull/17562), [#17578](https://github.com/NousResearch/hermes-agent/pull/17578))
- Fix: `bump_use()` wired into skill invocation + preload + skill_view (salvage #17782) ([#17932](https://github.com/NousResearch/hermes-agent/pull/17932))

### Self-improvement loop (background review fork)
- **Class-first skill-review prompt** — rubric-based grading rather than free-form "should this update" ([#16026](https://github.com/NousResearch/hermes-agent/pull/16026))
- **Active-update bias** — prefers updating skills the agent just loaded, handles `references/` + `templates/` sub-files ([#17213](https://github.com/NousResearch/hermes-agent/pull/17213))
- **Fork inherits parent's live runtime** — provider, model, credentials actually propagate now ([#16099](https://github.com/NousResearch/hermes-agent/pull/16099))
- **Scoped toolsets** — review fork restricted to memory + skills (no shell, no web) ([#16569](https://github.com/NousResearch/hermes-agent/pull/16569))
- **Clean shutdown** — background review memory providers exit properly (salvage #15289) ([#16204](https://github.com/NousResearch/hermes-agent/pull/16204))
- **Clean context** — prior-history tool messages excluded from review summary (salvage #14967) ([#15057](https://github.com/NousResearch/hermes-agent/pull/15057))

---

## 🧩 Skills Ecosystem

### Skill integrations — newly bundled or promoted
- **ComfyUI v5** — official CLI + REST + hardware-gated local install; **moved from optional to built-in** ([#17610](https://github.com/NousResearch/hermes-agent/pull/17610), [#17631](https://github.com/NousResearch/hermes-agent/pull/17631), [#17734](https://github.com/NousResearch/hermes-agent/pull/17734), [#17612](https://github.com/NousResearch/hermes-agent/pull/17612))
- **TouchDesigner-MCP** — **bundled by default** ([#16753](https://github.com/NousResearch/hermes-agent/pull/16753) — @kshitijk4poor), expanded with GLSL, post-FX, audio, geometry references ([#16624](https://github.com/NousResearch/hermes-agent/pull/16624)), 9 new reference docs ([#16768](https://github.com/NousResearch/hermes-agent/pull/16768) — @SHL0MS)
- **Humanizer** — strips AI-isms from text ([#16787](https://github.com/NousResearch/hermes-agent/pull/16787))
- **claude-design** — HTML artifact skill with disambiguation from other design skills ([#16358](https://github.com/NousResearch/hermes-agent/pull/16358))
- **design-md** — Google's DESIGN.md spec skill ([#14876](https://github.com/NousResearch/hermes-agent/pull/14876))
- **airtable** — salvaged skill + skill API keys wired into `.env` (#15838) ([#16291](https://github.com/NousResearch/hermes-agent/pull/16291))
- **pretext** — creative browser demos with @chenglou/pretext ([#17259](https://github.com/NousResearch/hermes-agent/pull/17259))
- **spike** + **sketch** — throwaway experiments + HTML mockups, adapted from gsd-build ([#17421](https://github.com/NousResearch/hermes-agent/pull/17421))

### Skills UX
- **Install skills from a direct HTTP(S) URL** — `hermes skills install <url>` ([#16323](https://github.com/NousResearch/hermes-agent/pull/16323))
- **`/reload-skills`** slash command (salvage #17670) ([#17744](https://github.com/NousResearch/hermes-agent/pull/17744))
- **`hermes skills list`** shows enabled/disabled status ([#16129](https://github.com/NousResearch/hermes-agent/pull/16129))
- **`skill_manage` refuses writes on pinned skills** ([#17562](https://github.com/NousResearch/hermes-agent/pull/17562))
- **`skill_manage` edits external_dirs skills in place** (salvage #9966) ([#17512](https://github.com/NousResearch/hermes-agent/pull/17512), [#17289](https://github.com/NousResearch/hermes-agent/pull/17289))
- Fix: inline-shell rendering in `skill_view` ([#15376](https://github.com/NousResearch/hermes-agent/pull/15376))
- Fix: exclude `.archive/` from skill index walk (salvage #17639) ([#17931](https://github.com/NousResearch/hermes-agent/pull/17931))
- Fix: dedicated docs page per bundled + optional skill ([#14929](https://github.com/NousResearch/hermes-agent/pull/14929))
- Fix: `google-workspace` shared HERMES_HOME helper + ship deps as optional extra ([#15405](https://github.com/NousResearch/hermes-agent/pull/15405))
- Fix: auto-wrap ASCII-art code blocks in generated skill pages ([#16497](https://github.com/NousResearch/hermes-agent/pull/16497))
- Point agent at `hermes-agent` skill + docs site for Hermes questions ([#16535](https://github.com/NousResearch/hermes-agent/pull/16535))

---

## 🏗️ Core Agent & Architecture

### Provider & Model Support

#### New providers
- **GMI Cloud** — first-class API-key provider on par with Arcee/Kilocode/Xiaomi (salvage of #11955 — @isaachuangGMICLOUD) ([#16663](https://github.com/NousResearch/hermes-agent/pull/16663))
- **Azure AI Foundry** — auto-detection, full wiring ([#15845](https://github.com/NousResearch/hermes-agent/pull/15845))
- **LM Studio** — upgraded from custom-endpoint alias to first-class provider: dedicated auth, doctor checks, reasoning transport, live `/models` (salvage of #17061 — @kshitijk4poor) ([#17102](https://github.com/NousResearch/hermes-agent/pull/17102))
- **MiniMax OAuth** — PKCE browser flow with full OAuth integration (salvage #15203) ([#17524](https://github.com/NousResearch/hermes-agent/pull/17524))
- **Tencent Tokenhub** — new provider (salvage of #16860) ([#16960](https://github.com/NousResearch/hermes-agent/pull/16960))

#### Model catalog
- **Remote model catalog manifest** — OpenRouter + Nous Portal catalogs pulled from remote manifest so new models show up without a release ([#16033](https://github.com/NousResearch/hermes-agent/pull/16033))
- `openai/gpt-5.5` and `gpt-5.5-pro` added to OpenRouter + Nous Portal ([#15343](https://github.com/NousResearch/hermes-agent/pull/15343))
- `deepseek-v4-pro` and `deepseek-v4-flash` added ([#14934](https://github.com/NousResearch/hermes-agent/pull/14934))
- `qwen3.6-plus` added to Alibaba-supported models ([#16896](https://github.com/NousResearch/hermes-agent/pull/16896))
- Gemini free-tier keys blocked at setup with 429 guidance surfacing ([#15100](https://github.com/NousResearch/hermes-agent/pull/15100))

#### Model configuration
- **Configurable `prompt_caching.cache_ttl`** — 5m default, 1h opt-in (salvage #12659) ([#15065](https://github.com/NousResearch/hermes-agent/pull/15065))
- `/fast` whitelist broadened to all OpenAI + Anthropic models ([#16883](https://github.com/NousResearch/hermes-agent/pull/16883))
- `auxiliary.extra_body.reasoning` translates into Codex Responses API ([#17004](https://github.com/NousResearch/hermes-agent/pull/17004))
- `hermes fallback` command for managing fallback providers ([#16052](https://github.com/NousResearch/hermes-agent/pull/16052))

### Agent Loop & Conversation
- **Native multimodal image routing** — based on model vision capability, not provider defaults ([#16506](https://github.com/NousResearch/hermes-agent/pull/16506))
- **Delegate `child_timeout_seconds` default bumped to 600s** ([#14809](https://github.com/NousResearch/hermes-agent/pull/14809))
- **Diagnostic dump when subagent times out with 0 API calls** ([#15105](https://github.com/NousResearch/hermes-agent/pull/15105))
- **Gateway busts cached agent on compression/context_length config edits** ([#17008](https://github.com/NousResearch/hermes-agent/pull/17008))
- **Opt-in runtime-metadata footer on final replies** ([#17026](https://github.com/NousResearch/hermes-agent/pull/17026))
- `/reload-mcp` awareness — rebuild cached agents + prompt-cache cost confirmation ([#17729](https://github.com/NousResearch/hermes-agent/pull/17729))
- Fix: repair CamelCase + `_tool` suffix tool-call emissions ([#15124](https://github.com/NousResearch/hermes-agent/pull/15124))
- Fix: retry on `json.JSONDecodeError` instead of treating as local validation error ([#15107](https://github.com/NousResearch/hermes-agent/pull/15107))
- Fix: handle unescaped control chars in `tool_call.arguments` ([#15356](https://github.com/NousResearch/hermes-agent/pull/15356))
- Fix: ordering fix in `_copy_reasoning_content_for_api` — cross-provider reasoning isolation (@Zjianru) ([#15749](https://github.com/NousResearch/hermes-agent/pull/15749))
- Fix: inject empty `reasoning_content` for DeepSeek/Kimi `tool_calls` unconditionally (@Zjianru) ([#15762](https://github.com/NousResearch/hermes-agent/pull/15762))
- Fix: persist streamed `reasoning_content` on assistant turns (#16844) ([#16892](https://github.com/NousResearch/hermes-agent/pull/16892))
- Fix: cancel coroutine on timeout so worker thread exits; full traceback on tool failure ([#17428](https://github.com/NousResearch/hermes-agent/pull/17428))
- Fix: isolate `get_tool_definitions` quiet_mode cache + dedup LCM injection (#17335) ([#17889](https://github.com/NousResearch/hermes-agent/pull/17889))
- Fix: serialize concurrent `hermes_tools` RPC calls from `execute_code` (#17770) ([#17894](https://github.com/NousResearch/hermes-agent/pull/17894), [#17902](https://github.com/NousResearch/hermes-agent/pull/17902))
- Fix: rename `[SYSTEM:` → `[IMPORTANT:` in all user-injected markers (dodges Azure content filter) ([#16114](https://github.com/NousResearch/hermes-agent/pull/16114))

### Compression
- **Retry summary on main model for unknown errors before giving up** ([#16774](https://github.com/NousResearch/hermes-agent/pull/16774))
- **Notify users when configured aux model fails even if main-model fallback recovers** ([#16775](https://github.com/NousResearch/hermes-agent/pull/16775))
- `/compress` wrapped in `_busy_command` to block input during compression ([#15388](https://github.com/NousResearch/hermes-agent/pull/15388))
- Fix: reserve system + tools headroom when aux binds threshold ([#15631](https://github.com/NousResearch/hermes-agent/pull/15631))
- Fix: use text-char sum for multimodal token estimation in `_find_tail_cut_by_tokens` ([#16369](https://github.com/NousResearch/hermes-agent/pull/16369))

### Session, Memory & State
- **Trigram FTS5 index for CJK search, replace LIKE fallback** (@alt-glitch) ([#16651](https://github.com/NousResearch/hermes-agent/pull/16651))
- **Index `tool_name` + `tool_calls` in FTS5, with repair + migration** (salvages #16866) ([#16914](https://github.com/NousResearch/hermes-agent/pull/16914))
- **Checkpoints: auto-prune orphan and stale shadow repos at startup** ([#16303](https://github.com/NousResearch/hermes-agent/pull/16303))
- **Memory providers notified on mid-process session_id rotation** (#6672) ([#17409](https://github.com/NousResearch/hermes-agent/pull/17409))
- Fix: quote underscored terms in FTS5 query sanitization ([#16915](https://github.com/NousResearch/hermes-agent/pull/16915))
- Fix: resolve viking_read 500/412 on file URIs + pseudo-summary URIs (salvage #5886) ([#17869](https://github.com/NousResearch/hermes-agent/pull/17869))
- Fix: skip external-provider sync on interrupted turns ([#15395](https://github.com/NousResearch/hermes-agent/pull/15395))
- Fix: close embedded Hindsight async client cleanly (salvage #14605) ([#16209](https://github.com/NousResearch/hermes-agent/pull/16209))
- Fix: pass session transcript to `shutdown_memory_provider` on gateway + CLI (#15165) ([#16571](https://github.com/NousResearch/hermes-agent/pull/16571))
- Fix: write-origin metadata seam ([#15346](https://github.com/NousResearch/hermes-agent/pull/15346))
- Fix: preserve symlinks during atomic file writes ([#16980](https://github.com/NousResearch/hermes-agent/pull/16980))
- Refactor: remove `flush_memories` entirely ([#15696](https://github.com/NousResearch/hermes-agent/pull/15696))

### Auxiliary models
- Fix: surface auxiliary failures in UI (previously silent) ([#15324](https://github.com/NousResearch/hermes-agent/pull/15324))
- Fix: surface title-gen auxiliary failures instead of silently dropping ([#16371](https://github.com/NousResearch/hermes-agent/pull/16371))
- Fix: generalize unsupported-parameter detector and harden `max_tokens` retry ([#15633](https://github.com/NousResearch/hermes-agent/pull/15633))

---

## 📱 Messaging Platforms (Gateway)

### New Platforms
- **Microsoft Teams (19th platform)** — as a plugin, + xdist collision guard ([#17828](https://github.com/NousResearch/hermes-agent/pull/17828))
- **Yuanbao (Tencent 元宝, 18th platform)** — native adapter with text + media delivery ([#16298](https://github.com/NousResearch/hermes-agent/pull/16298), [#17424](https://github.com/NousResearch/hermes-agent/pull/17424), [#16880](https://github.com/NousResearch/hermes-agent/pull/16880))

### Pluggable Gateway Platforms
- **Drop-in messaging adapters** — the gateway is now a plugin host for platforms (salvage of #17664) ([#17751](https://github.com/NousResearch/hermes-agent/pull/17751))

### Telegram
- **Chat allowlists for groups and forums** (@web3blind) ([#15027](https://github.com/NousResearch/hermes-agent/pull/15027))
- **Send fresh finals for stale preview streams** (port openclaw#72038) ([#16261](https://github.com/NousResearch/hermes-agent/pull/16261))
- **Render markdown tables as row-group bullets + prompt hint** ([#16997](https://github.com/NousResearch/hermes-agent/pull/16997))
- Document fallback in centralized audio routing ([#17833](https://github.com/NousResearch/hermes-agent/pull/17833))
- Native multi-image sending ([#17909](https://github.com/NousResearch/hermes-agent/pull/17909))

### Discord
- **Opt-in toolsets + ID injection + tool split + Feishu wiring** (salvage #15457, #15458) ([#15610](https://github.com/NousResearch/hermes-agent/pull/15610), [#15613](https://github.com/NousResearch/hermes-agent/pull/15613))
- Fix: coerce `limit` parameter to int before `min()` call ([#16319](https://github.com/NousResearch/hermes-agent/pull/16319))

### Slack
- **Register every gateway command as a native slash (Discord/Telegram parity)** ([#16164](https://github.com/NousResearch/hermes-agent/pull/16164))
- **`strict_mention` config** — prevents thread auto-engagement ([#16193](https://github.com/NousResearch/hermes-agent/pull/16193))
- **`channel_skill_bindings`** — bind specific skills to specific Slack channels ([#16283](https://github.com/NousResearch/hermes-agent/pull/16283))

### Signal
- **Native formatting** — markdown → bodyRanges, reply quotes, reactions ([#17417](https://github.com/NousResearch/hermes-agent/pull/17417))
- Native multi-image sending ([#17909](https://github.com/NousResearch/hermes-agent/pull/17909))

### Feishu / Mattermost / Email / Signal
- All participate in **native multi-image sending** ([#17909](https://github.com/NousResearch/hermes-agent/pull/17909))

### Gateway Core
- **Centralized audio routing + FLAC support + Telegram doc fallback** ([#17833](https://github.com/NousResearch/hermes-agent/pull/17833))
- **Native multi-image sending** across Telegram, Discord, Slack, Mattermost, Email, Signal ([#17909](https://github.com/NousResearch/hermes-agent/pull/17909))
- **Make hygiene hard message limit configurable** ([#17000](https://github.com/NousResearch/hermes-agent/pull/17000))
- **Opt-in runtime-metadata footer on final replies** ([#17026](https://github.com/NousResearch/hermes-agent/pull/17026))
- **`pre_gateway_dispatch` hook** — plugins can intercept before dispatch ([#15050](https://github.com/NousResearch/hermes-agent/pull/15050))
- **`pre_approval_request` / `post_approval_response` hooks** ([#16776](https://github.com/NousResearch/hermes-agent/pull/16776))
- Fix: timeouts — guard `load_config()` call against runtime exceptions ([#16318](https://github.com/NousResearch/hermes-agent/pull/16318))
- Fix: support passing handler tools via registry ([#15613](https://github.com/NousResearch/hermes-agent/pull/15613))

---

## 🔧 Tool System

### Plugin-first architecture
- **Pluggable gateway platforms** — platforms can ship as plugins ([#17751](https://github.com/NousResearch/hermes-agent/pull/17751))
- **Microsoft Teams as first plugin-shipped platform** ([#17828](https://github.com/NousResearch/hermes-agent/pull/17828))
- **`pre_gateway_dispatch` hook** ([#15050](https://github.com/NousResearch/hermes-agent/pull/15050))
- **`pre_approval_request` + `post_approval_response` hooks** ([#16776](https://github.com/NousResearch/hermes-agent/pull/16776))
- **`duration_ms` on `post_tool_call`** (inspired by Claude Code 2.1.119) ([#15429](https://github.com/NousResearch/hermes-agent/pull/15429))
- **Bundled plugins**: Spotify ([#15174](https://github.com/NousResearch/hermes-agent/pull/15174)), Google Meet ([#16364](https://github.com/NousResearch/hermes-agent/pull/16364)), Langfuse observability ([#16917](https://github.com/NousResearch/hermes-agent/pull/16917)), hermes-achievements ([#17754](https://github.com/NousResearch/hermes-agent/pull/17754))
- **Page-scoped plugin slots for built-in dashboard pages** ([#15658](https://github.com/NousResearch/hermes-agent/pull/15658))
- **Declarative plugin installation for NixOS module** (@alt-glitch) ([#15953](https://github.com/NousResearch/hermes-agent/pull/15953))

### Browser
- **CDP supervisor** — dialog detection + response + cross-origin iframe eval ([#14540](https://github.com/NousResearch/hermes-agent/pull/14540))
- **Auto-spawn local Chromium for LAN/localhost URLs** when cloud provider is configured ([#16136](https://github.com/NousResearch/hermes-agent/pull/16136))

### Execute code / Terminal
- **Vercel Sandbox backend** for `execute_code` / terminal (@kshitijk4poor) ([#17445](https://github.com/NousResearch/hermes-agent/pull/17445))
- **Collapse subagent `task_id`s to shared container** ([#16177](https://github.com/NousResearch/hermes-agent/pull/16177))
- **Docker: run container as host user** to avoid root-owned bind mounts (@benbarclay) ([#17305](https://github.com/NousResearch/hermes-agent/pull/17305))
- Fix: safely quote `~/` subpaths in wrapped `cd` commands ([#15394](https://github.com/NousResearch/hermes-agent/pull/15394))
- Fix: close file descriptor in `LocalEnvironment._update_cwd` ([#17300](https://github.com/NousResearch/hermes-agent/pull/17300))
- Fix: SSH — prevent tar from overwriting remote home dir permissions ([#17898](https://github.com/NousResearch/hermes-agent/pull/17898), [#17867](https://github.com/NousResearch/hermes-agent/pull/17867))

### Image generation
- See Provider section for updates; no new image providers this window.

### TTS / Voice
- **Pluggable TTS provider registry** under `tts.providers.<name>` ([#17843](https://github.com/NousResearch/hermes-agent/pull/17843))
- **Piper** as native local TTS provider (closes #8508) ([#17885](https://github.com/NousResearch/hermes-agent/pull/17885))
- **Voice mode CLI parity in the TUI** — VAD loop + TTS + crash forensics ([#14810](https://github.com/NousResearch/hermes-agent/pull/14810))
- Fix: vision — use HERMES_HOME-based cache dir instead of cwd ([#17719](https://github.com/NousResearch/hermes-agent/pull/17719))

### Cron
- **Honor `hermes tools` config for the cron platform** ([#14798](https://github.com/NousResearch/hermes-agent/pull/14798))
- **Per-job `workdir`** — project-aware cron runs ([#15110](https://github.com/NousResearch/hermes-agent/pull/15110))
- **`context_from` field** — chain cron job outputs ([#15606](https://github.com/NousResearch/hermes-agent/pull/15606))
- Fix: promote `croniter` to a core dependency ([#17577](https://github.com/NousResearch/hermes-agent/pull/17577))

### Web search
- **Expose `limit` for `web_search`** ([#16934](https://github.com/NousResearch/hermes-agent/pull/16934))

### Maps
- Fix: include seconds in timezone UTC offset output ([#16300](https://github.com/NousResearch/hermes-agent/pull/16300))

### Approvals
- **Hardline blocklist for unrecoverable commands** ([#15878](https://github.com/NousResearch/hermes-agent/pull/15878))
- Perf: precompile DANGEROUS_PATTERNS and HARDLINE_PATTERNS ([#17206](https://github.com/NousResearch/hermes-agent/pull/17206))

### ACP
- **Advertise and forward image prompts** ([#18030](https://github.com/NousResearch/hermes-agent/pull/18030))

### API Server
- **POST `/v1/runs/{run_id}/stop`** (salvage of #15656) ([#15842](https://github.com/NousResearch/hermes-agent/pull/15842))
- **Expose run status for external UIs** (#17085) ([#17458](https://github.com/NousResearch/hermes-agent/pull/17458))

### Nix
- **Declarative plugin installation for NixOS module** (@alt-glitch) ([#15953](https://github.com/NousResearch/hermes-agent/pull/15953))
- Fix: use `--rebuild` in fix-lockfiles to bypass cached FOD store paths ([#15444](https://github.com/NousResearch/hermes-agent/pull/15444))
- Fix: `extraPackages` now actually works via per-user profile ([#17047](https://github.com/NousResearch/hermes-agent/pull/17047))
- Fix: refresh web/ npm-deps hash to unblock main builds ([#17174](https://github.com/NousResearch/hermes-agent/pull/17174))
- Fix: replace magic-nix-cache with Cachix ([#17928](https://github.com/NousResearch/hermes-agent/pull/17928))

---

## 🖥️ TUI

### New features
- **LaTeX rendering** (@austinpickett) ([#17175](https://github.com/NousResearch/hermes-agent/pull/17175))
- **`/reload` .env hot-reload** — ported from the classic CLI ([#17286](https://github.com/NousResearch/hermes-agent/pull/17286))
- **Pluggable busy-indicator styles** (@OutThisLife, #13610) ([#17150](https://github.com/NousResearch/hermes-agent/pull/17150))
- **Opt-in auto-resume of the most recent session** (@OutThisLife) ([#17130](https://github.com/NousResearch/hermes-agent/pull/17130))
- **Expanded light-terminal auto-detection** — `HERMES_TUI_THEME` + background hex (@OutThisLife) ([#17113](https://github.com/NousResearch/hermes-agent/pull/17113))
- **Delete sessions from `/resume` picker with `d`** (@OutThisLife) ([#17668](https://github.com/NousResearch/hermes-agent/pull/17668))
- **Line-by-line scroll on modified mouse wheel** (@OutThisLife) ([#17669](https://github.com/NousResearch/hermes-agent/pull/17669))
- **Delete queued message while editing with ctrl-x / cancel with esc** (@OutThisLife) ([#16707](https://github.com/NousResearch/hermes-agent/pull/16707))
- **Per-section visibility for the details accordion** (@OutThisLife) ([#14968](https://github.com/NousResearch/hermes-agent/pull/14968))
- **Voice mode CLI parity** — VAD loop + TTS + crash forensics ([#14810](https://github.com/NousResearch/hermes-agent/pull/14810))
- **Contextual first-touch hints ported to TUI** — `/busy`, `/verbose` ([#16054](https://github.com/NousResearch/hermes-agent/pull/16054))
- **Mini help menu on `?` in the input field** (@ethernet8023) ([#18043](https://github.com/NousResearch/hermes-agent/pull/18043))

### Fixes
- Fix: proactive mouse disable on ConPTY + `/mouse` toggle command (@kevin-ho, WSL2 ghost-mouse fix) ([#15488](https://github.com/NousResearch/hermes-agent/pull/15488))
- Fix: restore skills search RPC ([#15870](https://github.com/NousResearch/hermes-agent/pull/15870))
- Perf: cache text measurements across yoga flex re-passes ([#14818](https://github.com/NousResearch/hermes-agent/pull/14818))
- Perf: stabilize long-session scrolling ([#15926](https://github.com/NousResearch/hermes-agent/pull/15926))
- Perf: lazily seed virtual history heights ([#16523](https://github.com/NousResearch/hermes-agent/pull/16523))
- Perf: cut visible cold start ~57% with lazy agent init ([#17190](https://github.com/NousResearch/hermes-agent/pull/17190))

---

## 🖱️ CLI & User Experience

### New commands
- **`hermes -z <prompt>`** — non-interactive one-shot mode ([#15702](https://github.com/NousResearch/hermes-agent/pull/15702))
- **`hermes -z` with `--model` / `--provider` / `HERMES_INFERENCE_MODEL`** ([#15704](https://github.com/NousResearch/hermes-agent/pull/15704))
- **`hermes update --check`** preflight flag ([#15841](https://github.com/NousResearch/hermes-agent/pull/15841))
- **`hermes fallback`** command for managing fallback providers ([#16052](https://github.com/NousResearch/hermes-agent/pull/16052))
- **`/busy`** slash command for busy input mode ([#15382](https://github.com/NousResearch/hermes-agent/pull/15382))
- **`/busy` input mode 'steer'** as a third option ([#16279](https://github.com/NousResearch/hermes-agent/pull/16279))
- **`/btw` as alias for `/background`** ([#16053](https://github.com/NousResearch/hermes-agent/pull/16053))
- **`/reload-skills`** slash command (salvage #17670) ([#17744](https://github.com/NousResearch/hermes-agent/pull/17744))
- **Surface `/queue`, `/bg`, `/steer` in agent-running placeholder** ([#16118](https://github.com/NousResearch/hermes-agent/pull/16118))

### Setup / onboarding
- **Auto-reconfigure on existing installs** ([#15879](https://github.com/NousResearch/hermes-agent/pull/15879))
- **Contextual first-touch hints for `/busy` and `/verbose`** ([#16046](https://github.com/NousResearch/hermes-agent/pull/16046))
- **Cost-saving tips from the April 30 tip-of-the-day** ([#17841](https://github.com/NousResearch/hermes-agent/pull/17841))
- **Hyperlink startup banner title to the latest GitHub Release** ([#14945](https://github.com/NousResearch/hermes-agent/pull/14945))

### Update / backup
- **Snapshot pairing data before `git pull`** ([#16383](https://github.com/NousResearch/hermes-agent/pull/16383))
- **Auto-backup HERMES_HOME before `hermes update`** (opt-in, off by default) ([#16539](https://github.com/NousResearch/hermes-agent/pull/16539), [#16566](https://github.com/NousResearch/hermes-agent/pull/16566))
- **Exclude `checkpoints/` from backups** ([#16572](https://github.com/NousResearch/hermes-agent/pull/16572))
- **Exclude SQLite WAL/SHM/journal sidecars from backups** ([#16576](https://github.com/NousResearch/hermes-agent/pull/16576))
- **Installer FHS layout for root installs on Linux** ([#15608](https://github.com/NousResearch/hermes-agent/pull/15608))
- Fix: kill stale dashboards instead of warning ([#17832](https://github.com/NousResearch/hermes-agent/pull/17832))
- Fix: show correct update status on nix-built hermes ([#17550](https://github.com/NousResearch/hermes-agent/pull/17550))

### Slash-command housekeeping
- Refactor: drop `/provider`, `/plan` handler, and clean up slash registry ([#15047](https://github.com/NousResearch/hermes-agent/pull/15047))
- Refactor: drop `persist_session` plumbing + fix broken `/btw` mid-turn bypass ([#16075](https://github.com/NousResearch/hermes-agent/pull/16075))

### OpenClaw migration (for folks coming from OpenClaw)
- **Hardened OpenClaw import** — plan-first apply, redaction, pre-migration backup ([#16911](https://github.com/NousResearch/hermes-agent/pull/16911))
- Fix: case-preserving brand rewrite + one-time `~/.openclaw` residue banner ([#16327](https://github.com/NousResearch/hermes-agent/pull/16327))
- Fix: resolve `openclaw` workspace files from `agents.defaults.workspace` ([#16879](https://github.com/NousResearch/hermes-agent/pull/16879))
- Fix: resolve model aliases against real OpenClaw catalog schema (salvage #16778) ([#16977](https://github.com/NousResearch/hermes-agent/pull/16977))

---

## 📊 Web Dashboard

- **Models tab** — rich per-model analytics ([#17745](https://github.com/NousResearch/hermes-agent/pull/17745))
- **Configure main + auxiliary models from the Models page** ([#17802](https://github.com/NousResearch/hermes-agent/pull/17802))
- **Dashboard Chat tab — xterm.js + JSON-RPC sidecar** (supersedes #12710 + #13379, @OutThisLife) ([#14890](https://github.com/NousResearch/hermes-agent/pull/14890))
- **Dashboard layout refresh** (@austinpickett) ([#14899](https://github.com/NousResearch/hermes-agent/pull/14899))
- **`--stop` and `--status` flags** on the dashboard CLI ([#17840](https://github.com/NousResearch/hermes-agent/pull/17840))
- **Page-scoped plugin slots for built-in pages** ([#15658](https://github.com/NousResearch/hermes-agent/pull/15658))
- Fix: replace all buttons for design system buttons ([#17007](https://github.com/NousResearch/hermes-agent/pull/17007))

---

## ⚡ Performance

- **TUI visible cold start cut ~57%** via lazy agent init ([#17190](https://github.com/NousResearch/hermes-agent/pull/17190))
- **Lazy-import OpenAI, Anthropic, Firecrawl, account_usage** ([#17046](https://github.com/NousResearch/hermes-agent/pull/17046))
- **mtime-cache `load_config()` and `read_raw_config()`** ([#17041](https://github.com/NousResearch/hermes-agent/pull/17041))
- **Memoize `get_tool_definitions()` + TTL-cache `check_fn` results** ([#17098](https://github.com/NousResearch/hermes-agent/pull/17098))
- **Precompile DANGEROUS_PATTERNS and HARDLINE_PATTERNS** ([#17206](https://github.com/NousResearch/hermes-agent/pull/17206))
- **Cache Ink text measurements across yoga flex re-passes** ([#14818](https://github.com/NousResearch/hermes-agent/pull/14818))
- **Stabilize long-session scrolling** ([#15926](https://github.com/NousResearch/hermes-agent/pull/15926))
- **Lazily seed virtual history heights** ([#16523](https://github.com/NousResearch/hermes-agent/pull/16523))

---

## 🔒 Security & Reliability

- **Secret redaction off by default** — stops corrupting patches / API payloads with fake-key substitutions. Opt in via `redaction.enabled: true` ([#16794](https://github.com/NousResearch/hermes-agent/pull/16794))
- **`[SYSTEM:` → `[IMPORTANT:`** in all user-injected markers (Azure content filter dodge) ([#16114](https://github.com/NousResearch/hermes-agent/pull/16114))
- **Hardline blocklist for unrecoverable commands** ([#15878](https://github.com/NousResearch/hermes-agent/pull/15878))
- **Canonical `mask_secret` helper; fix status.py DIM drift** ([#17207](https://github.com/NousResearch/hermes-agent/pull/17207))
- **Sweep expired paste.rs uploads on a real timer** ([#16431](https://github.com/NousResearch/hermes-agent/pull/16431))
- **Preserve symlinks during atomic file writes** ([#16980](https://github.com/NousResearch/hermes-agent/pull/16980))
- **Probe `/dev/tty` by opening it, not bare existence** ([#17024](https://github.com/NousResearch/hermes-agent/pull/17024))

---

## 🐛 Notable Bug Fixes

This window includes 360 `fix:` PRs. Selected highlights from across the stack:

- **Background review fork inherits parent's live runtime** — provider/model/creds now propagate correctly ([#16099](https://github.com/NousResearch/hermes-agent/pull/16099))
- **Hindsight configurable `HINDSIGHT_TIMEOUT` env var** ([#15077](https://github.com/NousResearch/hermes-agent/pull/15077))
- **Tools: normalize numeric entries + clear stale `no_mcp` in `_save_platform_tools`** ([#15607](https://github.com/NousResearch/hermes-agent/pull/15607))
- **MCP: rewrite `definitions` refs to `$defs` in input schemas** — closes provider-side 400s
- **Azure content filter compatibility** — renamed `[SYSTEM:` markers so Azure's content filter stops flagging them ([#16114](https://github.com/NousResearch/hermes-agent/pull/16114))
- **Vision cache uses HERMES_HOME instead of cwd** ([#17719](https://github.com/NousResearch/hermes-agent/pull/17719))
- **FTS5 search** — tool_name + tool_calls indexing with repair + migration ([#16914](https://github.com/NousResearch/hermes-agent/pull/16914))
- **Streaming reasoning persists on assistant turns** ([#16892](https://github.com/NousResearch/hermes-agent/pull/16892))
- **execute_code concurrent RPC serialization** (#17770) ([#17894](https://github.com/NousResearch/hermes-agent/pull/17894), [#17902](https://github.com/NousResearch/hermes-agent/pull/17902))
- **Background reviewer scoped to memory + skills toolsets** — no more accidental web/shell escapes ([#16569](https://github.com/NousResearch/hermes-agent/pull/16569))
- **Compression recovery** — retry on main before giving up; notify user when aux fails ([#16774](https://github.com/NousResearch/hermes-agent/pull/16774), [#16775](https://github.com/NousResearch/hermes-agent/pull/16775))
- **`croniter` promoted to a core dependency** ([#17577](https://github.com/NousResearch/hermes-agent/pull/17577))
- **Discord tool `limit` parameter coerced to int** before `min()` call ([#16319](https://github.com/NousResearch/hermes-agent/pull/16319))
- **Yuanbao messaging platform entrance fix** ([#16880](https://github.com/NousResearch/hermes-agent/pull/16880))
- **ACP advertise and forward image prompts** ([#18030](https://github.com/NousResearch/hermes-agent/pull/18030))
- **DeepSeek / Kimi reasoning content isolation** across cross-provider histories (@Zjianru) ([#15749](https://github.com/NousResearch/hermes-agent/pull/15749), [#15762](https://github.com/NousResearch/hermes-agent/pull/15762))
- **Preserve reasoning_content replay on DeepSeek v4 + Kimi/Moonshot thinking** ([#18045](https://github.com/NousResearch/hermes-agent/pull/18045))

The vast majority of the 360 fixes landed in the streaming/compression/tool-calling paths across all providers — DeepSeek, Kimi, Moonshot, GLM, Qwen, MiniMax, Gemini, Anthropic, OpenAI — alongside TUI polish (resize, scroll, sticky-prompt) and gateway platform-specific edge cases.

---

## 🧪 Testing & CI

- Hermetic test parity (`scripts/run_tests.sh`) held across this window
- **Microsoft Teams xdist collision guard** — prevents worker collisions when Teams platform tests run in parallel ([#17828](https://github.com/NousResearch/hermes-agent/pull/17828))
- Chore: remove unused imports and dead locals (ruff F401, F841) ([#17010](https://github.com/NousResearch/hermes-agent/pull/17010))

---

## 📚 Documentation

- **Curator feature page** added to docs site ([#17563](https://github.com/NousResearch/hermes-agent/pull/17563))
- **Document pin also blocking `skill_manage` writes** ([#17578](https://github.com/NousResearch/hermes-agent/pull/17578))
- **Direct-URL skill install documented** across features, reference, guide, and `hermes-agent` skill ([#16355](https://github.com/NousResearch/hermes-agent/pull/16355))
- **Hooks tutorial — build a BOOT.md startup checklist** (replaces the removed built-in hook) ([#17202](https://github.com/NousResearch/hermes-agent/pull/17202))
- **ComfyUI docs: ask local vs cloud FIRST before hardware check** ([#17612](https://github.com/NousResearch/hermes-agent/pull/17612))
- **Obliteratus skill: link YouTube video guide in SKILL.md** ([#15808](https://github.com/NousResearch/hermes-agent/pull/15808))
- Per-skill docs pages generated for bundled + optional skills; ASCII art code blocks auto-wrapped ([#14929](https://github.com/NousResearch/hermes-agent/pull/14929), [#16497](https://github.com/NousResearch/hermes-agent/pull/16497))

---

## ⚖️ Removed / Reverted

- **Kanban multi-profile collaboration board** — landed in #16081, reverted in ([#16098](https://github.com/NousResearch/hermes-agent/pull/16098)) while the design is reworked
- **computer-use cua-driver** — 3 preparatory PRs landed then were reverted in ([#16927](https://github.com/NousResearch/hermes-agent/pull/16927))
- **BOOT.md built-in hook** removed ([#17093](https://github.com/NousResearch/hermes-agent/pull/17093)); the hooks tutorial ([#17202](https://github.com/NousResearch/hermes-agent/pull/17202)) shows how to build the same workflow yourself with a shell hook
- **`/provider` + `/plan` slash commands dropped** ([#15047](https://github.com/NousResearch/hermes-agent/pull/15047))
- **`flush_memories` removed entirely** ([#15696](https://github.com/NousResearch/hermes-agent/pull/15696))

---

## 👥 Contributors

### Core
- **@teknium1** (Teknium)

### Top Community Contributors (by merged PR count since v0.11.0)

- **@OutThisLife** (Brooklyn) — 52 PRs · TUI — light-terminal detection + pluggable busy styles + auto-resume + session-delete from /resume + mouse-wheel scrolling + xterm.js dashboard Chat tab + cold-start cut + accordion polish
- **@kshitijk4poor** — 12 PRs · LM Studio first-class provider (salvage), Vercel Sandbox backend, GMI Cloud salvage, bundled-by-default touchdesigner-mcp, many tool-call / reasoning fixes
- **@helix4u** — 10 PRs · MCP schema robustness, assorted stability fixes
- **@alt-glitch** — 8 PRs · trigram FTS5 CJK search, declarative Nix plugin install, matrix/feishu hints and fixes
- **@ethernet8023** — 4 PRs
- **@austinpickett** — 4 PRs · LaTeX rendering in TUI, dashboard layout refresh
- **@benbarclay** — 3 PRs · Docker run-as-host-user so bind mounts don't get root-owned
- **@vominh1919** — 2 PRs
- **@stephenschoettler** — 2 PRs
- **@kevin-ho** — ConPTY mouse-injection fix (#15488)
- **@Zjianru** — cross-provider reasoning_content isolation + DeepSeek/Kimi empty-reasoning injection (#15749, #15762)
- **@web3blind** — Telegram chat allowlists for groups and forums (#15027)
- **@SHL0MS** — 9 new TouchDesigner-MCP reference docs (#16768)
- **@0xDevNinja** — curator `restore_skill` nested-archive fix (#17951)
- **@y0shua1ee** — curator `use` activity fix (#17953)

### Also contributing
Salvaged or co-authored work from **@isaachuangGMICLOUD** (GMI Cloud), earlier upstream PRs from the original author of each salvage chain, and a long tail of one-shot fixes, documentation nudges, and skill contributions from the community.

### All Contributors (alphabetical, excluding @teknium1)

@0xbyt4, @0xharryriddle, @0xDevNinja, @0z1-ghb, @5park1e, @A-FdL-Prog, @aj-nt, @akhater, @alblez, @alexg0bot,
@alexzhu0, @AllardQuek, @alt-glitch, @amanning3390, @amanuel2, @AndreKurait, @andrewhosf, @Andy283, @andyylin,
@angel12, @AntAISecurityLab, @ash, @austinpickett, @badgerbees, @BadTechBandit, @Bartok9, @beenherebefore,
@beesrsj2500, @BeliefanX, @benbarclay, @benjaminsehl, @BlackishGreen33, @bloodcarter, @BlueBirdBack,
@briandevans, @brooklynnicholson, @bsgdigital, @buray, @bwjoke, @camaragon, @cdanis, @cgarwood82,
@charles-brooks, @chen1749144759, @chengoak, @ching-kaching, @Contentment003111, @crayfish-ai, @CruxExperts,
@cyclingwithelephants, @dandaka, @danklynn, @ddupont808, @dhabibi, @difujia, @dimitrovi, @dlkakbs,
@dontcallmejames, @EKKOLearnAI, @emozilla, @ericnicolaides, @Erosika, @ethernet8023, @exiao, @Feranmi10,
@flobo3, @foxion37, @georgeglessner, @georgex8001, @ghostmfr, @H-Ali13381, @HangGlidersRule, @harryplusplus,
@haru398801, @heathley, @hejuntt1014, @hekaru-agent, @helix4u, @Heltman, @HenkDz, @heyitsaamir, @hharry11,
@hhhonzik, @hhuang91, @HiddenPuppy, @htsh, @iamagenius00, @in-liberty420, @innocarpe, @irispillars, @iRonin,
@isaachuangGMICLOUD, @Ito-69, @j3ffffff, @jackjin1997, @jakubkrcmar, @Jason2031, @JayGwod, @jerome-benoit,
@johnncenae, @Kailigithub, @keiravoss94, @kevin-ho, @knockyai, @konsisumer, @kshitijk4poor, @kunlabs, @l0hde,
@Leihb, @leoneparise, @LeonSGP43, @liizfq, @liuhao1024, @loongzhao, @lsdsjy, @luyao618, @ma-pony, @Magaav,
@MagicRay1217, @math0r-be, @MattMaximo, @maxims-oss, @MaxyMoos, @maymuneth, @mcndjxlefnd, @memosr,
@MestreY0d4-Uninter, @mewwts, @Mirac1eSky, @MorAlekss, @mrhwick, @mrunmayee17, @mssteuer, @Nanako0129,
@nazirulhafiy, @Nerijusas, @Nicecsh, @nicoloboschi, @nightq, @ningfangbin, @octo-patch, @Octopus,
@OutThisLife, @Paperclip, @pein892, @perlowja, @prasadus92, @qike-ms, @qiyin-code, @Readon, @ReginaldasR,
@revaraver, @rfilgueiras, @rmoen, @romanornr, @rugvedS07, @rylena, @samrusani, @Sanjays2402, @sasha-id,
@Satoshi-agi, @scheidti, @scotttrinh, @season179, @SeeYangZhi, @sgaofen, @shamork, @shannonsands, @SHL0MS,
@simbam99, @Societus, @socrates1024, @Sonoyunchu, @sprmn24, @stephenschoettler, @tangyuanjc, @TechPrototyper,
@tekgnosis-net, @ThomassJonax, @tmimmanuel, @tochukwuada, @Tosko4, @Tranquil-Flow, @twozle, @txbxxx,
@UgwujaGeorge, @Versun, @vlwkaos, @voidborne-d, @vominh1919, @Wang-tianhao, @Wangshengyang2004, @web3blind,
@westers, @Wysie, @xandersbell, @xiahu88988, @XieNBi, @xinbenlv, @xnbi, @y0shua1ee, @yatesjalex, @yes999zc,
@yeyitech, @Yoimex, @YueLich, @Yukipukii1, @zhiyanliu, @zicochaos, @Zjianru, @zkl2333, @zons-zhaozhy,
@ztexydt-cqh.

Also: @Siddharth Balyan, @YuShu.

---

**Full Changelog**: [v2026.4.23...v2026.4.30](https://github.com/NousResearch/hermes-agent/compare/v2026.4.23...v2026.4.30)
