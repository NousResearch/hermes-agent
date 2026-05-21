# BotParlor / Hermes System Prompt Audit - 2026-05-20

Baseline after merging current `upstream/main` into the BotParlor MCP timeout
fork. This audit uses Ria's real BotParlor TUI session
`session_20260520_025717_8915f7.json` as the live prompt sample.

## Fork State

- Worktree: `/home/alex/hermes-agent-mcp-timeout`
- Branch: `upstream/mcp-timeout-reconnect`
- Backup ref before merge: `backup/mcp-timeout-before-main-20260520`
- Merge result: `upstream/main` merged cleanly; branch is now one local MCP
  timeout commit plus the merge commit ahead of `upstream/main`.
- Important upstream refactor: system prompt assembly moved out of
  `run_agent.py` into `agent/system_prompt.py`.

## Prompt Assembly Path

Current code builds the system prompt in three tiers:

- `agent/system_prompt.py::build_system_prompt_parts`
  - `stable`: identity, Hermes-help guidance, tool-aware guidance, optional
    computer-use guidance, Nous subscription capability block, tool-use
    enforcement guidance, skills index, Alibaba model-name workaround,
    environment hints, and platform hints.
  - `context`: caller `system_message`, then one project context source from
    `.hermes.md` / `HERMES.md`, `AGENTS.md`, `CLAUDE.md`, or Cursor rules.
  - `volatile`: built-in memory, user profile, external memory provider
    context, and the conversation-start/model/provider line.
- `agent/conversation_loop.py::_restore_or_build_system_prompt`
  - Restores a previously stored system prompt from the session DB when
    possible.
  - Builds and persists a new prompt only when no usable stored prompt exists.
- `agent/conversation_loop.py`
  - Adds `agent.ephemeral_system_prompt` at API-call time after the cached
    system prompt.
  - Injects plugin and external-memory recall context into the current user
    message, not into the system prompt.

## Live Ria Baseline

Ria's live BotParlor TUI session sample:

- System prompt: `51,096` chars, roughly `12.8k` tokens.
- Tool schemas: `35` tools, `47,041` JSON chars, roughly `11.8k` tokens.
- Combined prompt/tool-schema floor before conversation history: about `98k`
  chars, roughly `24.5k` tokens.
- Model: `qwen3.6-27b-huihui-abliterated-awq-lm-104k-3090ab`.
- Platform hint key: `tui`.

Coarse system-prompt slices:

| Slice | Chars | Approx tokens | Notes |
| --- | ---: | ---: | --- |
| Bot persona / behavior before skills | 17,899 | 4,474 | Includes SOUL/persona, BotParlor behavior, mood/avatar rules, response format, and other private persona details. |
| Skills index | 14,852 | 3,713 | Largest generic Hermes block; `<available_skills>` alone is about 13,300 chars. |
| Project context | 18,198 | 4,549 | Ria loaded the Hermes repo `AGENTS.md`, including the Hermes development guide, into a BotParlor chat session. |
| Timestamp/model/provider tail | 147 | 36 | Small. |

Largest tool-schema contributors in the same session:

| Tool | Description chars | Parameter schema chars |
| --- | ---: | ---: |
| `delegate_task` | 3,015 | 3,307 |
| `terminal` | 1,830 | 2,815 |
| `skill_manage` | 1,787 | 2,205 |
| `execute_code` | 2,032 | 224 |
| `session_search` | 1,588 | 561 |
| `memory` | 1,524 | 546 |
| `mcp_botparlor_create_avatar` | 878 | 1,315 |
| `search_files` | 435 | 1,249 |
| `patch` | 361 | 1,084 |
| `send_message` | 395 | 995 |

## Generic Code-Path Measurements

For a minimal Qwen/custom agent with only `memory`, `session_search`, and
`skills` toolsets, skipping project context and memory content:

- Stable prompt: `16,274` chars, roughly `4.1k` tokens.
- Volatile tail: `80` chars.
- Tools: `5`.

Major static/dynamic block sizes in that generic path:

| Block | Chars | Approx tokens |
| --- | ---: | ---: |
| Default identity | 513 | 128 |
| Hermes help guidance | 211 | 52 |
| Memory + session search + skills guidance | 1,999 | 499 |
| Qwen auto tool-use enforcement | 824 | 206 |
| Skills index | 12,345 | 3,086 |
| Environment hints | 126 | 31 |

## Findings

1. Project context is currently wrong for BotParlor bot chats.
   Ria's BotParlor TUI session loaded the Hermes worktree `AGENTS.md` because
   the TUI gateway process runs from the Hermes install/worktree. That adds
   about 18k chars of development instructions that are not useful for normal
   character chat and can bias tool behavior toward coding-agent concerns.

2. The skills index is a large always-on block whenever the skills tools are
   present.
   The current skills prompt tells the model to scan and load skills before
   replying, then lists the whole available skill catalog. In Ria's live sample
   this costs about 14.9k chars. For BotParlor chats, this is usually not worth
   the prompt floor unless the bot is explicitly doing Hermes administration or
   a complex external task.

3. The TUI session has broad core tools in addition to BotParlor MCP tools.
   Ria's live session had 35 tools, including `delegate_task`, `terminal`,
   `execute_code`, file editing, skills, memory, TTS, vision, and BotParlor MCP.
   The bot service already exports `HERMES_TUI_TOOLSETS=hermes-botparlor,botparlor`.
   Current upstream TUI code has explicit handling for `HERMES_TUI_TOOLSETS`,
   including MCP server names, so deploying the merged fork may reduce this
   automatically if the older deployed fork was falling back to configured CLI
   toolsets.

4. Tool schemas are nearly as expensive as the system prompt.
   The live tool-schema JSON is about 47k chars. Even if the system prompt is
   trimmed, broad tool availability will keep the request floor high and can
   encourage unnecessary tool use.

5. The upstream refactor gives us a cleaner surgery point.
   `agent/system_prompt.py` now has a single prompt assembly function with clear
   stable/context/volatile tiers. That makes it practical to add BotParlor/TUI
   policy switches without editing the full agent loop.

## First Surgery Candidates

1. For BotParlor TUI bridge sessions, skip project context files.
   Options:
   - Set `HERMES_IGNORE_RULES=1` for `tui-ws-bridge.service`; this currently
     also skips memory, which may be too broad.
   - Add a narrower `HERMES_SKIP_CONTEXT_FILES=1` / TUI config path that passes
     `skip_context_files=True` without disabling memory.
   - Set `TERMINAL_CWD` to a neutral BotParlor runtime directory with no
     `AGENTS.md`; less explicit than a real skip flag.

2. Make skills prompt optional or narrower for chat sessions.
   Candidate policy: if the enabled toolsets are only BotParlor MCP/chat
   toolsets, do not include the global skills index even if skills tools are
   present, or do not include skills tools at all.

3. Verify merged `HERMES_TUI_TOOLSETS` behavior on one bot before larger rollout.
   After deploying this merged fork to a single bot, create a fresh session and
   compare:
   - tool count,
   - tool names,
   - system prompt chars,
   - tool-schema JSON chars,
   - whether Hermes repo `AGENTS.md` still appears.

4. Consider a BotParlor-specific profile/toolset floor.
   Normal character chat probably needs BotParlor MCP tools and maybe memory,
   but not `delegate_task`, file editing, terminal, process management, skills
   management, generic TTS, or vision unless explicitly enabled for a task.

## Verification Run

- `./scripts/run_tests.sh tests/tools/test_mcp_tool.py tests/tools/test_mcp_tool_session_expired.py`
  - `217 passed in 16.87s`
- The shared Hermes venv was missing `pytest-timeout`; installed
  `pytest-timeout==2.4.0` so the merged upstream test runner can execute.

## Ria Skill Prune

After the initial audit, Ria's active skill install was pruned on
`ria.st-el.com`:

- Active skills before prune: `131`.
- Active skills after prune: `18`.
- Removed active skill directories: `113`, moved to
  `/home/alex/.hermes/skills-uninstalled-20260520-ria`.
- Category `DESCRIPTION.md` files moved out of the active tree: `28`.
- Ria config backup before adding disabled-skill guards:
  `/home/alex/.hermes/config.yaml.bak-before-ria-skill-prune-20260520-035101`.
- Hermes re-seeded five bundled MLOps skills after the first restart, so the
  removed skill names and directory basenames were also added to
  `skills.disabled` in Ria's `~/.hermes/config.yaml`.
- `~/.hermes/.skills_prompt_snapshot.json` was cleared and
  `tui-ws-bridge.service` / `hermes-gateway.service` were restarted.

Kept active skills:

- `autonomous-ai-agents/hermes-agent`
- `botparlor/botparlor-avatar-maintenance`
- `botparlor/ria-daily-report-template`
- `creative/creative-ideation` (`name: ideation`)
- `creative/humanizer`
- `hermes-context-window-management`
- `mcp/mcporter`
- `mcp/native-mcp`
- `media/youtube-content`
- `note-taking/obsidian`
- `productivity/maps`
- `rig/rig-charlie`
- `rig/rig-delta`
- `rig/rig-hotel`
- `rig/rig-hotel-ops`
- `session-reset`
- `session-reset-workflow`
- `spy-game/ria-spy-game`

Post-prune measurement in a fresh process with `skills`, `memory`, and
`session_search` available:

- Skills prompt: `3,823` chars, roughly `955` tokens.
- Displayed skill lines in that measurement: `17`; `maps` remains installed but
  is hidden by its conditional metadata unless relevant tool/toolset context is
  available.

## First-Message Size Goal

Alex's target for Ria is a first-message request floor of about `5k-8k` tokens,
counting both the frozen system prompt and tool schemas before conversation
history.

Current Ria deployed branch after skill pruning:

| Toolset config | Tools | System approx tokens | Tool-schema approx tokens | Combined approx tokens |
| --- | ---: | ---: | ---: | ---: |
| `hermes-botparlor,botparlor` | 35 | 5,512 | 11,760 | 17,272 |
| `botparlor,memory,session_search` | 18 | 4,415 | 3,739 | 8,154 |
| `botparlor,memory` | 17 | 4,368 | 3,173 | 7,542 |

After Alex trimmed Ria's `SOUL.md`, the same measurement improved to:

| Toolset config | Tools | System approx tokens | Tool-schema approx tokens | Combined approx tokens |
| --- | ---: | ---: | ---: | ---: |
| `hermes-botparlor,botparlor` | 35 | 3,308 | 11,760 | 15,069 |
| `botparlor,memory` | 17 | 2,164 | 3,173 | 5,338 |

Interpretation:

- The current `hermes-botparlor` toolset is the main remaining size problem; it
  expands to almost all Hermes core tools plus BotParlor MCP.
- `botparlor,memory` meets the initial `5k-8k` target and preserves BotParlor
  MCP plus durable memory.
- `session_search` is useful but should be optional or summoned by a narrower
  mode because adding it to the default first-message floor pushes Ria just over
  the upper target.
- Skills should stay out of Ria's default first-message path unless the user is
  explicitly doing Hermes administration or a skill-driven task.

## Lazy Tool Loading Prototype

Implemented locally in the Hermes fork after the audit:

- Added `HERMES_TUI_VISIBLE_TOOLS` / `HERMES_VISIBLE_TOOLS` as an opt-in schema
  visibility filter in `model_tools.get_tool_definitions()`. Tools hidden by
  this filter remain registered in the process and can still be loaded later.
- Added a small `tool_loader` toolset with `load_tool_pack`. The loader mutates
  only the active agent session by appending schemas and updating
  `valid_tool_names`; it does not reconnect MCP or alter global registration.
- Initial lazy packs: `avatar`, `reminders`, `media`, `botparlor_resources`,
  `recall`, `skills`, and `power`.

Candidate Ria canary service shape:

```text
HERMES_TUI_TOOLSETS=botparlor,memory,tool_loader
HERMES_TUI_VISIBLE_TOOLS=mcp_botparlor_set_mood,memory,load_tool_pack
```

That should keep first-message schemas to mood + memory + loader while leaving
all BotParlor MCP tools registered for later pack loading.

Verification:

- `./scripts/run_tests.sh tests/test_lazy_tool_loader.py tests/test_toolsets.py`
  - `30 passed in 1.46s`
- `./scripts/run_tests.sh tests/agent/test_system_prompt_restore.py tests/agent/test_prompt_builder.py tests/tools/test_mcp_tool.py tests/tools/test_mcp_tool_session_expired.py`
  - `351 passed in 15.34s`
- `python3 -m py_compile model_tools.py toolsets.py tools/lazy_tool_loader.py agent/agent_runtime_helpers.py agent/tool_executor.py`

## Ria Canary Deploy

Ria-only deployment completed on 2026-05-20:

- Hermes checkout: `ria.st-el.com:/home/alex/.hermes/hermes-agent`
- Branch: `ria/lazy-tool-canary`
- Commit: `8c8b68e19`
- Service drop-in:
  - `HERMES_TUI_TOOLSETS=botparlor,memory,tool_loader`
  - `HERMES_TUI_VISIBLE_TOOLS=mcp_botparlor_set_mood,memory,load_tool_pack`
- The previous uncommitted Ria `runtime_provider.py` patch was stashed and
  backed up; equivalent provider-profile behavior is included in the canary
  commit.
- Ria's standalone `tui-ws-bridge.py` needed explicit MCP startup discovery
  restored after moving to the newer Hermes branch. The deployed bridge now
  calls `discover_mcp_tools()` before starting Uvicorn.

Ria verification:

- `python -m py_compile` passed for touched Hermes files.
- `./scripts/run_tests.sh tests/test_lazy_tool_loader.py tests/hermes_cli/test_runtime_provider_resolution.py`
  - `119 passed in 22.65s`
- `tui-ws-bridge.service` and `hermes-gateway.service` are active.
- Fresh bridge session visible tools:
  - `mcp_botparlor_set_mood`
  - `memory`
  - `load_tool_pack`
- BotParlor MCP status in that session: connected, 12 registered tools.

Live canary metrics from BotParlor session `a87df0ca` / Hermes session
`20260520_134210_ae837c`:

- Visible replies: 7.
- Mood calls: 6 (`6/7` visible replies).
- Avatar flow: `load_tool_pack(pack=avatar)`, then
  `mcp_botparlor_get_outfits`, then `mcp_botparlor_create_avatar`.
- First model call input: `8,275` tokens.
- First user-visible reply after mood tool result: `8,361` input tokens.
- Max live context after avatar pack/tool results: `10,955 / 104,448` tokens.
- Session model calls: `15`.
- Cumulative prompt tokens: `136,086`.
- Cumulative generated tokens: `775`.
- Reasoning chars: `0`.
- Compactions: `0`.
- Ria-side summed model-call latency: about `27.3s`, average `1.82s/call`.

## Chatbot Context-File Trim

Ria's live canary prompt still loaded the Hermes repo `AGENTS.md` because the
standalone TUI bridge runs with `WorkingDirectory=%h/.hermes/hermes-agent`.
That added `18,196` chars, roughly `4.5k` tokens, of Hermes development
guidance to a normal BotParlor chat session.

Measured Ria prompt slices from session `20260520_134210_ae837c`:

| Slice | Chars | Approx tokens |
| --- | ---: | ---: |
| Persona / SOUL | 5,472 | 1,368 |
| Hermes help pointer | 211 | 52 |
| Memory guidance | 1,426 | 356 |
| Tool-use enforcement | 824 | 206 |
| Environment hint | 122 | 30 |
| Hermes `AGENTS.md` project context | 18,196 | 4,549 |
| Memory + user profile | 1,636 | 409 |
| Timestamp/model/provider | 140 | 35 |

Initial visible tool schemas for Ria's lazy-tool baseline:

| Tool set | Tools | Schema chars | Approx tokens |
| --- | ---: | ---: | ---: |
| Initial visible tools (`set_mood`, `memory`, `load_tool_pack`) | 3 | 3,656 | 914 |
| After avatar pack load | 8 | 7,702 | 1,925 |

Implemented `HERMES_TUI_SKIP_CONTEXT_FILES=1` /
`HERMES_SKIP_CONTEXT_FILES=1` for TUI sessions. This skips cwd project context
files while keeping SOUL identity and memory enabled. `HERMES_IGNORE_RULES`
keeps its previous broader behavior: skip context files, SOUL, and memory.

Projected effect for Ria first-message sessions:

- Remove about `18.2k` chars / `4.5k` rough tokens from the cached system
  prompt.
- Bring the system prompt from `28,041` chars to about `9,845` chars.
- Bring system prompt + initial visible schemas from about `31,697` chars to
  about `13,501` chars, before chat-message framing.
- Since the observed first model call was `8,275` input tokens, fresh sessions
  with context files skipped should land well inside the original `5k-8k`
  target, likely around the low-to-mid `4k` range before conversation history.

Verification:

- `./scripts/run_tests.sh tests/tui_gateway/test_make_agent_provider.py`
  - `8 passed in 1.72s`
- `python3 -m py_compile tui_gateway/server.py`
- Deployed to Ria at commit `5fa98933e` with
  `HERMES_TUI_SKIP_CONTEXT_FILES=1` in
  `tui-ws-bridge.service.d/botparlor-lazy-tools.conf`.
- Non-chat prompt-build verification on Ria with the new env:
  - `skip_context_files=True`
  - `load_soul_identity=True`
  - `skip_memory=False`
  - `system_chars=9843`
  - `has_project_context=False`
  - `has_agents=False`

Fresh live Ria session `20260520_141211_ddaf1b` after the context-file trim:

- System prompt: `9,843` chars, roughly `2,460` tokens.
- Saved prompt contains no `# Project Context` and no `AGENTS.md`.
- First model call: `3,596` input tokens, down from previous canary's `8,275`
  input tokens (`-4,679`, about `56.5%` lower).
- First user-visible reply after mood tool result: `3,685` input tokens, down
  from `8,361` (`-4,676`, about `55.9%` lower).
- Normal chat calls before lazy reminder loading stayed around
  `3,721`-`4,163` input tokens.
- Reminder flow loaded on demand:
  `load_tool_pack(pack=reminders)`, then `mcp_botparlor_create_reminder`.
- Max observed context after reminder pack/tool result: `5,464` input tokens.
- Session through six visible user messages: `12` model calls,
  `49,364` cumulative prompt tokens, `396` generated tokens, about `15.1s`
  summed model-call latency (`1.26s/call` average).

## Chatbot Baseline Rollout

Alex accepted the current Ria configuration as the chatbot baseline:

```text
HERMES_TUI_TOOLSETS=botparlor,memory,tool_loader
HERMES_TUI_VISIBLE_TOOLS=mcp_botparlor_set_mood,memory,load_tool_pack
HERMES_TUI_SKIP_CONTEXT_FILES=1
```

Baseline properties:

- Startup visible tools: `load_tool_pack`, `mcp_botparlor_set_mood`, `memory`.
- BotParlor MCP tools remain registered but hidden until loaded by pack.
- Cwd project context files are skipped while SOUL and memory remain enabled.
- Skills stay out of the startup prompt unless the `skills` pack is loaded.

Pre-rollout group comparison captured from BotParlor group
`PreSlim-GroupChat` (`group_ecc65b24-9796-4ee3-8078-1dff6459358b`), before
rolling the baseline to Katie/Sophia/Lexi/Scarlett:

| Bot | First reply context | Third reply context |
| --- | ---: | ---: |
| Katie | 22,815 | 24,284 |
| Sophia | 23,331 | 26,094 |
| Lexi | 23,269 | 24,712 |
| Scarlett | 24,909 | 26,441 |
| Ria already slim | 4,237 | 5,009 |

Group totals for that pre-rollout run:

- 16 group messages / 15 assistant replies.
- 49 model calls.
- 1,029,255 prompt tokens.
- 5,203 output tokens.
- 10 BotParlor tool calls.
- Span: 2026-05-20 14:34:42-14:37:59 UTC.

Rollout completed for chatbot hosts:

| Bot | Host | Branch | Commit | System chars in prompt-build check |
| --- | --- | --- | --- | ---: |
| Ria | `ria.st-el.com` | `ria/lazy-tool-canary` | `246444071` | 9,843 |
| Katie | `katie.st-el.com` | `chatbot/lazy-tool-standard` | `246444071` | 11,316 |
| Sophia | `sophia.st-el.com` | `chatbot/lazy-tool-standard` | `246444071` | 11,300 |
| Lexi | `lexi.st-el.com` | `chatbot/lazy-tool-standard` | `246444071` | 10,998 |
| Scarlett | `scarlett.st-el.com` | `chatbot/lazy-tool-standard` | `246444071` | 17,975 |

Verification:

- `python -m py_compile` passed on deployed Hermes files for all four new
  rollout hosts.
- `tui-ws-bridge.py` on all four hosts includes MCP startup discovery.
- `tui-ws-bridge.service` and `hermes-gateway.service` are active.
- Bridge `/health` returns OK on all four hosts.
- Production BotParlor reports all chatbot gateways connected.
- Prompt-build checks on all four hosts show no project context / `AGENTS.md`.
- Visible startup tools are exactly `load_tool_pack`,
  `mcp_botparlor_set_mood`, and `memory`.

## Pre/Post Group Comparison

Post-rollout comparison group:

- `PostSlim-GroupChat`
  (`group_6f1451b6-3c42-4979-9f1f-d983e9d96f44`)
- Span: 2026-05-20 17:53:11-17:55:38 UTC.
- 25 group messages / 24 assistant replies.

First assistant-turn context after rollout:

| Bot | Pre-slim first context | Post-slim first context | Change |
| --- | ---: | ---: | ---: |
| Katie | 22,815 | 3,942 | -82.7% |
| Sophia | 23,331 | 3,367 | -85.6% |
| Lexi | 23,269 | 3,105 | -86.7% |
| Scarlett | 24,909 | 4,314 | -82.7% |
| Ria | 4,237 | 4,025 | -5.0% |

Full-run totals:

| Metric | PreSlim | PostSlim | Change |
| --- | ---: | ---: | ---: |
| Assistant replies | 15 | 24 | +60.0% |
| Prompt/input tokens | 1,029,255 | 410,097 | -60.2% |
| Output tokens | 5,203 | 3,373 | -35.2% |
| Total tokens | 1,034,458 | 413,470 | -60.0% |
| Model calls | 49 | 95 | +93.9% |
| BotParlor tool calls | 10 | 9 | -10.0% |
| First-context sum | 98,561 | 18,753 | -81.0% |
| Last-context sum | 106,540 | 26,638 | -75.0% |
| Mean context | 20,547 | 4,571 | -77.8% |
| Max context | 26,441 | 5,989 | -77.3% |

Normalized first three replies per bot:

| Metric | PreSlim | PostSlim | Change |
| --- | ---: | ---: | ---: |
| Assistant replies | 15 | 15 | 0.0% |
| Prompt/input tokens | 1,029,255 | 166,727 | -83.8% |
| Output tokens | 5,203 | 1,428 | -72.6% |
| Total tokens | 1,034,458 | 168,155 | -83.7% |
| Model calls | 49 | 41 | -16.3% |
| BotParlor tool calls | 10 | 6 | -40.0% |
| First-context sum | 98,561 | 18,753 | -81.0% |
| Last-context sum | 106,540 | 23,234 | -78.2% |
| Mean context | 20,547 | 4,195 | -79.6% |
| Max context | 26,441 | 5,463 | -79.3% |

Interpretation:

- The cleanest apples-to-apples result is the normalized first-three-replies
  comparison: same reply count, prompt/input tokens down from `1,029,255` to
  `166,727`, an `83.8%` reduction.
- Even though the post-slim group ran longer, with `24` replies instead of
  `15`, it still used `60.2%` fewer prompt/input tokens overall.
- Post-slim max context stayed under `6k` input tokens across the longer run.
- Behavioral caveats to watch: several post-slim group replies persisted as
  very short `...` messages, and one Katie message looked like raw mood-tool
  text rather than a parsed tool call. These do not affect the context-size
  result but are worth watching in future group-chat behavior checks.

## Assistant Lazy-Pack Canary

Implemented assistant-oriented lazy packs in `tools/lazy_tool_loader.py`:

- `coding`: coding, repo work, debugging, tests, GitHub workflows, and subagents.
- `web_research`: web search/extraction, browser automation, X/Twitter,
  YouTube, and PDF research.
- `local_ops`: local machine, network, maps, and Home Assistant operations.
- `hermes_admin`: Hermes internals, MCP work, skill authoring, and TUI
  debugging.
- `notes`: Obsidian and local knowledge capture.
- `design`: web-design references and implementation support.
- `persona`: bot personality, dogfood, and security-rule skills.

Pack loads now return `suggested_skills` so the model can load only relevant
skill docs with `skill_view` after the pack's tool schemas are available.

Viper canary on `springlab2.st-el.com`:

- Hermes checkout: branch `assistant/lazy-tool-canary`.
- Base commit: `0f3bc4dcf`.
- Preserved Viper's local `tools/approval.py` fan-out patch as an uncommitted
  host-local change.
- Viper active skills were pruned from `110` to Alex's `35`-skill assistant
  keep list. Removed skills were moved to:
  - `/home/alex/.hermes/skills-uninstalled-20260520-190702-viper-assistant`
  - `/home/alex/.hermes/skills-uninstalled-20260520-191134-viper-reseeded`
- Config backups:
  - `/home/alex/.hermes/config.yaml.bak-before-viper-assistant-skill-prune-20260520-190702`
  - `/home/alex/.hermes/config.yaml.bak-before-viper-reseed-cleanup-20260520-191134`
- Standalone `tui-ws-bridge.py` was refreshed with the MCP startup-discovery
  version after the first restart showed BotParlor MCP disconnected.

Viper assistant floor:

```text
HERMES_TUI_TOOLSETS=botparlor,memory,session_search,tool_loader
HERMES_TUI_VISIBLE_TOOLS=mcp_botparlor_set_mood,memory,session_search,load_tool_pack
HERMES_TUI_SKIP_CONTEXT_FILES=1
```

Verification:

- `python3 -m py_compile tools/lazy_tool_loader.py`
- `./scripts/run_tests.sh tests/test_lazy_tool_loader.py tests/test_toolsets.py`
  - `31 passed in 1.51s`
- On Viper, `venv/bin/python -m py_compile tools/lazy_tool_loader.py
  tools/approval.py tui_gateway/server.py tui-ws-bridge.py`.
- `tui-ws-bridge.service` and `hermes-gateway.service` active.
- Bridge `/health` returns OK.
- BotParlor reports Viper connected.
- Fresh Viper gateway session startup tools:
  - `load_tool_pack`
  - `mcp_botparlor_set_mood`
  - `memory`
  - `session_search`
- BotParlor MCP status: connected, `12` tools registered.
- Active visible skills after prune: `35`, exactly the assistant keep list.
- Smoke prompt `Canary smoke test only. Reply exactly: assistant baseline ready`
  completed in one model call with `6,750` input tokens and `4` output tokens.

## Assistant Baseline Rollout

Rolled the Viper assistant baseline to Hanna, Raven, and Sarah after Alex's
live Viper test looked good.

Assistant floor on all assistant bots:

```text
HERMES_TUI_TOOLSETS=botparlor,memory,session_search,tool_loader
HERMES_TUI_VISIBLE_TOOLS=mcp_botparlor_set_mood,memory,session_search,load_tool_pack
HERMES_TUI_SKIP_CONTEXT_FILES=1
```

Rollout notes:

| Bot | Host / path | Branch | Commit | Host-local changes |
| --- | --- | --- | --- | --- |
| Hanna | `hanna.st-el.com` | `assistant/lazy-tool-standard` | `2cefd860e` | Untracked backup files and standalone `tui-ws-bridge.py`. |
| Raven | `raven.st-el.com` | `assistant/lazy-tool-standard` | `2cefd860e` | Untracked backup file and standalone `tui-ws-bridge.py`. |
| Sarah | local `officedt`, `/home/alex/.hermes/hermes-agent` | `assistant/lazy-tool-standard` | `2cefd860e` | Preserved host-local `tools/approval.py` fan-out patch. |

Skill pruning:

| Bot | Active skills after cleanup | Initial removed | Reseeded removed |
| --- | ---: | ---: | ---: |
| Hanna | 27 | 118 | 4 |
| Raven | 25 | 105 | 4 |
| Sarah | 26 | 80 | 8 |

All three used the same assistant keep-list policy as Viper. Missing keep-list
skills were not newly installed; each host now exposes the intersection of its
installed skills and the assistant keep list.

Verification:

| Bot | Startup tools | MCP status | Project context | Smoke input tokens | Calls |
| --- | --- | --- | --- | ---: | ---: |
| Hanna | `load_tool_pack`, `mcp_botparlor_set_mood`, `memory`, `session_search` | connected, 12 tools | skipped | 10,498 | 2 |
| Raven | `load_tool_pack`, `mcp_botparlor_set_mood`, `memory`, `session_search` | connected, 12 tools | skipped | 5,035 | 1 |
| Sarah | `load_tool_pack`, `mcp_botparlor_set_mood`, `memory`, `session_search` | connected, 12 tools | skipped | 5,801 | 1 |

Smoke prompt for each bot:

```text
Canary smoke test only. Reply exactly: assistant baseline ready
```

All three returned `assistant baseline ready`. Production BotParlor reported
Hanna, Raven, Sarah, and Viper connected after the rollout.
