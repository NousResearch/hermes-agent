# Telegram Buddy Personal Operator Design

Date: 2026-06-04

## Purpose

Hermes Telegram Buddy should become a practical personal operator, not a generic chatbot and not a CLI agent merely viewed through Telegram. The first priority is useful life-assistant behavior: research, organize, compare, inspect, draft, summarize, generate artifacts, reason through decisions, and operate trusted local/personal tooling with fewer unnecessary approval prompts.

The Buddy personality should support usefulness with warmth, continuity, and good chat ergonomics. It should not become a roleplay companion or a broad business automation profile.

## Local Context Findings

The current Hermes profile loads only the `hermes-cli` and `delegation` toolsets. Hermes has native MCP support through `mcp_servers` in `~/.hermes/config.yaml`, but no MCP servers are currently configured there.

The Telegram transcript audit found 9 Telegram sessions, 196 stored messages, 19 user turns, and 94 tool messages. Tool output dominates the experience. The strongest UX failure observed was the Houston rental interview, where Hermes produced a long intake-style response that the user explicitly called a poor Telegram interview. The improved follow-up was better because it switched to one short question, easy choices, and a concise summary.

`~/.hermes/SOUL.md` still contains the default template comments, so Hermes has no strong user-specific Buddy identity in its primary identity channel.

The local machine already has many MCP and CLI capabilities available through other tools and configs, including GitHub, filesystem, fetch, memory, sequential thinking, Playwright, Docker MCP, GitNexus, Notion, n8n, Azure, Microsoft Graph, HaloPSA, ConnectWise, ITGlue, and Obsidian-related tooling. The Buddy profile should therefore be curated instead of loading everything.

## Explicit Exclusions

The Telegram Buddy profile must not load business/client-system MCPs by default.

Excluded MCP/tooling categories:

- n8n
- Notion
- Azure
- Microsoft Graph
- HaloPSA
- ConnectWise
- ITGlue
- Other client, PSA, RMM, MSP, or business-system MCPs

Those systems can remain available in separate specialist profiles or explicit one-off sessions, but they should not be part of the personal Telegram Buddy profile.

## Recommended Approach

Use a Personal Operator Profile.

This profile loads only personal/local tools with a defined trust policy. It can take routine trusted actions automatically inside approved personal scopes, while still requiring approval for destructive, external, credential-sensitive, or business-system actions.

Rejected alternatives:

- Tool Router Profile: safer and leaner, but would preserve too much tool negotiation and reduce the Buddy feel.
- Full Autopilot Profile: too risky for the current local MCP landscape, especially with sensitive business MCPs and at least one observed config pattern that passes a secret-like value through command arguments.

## MCP And CLI Stack

Start small.

Initial MCP candidates:

- `clearthought`: preferred package `@waldzellai/clear-thought-onepointfive`; fallback `@waldzellai/clear-thought` if 1.5 does not probe cleanly.
- `filesystem-personal`: scoped to approved personal/project folders only, not full home.
- `fetch` or Hermes web tools: research with source links.
- `github` MCP or `gh`: personal/open-source repo reads and low-risk local repo operations.
- `playwright`: browser inspection and local app verification.

Optional later:

- Obsidian or memory MCP, only after memory hygiene and retention rules are agreed.

Useful local CLIs already present:

- `uvx`
- `npx`
- `pnpm`
- `bun`
- `gh`
- `git`
- `claude`
- `codex`
- `gemini`
- `brew`
- `python3`

Clear Thought should be treated as a reasoning aid, not as a replacement for Hermes skills. Use it for complex decisions, debugging, trade-off analysis, planning, and systematic review.

## Spark Subagent Policy

GPT-5.3-Codex-Spark subagents should be a deliberate part of the Buddy architecture. Their job is to make Hermes feel more capable and responsive by handling bounded side work while the main Buddy keeps the user-facing conversation coherent.

Use Spark subagents for:

- Fast mechanical checks, such as file inventories, grep-style searches, schema extraction, and simple comparisons.
- Parallel research extraction where each subagent has a separate, concrete question.
- Verification sidecars, such as checking whether generated reports exist, whether a local app renders, or whether an excluded MCP is absent.
- Lightweight codebase exploration before the main Buddy summarizes implications.
- Drafting small artifacts that the main Buddy reviews and integrates.

Do not use Spark subagents for:

- Final user-facing judgment.
- Sensitive credential or `.env` inspection.
- Business/client-system work excluded from the Buddy profile.
- Broad autonomous implementation without a written plan.
- Actions that require approval under the trust policy.
- Deep chained delegation or subagents that spawn further subagents.

Design constraints:

- The main Buddy remains the orchestrator and owns the final answer.
- Subagent tasks must be concrete, bounded, and non-overlapping.
- Spark should be preferred for small/fast sidecars; stronger Codex models should remain available for ambiguous synthesis, architecture, or high-stakes reasoning.
- Telegram should not receive raw subagent logs. The main Buddy should summarize what subagents checked, what they found, and what changed the decision.
- Spark subagents should be used proactively when they reduce latency, improve coverage, or let multiple independent checks run in parallel, but not when a direct local tool call is simpler.

## Trust And Permission Model

The Buddy profile should reduce approval fatigue through a trusted routine actions policy.

Allowed automatically inside approved personal scopes:

- Read, search, and list files in allowlisted folders.
- Create notes, reports, checklists, CSV files, HTML files, Markdown files, and other review artifacts.
- Edit files Buddy created during the current task.
- Run low-risk inspection commands such as `ls`, `find`, `git status`, `gh issue view`, version checks, test/report generation, and local read-only analysis.
- Use Clear Thought, web research, GitHub reads, Playwright inspection, and local browser verification.

Still requires approval:

- Delete, overwrite, bulk move, or chmod files.
- Install global packages or modify shell, profile, login, security, or launch settings.
- Push to GitHub, open PRs, publish packages, deploy, or spend money.
- Send messages, emails, posts, or external communications.
- Access secrets, `.env`, keychains, credentials, business systems, or client data.
- Act outside allowlisted paths.

The policy should be represented both in prompt guidance and, where Hermes supports it, in configuration or permission gates. Prompt-only guidance is useful but not sufficient for high-risk actions.

## Telegram UX Rules

Telegram Buddy should use mobile-chat ergonomics.

Default behavior:

- Ask one question at a time.
- Prefer multiple-choice when it reduces decision fatigue.
- Keep messages short by default.
- Offer expansion instead of dumping full reports into chat.
- Avoid large intake forms.
- Hide or summarize tool logs.
- Send artifacts when the output is too large for chat.
- Summarize after mini-rounds with what was learned and the next useful move.
- Narrate operator work briefly: what was checked, what was found, and what happens next.

For interviews and decision support, Buddy should guide with quick replies, lightweight ranking, and small rounds. It should avoid placing the burden of structure on the user.

For operator tasks, Buddy should translate CLI-style work into chat-native updates and concise outcomes.

## Memory Model

Memory should be useful, compact, and consent-aware.

Store:

- Durable communication preferences.
- Decision criteria.
- Recurring tools and trusted folders.
- Active personal projects.
- House-hunting preferences and other stable life-planning criteria.
- Preferred report formats.
- Approval preferences.
- Reusable procedures, as skills or procedural memory.

Do not store:

- Raw transcripts.
- Secrets, tokens, `.env` content, or credential material.
- Client/business details from excluded systems.
- One-off task progress likely to be stale.
- Guesses about private people without consent.

Use three memory layers:

- `SOUL.md`: identity and communication posture.
- Hermes memory: compact durable facts.
- Session search: episodic recall from local transcripts when the user references prior work.

## Rollout Plan

Phase 1: Identity and behavior

- Rewrite `~/.hermes/SOUL.md` for practical Telegram Buddy behavior.
- Add a compact Telegram Buddy operating policy as a skill or doc.
- Tune identity away from generic assistant mode.

Phase 2: Clear Thought MCP

- Install/configure Clear Thought in Hermes `mcp_servers`.
- Probe tools.
- Prefer a pinned or local install over floating `npx -y` if practical.
- Fall back from `@waldzellai/clear-thought-onepointfive` to `@waldzellai/clear-thought` if needed.

Phase 3: Personal tool profile

- Add only approved personal MCPs and CLIs.
- Keep excluded business MCPs unloaded.
- Define allowlisted personal folders.
- Encode the trusted routine actions policy.
- Encode Spark subagent usage rules so bounded sidecars are used appropriately without leaking raw subagent chatter into Telegram.

Phase 4: UX verification

- Replay or simulate the Houston housing interview failure.
- Verify one-question-at-a-time behavior.
- Verify concise progress updates and artifact delivery.
- Verify routine local actions do not trigger excessive approval prompts.
- Verify excluded MCPs are not loaded.
- Verify Spark subagents are used for appropriate sidecar tasks and not used for excluded/sensitive work.

## Verification Criteria

The design is successful when:

- Telegram Buddy can handle practical personal tasks with short, useful interaction.
- Clear Thought is available for systematic reasoning.
- Business/client MCPs are absent from the Buddy tool surface.
- Routine approved local actions proceed without repeated approval prompts.
- Risky actions still stop for approval.
- GPT-5.3-Codex-Spark subagents are used for bounded sidecars where they improve speed or coverage, while the main Buddy keeps orchestration and final judgment.
- The housing-interview transcript failure mode is not repeated.
- `SOUL.md` and memory guidance create a recognizable Buddy posture without fake intimacy or over-personalization.

## Sources Consulted

- LettaBot: persistent memory across Telegram, Slack, Discord, WhatsApp, and Signal.
- Mem0: structured long-term memory for multi-session coherence with lower latency and token cost than full-context approaches.
- MemMachine: episodic ground-truth preservation plus profile memory for personalized agents.
- python-telegram-bot InlineKeyboardButton docs: Telegram-native callback buttons for low-friction choices.
- Clear Thought MCP package documentation: systematic thinking, mental models, debugging, decision frameworks, and sequential reasoning operations.
