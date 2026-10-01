# 02 — Gap Analysis: Aro vs the Leaders

Sep 2026 · capability matrix + moats/gaps + verdict. Subject: Aro (SamAgent fork of Hermes Agent) vs OpenAI Codex App, Cursor, Zed, Claude Code.

Scale: ✅ strong · 🟡 partial · ❌ absent.

## Capability matrix

| Capability | **Aro** | Codex App | Cursor | Zed | Claude Code |
|---|---|---|---|---|---|
| Self-improving memory / skills | ✅ curated memory, autonomous skill creation, skill curator, FTS5 session search, Honcho user modeling | ❌ | 🟡 project rules, persistent memory (basic) | ❌ | 🟡 CLAUDE.md memory files, no skill synthesis |
| Model freedom | ✅ **39 providers** (OpenRouter, Anthropic, OpenAI, Gemini, DeepSeek, Ollama local, Bedrock, …) | ❌ OpenAI only | 🟡 multi-provider but pushes own models | 🟡 Anthropic/OpenAI/Ollama + ACP agents | 🟡 Anthropic-first, others possible |
| Messaging platforms | ✅ **~30** (Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Email, Teams, …) single gateway | ❌ | ❌ | ❌ | ❌ |
| Serverless / remote backends | ✅ **7**: local, Docker, SSH, Singularity, **Modal, Daytona, Vercel Sandbox** | 🟡 OpenAI cloud only | 🟡 background agents (Cursor cloud) | ❌ | ❌ |
| Cron / automation | ✅ built-in scheduler, delivery to any platform | ❌ | ❌ | ❌ | 🟡 headless + external cron |
| Code editor + LSP | ❌ (ACP adapter serves VS Code/Zed/JetBrains) | ❌ (app is chat+diffs, not editor) | ✅ **VS Code fork, best-in-class** | ✅ **Rust editor, fastest** | 🟡 IDE extensions, terminal-first |
| Inline diff review | 🟡 Git review + worktrees exist in backend; UI polish trails | ✅ **diff-first, reference design** | ✅ inline edits + accept/reject | ✅ inline assist + ACP edits | ✅ terminal diffs + IDE inline |
| Tab autocomplete | ❌ | ❌ | ✅ **best-in-class** | ✅ | 🟡 |
| Parallel tasks board | 🟡 kanban + live subagents exist; not the primary surface | ✅ **task threads, local+cloud** | 🟡 background agents list | ❌ | ❌ multiple terminals only |
| Semantic codebase indexing | ❌ (grep/ripgrep-based search) | 🟡 | ✅ embeddings index | ✅ | 🟡 |
| Onboarding | ❌ Python 3.11+ runtime, ~45 optional extras, CLI-first setup | ✅ sign-in and go | ✅ installer + login | ✅ one binary | ✅ npm i, API key |
| Usage telemetry | 🟡 `aro usage` exists; not ambient/visible in UI | 🟡 | 🟡 | ❌ (fails before showing limits) | 🟡 terse |
| Design polish | 🟡 feature-rich desktop, dated chrome | ✅ | ✅ | ✅ | 🟡 (terminal aesthetic) |
| Keyboard-first UX | 🟡 palette exists in desktop | 🟡 | ✅ | ✅ | ✅ |

## Aro's moats (defensible, rare, or unique)

1. **The learning loop.** Memory curation + autonomous skill creation + skill curator + session search (FTS5) + user modeling. *Every* competitor's agent resets to zero; Aro compounds. Nobody else ships this today.
2. **Model-agnostic.** 39 providers incl. local Ollama — swap with `aro model`. Every Big-Co tool is a funnel to its own models; Aro is Switzerland.
3. **Lives in messaging.** ~30 platforms, one gateway process, voice-memo transcription, cross-platform continuity. The agent is reachable where users already are — no app install required on the client side.
4. **Serverless execution.** Modal/Daytona hibernate when idle: a $0/month agent that scales to a GPU cluster. Devin-style cloud agents without Devin pricing.
5. **Cron automation.** Scheduled reports/audits delivered to any platform. Agents that act while you sleep.

These moats are currently **invisible** — the desktop UI buries them behind 2023-era chrome (P5 of the redesign exists to fix exactly this).

## Aro's gaps

| Gap | Severity | Why it matters | Plan |
|---|---|---|---|
| No editor surface / LSP | High | Coding agents win in editors; Aro only rents one via ACP | P3 repo tree + review mode; keep ACP for real editing |
| No tab autocomplete | Medium | Highest-frequency coding touchpoint; Cursor/Zed moat | Out of scope near-term; ACP + editor partnership covers it |
| Diff-review polish | High | Diff-first review is the winning review paradigm (Codex) | P3 PR-style review, per-hunk accept/reject — backend already has worktrees/checkpoints |
| No semantic indexing | Medium | Big repos defeat grep; competitors embed | Post-P5 (embeddings via existing providers) |
| Onboarding friction | High | Python runtime + 45 extras lose the 5-minute test | P6 first-run wizard; light/bundled desktop variants |
| Design polish | High | First-impression filter; feature parity reads as toy without it | P0–P2 tokens + chat canvas + shell |
| Ambient usage/cost UX | Medium | Zed's anti-pattern: fail-then-inform | P1 cost chips on every tool card; P6 usage dashboard |
| Parallelism not first-class surface | Medium | Kanban/subagents exist but hidden | P4 tasks board unifies them |

## Verdict — which tool wins which use-case

| Use case | Best today | Why |
|---|---|---|
| Heavy agentic coding sessions | **Claude Code** | Deepest loop, best long-session discipline |
| Editor-integrated daily coding | **Cursor** | Tab + Composer + polish |
| Fast, hackable OSS editing | **Zed** | Performance + ACP ecosystem |
| Fire-and-forget parallel tasks w/ review | **Codex App** | Task threads + diff-first UX |
| Any-model, self-hosted terminal agent | **Gemini CLI / OpenCode / Aro** | Aro wins on surfaces + loop, loses on polish |
| **Agent that lives in WhatsApp/Telegram/Slack 24/7** | **Aro — uncontested** | Only product with a 30-platform gateway |
| **Agent that gets smarter about you over months** | **Aro — uncontested** | Only product with the learning loop |
| **Agent on serverless infra, $0 idle** | **Aro — uncontested** | Modal/Daytona backends |
| **Scheduled autonomous work** | **Aro** | Cron + platform delivery |

**Where Aro can win:** not the editor war — the *ambient agent* war. The wedge: Codex-app-class desktop polish (P0–P4) + make the moats visible (P5) + frictionless onboarding (P6). Then Aro is the only agent that is simultaneously your terminal tool, your desktop app, your chat contact, and your cron worker — on whichever of 39 models you choose.
