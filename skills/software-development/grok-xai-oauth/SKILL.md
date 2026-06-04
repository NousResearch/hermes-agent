---
name: grok-xai-oauth
description: "Use when you want to drive Hermes with your Grok/SuperGrok or X Premium+ subscription via the official xAI OAuth provider. No XAI_API_KEY needed. Excellent for native X search with citations, Imagine image/video, and high-quality reasoning."
version: 1.0.0
author: trefong (via Grok Build)
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [xai, grok, oauth, inference, x-search, image-gen, video-gen, reasoning]
    related_skills: [hermes-agent-skill-authoring, subagent-driven-development, writing-plans, mcp]
    requires_toolsets: []
---

# Grok (xAI) via Official OAuth in Hermes

Connect your existing Grok access (grok.com SuperGrok subscription or X Premium+) directly to Hermes using the official browser-based OAuth flow. This is the integration announced by xAI at https://x.ai/news/grok-hermes.

**Why this matters:** You get frontier Grok 4.3 reasoning, native server-side X search (with citations), Grok Imagine (images + video), and TTS — all without managing an `XAI_API_KEY`, and the tokens refresh automatically.

## When to Use

Use this skill when:
- You have (or want) a Grok subscription and want to drive Hermes with it.
- You want the best possible real-time X (Twitter) search and discussion synthesis (Grok runs the search server-side).
- You need high-quality image or short video generation from the agent (`grok imagine` flows).
- You're doing long-horizon reasoning, planning, or agentic coding where Grok's style and tool use shine.
- You want a single subscription powering both chat interfaces *and* a persistent local/self-hosted autonomous agent (Hermes).

**Don't use (or combine carefully) when:**
- You need the absolute lowest latency or highest volume (local models or cheaper providers may be better for bulk work).
- You are on a very tight budget (SuperGrok Heavy tiers for Grok Build are premium).
- The task is purely local file/system work with no need for web/X/image capabilities (a fast local model + Hermes tools is often sufficient).

## Quick Setup (One-Time)

1. Make sure you have a qualifying subscription (SuperGrok on grok.com or X Premium+ linked to the same X account).
2. In Hermes:

```bash
hermes model
```

Select **xAI Grok OAuth (SuperGrok / X Premium+)**.

3. A browser window opens to accounts.x.ai — log in and authorize.

4. Hermes stores the tokens locally and refreshes in the background. No `XAI_API_KEY` required for this provider.

Run the bundled helper for a quick local check:

```bash
bash ${HERMES_SKILL_DIR}/scripts/check-grok-oauth.sh
```

(When the skill is active the token is substituted.)

Verify the brain:

```bash
hermes chat -q "What model am I using right now? Can you search X for recent xAI announcements?"
```

You should see Grok identify itself and return synthesized X results with citations.

See the official xAI post for the latest: https://x.ai/news/grok-hermes and Hermes docs: https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth

## Recommended Models & Capabilities

When the xAI OAuth provider is active, prefer these for different jobs:

- **grok-4.3** (or latest fast/reasoning variant): Default for most agent work, planning, coding, analysis. Excellent at following complex procedures and tool use.
- **Image / video**: Use Grok Imagine flows (the agent will route appropriately when you ask for visuals). Great for mockups, diagrams, storyboards.
- **TTS**: Grok Text-to-Speech for voice responses in messaging bridges (Telegram, etc.).
- **X search**: This is a standout. The `x_search` tool (or natural language "search X for...") is powered by Grok running server-side. Prefer this over generic web_search when you care about current discourse, reactions, or claims *on X*.

You can still mix providers in one Hermes session (different subagents or explicit model selection) — this is one of Hermes' superpowers.

## Strong Workflows with Grok + Hermes

### 1. Persistent Research Agent with Live X Context
Hermes remembers across days/weeks. Pair it with Grok's X search:

- "Monitor sentiment on X about [topic] overnight and summarize in my ledger tomorrow at 8am."
- Use Hermes cron + the grok-powered X search.

Real-world example (from community): People are already using Hermes + Grok X search for things like predicting 2026 World Cup winners by combining X sentiment, stats, odds, and NotebookLM – then turning results into content/scripts. Your Hermes on Grok can do the X part natively with citations and memory.

### 2. Agentic Coding with Visuals
- Ask Hermes (on Grok) to plan a feature.
- During implementation, have it generate UI mockups or architecture diagrams with Grok Imagine.
- Subagents can critique the images + the code.

### 3. Multi-Model Orchestration (Grok as the "Brain")
Use Grok for high-level planning, decomposition, and final synthesis. Delegate narrow execution (file edits, tests, browser) to faster/cheaper/local models via `delegate_task` or explicit model routing. This is cost-effective and plays to each model's strengths.

See the `subagent-driven-development` skill for a proven pattern — simply dispatch some subagents with the Grok provider and others with a local/fast one.

### 4. Evidence Ledger + Grok (Advanced)
If you run a bridge like grok-concierge (which injects an Evidence Ledger into Grok prompts and exposes it to Hermes), have Hermes on Grok read/write the ledger for long-term project memory that survives context windows.

### 5. Best-of-N + Grok Reasoning (Grok Build style)
When a decision or design has high uncertainty, run several parallel subagents on Grok, then synthesize.

This directly ports the `best-of-n` pattern from Grok Build's skill system:

```python
# In a Hermes session on the Grok provider
todo([
    {"id": "idea-1", "content": "Variant A: conservative, minimal change", "status": "pending"},
    {"id": "idea-2", "content": "Variant B: aggressive, high impact", "status": "pending"},
    {"id": "idea-3", "content": "Variant C: hybrid with feature flag", "status": "pending"},
])

# Dispatch N parallel subagents (fresh context each)
for variant in ["A", "B", "C"]:
    delegate_task(
        goal=f"Generate detailed proposal for {variant}",
        context="Full problem description + constraints here. Be creative but realistic.",
        toolsets=["terminal", "file"],
        # In Hermes with Grok provider active, this uses Grok 4.3
    )

# Then a single reviewer subagent (also on Grok) picks or merges
delegate_task(
    goal="Review all three proposals using best-of-n criteria: feasibility, user value, maintenance cost, risk. Output the winner + why, or a synthesized 4th option.",
    context="The outputs from the three idea subagents...",
)
```

Grok excels at the comparative judgment step. Use this for architecture decisions, API designs, or refactor strategies inside Hermes.

See also: port the full `best-of-n`, `implement`, `review`, and `check-work` disciplines from `~/.grok/skills/` as reusable Hermes skills. They compose beautifully with a strong reasoning provider like Grok.

## Grok-Specific Tips & Gotchas

- **X search is native and citable**: Ask naturally. Grok returns synthesized answers + post citations. This is often better than raw web_search for timely topics.
- **Image generation consumes quota differently**: Be explicit when you want visuals ("generate a diagram... using Imagine"). Review generated assets before committing them to plans.
- **Token usage**: Grok is powerful but premium. Use Hermes' memory compression, context engineering, and subagent isolation aggressively so you only pay for the "thinking" steps that matter.
- **OAuth session**: If the agent complains about auth, run `hermes model` again or check `~/.hermes/auth.json` (or the shared nous auth). Hermes handles refresh for you in most cases.
- **Provider switching**: You can have the main session on a local model and only escalate specific `delegate_task` calls to the Grok OAuth provider for the hard reasoning steps.

## Security & Privacy Notes (Important)

- The OAuth tokens live in your Hermes home (`~/.hermes/...`). They are only as secure as your machine.
- Never commit auth files.
- When using container backends (Docker, Modal, etc.), Hermes mounts only what's necessary. Review `hermes doctor` output.
- Grok (via xAI) will see the prompts you send through the provider. This is the same as using grok.com or the X Grok interface.
- For highly sensitive work, prefer local models + tools even if slower.

Follow all the normal Hermes security practices in CONTRIBUTING.md (shlex.quote, path realpath checks, approval flows for dangerous commands).

## Common Pitfalls

1. Forgetting that the current Hermes session may need restart or `hermes model` re-selection after first OAuth login for the provider to be fully active in long-running TUI/gateway sessions.
2. Over-using Grok for everything instead of mixing with fast local models for rote work (burns quota and slows the agent).
3. Assuming X search results are "facts" — always treat as "what people are saying on X right now" and cross-verify for important claims.
4. Generating many images/videos in one loop without review — quota and cost add up fast.
5. Not using subagents + fresh context when doing big tasks on a high-context model like Grok 4.3 (you lose the isolation benefit).

## Verification Checklist (After Setup or Major Task)

- [ ] `hermes model` shows the xAI Grok OAuth provider as selected / available.
- [ ] A simple chat confirms it's Grok ("I am Grok, built by xAI").
- [ ] X search returns results with post citations and links.
- [ ] Image generation request produces assets that appear in your Hermes media handling.
- [ ] Long-running cron or gateway session still has valid (refreshed) auth after hours.
- [ ] Cost/usage feels reasonable for the value (use Hermes usage tracking + your grok.com account dashboard).
- [ ] Mixed-provider workflows (Grok for planning + local for execution) work cleanly.

## One-Shot Recipes

**"Connect Grok and do a research spike on X"**

```bash
hermes model   # pick xAI Grok OAuth
hermes chat -q "Search X for the latest discussion around Hermes Agent + Grok integration. Summarize top themes and link key threads. Then propose 3 high-value Hermes skills that would shine with Grok as the brain."
```

**"Use Grok + subagents for a feature"**

Load `writing-plans` then `subagent-driven-development`, but dispatch the planner and at least one reviewer subagent explicitly on the Grok provider while execution subagents use a fast local model.

**"Generate visuals during planning"**

In a planning session on Grok: "For the architecture we just discussed, generate 2-3 diagram options using Grok Imagine. Save the best one and describe why."

## Grok Build Synergy (This CLI)

This environment (Grok Build) is xAI's agentic CLI for software engineering. It already understands your `~/.grok/skills/`, AGENTS.md, MCP servers (including direct GitHub and Notion), subagents, plan mode, worktrees, and review loops.

**Powerful combo:**
- Do heavy lifting and research here in Grok Build (using your local MCPs, best-of-n ideation, systematic review with `check-work` / review skills, multi-file edits).
- Hand off long-running autonomous execution, cron jobs, messaging bridges, or persistent memory work to Hermes running on the same Grok provider.
- Port useful patterns from `~/.grok/skills/` (e.g. `best-of-n`, `implement`, `review`, `check-work`, `design`) into Hermes skills so your autonomous agent can use the same disciplines.
- Use the Evidence Ledger / concierge patterns from your grok-concierge project as inspiration for memory that survives across both tools.

Example handoff:
1. Use Grok Build + subagents here to produce a rock-solid plan + initial code + tests.
2. Commit.
3. Ask Hermes (on Grok): "Take the plan in docs/plans/xxx.md and the current branch. Execute the remaining autonomous pieces (deploy checks, monitoring cron, user-facing Telegram briefings) using the persistent memory setup."

This gives you the best of both: precise, review-heavy coding in the CLI + always-on autonomous agent with memory.

## Contributing Back

This integration is new (May 2026). The fastest way to help xAI and the Hermes project is:

- Use it heavily on real work.
- Report friction in Hermes (GitHub issues on NousResearch/hermes-agent).
- Send direct feedback to xAI from inside Grok Build (type `/feedback` in this CLI) or the main Grok interfaces.
- Author and PR more skills that demonstrate powerful Grok + Hermes combinations (this skill is one example).

If you maintain a custom bridge (like grok-concierge), consider open-sourcing patterns that make the combo even better.

## References & Sources

- xAI announcement: https://x.ai/news/grok-hermes
- Hermes OAuth guide: https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth
- Your local Hermes source (for deeper provider code): `~/.hermes/hermes-agent/providers/` and agent adapters.
- Peer skills in this repo: `subagent-driven-development`, `hermes-agent-skill-authoring`, `writing-plans`.

---

**Made with Grok Build + Hermes on macOS.** If you improve this skill, please PR it back to the Hermes repo so everyone using the xAI integration benefits.
