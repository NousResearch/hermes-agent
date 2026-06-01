# Hermes Agent Has Had "Routines" Since March

Anthropic just announced [Claude Code Routines](https://claude.com/blog/introducing-routines-in-claude-code) — scheduled tasks, GitHub event triggers, and API-triggered agent runs. Bundled prompt + repo + connectors, running on their infrastructure.

It's a good feature. We shipped it two months ago.

---

## The Three Trigger Types — Side by Side

Claude Code Routines offers three ways to trigger an automation:

**1. Scheduled (cron)**
> "Every night at 2am: pull the top bug from Linear, attempt a fix, and open a draft PR."

Hermes equivalent — works today:
```bash
hermes cron create "0 2 * * *" \
  "Pull the top bug from the issue tracker, attempt a fix, and open a draft PR." \
  --name "Nightly bug fix" \
  --deliver telegram
```

**2. GitHub Events (webhook)**
> "Flag PRs that touch the /auth-provider module and post to #auth-changes."

Hermes equivalent — works today:
```bash
hermes webhook subscribe auth-watch \
  --events "pull_request" \
  --prompt "PR #{pull_request.number}: {pull_request.title} by {pull_request.user.login}. Check if it touches the auth-provider module. If yes, summarize the changes." \
  --deliver slack
```

**3. API Triggers**
> "Read the alert payload, find the owning service, post a triage summary to #oncall."

Hermes equivalent — works today:
```bash
hermes webhook subscribe alert-triage \
  --prompt "Alert: {alert.name} — Severity: {alert.severity}. Find the owning service, investigate, and post a triage summary with proposed first steps." \
  --deliver slack
```

Every use case in their blog post — backlog triage, docs drift, deploy verification, alert correlation, library porting, bespoke PR review — has a working Hermes implementation. No new features needed. It's been shipping since March 2026.

---

## What's Different

| | Claude Code Routines | Hermes Agent |
|---|---|---|
| **Scheduled tasks** | ✅ Schedule-based | ✅ Any cron expression + human-readable intervals |
| **GitHub triggers** | ✅ PR, issue, push events | ✅ Any GitHub event via webhook subscriptions |
| **API triggers** | ✅ POST to unique endpoint | ✅ POST to webhook routes with HMAC auth |
| **MCP connectors** | ✅ Native connectors | ✅ Full MCP client support |
| **Script pre-processing** | ❌ | ✅ Python scripts run before agent, inject context |
| **Skill chaining** | ❌ | ✅ Load multiple skills per automation |
| **Daily limit** | 5-25 runs/day | **Unlimited** |
| **Model choice** | Claude only | **Any model** — Claude, GPT, Gemini, DeepSeek, Qwen, local |
| **Delivery targets** | GitHub comments | Telegram, Discord, Slack, SMS, email, GitHub comments, webhooks, local files |
| **Infrastructure** | Anthropic's servers | **Your infrastructure** — VPS, home server, laptop |
| **Data residency** | Anthropic's cloud | **Your machines** |
| **Cost** | Pro/Max/Team/Enterprise subscription | Your API key, your rates |
| **Open source** | No | **Yes** — MIT license |

---

## Things Hermes Does That Routines Can't

### Script Injection

Run a Python script *before* the agent. The script's stdout becomes context. The script handles mechanical work (fetching, diffing, computing); the agent handles reasoning.

```bash
hermes cron create "every 1h" \
  "If CHANGE DETECTED, summarize what changed. If NO_CHANGE, respond with [SILENT]." \
  --script ~/.hermes/scripts/watch-site.py \
  --name "Pricing monitor" \
  --deliver telegram
```

The `[SILENT]` pattern means you only get notified when something actually happens. No spam.

### Multi-Skill Workflows

Chain specialized skills together. Each skill teaches the agent a specific capability, and the prompt ties them together.

```bash
hermes cron create "0 8 * * *" \
  "Search arXiv for papers on language model reasoning. Save the top 3 as Obsidian notes." \
  --skills "arxiv,obsidian" \
  --name "Paper digest"
```

### Deliver Anywhere

One automation, any destination:

```bash
--deliver telegram                      # Telegram home channel
--deliver discord                       # Discord home channel  
--deliver slack                         # Slack channel
--deliver sms:+15551234567              # Text message
--deliver telegram:-1001234567890:42    # Specific Telegram forum topic
--deliver local                         # Save to file, no notification
```

### Model-Agnostic

Your nightly triage can run on Claude. Your deploy verification can run on GPT. Your cost-sensitive monitors can run on DeepSeek or a local model. Same automation system, any backend.

---

## The Limits Tell the Story

Claude Code Routines: **5 routines per day** on Pro. **25 on Enterprise.** That's their ceiling.

Hermes has no daily limit. Run 500 automations a day if you want. The only constraint is your API budget, and you choose which models to use for which tasks.

A nightly backlog triage on Sonnet costs roughly $0.02-0.05. A monitoring check on DeepSeek costs fractions of a cent. You control the economics.

---

## Get Started

Hermes Agent is open source and free. The automation infrastructure — cron scheduler, webhook platform, skill system, multi-platform delivery — is built in.

```bash
pip install hermes-agent
hermes setup
```

Set up a scheduled task in 30 seconds:
```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly AI news digest. Search the web for major announcements, trending repos, and notable papers. Keep it under 500 words with links." \
  --name "Weekly digest" \
  --deliver telegram
```

Set up a GitHub webhook in 60 seconds:
```bash
hermes gateway setup    # enable webhooks
hermes webhook subscribe pr-review \
  --events "pull_request" \
  --prompt "Review PR #{pull_request.number}: {pull_request.title}" \
  --skills "github-code-review" \
  --deliver github_comment
```

Full automation templates gallery: [hermes-agent.nousresearch.com/docs/guides/automation-templates](https://hermes-agent.nousresearch.com/docs/guides/automation-templates)

Documentation: [hermes-agent.nousresearch.com](https://hermes-agent.nousresearch.com)

GitHub: [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)

---

*Hermes Agent is built by [Nous Research](https://nousresearch.com). Open source, model-agnostic, runs on your infrastructure.*
