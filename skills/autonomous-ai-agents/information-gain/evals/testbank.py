#!/usr/bin/env python3
"""testbank.py — categorized prompt bank spanning how Hermes is actually used.

Two pools:
  LIFE  — generic advice questions (the original Phase-1 baseline / control). Homogeneous,
          non-derivable user-intent uncertainty — the degenerate corner where U looks inert.
  BANK  — agentic / tool-access / coding tasks, the REAL target domain. Each carries a mix of
          feasibility uncertainty (is there a tool / do I have access? — partly discoverable),
          intent uncertainty (which / how — user-only), and research-resolvable uncertainty
          (the agent can just go find out). Tagged by `cat` so value structure can be analyzed
          per category.

Used by evals/score_scan.py (cheap value-distribution scan, no realized_change) and
evals/validate_evsi.py (realized-change calibration). Keep prompts genuinely UNDERSPECIFIED.
"""

LIFE = [
    {"id": "buy-rent", "cat": "life", "problem": "Should I buy or rent a home?"},
    {"id": "gtm-plan", "cat": "life", "problem": "Write a go-to-market plan for a new B2B SaaS product."},
    {"id": "remote-hybrid", "cat": "life",
     "problem": "Summarize the main trade-offs of remote vs hybrid work for a 200-person company."},
]

BANK = [
    # comms — retrieve
    {"id": "telegram-updates", "cat": "comms-retrieve", "problem": "Get the latest updates from my Telegram channels."},
    {"id": "slack-catchup", "cat": "comms-retrieve", "problem": "Catch me up on what I missed in Slack today."},
    {"id": "whatsapp-unread", "cat": "comms-retrieve", "problem": "What are my unread WhatsApp messages about?"},
    # comms — send
    {"id": "whatsapp-send", "cat": "comms-send", "problem": "Send a WhatsApp message to my team about the deploy."},
    {"id": "slack-announce", "cat": "comms-send", "problem": "Announce the new feature launch in the right Slack channel."},
    # email
    {"id": "gmail-triage", "cat": "email", "problem": "Summarize my important unread Gmail from this week."},
    {"id": "gmail-reply", "cat": "email", "problem": "Draft a reply to the latest email from my biggest customer."},
    {"id": "gmail-find", "cat": "email", "problem": "Find the invoice from our cloud provider last month."},
    # calendar
    {"id": "cal-schedule", "cat": "calendar", "problem": "Schedule a 30-minute meeting with Alex next week."},
    {"id": "cal-week", "cat": "calendar", "problem": "Summarize my calendar for the coming week and flag conflicts."},
    # web research
    {"id": "research-ratelimit", "cat": "web-research",
     "problem": "Research current best practices for rate-limiting a public HTTP API and summarize them."},
    {"id": "research-compare", "cat": "web-research",
     "problem": "Compare the top vector databases for a RAG app and recommend one."},
    # code — feature
    {"id": "add-auth", "cat": "code-feature", "problem": "Add authentication to my web app."},
    {"id": "add-export", "cat": "code-feature", "problem": "Add CSV export to the reports page."},
    # code — debug
    {"id": "fix-test", "cat": "code-debug", "problem": "Fix the failing test in the CI pipeline."},
    {"id": "debug-slow", "cat": "code-debug", "problem": "An API endpoint is slow — find out why and fix it."},
    # code — review
    {"id": "review-pr", "cat": "code-review", "problem": "Review my open pull request for issues."},
    {"id": "security-audit", "cat": "code-review", "problem": "Audit this service for security vulnerabilities."},
    # devops
    {"id": "deploy-app", "cat": "devops", "problem": "Deploy the latest version of my app."},
    {"id": "setup-ci", "cat": "devops", "problem": "Set up CI/CD for my repository."},
    # system / files
    {"id": "organize-files", "cat": "system-files", "problem": "Organize the files in my Downloads folder."},
    {"id": "find-doc", "cat": "system-files", "problem": "Find the document where I wrote the Q3 plan."},
    # data
    {"id": "analyze-csv", "cat": "data", "problem": "Analyze this sales CSV and tell me what's interesting."},
    {"id": "query-db", "cat": "data", "problem": "Get the top 10 customers by revenue from the database."},
    # docs / content
    {"id": "summarize-pdf", "cat": "docs", "problem": "Summarize this research paper PDF."},
    {"id": "write-brief", "cat": "docs", "problem": "Write a project brief for the new initiative."},
    # automation / integration
    {"id": "sync-notion-cal", "cat": "automation", "problem": "Sync new Notion tasks to my calendar automatically."},
    {"id": "standup-bot", "cat": "automation", "problem": "Build a bot that posts daily standup reminders."},
    # knowledge (well-specified-ish — low-end control)
    {"id": "explain-oauth", "cat": "knowledge", "problem": "Explain how OAuth2 works."},
    # planning
    {"id": "plan-day", "cat": "planning", "problem": "Plan my day based on my priorities and calendar."},
    # finance
    {"id": "portfolio-check", "cat": "finance", "problem": "Check my portfolio and flag anything I should worry about."},
]

ALL = LIFE + BANK
BY_ID = {p["id"]: p for p in ALL}

# A representative cross-category subset for the (expensive) realized-change calibration.
REALIZED_SUBSET = [
    "telegram-updates", "gmail-triage", "cal-schedule", "research-ratelimit",
    "add-auth", "fix-test", "security-audit", "deploy-app",
    "organize-files", "query-db", "write-brief", "standup-bot",
]
