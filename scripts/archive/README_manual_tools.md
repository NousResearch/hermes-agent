# Manual Tools in ~/.hermes/scripts/

These scripts are retained as manual tools for specific operations. They are not referenced by active cron jobs but are useful for ad-hoc tasks.

## Content Approval Workflow
- `approve_draft.sh` - Approve a content draft
- `reject_draft.sh` - Reject a content draft
- `view_draft.sh` - View a content draft

## Utility Scripts
- `kill_modal.sh` - Kill Modal processes
- `secret-vault.sh` - Interact with secret vault
- `git-fetch-all.sh` - Fetch all git remotes
- `profile-tui.py` - Diagnostic tool for Hermes profiles
- `gateway-health.py` - Diagnostic tool for Hermes gateway health
- `cron-output-lint.py` - Linting tool for cron job prompts (used after prompt changes)
- `hermes-rollback.sh` - Rollback Hermes Agent to previous version
- `hermes-docs-check.sh` - Check Hermes documentation integrity
- `outlook_fetch.py` - Quick calendar fetch using MSAL cache
- `contributor_audit.py` - Release audit tool for git contributors
- `quarterly-audit.py` - Designed for no_agent cron on Q-start (not yet wired)
- `check-windows-footguns.py` - Dev tool for Windows path issues
- `build_model_catalog.py` - Rebuild Hermes model catalog
- `build_skills_index.py` - Rebuild skills index
- `lint_diff.py` - Lint only changed lines in a diff
- `token_health.py` - Token usage health check
- `tavily_health_probe.py` - Tavily search health check
- `rate_limited_executor.py` - Rate-limited command executor
- `direct_fetch.py` - Direct URL fetch utility
- `sample_and_compress.py` - Log sampling and compression
- `toon_utils.py` - TOON encoding utilities
- `update_cron_prompts.py` - Update cron prompt templates
- `update_investigation.py` - Update investigation data
- `uptime_ping.py` - Uptime monitoring ping
- `system_report.py` - System report generator
- ~~`misa-auto-join.py`~~ — removed (script was orphaned, no cron refs)
- `post_remediation_audit.sh` - Post-remediation audit
- `skill-broker-ledger.py` - Skill broker ledger management
- `skill-broker-revoke-hook.py` - Skill broker revocation hook
- `self-eval-reminder.py` - Self-evaluation reminder
- `worker-failure-analysis.py` - Worker failure analysis
- `mailbox_cleaner_mcp.py` - Mailbox cleaner MCP utility
- `analyze_livetest.py` - Live test analysis
- `tool_search_livetest.py` - Tool search test
- `LIVETEST_README.md` - Test documentation

## Internal Library Code
- `lib/` - Python modules imported by cron scripts (not standalone scripts)
- `tests/` - Test scripts

## Archive
Dead/superseded scripts archived in `archive/`:
- calendar scripts (superseded by calendar_brief_noagent.sh)
- research_digest.py.bak (backup, safe to delete)
- kensei_rss_watcher.py (RSS watcher retired)
- workflow-router-check.py (workflow router removed)
- triage-*.py (superseded by triage-processor cron)
- setup-specialist-bots.sh (superseded by multi-bot systemd architecture)
- installer scripts (copies of Hermes installer)
- Various one-off diagnostic scripts
