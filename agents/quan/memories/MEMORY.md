Infra decisions are locked in ~/kensei-system-review.md (32 findings, 2026-05-08). Future agents should follow NSP priority order and update Status during implementation, not re-audit unless overridden.
§
Claude Code setup: 30 plugins, ~/.claude/CLAUDE.md uses British English/no em-dashes, 4 hooks, kensei-code-reviewer. KENSEI skin lives at ~/.hermes/skins/kensei.yaml; local CLI supports response_text/reasoning colours.
§
CoachOS renamed to CoachSense. Postiz/content engine v2 needs social accounts fully wired for multi-platform publishing.
§
xurl is installed and working at /home/kensei/.local/bin/xurl. Kept dormant by policy because X API Basic is not being paid for. Zero-cost alternative: paste tweet link, get manual reply suggestions.
§
GitRadar (code-discovery-pipeline v3.1.0) fully hardened 2026-05-15: zero-repo guard in classify cron, stale cache cleared, wiki repo arsenal created (repos/, SCHEMA.md frontmatter, adoption lifecycle, _meta/repo-index.md, 9 seeded repo pages). Upstream monitor script + weekly cron (Mon 10:00) tracks forked/extracted/monitoring repos. Discovery script template synced. Skill updates: requesting-code-review (5 explicit quality gates), hermes-update (upstream risk assessment pre-flight). GitNexus is installed at v1.6.4 but not integrated with pipeline.
§
X-algorithm (xai-org/x-algorithm) analysed 2026-05-16. BangerInitialScreen: 0.4 quality threshold, slop_score detection. Phoenix scorer: reply chains 150x, bookmarks 10x, likes 0.5x. Multimodal embedder: text+image combined. Patched into sahil-twitter-voice and content-pipeline. reply_suggester.py created but dormant — needs xurl + $100/mo X API Basic. User chose manual zero-cost alternative (paste link, get reply options).
§
deepseek-v4-flash on Ollama Cloud returns 404 (gateway prepends deepseek/ prefix). Swap to deepseek-v4-pro. Always audit ALL profiles after fixing one. On 2026-05-16, 4 profiles on broken flash: ops-lead, content-lead, triage-router, market-scanner. Google Workspace MCP fixed with --single-user flag for OAuth callback port conflict.