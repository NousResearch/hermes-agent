# Secret Surface Inventory

This pass focused on skill frontmatter that references secrets or environment variables and updated user-facing setup text to prefer Bitwarden Secrets Manager where plaintext `.env` instructions were present.

## Patched in this pass
- Core / plugins / adjacent docs: hermes-agent, google_meet, hyperliquid, siyuan
- Productivity: airtable, linear, notion, teams-meeting-pipeline, gif-search, canvas, shop-app, telephony
- Creative / devops / automation: comfyui, webhook-subscriptions, pinggy-tunnel, watchers, kanban-codex-lane, 1password
- GitHub: github-auth, github-code-review, github-issues, github-pr-workflow, github-repo-management, oss-forensics
- MLOps: weights-and-biases, huggingface-hub, lambda-labs, modal
- Research / messaging: agentmail, openclaw-migration, duckduckgo-search, osint-investigation, parallel-cli, searxng-search

## Remaining high-signal surfaces to review next
The broader scan still finds env-dependent skills in optional skill trees and profile-local mirrors. Examples:
- devops/cli (`INFSH_API_KEY`)
- mlops/modal and mlops/huggingface-hub (token-based auth fallbacks)
- optional skill trees with api-token driven workflows not yet normalised to required_environment_variables
- profile-local mirror files under `~/.hermes/profiles/coding/skills` for any divergent entries

## Notes
- No secrets or token values are recorded here.
- Plugin and cron surfaces remain in the downstream board tasks for this rollout.
