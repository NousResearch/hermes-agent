# Secret Surface Inventory

This branch audits the secret and environment-variable guidance it changes. Secret values belong in Bitwarden Secrets Manager where available, with `${HERMES_HOME:-~/.hermes}/.env` as the documented fallback. Non-secret URLs, identifiers, addresses, paths, ports, and selectors remain local configuration rather than secret-manager entries.

## Final changed surface

### Setup metadata and adjacent guidance

- Optional skills: `pinggy-tunnel`, `watchers`, `agentmail`, `openclaw-migration`, `lambda-labs`, `modal`, `canvas`, `siyuan`, `osint-investigation`, `parallel-cli`, `1password`, and `oss-forensics`
- Built-in skills: `comfyui`, `gif-search`, `weights-and-biases`, `huggingface-hub`, `airtable`, `notion`, and `teams-meeting-pipeline`
- GitHub skills: `github-auth`, `github-code-review`, `github-issues`, `github-pr-workflow`, and `github-repo-management`

### Guidance-only surfaces

- `optional-skills/productivity/telephony/SKILL.md`
- `plugins/google_meet/SKILL.md`
- `skills/autonomous-ai-agents/hermes-agent/SKILL.md`
- `skills/autonomous-ai-agents/hermes-agent/references/webhooks.md`

## Audit corrections

- Optional or fallback credentials use `optional: true` and `required_for` so they do not block skill loading.
- DuckDuckGo and SearXNG remain keyless fallbacks; their unrelated `FIRECRAWL_API_KEY` declarations were removed from the proposed patch.
- `HYPERLIQUID_USER_ADDRESS`, `CANVAS_BASE_URL`, `SIYUAN_URL`, Microsoft Graph tenant/client IDs, Notion keyring mode, and telephony identifiers/settings are documented as local non-secret configuration.
- The AgentMail MCP example references `${AGENTMAIL_API_KEY}` instead of embedding a literal credential placeholder.
- Semantic regression coverage lives in `tests/skills/test_secret_surface_guidance.py`.

## Scope boundary

This is an inventory of the branch's audited surface, not a claim that every environment-dependent skill in the repository has been normalized. A high-signal follow-up candidate is `optional-skills/devops/cli` (`INFSH_API_KEY`). Profile-local skill mirrors are outside this repository and are intentionally excluded.

No secret or token values are recorded here.
