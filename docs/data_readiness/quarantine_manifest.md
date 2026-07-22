# SKILL QUARANTINE MANIFEST

Status definitions:
- verified: installed/manifest/version/source/hash audited
- quarantine: not yet verified; execute disabled pending review
- rejected: failed verification or removed

| Skill | Status | Tier | Evidence | Notes |
|-------|--------|------|----------|-------|
| autonomous-ai-agents/hermes-agent | quarantine | runtime | no audit entry | exclude until signature review complete |
| productivity/google-workspace | quarantine | integration | no audit chain | sensitive auth surface |
| computer-use | quarantine | privileged | no manifest hash | high blast radius |
| yuanbao | quarantine | external | no verified source | external platform integration |
| smart-home | quarantine | external | no audit | remove runtime access |
| data-science | quarantine | external | no manifest | workspace-visible paths |
| mlops | quarantine | external | no audit | external API dependencies |
| social-media | quarantine | external | no verified source | outbound exposure surface |
| creative/touchdesigner-mcp | quarantine | external | no verified source | third-party integration |
| creative/popular-web-designs | quarantine | external | no audit | dynamic asset surface |

Criteria for lift from quarantine:
- manifest with source, version, and sha256/hash
- known install/test provenance
- required capability scoped only
- Arthur approval logged

Automatic exclusions:
- Any skill modifying core files without approval
- Any skill with broad self-modify or download surface
