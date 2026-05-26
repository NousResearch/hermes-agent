# Multi-Tenant Fork Notice

This is **hermes-agent-mt** — a thin fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) patched for multi-tenant deployments managed by [Hermes Swarm Map](https://github.com/NimbleCoAI/hermes-swarm-map).

## What's Different

This fork adds **2 core patches** and **~8 adapter improvements** on top of upstream:

### Core Patches
1. **Memory context scoping** — Memory writes are scoped per-context (group/DM), not global. Each conversation thread maintains isolated memory.
2. **Context ID sanitization** — Platform-specific context IDs are normalized for safe filesystem paths.

### Adapter Improvements
- Signal: UUID-based allowlisting, group invite policy, voice memo detection, profile name setting
- Mattermost: Channel join/leave gating, per-channel allowlist, mention gating
- Telegram: Group session isolation, admin resolution

### Plugins (installed by HSM)
- `swarm_map_policy` — Group access control via HSM policy endpoint
- `boot_md` — Startup checklist execution
- `lifecycle-notify` — Startup notification hook

## Upstream Relationship

- **Upstream:** [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
- **Sync:** Weekly automated rebase via CI workflow
- **Goal:** Minimize diff. Patches that benefit upstream are submitted as PRs.
- **Rebase journal:** See `docs/rebase-journal.md`

## Using This Fork

```bash
# Docker (recommended)
docker pull ghcr.io/nimblecoai/hermes-agent:latest

# Or build from source
git clone https://github.com/NimbleCoAI/hermes-agent.git
cd hermes-agent
pip install -e ".[all]"
```

For multi-tenant management, use [Hermes Swarm Map](https://github.com/NimbleCoAI/hermes-swarm-map).

## License

Same as upstream: MIT License.
