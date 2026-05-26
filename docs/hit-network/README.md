# Hit Network Augment Docs

This subtree documents Hit Network specific behavior layered on top of upstream hermes-agent. Use it to understand what differs from NousResearch/main.

Documents:
- architecture.md: summary of fork deltas sourced from hardened upstream diff
- fork-policy.md: when to PR upstream vs keep local
- skills-loading.md: how ~/.hermes/skills/hit-network flows into the agent
- prompt-builder-integration.md: prompt-builder markers and how delegate_task consumes them
- cron-conventions.md: how we register and operate crons under Hermes
