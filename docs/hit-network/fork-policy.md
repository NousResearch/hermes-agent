# Fork Policy (Hit Network)

- Generally useful changes: open an upstream PR first and keep the local patch while PR is open.
- Hit Network specific changes (gbrain, Convex, internal gateways): keep local with a `# HIT-NETWORK-PATCH` comment marker.
- Follow skills: upstream-fork-integration-pattern, upstream-migration-safety-gate.
- Rebase strategy: fetch upstream --prune weekly, rebase feature branches, resolve conflicts locally, never force-push upstream.
