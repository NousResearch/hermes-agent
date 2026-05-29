# Risk Register

| Risk | Severity | Status | Mitigation and gate |
|---|---|---|---|
| Secrets or personal runtime state are copied into the package. | Critical | Blocking | Use fresh `HERMES_HOME`; examples use placeholders only; redaction and static scans must pass. |
| Work touches a live customer host such as `<LIVE_REMOTE_USER>@<LIVE_REMOTE_HOST>` or remote `~/.hermes`. | Critical | Blocking | No live commands in package verification; preflight rejects live-path targets; final evidence reports local-only commands. |
| Dobby weights are bundled or implied. | Critical | Blocking | BYO endpoint language in docs and examples; artifact scan rejects large model files. |
| Honcho server is bundled or required. | High | Blocking for default package | Native memory tests must pass without Honcho config; Honcho remains optional external setup only. |
| Webhook inbox accepts unsigned or replayed payloads. | High | Manageable | HMAC required per route; tests cover missing, bad, replayed, oversized, and stale payloads. |
| Attachment handling reads sensitive files before review. | High | Manageable | Metadata-first quarantine; approval token required; tests cover denied and expired approvals. |
| Repo helper performs writes, commits, pushes, deploys, or destructive git. | High | Manageable | Read-only/propose-only default; denylist plus allowlist tests; no shell passthrough without policy gate. |
| Memory consent, forget, export, or delete is incomplete. | High | Manageable | Temp-home privacy tests cover each flow and prove package-owned scope. |
| Broad integrations expand the attack surface. | Medium | Manageable | Default config enables Discord and signed webhooks only; examples omit other platform env vars. |
| Discord permissions or mentions create spam or unsafe visibility. | Medium | Manageable | Preflight checks intents/channel access; allowed mentions remain restrictive by default. |
| Quota/status claims call live providers during tests. | Medium | Manageable | Mock model/quota probes; live checks are optional operator diagnostics only. |
| Demo kit leaks real data. | Medium | Manageable | Synthetic fixtures only; static scan for personal paths and secret-shaped values. |
| Existing checkout divergence causes integration drift. | Medium | Open | Stage 3A used local `origin/main` without fetch. Later workers should decide whether to fetch/rebase before code work. |
| Parallel workers collide in command routing or config files. | Medium | Open | Use `IMPLEMENTATION_SLICES.md`; S2 owns central command wiring, S1 owns package scripts/examples. |
| Product claims outrun verified implementation. | Medium | Manageable | `TRACEABILITY.md` gates must be filled with test commands before publish. |
