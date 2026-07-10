# Fork Guide for AI Agents

You are in **zapabob/hermes-agent**, an engineering fork of NousResearch/hermes-agent.
Read this file when the task touches **local-only** behaviour, upstream sync, or
Windows operations.

## Decision tree

```
Need to change Hermes core agent / gateway?
  → Prefer upstream-compatible fix; use official_with_overlay in merge policy if fork delta required.

Need new user-facing capability?
  → Plugin or skill first (see extensions/AGENTS.md).

Need to merge upstream/main?
  → harness/AGENTS.md — never hand-merge toolsets.py without overlay sanitizer.

Need to restart services or fix desktop?
  → operations/AGENTS.md

Scratch media / logs at repo root?
  → local-workspace/AGENTS.md — gitignored; do not commit.
```

## Sacred upstream constraints (still apply here)

- Do not break per-conversation prompt caching mid-session.
- Do not add core model tools without footprint review (see root AGENTS.md).
- Do not add new `HERMES_*` env vars for non-secret config — use `config.yaml`.

## Fork-specific defaults on this machine

- `HERMES_HOME`: `~/.hermes` (profiles under `~/.hermes/profiles/`)
- Web search default: CloakBrowser plugin (`plugins/web/cloakbrowser/`)
- FreeLLMAPI proxy: `plugins/model-providers/freellmapi/` @ `http://127.0.0.1:3001/v1`
- Gateway single-user: `GATEWAY_ALLOW_ALL_USERS=true` in `~/.hermes/.env`
- HF cache: `H:\elt_data\hf-cache\` when set

## What not to put in upstream PRs

Exclude from PRs to NousResearch/hermes-agent:

- `_docs/` implementation logs
- `fork/` navigation docs (this tree)
- Fork-only plugins and merge overlay policy
- Unrelated Windows path literals in cron scripts

Cherry-pick upstream-worthy fixes onto `upstream/main` branches only.

## Child guides

| File | When to read |
|------|----------------|
| [harness/AGENTS.md](harness/AGENTS.md) | Merge, overlay, conflict resolution |
| [extensions/AGENTS.md](extensions/AGENTS.md) | Plugins, toolsets, VRChat/voice/harness tools |
| [operations/AGENTS.md](operations/AGENTS.md) | Stack restart, Tailscale, cron scripts |
| [local-workspace/AGENTS.md](local-workspace/AGENTS.md) | Root clutter, gitignore boundaries |
