# Unitalk Desktop Documentation

Developer documentation for the Unitalk Desktop distribution fork of [Hermes Agent](https://github.com/NousResearch/hermes-agent).

## Start Here

- **[CONFLICTS.md](./CONFLICTS.md)** — Merge conflict guide: what breaks when pulling upstream, how to resolve it, and what to verify after. **Read this first before any upstream merge.**

## Core Docs

- **[REBRANDING.md](./REBRANDING.md)** — Rebranding methodology: the two-layer naming concept (AI Collaborator vs Unitalk), substitution rules table, what to never modify, verification steps, and discovered pitfalls.

- **[DECISIONS.md](./DECISIONS.md)** — Architecture decisions log: why "Collaborator" not "Agent", why 11 brand themes, why browser-auth only, why Unitalk first in providers, why `DEFAULT_SKIN_NAME = 'unitalk'`, and why the welcome plugin exists.

## Automation

- **[`scripts/apply-unitalk-branding.sh`](../scripts/apply-unitalk-branding.sh)** — Idempotent rebranding script for fresh upstream pulls. Replaces user-facing strings, fixes onboarding, applies themes, and runs verification checks. Run after every upstream merge.

## Reference

| Topic | File |
|---|---|
| Agent system prompt assembly | `agent/prompt_builder.py` |
| Onboarding text and greetings | `agent/onboarding_text.py` |
| Theme presets (11 brand themes) | `apps/desktop/src/themes/presets.ts` |
| Provider list and ordering | `apps/desktop/src/components/onboarding/providers.tsx` |
| Welcome plugin (dashboard) | `apps/desktop/src/plugins/welcome/` |
| i18n strings (8+ locales) | `apps/desktop/src/i18n/` |
| Shared metrics consent | `apps/desktop/src/i18n/*_shared_metrics.ts` |
| First-run intro copy | `apps/desktop/src/i18n/*_first_run.ts` |

## Quick Checks

```bash
# After any branding change, verify:
grep -r "Hermes Agent" apps/desktop/src/ --include="*.ts" --include="*.tsx" | grep -v node_modules  # must be empty
grep -r "HERMES_HOME" agent/ --include="*.py"  # must exist
python3 -m py_compile agent/prompt_builder.py  # must compile
```