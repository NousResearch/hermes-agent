# Sources and provenance

Upstream: https://github.com/cloudflare/security-audit-skill — `skills/security-audit/`,
commit `c1c8a8c1471069fb0e188eeaff69b8e8db6564a8` (2026-09-14), MIT (see `LICENSE`).
Background: https://blog.cloudflare.com/build-your-own-vulnerability-harness

## What was changed for Hermes

| Upstream | Here | Why |
|---|---|---|
| `SKILL.md` frontmatter (`name`, `description`) | Hermes frontmatter + "Hermes adaptation notes" block, `When to Use`, `Prerequisites` | authoring standards; map "Task tool"/agent roles onto `delegate_task` |
| `SKILL.md` body | verbatim, with `](X.md)` links pointing at `references/X.md` and validator names prefixed `scripts/` | files moved into the Hermes skill layout |
| 13 companion `*.md` | `references/*.md`, verbatim except `<skill-dir>/validate-*.cjs` → `<skill-dir>/scripts/validate-*.cjs` (2 files, 3 lines) | validators moved |
| `report-schema.json`, `validate-*.cjs`, `validate-*.test.cjs` | `scripts/`, byte-identical | `validate-findings.cjs` resolves the schema via `__dirname` |

Re-sync: copy the upstream directory over `references/` + `scripts/`, re-apply the
two path rewrites above, refresh the pinned commit in `SKILL.md`, and run
`node --test scripts/*.test.cjs` (65 tests at the pinned commit).
