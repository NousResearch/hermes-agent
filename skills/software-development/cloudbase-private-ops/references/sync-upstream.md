# Syncing with Official CloudBase Skill

This private skill is an overlay. Do not fork the whole official `cloudbase` skill unless absolutely necessary.

## Source-of-truth split

- Official `cloudbase`: product capabilities, generic MCP/tool docs, platform guides.
- `cloudbase-private-ops`: this user's production safety policy, project-local MCP isolation, route-A wrapper, operation levels.

When rules conflict for this user's production environment, this overlay wins.

## Recommended sync process

1. Update or inspect official CloudBase skill through normal Hermes skill update flow.
2. Check official version and changed references.
3. Review only areas that may affect this overlay:
   - MCP setup examples;
   - mcporter call syntax;
   - CloudBase function deployment schema;
   - auth/environment binding guidance;
   - database/storage write tools.
4. If official docs add useful generic guidance, link to it or summarize it in this overlay.
5. Do not merge private route-A policy back into official skill unless it is intentionally becoming general public guidance.

## Local review record

Maintain this section when syncing:

```text
upstream skill: cloudbase
upstream version reviewed: 2.18.0
private overlay version: 1.0.0
last reviewed: 2026-05-18
review notes: initial overlay extracted from Vibe Photoing + meme migration validation
```

## Minimal official skill patch

A small pointer inside official `cloudbase` is acceptable:

```text
For this profile's production CloudBase operations, load `cloudbase-private-ops` after this skill. Its project-local MCP isolation and route-A rules override generic CLI/MCP examples.
```

Avoid large edits to official `cloudbase/SKILL.md` because they create upstream merge conflicts.

## What to verify after sync

- The official skill still loads.
- This overlay still loads.
- `related_skills: [cloudbase]` remains valid.
- Any new official recommendation involving `tcb`, global MCP, direct mcporter, or `auth.set_env` is either not production-relevant or explicitly overridden here.
