# BO-004 Kanban-native execution and closeout skill default

Authority: Kanban `BO-004` (`t_f6ae95f2`). Linear is compatibility/reference only for this work item.

## Default skill

The default skill for new Kanban-authority work is:

- skill name: `kanban-native-work-execution`
- file: `~/.hermes/skills/autonomous-ai-agents/kanban-native-work-execution/SKILL.md`
- version verified during BO-004: `1.0.1`
- discoverability: `skill_view(name="kanban-native-work-execution")` succeeds and returns the skill content plus linked references

## What the skill must enforce

The skill is the default entrypoint for new work after the Kanban SSOT cutover where the namespace/domain policy allows Kanban authority.

It explicitly requires:

1. authority classification before mutation;
2. live Kanban state re-query before acting;
3. correct namespace use (`BO`, `DC`, `WS`, `RS`; no `HL` namespace);
4. routing verdict before broad reads, worktree creation, mutation, or executor handoff;
5. task-owned worktree/branch for code work;
6. focused verification, static checks, diff whitespace checks, and changed-diff secret-ish scan;
7. PR/GitHub as evidence, not work-management authority;
8. Kanban `worker_done -> review_ready -> closed` separation;
9. drift audit before final closure where available;
10. task-owned cleanup and canonical sync proof;
11. Linear only as legacy compatibility/projection when it remains useful;
12. explicit next-action disposition.

## Projection guard

GitHub PRs, dashboards, Discord summaries, wiki pages, and Linear references are evidence/projection/compatibility surfaces. They cannot override the Kanban task row, Kanban comments/events, or Kanban closeout evidence for Kanban-authority work.

If any projection claims a conflicting state, treat it as drift and fail closed until the Kanban authority item is reconciled.

## BO-004 validation performed

- Loaded `kanban-native-work-execution` with the Hermes skill loader.
- Verified the skill says new work defaults to Kanban authority.
- Patched the skill from `1.0.0` to `1.0.1` to make live Kanban state re-query and persisted routing verdict mandatory before acting.
- Verified the patched skill can still be loaded.
- No Linear authority card was created.
- No gateway restart/reload, secrets/env/BWS mutation, merge, or production action was performed.
