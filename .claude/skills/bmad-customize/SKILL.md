---
name: bmad-customize
description: Authors and updates customization overrides for installed BMad skills. Use when the user says 'customize bmad', 'override a skill', 'change agent behavior', or 'customize a workflow'.
---

# BMad Customize

Translate the user's intent into a correctly-placed TOML override file under `{project-root}/_bmad/custom/` for a customizable agent or workflow skill. Discover, route, author, write, verify.

Scope v1: per-skill `[agent]` overrides (`bmad-agent-<role>.toml` / `.user.toml`) and per-skill `[workflow]` overrides (`bmad-<workflow>.toml` / `.user.toml`). Central config (`{project-root}/_bmad/custom/config.toml`) is out of scope — point users at the [How to Customize BMad guide](https://docs.bmad-method.org/how-to/customize-bmad/).

When the target's `customize.toml` doesn't expose what the user wants, say so plainly. Don't invent fields.

## Preflight

- No `{project-root}/_bmad/` → BMad isn't installed. Say so, stop.
- `{project-root}/_bmad/scripts/resolve_customization.py` missing → continue, but Step 6 verify falls back to manual merge.
- Both present → proceed.

## Activation

Load `_bmad/config.toml` and `_bmad/config.user.toml` from `{project-root}` for `user_name` (default `BMad`) and `communication_language` (default `English`). Greet. If the user's invocation already names a target skill AND a specific change, jump to Step 3.

## Step 1: Classify intent

- **Directed** — specific skill + specific change → Step 3.
- **Exploratory** — "what can I customize?" → Step 2.
- **Audit/iterate** — wants to review or change something already customized → Step 2, lead with skills that have existing overrides; read the existing override in Step 3 before composing.
- **Cross-cutting** — could live on multiple surfaces → Step 3, choose agent vs workflow explicitly with the user.

## Step 2: Discovery

```
python3 {skill-root}/scripts/list_customizable_skills.py --project-root {project-root}
```

Use `--extra-root <path>` (repeatable) if the user has skills installed in additional locations.

Group the returned `agents` and `workflows` for the user; for each show name, description, whether `has_team_override` or `has_user_override` is true. Surface any `errors[]`. For audit/iterate intents, lead with already-overridden entries.

Empty list: show `scanned_roots`, ask whether skills live elsewhere (offer `--extra-root`); otherwise stop.

## Step 3: Determine the right surface

Read the target's `customize.toml`. Top-level `[agent]` or `[workflow]` block defines the surface.

If a team or user override already exists, read it first and summarize what's already overridden before composing.

**Cross-cutting intent — walk both surfaces with the user:**
- Every workflow a given agent runs → agent surface (e.g. `bmad-agent-pm.toml` with `persistent_facts`, `principles`).
- One workflow only → workflow surface (e.g. `bmad-prd.toml` with `activation_steps_prepend`).
- Several specific workflows → multiple workflow overrides in sequence, not an agent override.

**Single-surface heuristic:**
- Workflow-level: template swap, output path, step-specific behavior, or a named scalar already exposed (`*_template`, `on_complete`). Surgical, reliable.
- Agent-level: persona, communication style, org-wide facts, menu changes, behavior that should apply to every workflow the agent dispatches.

When ambiguous, present both with tradeoff, recommend one, let the user decide.

Intent outside the exposed surface (step logic, ordering, anything not in `customize.toml`): say so; offer `activation_steps_prepend`/`append` or `persistent_facts` as approximations, or recommend `bmad-builder` to create a custom skill.

## Step 4: Compose the override

Translate plain-English into TOML against the target's `customize.toml` fields. If an existing override was read, frame the change as additive.

Merge semantics:
- **Scalars** (`icon`, `role`, `*_template`, `on_complete`) — override wins.
- **Append arrays** (`persistent_facts`, `activation_steps_prepend`/`append`, `principles`) — team/user entries append in order.
- **Keyed arrays of tables** (menu items with `code` or `id`) — matching keys replace, new keys append.

Overrides are sparse: only the fields being changed. Never copy the whole `customize.toml`.

**Template swap** (`*_template` scalar): offer to copy the default template to `{project-root}/_bmad/custom/{skill-name}-{purpose}-template.md`, point the override at the new path, offer to help edit it.

## Step 5: Team or user placement

Under `{project-root}/_bmad/custom/`:
- `{skill-name}.toml` — team, committed. Policies, org conventions, compliance.
- `{skill-name}.user.toml` — user, gitignored. Personal tone, private facts, shortcuts.

Default by character (policy → team, personal → user), confirm before writing.

## Step 6: Show, confirm, write, verify

1. Show the full TOML. If the file exists, show a diff. Never silently overwrite.
2. Wait for explicit yes.
3. Write. Create `{project-root}/_bmad/custom/` if needed.
4. Verify:
   ```
   python3 {project-root}/_bmad/scripts/resolve_customization.py --skill <install-path> --key <agent-or-workflow>
   ```
   Show the merged output, point out the changed fields.

   **Resolver missing or fails:** read whichever layers exist — `<install-path>/customize.toml` (base), `{project-root}/_bmad/custom/{skill-name}.toml` (team), `{project-root}/_bmad/custom/{skill-name}.user.toml` (user) — apply base → team → user with the same merge rules (scalars override, tables deep-merge, `code`/`id`-keyed arrays merge by key, all other arrays append), describe how the changed fields resolve.

   **Verify shows override didn't land** (field unchanged, merge conflict, file not picked up): re-enter Step 4 with the verify output as context. Usually wrong field name, wrong merge mode (scalar vs array), or wrong scope.
5. Summarize what changed, where the file lives, how to iterate. Remind the user to commit team overrides.

## Complete when

- Override file written (or user explicitly aborted).
- User has seen resolver output (or manual fallback merge summary).
- User has acknowledged the summary.

Otherwise the skill isn't done — finish or tell the user they're exiting incomplete.

## When this skill can't help

- **Central config** (`{project-root}/_bmad/custom/config.toml`) — see the [How to Customize BMad guide](https://docs.bmad-method.org/how-to/customize-bmad/).
- **Step logic, ordering, behavior not in `customize.toml`** — open a feature request, or use `bmad-builder` to create a custom skill. Offer to help with either.
- **Skills without a `customize.toml`** — not customizable.
