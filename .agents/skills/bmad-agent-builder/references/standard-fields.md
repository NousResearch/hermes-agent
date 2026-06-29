# Standard Agent Fields

## Frontmatter Fields

Only these fields go in the YAML frontmatter block:

| Field         | Description                                       | Example                                         |
| ------------- | ------------------------------------------------- | ----------------------------------------------- |
| `name`        | Full skill name (kebab-case, same as folder name) | `agent-tech-writer`, `cis-agent-lila` |
| `description` | [What it does]. [Use when user says 'X' or 'Y'.]  | See Description Format below                    |

## Content Fields

These are used within the SKILL.md body — never in frontmatter:

| Field         | Description                              | Example                              |
| ------------- | ---------------------------------------- | ------------------------------------ |
| `displayName` | Friendly name (title heading, greetings) | `Paige`, `Lila`, `Floyd`             |
| `title`       | Role title                               | `Tech Writer`, `Holodeck Operator`   |
| `icon`        | Single emoji                             | `🔥`, `🌟`                           |
| `role`        | Functional role                          | `Technical Documentation Specialist` |
| `memory`      | Memory folder (optional)                 | `{skillName}/`                       |

### Memory Agent Fields (bootloader SKILL.md only)

These fields appear in memory agent SKILL.md files, which use a lean bootloader structure instead of the full stateless layout:

| Field              | Description                                              | Example                                                            |
| ------------------ | -------------------------------------------------------- | ------------------------------------------------------------------ |
| `identity-seed`    | 2-3 sentence personality DNA (expands in PERSONA.md)     | "Equal parts provocateur and collaborator..."                      |
| `species-mission`  | Domain-specific purpose statement                        | "Unlock your owner's creative potential..."                        |
| `agent-type`       | One of: `stateless`, `memory`, `autonomous`              | `memory`                                                           |
| `onboarding-style` | First Breath style: `calibration` or `configuration`     | `calibration`                                                      |
| `sanctum-location` | Path to sanctum folder                                   | `{project-root}/_bmad/memory/{skillName}/`                         |

### Sanctum Template Seed Fields (CREED, BOND, PERSONA templates)

These are content blocks the builder fills when emitting the sanctum templates. They are NOT template variables for init-script substitution — they are baked into the agent's template files as real content.

| Field                       | Destination Template    | Description                                                  |
| --------------------------- | ----------------------- | ------------------------------------------------------------ |
| `core-values`               | CREED-template.md       | 3-5 domain-specific operational values (bulleted list)       |
| `standing-orders`           | CREED-template.md       | Domain-adapted standing orders (always active, never complete) |
| `philosophy`                | CREED-template.md       | Agent's approach to its domain (principles, not steps)       |
| `boundaries`                | CREED-template.md       | Behavioral guardrails                                        |
| `anti-patterns-behavioral`  | CREED-template.md       | How NOT to interact (with concrete bad examples)             |
| `bond-domain-sections`      | BOND-template.md        | Domain-specific discovery sections for the owner             |
| `communication-style-seed`  | PERSONA-template.md     | Initial personality expression seed                          |
| `vibe-prompt`               | PERSONA-template.md     | Prompt for vibe discovery during First Breath                |

## Customization Surface (`customize.toml`)

Every agent ships a `customize.toml` alongside SKILL.md. The file has two parts: a metadata block that is always emitted, and an override surface that is emitted only when the author opted in during build.

### Metadata block (always present)

Consumed by the installer to populate `module.yaml:agents[]` and the central config's `[agents.<code>]` section. Required for every agent regardless of archetype.

| Field         | Type   | Required | Notes                                                                 |
| ------------- | ------ | -------- | --------------------------------------------------------------------- |
| `code`        | string | yes      | Stable identifier. Matches skill directory basename (no module prefix). |
| `name`        | string | optional | Display name. Empty string is valid for First-Breath-named agents.    |
| `title`       | string | yes      | Role title. Always fillable at build time.                            |
| `icon`        | string | yes      | Single emoji.                                                         |
| `description` | string | yes      | One-sentence summary of what the agent does.                          |
| `agent_type`  | string | yes      | One of `stateless`, `memory`, `autonomous`.                           |

**First-Breath-named agents:** leave `name = ""` at build time. The owner fills it post-activation in `{project-root}/_bmad/custom/config.toml`:

```toml
[agents.<code>]
name = "..."
```

UIs tolerate empty `name` and fall back to `title`.

### Override surface (emitted only when opted in)

Loaded via `{project-root}/_bmad/scripts/resolve_customization.py` at activation. Skip entirely for agents that did not opt in to customization.

| Field                      | Type          | Purpose                                                        |
| -------------------------- | ------------- | -------------------------------------------------------------- |
| `activation_steps_prepend` | array[string] | Steps run before standard activation. Overrides append.        |
| `activation_steps_append`  | array[string] | Steps run after greet, before user input. Overrides append.    |
| `persistent_facts`         | array[string] | Facts (literal or `file:` prefixed). Overrides append.         |

### Agent-specific scalars (lifted during Configurability Discovery)

Named by purpose and suffix. Override wins (scalar merge rule).

| Naming pattern          | Use for                                       | Example                                          |
| ----------------------- | --------------------------------------------- | ------------------------------------------------ |
| `<purpose>_template`    | File paths for templates the agent loads      | `style_guide_template = "resources/style.md"`    |
| `<purpose>_output_path` | Writable destinations                         | `report_output_path = "{project-root}/reports"`  |
| `on_<event>`            | Prompt or command executed at a hook point    | `on_session_close = ""`                          |

**Path resolution within scalar values:**

- Bare paths (e.g. `resources/style.md`) resolve from the skill root.
- `{project-root}/...` resolves from the project working directory — use for org-owned overrides.
- Config variables are used directly (they already contain `{project-root}`) — no double-prefix.

### How SKILL.md references the resolved values

After the resolver step runs, read customized values as `{agent.<name>}`:

```markdown
Load the style guide from `{agent.style_guide_template}`.
```

### Override files

Teams and users override without editing `customize.toml`:

- Team: `{project-root}/_bmad/custom/{skill-name}.toml`
- Personal: `{project-root}/_bmad/custom/{skill-name}.user.toml`

Both use the same `[agent]` block shape. Merge order: base (skill's `customize.toml`) → team → user.

The archetype defaults for when to emit the override surface at all live in `references/agent-quality-principles.md`.

## Overview Section Format

The Overview is the first section after the title — it primes the AI for everything that follows. Cover what the agent does, how it works (role, approach, modes), and the outcome it delivers, written as the agent's own destination rather than a description of the system.

## SKILL.md Description Format

```
{description of what the agent does}. Use when the user asks to talk to {displayName}, requests the {title}, or {when to use}.
```

## Path Rules

### Same-Folder References

Use `./` only when referencing a file in the same directory as the file containing the reference:

- From `references/build-process.md` → `./some-guide.md` (both in references/)
- From `scripts/scan.py` → `./utils.py` (both in scripts/)

### Cross-Directory References

Use bare paths relative to the skill root — no `./` prefix:

- `references/memory-system.md`
- `scripts/calculate-metrics.py`
- `assets/template.md`

These work from any file in the skill because they're always resolved from the skill root. **Never use `./` for cross-directory paths** — writing `./` before `scripts/foo.py` in a file that lives in `references/` is misleading because `scripts/` is not next to that file.

### Memory Files

Always use `{project-root}` prefix: `{project-root}/_bmad/memory/{skillName}/`

The memory `index.md` is the single entry point to the agent's memory system — it tells the agent what else to load (boundaries, logs, references, etc.). Load it once on activation; don't duplicate load instructions for individual memory files.

### Project-Scope Paths

Use `{project-root}/...` for any path relative to the project root:

- `{project-root}/_bmad/planning/prd.md`
- `{project-root}/docs/report.md`

### Config Variables

Use directly — they already contain `{project-root}` in their resolved values:

- `{output_folder}/file.md`
- Correct: `{bmad_builder_output_folder}/agent.md`
- Wrong: prefixing the same value with `{project-root}` again (double-prefix)
