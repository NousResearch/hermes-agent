# Standard Fields and Naming Conventions

Frontmatter, body fields, stage and hook naming, the Overview shapes, and path rules for skills the builder produces. The description format lives in `references/skill-quality-principles.md` and the full customize.toml surface lives in `references/customize-toml-guide.md`; this file points to them rather than restating them.

## Frontmatter fields

Only these two fields go in the YAML frontmatter block:

| Field | Description | Example |
| --- | --- | --- |
| `name` | Full skill name, hyphen-case, same as the folder name | `validate-json`, `cis-brainstorm` |
| `description` | A 5-8 word summary, then a trigger clause naming what the user says | See Description format below |

Nothing else belongs in frontmatter. Role, stages, hooks, and config all live in the body or in customize.toml.

## Body fields

These describe the skill inside SKILL.md, never in frontmatter:

| Field | Description | Example |
| --- | --- | --- |
| `role-guidance` | A brief expertise primer | "Act as a senior DevOps engineer" |
| `module-code` | Module code, only when the skill ships inside a module | `bmb`, `cis` |
| `input-format` | What the skill accepts | JSON file path, stdin text |
| `output-format` | What the skill returns | Validated JSON, error report |
| `composability` | How other skills call this one | "Called by quality scanners for validation" |

### Module capability handoff

When the skill ships inside a module, capture these as handoff fields for the module builder; the workflow-builder never authors module.yaml.

| Field | Description |
| --- | --- |
| `phase-name` | The module phase this skill belongs to |
| `after` / `before` | Ordering hints relative to sibling skills in the phase |
| `is-required` | Whether the phase requires this skill to complete |

## Stage naming

Stages get descriptive names that say what the stage is for, never numbered prefixes: a number implies a fixed order the model must march through and fights the outcome-driven shape, so name the stage by its goal and let routing or prose carry the order where it matters.

| Prefer | Over |
| --- | --- |
| `discover`, `plan`, `build` | `01-discover`, `02-plan`, `03-build` |
| `gather-input`, `draft`, `finalize` | `step-1-gather`, `step-2-draft` |

The same rule covers stage files on disk: `discover.md`, not `01-discover.md`. When a stage genuinely must precede another (a later stage consumes an earlier stage's output), state the dependency in the prose so the constraint is explicit, rather than relying on a number to imply it.

A simple utility usually needs no stages at all; it does one deterministic thing and returns. Reach for named stages only when the work has distinct phases a reader needs to navigate.

## Hook naming

Hook points use the `on_<event>` form, where the event names the moment the hook fires. The hook value is a prompt string or a command the skill runs at that point, empty by default.

| Hook | Fires |
| --- | --- |
| `on_complete` | After the skill finishes its work |
| `on_start` | Before the skill's first stage runs |
| `on_error` | When the skill hits an unrecoverable error |

Keep hooks to real moments the skill reaches. Do not invent hook points for events the skill never produces.

## customize.toml fields

customize.toml is the only customizability mechanism, emitted only when the author accepts the offer (default no). `references/customize-toml-guide.md` owns the whole surface: the universal baked defaults, the `<purpose>_template` / `<purpose>_output_path` / `on_<event>` naming patterns, the standards-not-options arrays, the three-layer merge rules, the override files, and the rule that SKILL.md must read `{workflow.<name>}` rather than a hardcoded path. Author against that file.

## Overview section

The Overview is the first section after the title and primes the model for everything that follows. State what the skill does, how it works, and the outcome it delivers.

| Skill type | Shape |
| --- | --- |
| Complex workflow | This skill helps you {outcome} through {approach}. Act as {role}, guiding users through {key stages}. The output is {deliverable}. |
| Simple workflow | This skill {what it does} by {approach}. Act as {role}. Use when {triggers}. Produces {output}. |
| Simple utility | This skill {what it does}. Use when {when to use}. Returns {output format}. |

## Description format

The frontmatter `description` is the primary trigger mechanism. Its two-part format, the explicit-vs-organic distinction, and the good/bad examples live in `references/skill-quality-principles.md` under "Description format." Default to explicit invocation unless the author describes organic activation during discovery.

## Role guidance

Every generated SKILL.md carries a brief role statement in the Overview or as a standalone line:

```markdown
Act as {role}. {brief expertise and approach}.
```

A skill may use a fuller identity and principles section when personality serves the work, but a single role line is enough for most.

## Path rules

### Skill-internal references

Use bare paths from the skill root for any file inside the skill, including a reference between two files in the same folder:

- `references/build-process.md`
- `references/standard-fields.md` referenced from another file in `references/`, still a bare path
- `scripts/validate.py`
- `assets/template.md`

The convention is universal: bare paths from the skill root. Never use a `./` prefix, which causes inconsistency and breaks under context compaction when the working directory shifts.

### Project-scope paths

Use `{project-root}/...` for any path relative to the project root:

- `{project-root}/_bmad/planning/prd.md`
- `{project-root}/docs/report.md`

### Anti-patterns

These are wrong; the fences keep the path linter from firing on them:

```text
{project-root}/{output_folder}/file.md   # WRONG, double-prefix; a config var already has {project-root}
_bmad/planning/prd.md                    # WRONG, bare _bmad needs a {project-root} prefix
./references/foo.md                       # WRONG, never use ./ for a skill-internal path
./scripts/foo.py                          # WRONG, bare paths from skill root only
```
