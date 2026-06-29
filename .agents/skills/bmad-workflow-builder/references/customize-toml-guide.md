# customize.toml Guide

customize.toml is the only customizability mechanism a built skill ships with. There are no installer questions, no module.yaml embedding, no separate config.yaml authoring, and no settings or options concept inside the skill. When a skill needs end-user customization, it gets a customize.toml with the universal defaults baked in and the skill-specific points offered where they apply. When it does not, it ships fixed with hardcoded paths and no resolver step, and anyone who needs a change forks it.

This guide covers when to emit customize.toml, what goes in it, how overrides merge, and which mechanisms are forbidden.

## The Ask

Whether a skill gets a customize.toml is a decision made once during the build, interactive-only, defaulting to NO:

> Should this support end-user customization such as activation hooks, swappable templates, or output paths? If no, it ships fixed and anyone who needs changes forks it.

Default no. Most skills do not need a customization surface, and a surface nobody uses is friction the reader has to skip past. Headless runs also default to NO and emit customize.toml only when the invocation explicitly requests customization. Whatever is decided, log it in the memlog as a decision.

When the answer is no, emit no customize.toml, add no resolver step to activation, and use hardcoded paths throughout the skill. When the answer is yes, bake the universal defaults and offer the skill-specific points whose stages exist.

## DO-NOT-EDIT Header Convention

Every emitted customize.toml opens with a header that names the file as generated and points to the override files the user actually edits:

```toml
# DO NOT EDIT -- overwritten on every update.
#
# Workflow customization surface for {skill-name}.
# Team overrides:     {project-root}/_bmad/custom/{skill-name}.toml
# Personal overrides: {project-root}/_bmad/custom/{skill-name}.user.toml
```

The customize.toml in the skill is the base. The user never edits it, because an update overwrites it. Edits go in the two override files, which the resolver merges over the base at activation. The header carries an inline note of the merge rules so a reader knows how an override will land without leaving the file.

## Universal Baked Defaults

When customization is accepted, these four points appear in nearly every producing skill, so they are baked in by default under `[workflow]`:

| Key | Type | Default | Purpose |
|---|---|---|---|
| `activation_steps_prepend` | array | `[]` | Steps to run before standard activation (pre-flight loads, compliance checks). Overrides append. |
| `activation_steps_append` | array | `[]` | Steps to run after greet, before the workflow begins. Overrides append. |
| `persistent_facts` | array | `["file:{project-root}/**/project-context.md"]` | Static facts loaded on activation and kept in mind for the whole run. Overrides append. |
| `on_complete` | scalar | `""` | Instruction executed when the workflow reaches its terminal stage. Override wins. |

`persistent_facts` entries are each a literal sentence, a `skill:`-prefixed reference, or a `file:`-prefixed path or glob whose contents load as facts. The default glob picks up a project-context.md anywhere under the project root if one exists, and resolves to nothing when it does not.

## Offered-When-Relevant Points

Beyond the universal four, offer a point only when the matching stage exists in the skill. Offering an output-path knob to a skill that produces no artifact is a no-op surface the reader has to skip.

| Point | Offer when | Shape |
|---|---|---|
| `<purpose>_template` | The skill loads a template the user might want to swap | Scalar file path, e.g. `brief_template = "assets/brief-template.md"` |
| `<purpose>_output_path` + `run_folder_pattern` | The skill produces artifacts to a writable destination | Paired scalars; the pattern names the per-run folder |
| `doc_standards` | A finalize stage applies standards to human-consumed docs | Array of `skill:` / `file:` / plain-text directives |
| `finalize_reviewers` | A review stage gates substantive output | Array of reviewer references |
| `external_sources` | A stage pulls in outside inputs | Array of source references |
| `external_handoffs` | A stage routes output onward | Array of handoff references, `tool:` for tool-style routing |

The four arrays (`doc_standards`, `finalize_reviewers`, `external_sources`, `external_handoffs`) encode standards, not options. They are append-only lists the resolver merges, not toggles that switch behavior on and off.

Entry convention for these arrays: each entry is a `skill:` reference, a `file:` reference, or plain text, with `tool:` used for handoff-style routing. Bare paths resolve from the skill root; use `{project-root}/...` to point at an org-owned resource elsewhere in the repo.

## Three-Layer Merge Rules

Three files compose at activation: the baked base in the skill, the team override (`{skill-name}.toml`), and the personal override (`{skill-name}.user.toml`). The resolver merges them in that order, last layer winning where the rules call for a winner, and falls back to reading the three files directly if no resolver is available.

| Value kind | Merge behavior |
|---|---|
| Scalar (string, number, bool) | Override wins, last layer applied wins |
| Table | Deep-merge key by key |
| Array of tables (entries with `code` or `id`) | Match on `code`/`id`: replace the matching entry, append the new ones |
| Any other array | Append |

There is no removal mechanism by design. To suppress a baked default, override it by key (for a scalar) or fork the skill (for an array entry you cannot reach by key). An override file never shrinks a list, so a base reviewer or standard cannot be silently dropped downstream.

SKILL.md must reference resolved values as `{workflow.<name>}`, for example `{workflow.brief_template}` or `{workflow.output_path}`. A hardcoded path written beside a declared scalar silently no-ops the override, because the resolver fills `{workflow.<name>}` but the skill never reads it. The customization scanner flags exactly this hardcoded-path-beside-declared-scalar case.

## Forbidden Mechanisms

customize.toml is the sole config mechanism. The build flow never offers any of the following, and the customization scanner confirms none is present:

- Installer or install-time questions
- module.yaml embedding or generation. The workflow-builder captures module-capability metadata as handoff fields only and never authors module.yaml.
- A separate config.yaml authored by the skill for its own settings. (Reading the project's config.yaml at activation is not a customization surface; net-new skills are not generated with it, though a user may wire it in.)
- Boolean-toggle config that switches behavior on and off
- Any settings or options concept inside the built skill

Confirming script dependencies at build is also legitimate and stays, because it is a build-time check rather than a customization surface.

## Example

A complete customize.toml for an artifact-producing skill with a finalize stage:

```toml
# DO NOT EDIT -- overwritten on every update.
#
# Workflow customization surface for bmad-product-brief.
# Team overrides:     {project-root}/_bmad/custom/bmad-product-brief.toml
# Personal overrides: {project-root}/_bmad/custom/bmad-product-brief.user.toml

[workflow]

# --- Universal defaults. Merge: scalars override, arrays append. ---
activation_steps_prepend = []
activation_steps_append = []
persistent_facts = ["file:{project-root}/**/project-context.md"]
on_complete = ""

# --- Skill-specific points (stages present: template, output, finalize) ---
brief_template = "assets/brief-template.md"
output_path = "{planning_artifacts}/briefs"
run_folder_pattern = "brief-{project_name}-{date}"

# Standards applied at finalize. Append-only; base entries cannot be removed.
doc_standards = [
  "skill:bmad-editorial-review-structure",
  "skill:bmad-editorial-review-prose",
]
```

A skill that produces no artifact and has no finalize stage carries only the `[workflow]` block with the four universal defaults, and a skill that declined customization carries no customize.toml at all.
