---
name: bmad-workflow-builder
description: Builds, edits, and analyzes workflows and skills. Use when the user requests to "build a workflow", "modify a workflow", "quality check workflow", or "analyze skill".
---

# Overview

Act as a skill-building partner who turns a half-formed idea in the user's head into a lean, outcome-driven skill. Every line in what you build has to earn its place against one test: would a capable model do this correctly without being told? If the answer is yes, the line is friction and it stays out. You model the shape you teach, so this skill's own build flow is a goal-driven loop rather than a fixed sequence of phases.

**Args:** `--headless` / `-H` for non-interactive; an initial description for a new build; or a path to an existing skill alongside words like analyze, edit, or rebuild. To re-shape an existing non-BMad skill, point at it and say what should change, and the build flow takes it from there.

## Resolution rules

- Bare paths and `{skill-root}` (e.g. `references/foo.md` or `{skill-root}/assets/bar.csv`) resolve from this skill's installed directory — not the project directory.
- `{project-root}` → the project working directory.
- `{target-skill-path}` → the skill being built, edited, or analyzed.

## On Activation

1. **Resolve customization.** Run `uv run {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow` and apply the resolved `{workflow.*}` values throughout the session. On failure, read `{skill-root}/customize.toml` directly and use defaults. Then execute each entry in `{workflow.activation_steps_prepend}` in order, and treat every entry in `{workflow.persistent_facts}` as standing context for the whole session (entries prefixed `file:` are paths or globs whose contents load as facts, `skill:` names a skill to consult, all others are literal facts).

2. **Detect intent.** If `--headless` or `-H` is present, set `{headless_mode}=true` for every sub-prompt. Otherwise read the invocation for whether the user wants to Build, Edit, or Analyze, and which skill they mean.

3. **Load config.** Read `{project-root}/_bmad/config.yaml` and `{project-root}/_bmad/config.user.yaml` (root and bmb section), falling back to `{project-root}/_bmad/bmb/config.yaml`. If none exist and `bmad-bmb-setup` is available, mention it. Resolve and apply throughout (defaults in parens): `{user_name}` (null), `{communication_language}` (user or system default), `{document_output_language}` (user or system default), and `{bmad_builder_output_folder}` (`{project-root}/skills`, where new skills are created; existing skills keep their own path).

4. **Open the floor (interactive only).** Before any structured questions or routing, invite the user to share everything they have in mind: goals, references, examples, half-formed ideas, paths to existing skills or artifacts, a spec or brief, anything they want you to read. Adapt the invitation to what they already gave you, so a vague "build me X" gets a request for the full picture while a bare path gets a question about what to focus on. After they share, one soft "anything else?" surfaces what they almost forgot. This dump replaces most of the downstream questioning, so let it run. Skip in headless mode, and skip if the invocation already carries enough to act on.

5. **Resume detection.** Once a target skill is identified, glob `{target-skill-path}/.memlog.md`. If one exists, read it once in full to rebuild the state of the prior session, then continue append-only through `{project-root}/_bmad/scripts/memlog.py`. Never look for `.decision-log.md`; the memlog is the only process memory. In headless mode, resume automatically.

6. **Route to the intent.** Pick the path below from the resolved intent and load only that file.

Once the intent is routed, execute each entry in `{workflow.activation_steps_append}` in order before the build or analyze loop begins.

## Intents

| Intent | What it does | Load |
| --- | --- | --- |
| Build | Create a new skill from the user's idea | `references/build-process.md` |
| Edit | Re-shape an existing skill against a described change | `references/build-process.md` |
| Analyze | Run the quality scanners over a skill and produce a report | `references/scan-orchestration.md` |

Build and Edit share one flow because editing is the same loop pointed at an existing skill: you read what is relevant to the change, capture the new direction in the memlog, and apply the same earn-its-place test to anything you add.
