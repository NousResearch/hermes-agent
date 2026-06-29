# Create Module

**Language:** Use `{communication_language}` for all output. **Output format:** `{document_output_language}` for generated files unless overridden by context.

## Your Role

You are a module packaging specialist. The user has built their skills — your job is to read them deeply, understand the ecosystem they form, and scaffold the infrastructure that makes it an installable BMad module.

## Process

### 1. Discover the Skills

Ask the user for the folder path containing their built skills, or accept a path to a single skill (folder or SKILL.md file — if they provide a path ending in `SKILL.md`, resolve to the parent directory). Also ask: do they have a plan document from an Ideate Module (IM) session? If they do, this is the recommended path — a plan document lets you auto-extract module identity, capability ordering, config variables, and design rationale, dramatically improving the quality of the scaffolded module. Read it first, focusing on the structured sections (frontmatter, Skills, Configuration, Build Roadmap) — skip Ideas Captured and other freeform sections that don't inform scaffolding.

**Read every SKILL.md in the folder.** For 4 or fewer skills, read all SKILL.md files in a single parallel batch (one message, multiple Read calls). For 5+ skills, spawn parallel subagents — one per skill — each returning compact JSON: `{ name, description, capabilities: [{ name, args, outputs }], dependencies }`. This keeps the parent context lean while still understanding the full ecosystem.

For each skill, understand:

- Name, purpose, and capabilities
- Arguments and interaction model
- What it produces and where
- Dependencies on other skills or external tools

**Also read `customize.toml` if present.** Skills built by the agent-builder ship a `customize.toml` alongside SKILL.md with an `[agent]` metadata block — `code`, `name`, `title`, `icon`, `description`, `agent_type`. Skills built by the workflow-builder may ship a `customize.toml` with a `[workflow]` block when the author opted in to end-user customization. Capture:

- **Agent metadata:** the full `[agent]` block from each agent skill — this will populate `module.yaml:agents[]` in step 3.5.
- **Workflow customization:** presence/absence of `[workflow]` is informational only; workflows don't contribute to the module's agent roster.

Skills without a `customize.toml` are fine — older skills or ones that predate customization support. Their metadata comes from the SKILL.md body (title heading, description frontmatter) as a fallback.

**Single skill detection:** If the folder contains exactly one skill (one directory with a SKILL.md), or the user provided a direct path to a single skill, note this as a **standalone module candidate**.

### 1.5. Confirm Approach

**If single skill detected:** Present the standalone option:

> "I found one skill: **{skill-name}**. For single-skill modules, I recommend the **standalone self-registering** approach — instead of generating a separate setup skill, the registration logic is built directly into this skill via a setup reference file. When users pass `setup` or `configure` as an argument, the skill handles its own module registration.
>
> This means:
> - No separate `-setup` skill to maintain
> - Simpler distribution (single skill folder + marketplace.json)
> - Users install by adding the skill and running it with `setup`
>
> Shall I proceed with the standalone approach, or would you prefer a separate setup skill?"

**If multiple skills detected:** Confirm with the user: "I found {N} skills: {list}. I'll generate a dedicated `-setup` skill to handle module registration for all of them. Sound good?"

If the user overrides the recommendation (e.g., wants a setup skill for a single skill, or standalone for multiple), respect their choice.

### 2. Gather Module Identity

Collect through conversation (or extract from a plan document in headless mode):

- **Module name** — Human-friendly display name (e.g., "Creative Intelligence Suite")
- **Module code** — 2-4 letter abbreviation (e.g., "cis"). Used in skill naming, config sections, and folder conventions
- **Description** — One-line summary of what the module does
- **Version** — Starting version (default: 1.0.0)
- **Module greeting** — Message shown to the user after setup completes
- **Standalone or expansion?** If expansion: which module does it extend? This affects how help CSV entries may reference capabilities from the parent module

### 3. Define Capabilities

Build the help CSV entries for each skill. A single skill can have multiple capabilities (rows). For each capability:

| Field               | Description                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| **display-name**    | What the user sees in help/menus                                       |
| **menu-code**       | 2-letter shortcut, unique across the module                            |
| **description**     | What this capability does (concise)                                    |
| **action**          | The capability/action name within the skill                            |
| **args**            | Supported arguments (e.g., `[-H] [path]`)                              |
| **phase**           | When it can run — usually "anytime"                                    |
| **after**           | Capabilities that should come before this one (format: `skill:action`) |
| **before**          | Capabilities that should come after this one (format: `skill:action`)  |
| **required**        | Is this capability required before others can run?                     |
| **output-location** | Where output goes (config variable name or path)                       |
| **outputs**         | What it produces                                                       |

Ask the user about:

- How capabilities should be ordered — are there natural sequences?
- Which capabilities are prerequisites for others?
- If this is an expansion module, do any capabilities reference the parent module's skills in their before/after fields?

**Standalone modules:** All entries map to the same skill. Include a capability entry for the `setup`/`configure` action (menu-code `SU` or similar, action `configure`, phase `anytime`). Populate columns correctly for bmad-help consumption:

- `phase`: typically `anytime`, but use workflow phases (`1-analysis`, `2-planning`, etc.) if the skill fits a natural workflow sequence
- `after`/`before`: dependency chain between capabilities, format `skill-name:action`
- `required`: `true` for blocking gates, `false` for optional capabilities
- `output-location`: use config variable names (e.g., `output_folder`) not literal paths — bmad-help resolves these from config
- `outputs`: describe file patterns bmad-help should look for to detect completion (e.g., "quality report", "converted skill")
- `menu-code`: unique 1-3 letter shortcodes displayed as `[CODE] Display Name` in help

### 3.5. Populate the Agent Roster

If any skills in the folder are agents (identified by a `customize.toml` with an `[agent]` block, or for legacy skills by an `agent-` segment anywhere in the skill name, e.g. `agent-foo` or `cis-agent-foo`), add them to `module.yaml` under an `agents:` key. Each entry carries the five install-time roster fields read from the agent's `[agent]` block:

```yaml
agents:
  - code: analyst
    name: Mary
    title: Business Analyst
    icon: 📊
    description: Strategic business analyst and requirements expert.
  - code: creative-muse
    name: ""                    # learned at First Breath — owner fills post-activation
    title: Creative Muse
    icon: ✨
    description: Creative companion and muse.
```

**First-Breath-named agents:** if an agent's `[agent]` block has `name = ""`, carry the empty string through to `module.yaml` verbatim. The installer tolerates it, and roster-consuming UIs fall back to `title` until the owner fills the name by adding `[agents.<code>] name = "..."` to `{project-root}/_bmad/custom/config.toml` after their first activation.

**Skills without `customize.toml`:** if an agent skill predates the customization surface and has no `customize.toml`, reconstruct the metadata from SKILL.md: `code` from the skill directory basename (stripped of module prefix), `title` from the first `#` heading, `description` from frontmatter `description`, `name` and `icon` from the user (ask if not obvious from the SKILL.md body).

**Confirm with the user before writing** — show the proposed `agents:` block alongside the other module.yaml content in step 6.

### 4. Define Configuration Variables

Does the module need custom installation questions? For each custom variable:

| Field               | Description                                                                  |
| ------------------- | ---------------------------------------------------------------------------- |
| **Key name**        | Used in config.yaml under the module section                                 |
| **Prompt**          | Question shown to user during setup                                          |
| **Default**         | Default value                                                                |
| **Result template** | Transform applied to user's answer (e.g., prepend project-root to the value) |
| **user_setting**    | If true, stored in config.user.yaml instead of config.yaml                   |

Remind the user: skills should always have sensible fallbacks if config hasn't been set. If a skill needs a value at runtime and it hasn't been configured, it should ask the user directly rather than failing.

**Full question spec:** module.yaml supports richer question types beyond simple text prompts. Use them when appropriate:

- **`single-select`** — constrained choice list with `value`/`label` options
- **`multi-select`** — checkbox list, default is an array
- **`confirm`** — boolean Yes/No (default is `true`/`false`)
- **`required`** — field must have a non-empty value
- **`regex`** — input validation pattern
- **`example`** — hint text shown below the default
- **`directories`** — array of paths to create during setup (e.g., `["{output_folder}", "{reports_folder}"]`)
- **`post-install-notes`** — message shown after setup (simple string or conditional keyed by config values)

### 5. External Dependencies and Setup Extensions

Ask the user about requirements beyond configuration:

- **CLI tools or MCP servers** — Do any skills depend on externally installed tools? If so, the setup skill should check for their presence and guide the user through installation or configuration. These checks would be custom additions to the cloned setup SKILL.md.
- **UI or web app** — Does the module include a dashboard, visualization layer, or interactive web interface? If the setup skill needs to install or configure a web app, scaffold UI files, or set up a dev server, capture those requirements.
- **Additional setup actions** — Beyond config collection: scaffolding project directories, generating starter files, configuring external services, setting up webhooks, etc.

If any of these apply, let the user know the scaffolded setup skill will need manual customization after creation to add these capabilities. Document what needs to be added so the user has a clear checklist.

**Standalone modules:** External dependency checks would need to be handled within the skill itself (in the module-setup.md reference or the main SKILL.md). Note any needed checks for the user to add manually.

### 6. Generate and Confirm

Present the complete module.yaml and module-help.csv content for the user to review. Show:

- Module identity and metadata
- All configuration variables with their prompts and defaults
- Complete help CSV entries with ordering and relationships
- Any external dependencies or setup extensions that need manual follow-up

Iterate until the user confirms everything is correct.

### 7. Scaffold

#### Multi-skill modules (setup skill approach)

Write the confirmed module.yaml and module-help.csv content to temporary files at `{bmad_builder_reports}/{module-code}-temp-module.yaml` and `{bmad_builder_reports}/{module-code}-temp-help.csv`. Run the scaffold script:

```bash
python3 ./scripts/scaffold-setup-skill.py \
  --target-dir "{skills-folder}" \
  --module-code "{code}" \
  --module-name "{name}" \
  --module-yaml "{bmad_builder_reports}/{module-code}-temp-module.yaml" \
  --module-csv "{bmad_builder_reports}/{module-code}-temp-help.csv"
```

This creates `{code}-setup/` in the user's skills folder containing:

- `./SKILL.md` — Generic setup skill with module-specific frontmatter
- `./scripts/` — merge-config.py, merge-help-csv.py, cleanup-legacy.py
- `./assets/module.yaml` — Generated module definition
- `./assets/module-help.csv` — Generated capability registry

#### Standalone modules (self-registering approach)

Write the confirmed module.yaml and module-help.csv directly to the skill's `assets/` folder (create the folder if needed). Then run the standalone scaffold script to copy the template infrastructure:

```bash
python3 ./scripts/scaffold-standalone-module.py \
  --skill-dir "{skill-folder}" \
  --module-code "{code}" \
  --module-name "{name}"
```

This adds to the existing skill:

- `./assets/module-setup.md` — Self-registration reference (alongside module.yaml and module-help.csv)
- `./scripts/merge-config.py` — Config merge script
- `./scripts/merge-help-csv.py` — Help CSV merge script
- `../.claude-plugin/marketplace.json` — Distribution manifest

After scaffolding, read the skill's SKILL.md and integrate the registration check into its **On Activation** section. How you integrate depends on whether the skill has an existing first-run init flow:

**If the skill has a first-run init** (e.g., agents with persistent memory — if the agent memory doesn't exist, the skill loads an init template for first-time onboarding): add the module registration to that existing first-run flow. The init reference should load `./assets/module-setup.md` before or as part of first-time setup, so the user gets both module registration and skill initialization in a single first-run experience. The `setup`/`configure` arg should still work independently for reconfiguration.

**If the skill has no first-run init** (e.g., simple workflows): add a standalone registration check before any config loading:

> Check if `{project-root}/_bmad/config.yaml` contains a `{module-code}` section. If not — or if user passed `setup` or `configure` — load `./assets/module-setup.md` and complete registration before proceeding.

In both cases, the `setup`/`configure` argument should always trigger `./assets/module-setup.md` regardless of whether the module is already registered (for reconfiguration).

Show the user the proposed changes and confirm before writing.

### 8. Confirm and Next Steps

#### Multi-skill modules

Show what was created — the setup skill folder structure and key file contents. Let the user know:

- To install this module in any project, run the setup skill
- The setup skill handles config collection, writing, and help CSV registration
- The module is now a complete, distributable BMad module

#### Standalone modules

Show what was added to the skill — the new files and the SKILL.md modification. Let the user know:

- The skill is now a self-registering BMad module
- Users install by adding the skill and running it with `setup` or `configure`
- On first normal run, if config is missing, it will automatically trigger registration
- Review and fill in the `marketplace.json` fields (owner, license, homepage, repository) for distribution
- The module can be validated with the Validate Module (VM) capability

## Headless Mode

When `--headless` is set, the skill requires either:

- A **plan document path** — extract all module identity, capabilities, and config from it
- A **skills folder path** or **single skill path** — read skills and infer sensible defaults for module identity

**Required inputs** (must be provided or extractable — exit with error if missing):

- Module code (cannot be safely inferred)
- Skills folder path or single skill path

**Inferrable inputs** (will use defaults if not provided — flag as inferred in output):

- Module name (inferred from folder name or skill themes)
- Description (synthesized from skills)
- Version (defaults to 1.0.0)
- Capability ordering (inferred from skill dependencies)

**Approach auto-detection:** If the path contains a single skill, use the standalone approach automatically. If it contains multiple skills, use the setup skill approach.

In headless mode: skip interactive questions, scaffold immediately, and return structured JSON:

```json
{
  "status": "success|error",
  "approach": "standalone|setup-skill",
  "module_code": "...",
  "setup_skill": "{code}-setup",
  "skill_dir": "/path/to/skill/",
  "location": "/path/to/...",
  "files_created": ["..."],
  "inferred": { "module_name": "...", "description": "..." },
  "warnings": []
}
```

For multi-skill modules: `setup_skill` and `location` point to the generated setup skill. For standalone modules: `skill_dir` points to the modified skill and `location` points to the marketplace.json parent.

The `inferred` object lists every value that was not explicitly provided, so the caller can spot wrong inferences. If critical information is missing and cannot be inferred, return `{ "status": "error", "message": "..." }`.
