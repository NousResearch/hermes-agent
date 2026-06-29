# Template Substitution Rules

The SKILL-template provides a minimal skeleton: frontmatter, overview, agent identity sections, memory, and the activation spine. The bootloader carries no standalone config-load step — `init-sanctum` bakes config into the sanctum, so wake.py loads it as part of the identity. Everything beyond the skeleton is crafted by the builder based on what was learned during discovery. Apply these rules deterministically via `uv run scripts/process-template.py <template> -o <dest> --var key=value... --true <condition>...` — one `--var` per token, one `--true` per conditional that holds. The script fails (exit 3) on any leftover `{if-...}` marker and reports remaining `{token}` placeholders as `tokens_remaining` for you to judge against the runtime-token set.

## Frontmatter

- `{module-code-or-empty}` → Module code prefix with hyphen (e.g., `cis-`) or empty for standalone. The `bmad-` prefix is reserved for official BMad creations; user agents should not include it.
- `{agent-name}` → Agent functional name (kebab-case)
- `{skill-description}` → Two parts: [4-6 word summary]. [trigger phrases]
- `{displayName}` → Friendly display name
- `{skillName}` → Full skill name with module prefix

## Conditionals

A `--true` condition keeps the block's content (markers stripped); anything else removes the whole block including markers.

- `{if-module}` / `{if-standalone}` → module-based vs standalone agent
- `{if-memory-agent}` / `{if-stateless-agent}` → memory and autonomous agents vs stateless
- `{if-evolvable}` → the owner can teach the agent new capabilities
- `{if-pulse}` → autonomous mode (PULSE enabled)
- `{if-customizable}` → the author opted in to the override surface

Module tokens, filled when `{if-module}` holds: `{module-code}` (no trailing hyphen, e.g. `cis`) and `{module-setup-skill}` (e.g. `cis-setup`).

## Template Selection

- **Stateless agent:** `assets/SKILL-template.md` (full identity, no Three Laws/Sacred Truth)
- **Memory/autonomous agent:** `assets/SKILL-template-bootloader.md` (lean bootloader with Three Laws, Sacred Truth, Stay in Character, the Persistent Memory directive, and the four-step "Invoke & hold" activation spine)

The activation is a fixed four-step spine, not a set of renumbered paths: (1) Wake via `scripts/wake.py`; (2) Become yourself; (3) Bind the standing rules; (4) Execute the Proper Mode. The Mode in step 4 is what varies — Waking and First Breath are always present; only Pulse Mode is conditional, wrapped in `{if-pulse}` for autonomous agents. The step numbers never shift, so there is no gap to renumber; keep `{if-pulse}` strictly around the Pulse Mode bullet.

## Customize.toml Emission

Every agent ships `customize.toml` alongside SKILL.md, from `assets/customize-template.toml`. Fill the `[agent]` metadata block from the metadata gathered during discovery:

- `{agent-code}` → stable identifier (skill dir basename without module prefix)
- `{agent-name-or-empty}` → display name, or empty string for First-Breath-named agents
- `{agent-title}` → role title
- `{agent-icon}` → single emoji
- `{agent-description}` → one-sentence description
- `{agent-type}` → `stateless` | `memory` | `autonomous`

When `{if-customizable}` holds, also add the resolver step to SKILL.md and reference lifted scalars as `{agent.<name>}` in the SKILL.md body — these resolve at runtime, so emit them verbatim. When it does not hold, `customize.toml` ships metadata-only and SKILL.md uses hardcoded paths with no resolver step.

## Beyond the Template

The builder determines the rest of the agent structure — capabilities, activation flow, sanctum templates, init script, First Breath, capability routing, external skills, scripts — based on the agent's requirements. The template intentionally does not prescribe these.

## Path References

Everything the builder emits follows the bare-path convention the lint gate enforces: skill-internal paths are written bare from the skill root (`references/first-breath.md`, `scripts/wake.py`, `scripts/init-sanctum.py`, `assets/PERSONA-template.md`), `./` appears only for a file in the same directory as the file referencing it, and project-scope paths carry `{project-root}/`. This applies equally to SKILL.md, capability prompts, the sanctum templates the init script copies, and the emitted `scripts/wake.py` (from `assets/wake-template.py`, parameterized with the agent's `{skillName}`).
