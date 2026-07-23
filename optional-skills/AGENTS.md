# Optional Hermes Skills

The authoring standards in `../skills/AGENTS.md` apply here in full. Read that
file before changing an optional skill.

This tree is for dependency-heavy, platform-specific, experimental, or niche
skills that should not be active for every installation. Installation is
explicit through `hermes skills install official/<category>/<skill>`.

Do not move a skill into the default `skills/` tree merely to simplify setup.
First prove that its dependencies, platform coverage, prompt footprint, and
general usefulness justify default activation.
