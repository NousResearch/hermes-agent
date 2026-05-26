# Skills Loading (Hit Network)

- Source directory: ~/.hermes/skills/hit-network/
- Discovery path: agent/skill_commands.py scans SKILL.md files across ~/.hermes/skills and any configured external_dirs in ~/.hermes/config.yaml.
- Command surface: each skill becomes a /slash command via scan_skill_commands(); platform-disabled filtering respects HERMES_PLATFORM and gateway session context.
- Frontmatter: name, description, triggers, when_to_use, tags; metadata.hermes.config variables are surfaced into the message via _inject_skill_config.
- Absolute skill_dir is included in the injected message so tools can run bundled scripts by absolute path.
