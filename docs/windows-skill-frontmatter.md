# Windows Skill Frontmatter Validation

Hermes skills use `SKILL.md` files with YAML frontmatter followed by Markdown
content. On Windows, an installation or update path must preserve real newline
characters. Literal `\n` sequences are not equivalent to line breaks.

## Validation

Run:

```powershell
pwsh -File .\scripts\validate-skill-frontmatter.ps1 `
  -SkillsRoot "$env:LOCALAPPDATA\hermes\skills"
```

Validate selected skills:

```powershell
pwsh -File .\scripts\validate-skill-frontmatter.ps1 `
  -SkillsRoot "$env:LOCALAPPDATA\hermes\skills" `
  -SkillName '3-statement-model','adversarial-ux-test','bioinformatics'
```

The validator checks:

- YAML opening delimiter at the first line;
- closing frontmatter delimiter;
- `name` and `description`;
- directory name matching the declared skill name;
- non-empty Markdown body;
- absence of literal `\n` sequences.

The validator is read-only. It does not rewrite installed files or silently
change skill content.

## Reproduction

If an installer or updater serializes newline escapes literally, the first
line may appear as:

```text
---\nmetadata:
```

The validator reports this as invalid instead of accepting a malformed file.

## Scope

This check is intentionally separate from skill content. A repair operation
should create a backup and require explicit user confirmation before writing
changes.
