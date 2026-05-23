## Skill Creation Guide

Use this guide when asked to create, import, or build a new skill.

There are two ways to create a skill — determine which applies before starting.

## Path A: Import from a source (GitHub link, file, etc.)

The user provides a URL or file path to existing skill content.

1. **Fetch the content** — download or copy the source file(s) into the sandbox.
2. **Identify the SKILL.md** — the source must contain a `SKILL.md` file with valid YAML frontmatter (`name` and `description`).
3. Determine the contents of the skill:
  - If the skill contains companion files (i.e. files other than SKILL.md), zip the skill's content. Ensure that the SKILL.md is at the root of the zip, not nested inside of a subdirectory. Then, call `create_skill` with the `file_path` pointing to the zip file.
  - If the skill is just a single SKILL.md file, pass the `.md` file path directly to `create_skill(file_path=...)`. The frontmatter will be automatically parsed for name and description, and the body becomes the skill content.

## SKILL.md format

Every skill must have a `SKILL.md` with YAML frontmatter:

```markdown
---
name: my-skill-name
description: One-line description of what this skill does and when to use it.
---

# Skill Title

Instructions for the agent go here...
```

**Frontmatter rules:**
- `name` (required): Lowercase alphanumeric + hyphens only. 1-64 chars. Must match `^[a-z0-9][a-z0-9-]*$`. No spaces, underscores, or uppercase.
- `description` (required): Max 1024 chars. Should explain what the skill does AND when to use it.
- Optional fields: `license`, `compatibility`, `allowed-tools`, `metadata`.

**Body:** Markdown instructions after the closing `---`. Structure with clear sections (e.g., Input, Process, Output) so agents can follow step-by-step.

## Path B: Create a new skill from scratch (document-backed)

The user describes what they want the skill to do.

1. **Draft the SKILL.md content** with a clear instruction body. Since you're providing name and description as separate parameters, the content does not need to contain frontmatter.
2. **Call create_skill** with `name`, `description`, and `initial_content` parameters (document-backed, editable).

### Capture Intent

Start by understanding the user's intent. The current conversation might already contain a routine the user wants to capture (e.g., they say "turn this into a skill"). If so, extract answers from the conversation history first — the tools used, the sequence of steps, corrections the user made, input/output formats observed. The user may need to fill the gaps, and should confirm before proceeding to the next step.

1. What should this skill enable Claude to do?
2. When should this skill trigger? (what user phrases/contexts)
3. What's the expected output format?

### Interview and Research

Proactively ask questions about edge cases, input/output formats, example files, success criteria, and dependencies.

Check available MCPs - if useful for research (searching docs, finding similar skills, looking up best practices), research in parallel via subagents if available, otherwise inline. Come prepared with context to reduce burden on the user.

### Write the skill content and metadata

Based on the user interview, fill in these components:

- **name**: Skill identifier
- **description**: When to trigger, what it does. This is the primary triggering mechanism - include both what the skill does AND specific contexts for when to use it. All "when to use" info goes here, not in the body. Note: currently Claude has a tendency to "undertrigger" skills -- to not use them when they'd be useful. To combat this, please make the skill descriptions a little bit "pushy". So for instance, instead of "How to build a simple fast dashboard to display internal Anthropic data.", you might write "How to build a simple fast dashboard to display internal Anthropic data. Make sure to use this skill whenever the user mentions dashboards, data visualization, internal metrics, or wants to display any kind of company data, even if they don't explicitly ask for a 'dashboard.'"
- **compatibility**: Required tools, dependencies (optional, rarely needed)
- **the rest of the skill :)**

### Skill Writing Guide

#### Anatomy of a Skill

```
skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter (name, description required)
│   └── Markdown instructions
└── Bundled Resources (optional)
    ├── scripts/    - Executable code for deterministic/repetitive tasks
    ├── references/ - Docs loaded into context as needed
    └── assets/     - Files used in output (templates, icons, fonts)
```

#### Progressive Disclosure

Skills use a three-level loading system:
1. **Metadata** (name + description) - Always in context (~100 words)
2. **SKILL.md body** - In context whenever skill triggers (<500 lines ideal)
3. **Bundled resources** - As needed (unlimited, scripts can execute without loading)

These word counts are approximate and you can feel free to go longer if needed.

**Key patterns:**
- Keep SKILL.md under 500 lines; if you're approaching this limit, add an additional layer of hierarchy along with clear pointers about where the model using the skill should go next to follow up.
- Reference files clearly from SKILL.md with guidance on when to read them
- For large reference files (>300 lines), include a table of contents

**Domain organization**: When a skill supports multiple domains/frameworks, organize by variant:
```
cloud-deploy/
├── SKILL.md (routine + selection)
└── references/
    ├── aws.md
    ├── gcp.md
    └── azure.md
```
Claude reads only the relevant reference file.

#### Principle of Lack of Surprise

This goes without saying, but skills must not contain malware, exploit code, or any content that could compromise system security. A skill's contents should not surprise the user in their intent if described. Don't go along with requests to create misleading skills or skills designed to facilitate unauthorized access, data exfiltration, or other malicious activities. Things like a "roleplay as an XYZ" are OK though.

#### Writing Patterns

Prefer using the imperative form in instructions.

**Defining output formats** - You can do it like this:
```markdown
## Report structure
ALWAYS use this exact template:
# [Title]
## Executive summary
## Key findings
## Recommendations
```

**Examples pattern** - It's useful to include examples. You can format them like this (but if "Input" and "Output" are in the examples you might want to deviate a little):
```markdown
## Commit message format
**Example 1:**
Input: Added user authentication with JWT tokens
Output: feat(auth): implement JWT-based authentication
```

#### Writing Style

Try to explain to the model why things are important in lieu of heavy-handed musty MUSTs. Use theory of mind and try to make the skill general and not super-narrow to specific examples. Start by writing a draft and then look at it with fresh eyes and improve it.


---

### Keep these in mind

**Importing:**
- **Never pass a directory path** to create_skill — only `.md` or `.zip` files are accepted.
- **Zip structure matters**: SKILL.md must be at the **root** of the zip, NOT inside a subdirectory. Use `zip skill.zip SKILL.md [other-files...]` from within the directory — NOT `zip -r skill.zip skill-directory/`.
- **For single `.md` files, prefer `file_path`**: Passing the `.md` path directly to `create_skill` is simpler and less error-prone than manually extracting frontmatter fields. Both approaches create an editable document-backed skill.
- When creating/importing the skill, just call the create_skill tool without talking about it (i.e. don't say "Now I'm going to call create_skill with the following parameters..."). The user doesn't need to know the implementation details of how the skill is created, just that it gets created.

**Naming:**
- Names must be lowercase with hyphens only. `my_skill` and `My Skill` will fail.
- Names must be unique per user — check if a skill with that name already exists before creating.

**Content:**
- Write clear, structured instructions the agent can follow step-by-step.
- Use markdown headings and lists for scanability.
- If the skill references external tools, list them explicitly.
