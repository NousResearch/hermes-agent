---
name: example-skill
description: An example skill demonstrating the skill file format and structure
---

# Example Skill

This is an example skill file that demonstrates how to create skills for the Hermes Agent.

## Skill File Format

Skills are markdown files with YAML frontmatter at the top:

```yaml
---
name: your-skill-name
description: A brief one-line description of what this skill does
---
```

The frontmatter fields:
- **name**: The identifier used to reference this skill (lowercase, hyphens for spaces)
- **description**: A brief description shown when listing skills (keep under 200 chars)

## Writing Effective Skills

### 1. Be Specific and Actionable

Good skills provide clear, actionable instructions:

```
When reviewing code:
1. Check for security vulnerabilities first
2. Verify error handling is comprehensive
3. Ensure tests cover edge cases
```

### 2. Include Examples

Show concrete examples of what you want:

```python
# Good: Descriptive variable names
user_authentication_token = get_token()

# Bad: Cryptic abbreviations  
uat = gt()
```

### 3. Define When to Use

Help the agent understand when this skill applies:

> Use this skill when: reviewing pull requests, auditing security, or checking code quality.

## Skill Categories

Consider organizing skills by purpose:

- **Conventions**: Coding standards, API patterns, naming rules
- **Workflows**: Step-by-step processes for deployments, reviews, releases
- **Knowledge**: Domain-specific information, system architecture, gotchas
- **Templates**: Boilerplate for common tasks, response formats

## Tips

1. Keep the description concise - it's shown in the skills list
2. Use headers to organize longer skills
3. Include code examples where helpful
4. Reference other skills if they're related
