# Skills System Improvements

This document describes the new features added to the Hermes skills system.

## 1. External Directories Filtering

### Overview

The `skills.external_dirs` config now supports advanced filtering with include/exclude glob patterns and category mapping, while maintaining full backward compatibility with the simple list-of-paths format.

### Configuration

#### Simple Format (Backward Compatible)

```yaml
skills:
  external_dirs:
    - /path/to/skills
    - ~/another/skills/path
```

#### Advanced Format (With Filtering)

```yaml
skills:
  external_dirs:
    - path: /path/to/skills
      include:
        - "anthropic-*"
        - "mcp-*"
      exclude:
        - "*test*"
        - "*sample*"
      category_map:
        "anthropic-*": "anthropic-tools"
        "mcp-*": "mcp-servers"
      enabled: true
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | string | required | Path to the external skills directory |
| `include` | list | `["*"]` | Glob patterns for skills to include |
| `exclude` | list | `[]` | Glob patterns for skills to exclude |
| `category_map` | dict | `{}` | Map glob patterns to Hermes categories |
| `enabled` | bool | `true` | Whether this directory is active |

### Examples

#### Only Include Specific Skills

```yaml
external_dirs:
  - path: /path/to/skills
    include:
      - "anthropic-*"
      - "mcp-*"
```

#### Exclude Test Skills

```yaml
external_dirs:
  - path: /path/to/skills
    exclude:
      - "*test*"
      - "*sample*"
```

#### Map to Categories

```yaml
external_dirs:
  - path: /path/to/skills
    category_map:
      "anthropic-*": "anthropic-tools"
      "mcp-*": "mcp-servers"
```

---

## 2. Batch Curator Commands

### Overview

New batch operations for the curator CLI allow you to pin, unpin, and archive multiple skills at once based on names, usage thresholds, or idle time.

### Commands

#### pin-batch

Batch pin skills by name, usage threshold, or category.

```bash
# Pin specific skills
hermes curator pin-batch --batch "skill-a,skill-b,skill-c"

# Pin skills with high usage
hermes curator pin-batch --by-usage 20

# Preview without making changes
hermes curator pin-batch --batch "skill-a,skill-b" --dry-run
```

#### unpin-batch

Batch unpin skills by name, usage threshold, or category.

```bash
# Unpin specific skills
hermes curator unpin-batch --batch "skill-a,skill-b"

# Unpin low-usage skills
hermes curator unpin-batch --by-usage 5
```

#### archive-batch

Batch archive skills by name, usage threshold, or idle days.

```bash
# Archive specific skills
hermes curator archive-batch --batch "old-skill1,old-skill2"

# Archive skills idle for 90+ days
hermes curator archive-batch --stale 90

# Skip confirmation
hermes curator archive-batch --batch "skill-a" -y
```

#### usage-filter

Show usage telemetry with filtering options.

```bash
# Show skills with minimum 10 uses
hermes curator usage-filter --min-usage 10

# Show agent-created skills as JSON
hermes curator usage-filter --provenance agent --json

# Show skills with usage between 5 and 50
hermes curator usage-filter --min-usage 5 --max-usage 50
```

---

## 3. Trigger Auto-Loading

### Overview

Skills can now declare trigger keywords in their SKILL.md frontmatter. When a user's message matches these triggers, the skill is automatically loaded and its content is injected into the conversation context.

### Configuration

#### Enable/Disable Auto-Loading

```yaml
skills:
  auto_load:
    enabled: true  # default: true
    max_skills: 5  # default: 5
```

#### Adding Triggers to a Skill

In your SKILL.md frontmatter:

```yaml
---
name: my-skill
triggers:
  - "deploy to srv1"
  - "ansible deploy"
  - "incus deploy"
---

# My Skill

This skill handles deployment tasks.
```

### How It Works

1. **Index Building**: When Hermes starts, it scans all SKILL.md files for `triggers:` in frontmatter
2. **Trigger Matching**: When a user sends a message, the trigger index is checked for matches
3. **Skill Loading**: Matching skills are loaded via `skill_view` and their content is injected
4. **Context Injection**: The loaded skill content is added to the user message via the `api_content` sidecar

### Best Practices

- **Be Specific**: Use specific phrases like "deploy to srv1" instead of generic ones like "deploy"
- **Multiple Triggers**: Add several triggers to catch different ways users might phrase the same request
- **Keep Triggers Short**: Shorter triggers are more reliable (1-3 words ideal)
- **Test Your Triggers**: Use `get_trigger_index_stats()` to verify your triggers are indexed

### Example

```yaml
---
name: rapidwebs-infra-deployment
triggers:
  - "deploy to srv1"
  - "ansible deploy rapidwebs"
  - "incus deploy"
  - "deploy infrastructure"
---

# Deployment Skill

This skill handles deploying to the RapidWebs infrastructure.
```

When a user says "deploy to srv1", this skill is automatically loaded and its content is available to the agent.

---

## Testing

### Running Tests

```bash
# Run all new tests
python3 -m pytest tests/agent/test_skill_trigger_index.py tests/test_batch_curator.py tests/test_external_dirs_filtering.py -v

# Run specific test file
python3 -m pytest tests/agent/test_skill_trigger_index.py -v
```

### Test Coverage

- **Trigger Index**: 12 tests covering index building, matching, and auto-loading
- **Batch Curator**: 12 tests covering pin/unpin/archive batch operations
- **External Dirs**: 15 tests covering config parsing, filtering, and integration

---

## Migration Guide

### Existing Configurations

No changes required. Existing `external_dirs` configurations continue to work as-is.

### Adding Filters

To add filtering to an existing directory:

```yaml
# Before
external_dirs:
  - /path/to/skills

# After
external_dirs:
  - path: /path/to/skills
    include: ["specific-*"]
    exclude: ["*test*"]
```

### Adding Triggers

To add triggers to an existing skill:

```yaml
---
name: my-existing-skill
triggers:
  - "trigger phrase 1"
  - "trigger phrase 2"
---

# Existing Skill Content
```

---

## Troubleshooting

### Auto-Loading Not Working

1. Check if auto-loading is enabled in config
2. Verify triggers are in the SKILL.md frontmatter
3. Use `get_trigger_index_stats()` to check if triggers are indexed

### Batch Commands Not Found

1. Ensure you're on the latest version with batch support
2. Check `hermes curator --help` for available commands

### External Dirs Not Filtering

1. Verify the config format is correct (dict, not string)
2. Check glob patterns match your skill names
3. Use `--dry-run` to preview changes
