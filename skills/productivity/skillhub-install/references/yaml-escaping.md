# YAML Escaping for Untrusted Input

When converting frontmatter from external sources (SkillHub, user input), values may contain YAML-breaking characters. This reference shows safe escaping patterns.

## The Problem

```yaml
# BROKEN — colon breaks parsing
description: Search: the web #1

# BROKEN — unescaped quotes
name: "My "Awesome" Skill"

# BROKEN — hash starts a comment
description: Fast search # optimized
```

## The Solution

```python
def yaml_escape(value: str) -> str:
    """Escape a string value for safe YAML output."""
    if not value:
        return '""'
    
    needs_quote = (
        ':' in value or 
        '#' in value or 
        "'" in value or 
        '"' in value or
        value.startswith(' ') or 
        value.endswith(' ') or
        value.startswith(('&', '*', '!', '%', '@', '`'))
    )
    
    if needs_quote:
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    
    return value
```

## Examples

| Input | Output | Why |
|-------|--------|-----|
| `simple` | `simple` | No special chars |
| `Search: the web` | `"Search: the web"` | Colon needs quoting |
| `Fast #1` | `"Fast #1"` | Hash needs quoting |
| `My "skill"` | `"My \"skill\""` | Quotes escaped |
| ` leading space` | `" leading space"` | Leading space |

## Usage in Frontmatter Conversion

```python
name = yaml_escape(extracted_name)
desc = yaml_escape(extracted_description)

new_fm = f'---\nname: {name}\ndescription: {desc}\n---'
```

## Related Patterns

- **JSON-safe escaping**: Use `json.dumps(value)` for JSON output
- **Shell-safe escaping**: Use `shlex.quote(value)` for shell commands
- **YAML library**: `yaml.safe_dump()` handles escaping automatically, but we build YAML strings manually for frontmatter control