# Pull Request Description Template

Use this template when the user asks to open or write a PR description.
Fill in each section — skip none unless explicitly told to.

---

## PR Title

```
<type>(<scope>): <short imperative description under 72 chars>
```

Examples:
- `feat(gateway): add reconnect backoff for WhatsApp session failures`
- `fix(cli): prevent crash in save_config_value when model is a string`
- `docs(skills): add git-workflow bundled skill`

---

## PR Body Template

```markdown
## What

[1–3 sentences. Describe the change concisely. What file/component changed and what it now does differently.]

## Why

[Root cause or missing capability. Why does this matter? Reference any related issue with `Closes #N` or `Related to #N`.]

## How

[Your approach. Any trade-offs or alternatives considered. If security-relevant, say so explicitly.]

## Testing

- **Tested on:** [OS, Python version — e.g. Ubuntu 24.04, Python 3.11]
- **Manual steps:**
  1. [Step to reproduce before]
  2. [Step to verify after]
- **Automated:** `pytest tests/<path> -v`

## Checklist

- [ ] `pytest tests/ -v` passes
- [ ] Tested manually on the changed code path
- [ ] No secrets or API keys in diff
- [ ] Cross-platform impact considered (Windows / macOS)
- [ ] PR is focused — one logical change only
```

---

## Commit Message Format (Conventional Commits)

```
<type>(<scope>): <description>

[optional body]

[optional footer: Closes #N | BREAKING CHANGE: description]
```

| Type | Use for |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Tests |
| `refactor` | Code restructure, no behavior change |
| `chore` | Build, CI, dependency updates |
| `perf` | Performance improvement |
| `security` | Security fix |

Scopes for hermes-agent: `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security`
