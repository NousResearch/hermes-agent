# PR Disclosure Examples — Generic Placeholders

Use the following patterns when including AI-agent disclosure in PR descriptions, issue comments, or review comments. **Always** use a generic placeholder — never substitute a real account name or handle.

## Disclosure in PR Body

```markdown
## AI Disclosure

This pull request was prepared with assistance from an AI agent
on behalf of `<username>`.
```

Or with environment variable:

```markdown
## AI Disclosure

This pull request was prepared with assistance from an AI agent
on behalf of `${GITHUB_USERNAME}`.
```

## Disclosure in Issue Comments

```markdown
> Filed by an AI agent on behalf of `<username>`.
```

## Disclosure in Code Review Comments

```markdown
> This review was prepared with AI assistance on behalf of `<username>`.
```

## Commit Message Footer

```
Co-authored-by: <username> <user@users.noreply.github.com>
AI-assisted: true
```

## Checking Before Posting

Before submitting any AI-assisted content to a public repository:

1. Verify that the disclosure line uses `<username>` or `${GITHUB_USERNAME}` — **not** a real account name.
2. If the repository requires a specific disclosure format, follow its `CONTRIBUTING.md`.
3. Never embed personal names, email addresses, or account handles in reusable templates or examples.
