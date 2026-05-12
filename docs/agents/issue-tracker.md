# Issue tracker: GitHub

Issues and PRDs for this repo live primarily as GitHub issues in `NousResearch/hermes-agent`. Use the `gh` CLI for issue operations.

Local markdown under `.scratch/<feature>/` may be used for drafts, private planning, or pre-GitHub issue shaping. If a skill says "publish to the issue tracker", publish to GitHub Issues unless the user explicitly asks for a local-only draft or another workflow.

Other trackers such as Linear, Jira, or client-specific systems are only used when explicitly requested by the user.

## Primary tracker

- **Tracker**: GitHub Issues
- **Repo**: `NousResearch/hermes-agent`
- **CLI**: `gh`
- **Default write target**: GitHub Issues

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`
- **Read an issue**: `gh issue view <number> --comments`
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments`
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v`. `gh` does this automatically when run inside the clone.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Local markdown scratch workflow

Use `.scratch/<feature>/` only for temporary local drafts, private notes, or issue bodies that are not ready to publish.

Suggested files:

```text
.scratch/<feature>/
├── issue.md
├── notes.md
└── acceptance-criteria.md
```

Do not treat local scratch files as the source of truth once a GitHub issue exists. Link back to the GitHub issue from any scratch notes that remain useful.
