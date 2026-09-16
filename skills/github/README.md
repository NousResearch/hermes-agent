# GitHub Skills

Complete GitHub workflow automation — from repository management to pull request reviews.

## Overview

This category contains 6 skills for comprehensive GitHub integration, covering authentication, repository management, issue tracking, pull request workflows, code review, and codebase analysis. Whether you're maintaining open source projects, managing team workflows, or automating contributions, these skills provide professional GitHub operations directly from Hermes.

## Available Skills

### Authentication & Setup

#### **github-auth**
GitHub authentication setup — HTTPS tokens, SSH keys, and GitHub CLI login.

**Use when:** Setting up GitHub access, configuring authentication, or switching between accounts.

**Key features:**
- HTTPS personal access token setup
- SSH key generation and configuration
- GitHub CLI (gh) authentication
- Multi-account management

---

### Repository Management

#### **github-repo-management**
Clone, create, and fork repositories; manage remotes and releases.

**Use when:** Managing repository lifecycle, working with forks, or handling releases.

**Key features:**
- Repository cloning, creation, and forking
- Remote management (add, remove, sync)
- Release creation and management
- Repository settings and configuration

---

### Issue Management

#### **github-issues**
Create, triage, label, and assign GitHub issues via gh CLI or REST API.

**Use when:** Automating issue tracking, triaging bugs, or managing project backlogs.

**Key features:**
- Issue creation with templates
- Label and milestone management
- Assignment and triage workflows
- Bulk operations and filtering

---

### Pull Request Workflow

#### **github-pr-workflow**
Complete PR lifecycle — branch creation, commits, PR opening, CI monitoring, and merging.

**Use when:** Contributing to projects, managing PR workflows, or automating merge processes.

**Key features:**
- Branch creation and management
- Commit and push operations
- PR creation with descriptions
- CI status monitoring
- Merge and cleanup workflows

---

#### **github-code-review**
Review pull requests — view diffs, leave inline comments via gh CLI or REST API.

**Use when:** Conducting code reviews, providing feedback, or automating review workflows.

**Key features:**
- Diff viewing and analysis
- Inline comment placement
- Review submission (approve, request changes, comment)
- Batch review operations

---

### Codebase Analysis

#### **codebase-inspection**
Inspect codebases using pygount — lines of code, language breakdown, and code ratios.

**Use when:** Analyzing project size, understanding language distribution, or generating codebase metrics.

**Key features:**
- Lines of code (LOC) counting
- Language breakdown and percentages
- Code, comment, and blank line ratios
- Multi-language support via pygount

---

## Quick Start

### Example: Contributing to a Project

```bash
# 1. Set up authentication (first time only)
/github-auth "Configure SSH keys for GitHub"

# 2. Fork and clone
/github-repo-management "Fork nodejs/node and clone to ~/projects/"

# 3. Create feature branch
/github-pr-workflow "Create branch fix-typo-in-readme from main"

# 4. Make changes, then open PR
/github-pr-workflow "Commit and push changes, open PR: 'Fix typo in README.md'"
```

### Example: Issue Triage Workflow

```bash
# 1. List new issues
/github-issues "Show open issues with label: bug, created in last 7 days"

# 2. Triage and label
/github-issues "Add label 'needs-reproduction' to issue #1234"

# 3. Assign to team member
/github-issues "Assign issue #1234 to @developer"
```

### Example: Code Review

```bash
# 1. View PR diff
/github-code-review "Show diff for PR #456"

# 2. Leave inline comments
/github-code-review "Comment on PR #456 line 42: Consider using const instead of let here"

# 3. Submit review
/github-code-review "Approve PR #456 with comment: LGTM, nice refactoring!"
```

### Example: Repository Analysis

```bash
# 1. Analyze current repository
/codebase-inspection "Count LOC and show language breakdown"

# 2. Compare before/after refactoring
/codebase-inspection "Show code metrics for src/ directory only"
```

## Skill Combinations

**Complete Contribution Workflow:**
1. Use `github-auth` to set up access (one-time)
2. Use `github-repo-management` to fork and clone
3. Use `github-pr-workflow` to create branch and PR
4. Use `github-code-review` to respond to review feedback

**Issue Management Pipeline:**
1. Use `github-issues` to create and triage issues
2. Use `github-pr-workflow` to create fixing PR
3. Use `github-code-review` to get review
4. Use `github-pr-workflow` to merge and close issue

**Open Source Maintenance:**
1. Use `github-issues` to triage incoming issues
2. Use `codebase-inspection` to analyze contributions
3. Use `github-code-review` to review PRs
4. Use `github-repo-management` to manage releases

**Project Analysis:**
1. Use `github-repo-management` to clone target repo
2. Use `codebase-inspection` to analyze structure
3. Use `github-issues` to review open issues and roadmap
4. Use `github-pr-workflow` to contribute improvements

## Common Workflows

### Opening Your First PR

```bash
# 1. Fork and clone (if not already done)
/github-repo-management "Fork username/project"

# 2. Create feature branch
/github-pr-workflow "Create branch add-dark-mode from main"

# 3. Make your changes in your editor...

# 4. Open PR
/github-pr-workflow "Commit with message 'Add dark mode support', push, and open PR with description"
```

### Maintainer Review Flow

```bash
# 1. List pending PRs
/github-code-review "List open PRs sorted by oldest"

# 2. Review specific PR
/github-code-review "Show diff for PR #789"

# 3. Leave feedback
/github-code-review "Request changes on PR #789: Please add unit tests for the new function"

# 4. After updates, approve and merge
/github-code-review "Approve PR #789"
/github-pr-workflow "Merge PR #789 with squash"
```

### Release Management

```bash
# 1. Analyze what changed
/codebase-inspection "Show LOC changes since last tag"

# 2. Create release
/github-repo-management "Create release v2.0.0 with notes from CHANGELOG.md"

# 3. Announce
/github-issues "Create issue: Release announcement for v2.0.0"
```

### Codebase Health Check

```bash
# 1. Get overall metrics
/codebase-inspection "Full codebase analysis with language breakdown"

# 2. Check test coverage ratio
/codebase-inspection "Compare LOC between src/ and tests/"

# 3. Identify large files
/codebase-inspection "List files over 500 lines"
```

## Authentication Setup

Most GitHub skills work through either:

**GitHub CLI (gh):**
```bash
# Install gh and authenticate
/github-auth "Set up gh CLI with OAuth"
```

**Personal Access Token:**
```bash
# Generate and configure PAT
/github-auth "Create personal access token for HTTPS"
```

**SSH Keys:**
```bash
# Generate and add SSH key
/github-auth "Set up SSH key for git operations"
```

Choose based on your workflow:
- **gh CLI** — Best for interactive operations
- **PAT** — Good for automation and scripts
- **SSH** — Preferred for git operations

## Best Practices

**Branch Naming:**
Use descriptive branch names: `feature/add-auth`, `fix/typo-readme`, `docs/api-updates`

**Commit Messages:**
Follow conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, etc.

**PR Descriptions:**
Include:
- What changed and why
- How to test
- Related issues (`Fixes #123`)
- Screenshots for UI changes

**Code Review:**
- Be specific and constructive
- Reference line numbers
- Suggest alternatives
- Acknowledge good work

## Troubleshooting

**Authentication Issues:**
```bash
/github-auth "Diagnose authentication problems"
```

**Push Rejected:**
```bash
/github-pr-workflow "Pull latest changes and rebase"
```

**Merge Conflicts:**
```bash
/github-pr-workflow "Show conflict details and resolution steps"
```

## Contributing

Found a bug or have an enhancement idea?

1. Open an issue describing the improvement
2. Fork the repository
3. Make changes to the relevant `SKILL.md`
4. Submit a pull request

## Related Categories

- **software-development/** - Development workflows and debugging
- **productivity/** - Task and project management tools
- **creative/** - Design and content creation
- **mlops/** - ML model and experiment management

---

**Questions?** Check the [Hermes Agent documentation](https://hermes-agent.nousresearch.com/docs/) or ask in the [Discord community](https://discord.gg/nousresearch).
