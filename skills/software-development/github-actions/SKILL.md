---
name: github-actions
description: Create, debug, and manage GitHub Actions CI/CD workflows. Write workflow YAML, diagnose failures, manage secrets, and optimize build times.
version: 1.0.0
author: Tugrul Guner
license: MIT
metadata:
  hermes:
    tags: [GitHub, CI/CD, Actions, Workflows, Automation, Testing, Deployment]
    related_skills: [github-pr-workflow, github-auth, docker]
---

# GitHub Actions CI/CD

Complete guide for creating, debugging, and managing GitHub Actions workflows.

## Prerequisites

- Authenticated with GitHub (see `github-auth` skill)
- Inside a git repository with a GitHub remote
- `gh` CLI installed (recommended but not required)

---

## 1. Workflow File Basics

Workflow files live in `.github/workflows/` and use YAML format.

### Minimal CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
```

### Common Triggers

```yaml
on:
  push:
    branches: [main, develop]
    paths:
      - "src/**"              # Only trigger on source changes
      - "!docs/**"            # Ignore docs changes
  pull_request:
    types: [opened, synchronize, reopened]
  schedule:
    - cron: "0 9 * * 1"      # Every Monday at 9am UTC
  workflow_dispatch:           # Manual trigger from GitHub UI
    inputs:
      environment:
        description: "Deploy target"
        required: true
        default: "staging"
        type: choice
        options: [staging, production]
```

---

## 2. Common Workflow Patterns

### Matrix Testing (Multiple Python/Node Versions)

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
```

### Caching Dependencies

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with:
      python-version: "3.11"
  - uses: actions/cache@v4
    with:
      path: ~/.cache/pip
      key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
      restore-keys: |
        ${{ runner.os }}-pip-
  - run: pip install -r requirements.txt
```

### Docker Build and Push

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Deploy on Tag

```yaml
on:
  push:
    tags: ["v*"]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - run: echo "Deploying ${{ github.ref_name }}"
      # Add your deploy steps here
```

### Lint + Test + Build Pipeline

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check .

  test:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install build
      - run: python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

---

## 3. Debugging Failed Workflows

### Via gh CLI

```bash
# List recent workflow runs
gh run list --limit 10

# View a specific failed run
gh run view RUN_ID

# View failed step logs
gh run view RUN_ID --log-failed

# View full logs
gh run view RUN_ID --log

# Re-run failed jobs
gh run rerun RUN_ID --failed

# Watch a running workflow
gh run watch
```

### Via curl (no gh)

```bash
# List workflow runs
OWNER_REPO=$(git remote get-url origin | sed -E 's|.*github.com[:/]||;s|\.git$||')
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/$OWNER_REPO/actions/runs?per_page=5" | \
  jq '.workflow_runs[] | {id, name, status, conclusion, created_at}'

# Get failed job logs
curl -sL -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/$OWNER_REPO/actions/runs/RUN_ID/logs" -o logs.zip
unzip logs.zip -d run-logs/
```

### Common Failure Causes

1. **"Resource not accessible by integration"** — Missing permissions. Add to workflow:
   ```yaml
   permissions:
     contents: read
     pull-requests: write
   ```

2. **Cache miss every run** — Check the cache key. Use `hashFiles()` on lock files.

3. **Flaky tests** — Use `retry` action or `continue-on-error: true` for non-critical steps.

4. **Rate limits** — Add delays between API calls. Use `GITHUB_TOKEN` instead of PAT when possible.

5. **"Node.js 16 actions are deprecated"** — Update action versions (e.g., `actions/checkout@v4`).

---

## 4. Secrets and Variables

### Managing Secrets

```bash
# Set a repository secret
gh secret set MY_SECRET

# Set from a file
gh secret set MY_SECRET < secret.txt

# Set for a specific environment
gh secret set MY_SECRET --env production

# List secrets (names only — values are never shown)
gh secret list

# Delete a secret
gh secret delete MY_SECRET
```

### Managing Variables (non-sensitive config)

```bash
# Set a repository variable
gh variable set MY_VAR --body "my-value"

# List variables
gh variable list
```

### Using in Workflows

```yaml
steps:
  - run: echo "Deploying to ${{ vars.DEPLOY_TARGET }}"
    env:
      API_KEY: ${{ secrets.API_KEY }}
```

---

## 5. Useful Workflow Snippets

### Cancel Previous Runs on New Push

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

### Run Only When Specific Files Change

```yaml
on:
  push:
    paths:
      - "src/**"
      - "tests/**"
      - "pyproject.toml"
```

### Conditional Steps

```yaml
steps:
  - name: Deploy to production
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    run: ./deploy.sh

  - name: Comment PR with preview URL
    if: github.event_name == 'pull_request'
    run: gh pr comment ${{ github.event.number }} --body "Preview: https://preview.example.com/${{ github.sha }}"
```

### Job Outputs (Pass Data Between Jobs)

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.value }}
    steps:
      - id: version
        run: echo "value=$(cat VERSION)" >> $GITHUB_OUTPUT

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploying version ${{ needs.build.outputs.version }}"
```

---

## 6. Optimization

### Speed Up Workflows

1. **Cache aggressively** — pip, npm, cargo, go modules
2. **Use `concurrency` to cancel stale runs**
3. **Split large test suites** with matrix strategy
4. **Use `paths` filter** to skip unnecessary runs
5. **Pin action versions by SHA** for security and reproducibility

### Reduce Costs

- Use `ubuntu-latest` (cheapest runner)
- Avoid running on every push to every branch — use path filters
- Set `timeout-minutes` to prevent runaway jobs:
  ```yaml
  jobs:
    test:
      timeout-minutes: 15
  ```

## Pitfalls

- **Secrets in logs**: Never `echo ${{ secrets.X }}`. GitHub masks them but shell expansion can leak.
- **Default GITHUB_TOKEN permissions**: Minimal by default since Feb 2023. Explicitly declare what you need.
- **actions/checkout depth**: Default is shallow clone (depth=1). Use `fetch-depth: 0` for git history operations.
- **Workflow file must be on default branch**: New workflows only trigger after being merged to main.
- **YAML indentation**: Spaces only, no tabs. 2-space indent is conventional.
- **Expression syntax**: `${{ }}` is GitHub expression. Shell `$()` is different — don't mix them up.
