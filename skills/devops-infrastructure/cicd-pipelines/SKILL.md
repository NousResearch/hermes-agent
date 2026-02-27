---
name: cicd-pipelines
description: Create, manage, and debug CI/CD pipelines with GitHub Actions, GitLab CI, and other platforms. Handle secrets, artifacts, workflow triggers, and pipeline optimization.
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [CI/CD, GitHub Actions, GitLab CI, DevOps, Automation, Pipelines]
    related_skills: [github-pr-workflow, kubernetes-management]
---

# CI/CD Pipelines

Manage continuous integration and deployment pipelines across multiple platforms.

---

## 1. GitHub Actions

### Workflow Syntax

```yaml
# .github/workflows/main.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:  # Manual trigger

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run tests
        run: npm test
      
      - name: Build
        run: npm run build
```

### Common Patterns

```yaml
# Matrix builds
strategy:
  matrix:
    node: [18, 20, 21]
    os: [ubuntu-latest, windows-latest]
    exclude:
      - node: 18
        os: windows-latest

# Conditional steps
- name: Deploy to production
  if: github.ref == 'refs/heads/main'
  run: ./deploy.sh

# Environment variables
env:
  NODE_ENV: production
  API_URL: ${{ vars.API_URL }}
  
# Secrets
- name: Deploy
  env:
    API_KEY: ${{ secrets.API_KEY }}
  run: |
    echo "$API_KEY" | base64 -d > key.json
    ./deploy.sh
```

### CLI Operations

```bash
# List workflows
gh workflow list

# View workflow
gh workflow view <workflow-name>

# Trigger workflow manually
gh workflow run <workflow-name>
gh workflow run deploy.yml -f environment=production

# Watch workflow
gh run watch

# List recent runs
gh run list
gh run list --workflow=ci.yml

# View specific run
gh run view <run-id>

# View run logs
gh run view <run-id> --log
gh run view <run-id> --log-failed

# Re-run failed jobs
gh run rerun <run-id>
gh run rerun <run-id> --failed

# Cancel run
gh run cancel <run-id>

# Download artifacts
gh run download <run-id>
gh run download <run-id> -n artifact-name
```

---

## 2. GitLab CI

### Basic Configuration

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

variables:
  NODE_VERSION: "20"
  DOCKER_IMAGE: "$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA"

build:
  stage: build
  image: node:$NODE_VERSION
  script:
    - npm ci
    - npm run build
  artifacts:
    paths:
      - dist/
    expire_in: 1 week

test:
  stage: test
  image: node:$NODE_VERSION
  script:
    - npm ci
    - npm test
  coverage: '/All files[^|]*\|[^|]*\s+([\d\.]+)/'

deploy:
  stage: deploy
  image: alpine/k8s:latest
  script:
    - kubectl apply -f k8s/
  environment:
    name: production
    url: https://app.example.com
  only:
    - main
```

### CLI Operations

```bash
# Validate configuration
gitlab-ci-lint

# List pipelines
glab pipeline list

# View pipeline
glab pipeline status
glab pipeline view <pipeline-id>

# Retry pipeline
glab pipeline retry <pipeline-id>

# Cancel pipeline
glab pipeline cancel <pipeline-id>

# View jobs
glab ci view
glab ci status

# Trace job logs
glab ci trace <job-name>
```

---

## 3. Secrets Management

### GitHub Secrets

```bash
# Set repository secret
gh secret set API_KEY
gh secret set API_KEY --body "secret-value"
gh secret set API_KEY --env-file .env

# List secrets
gh secret list

# Delete secret
gh secret delete API_KEY

# Set organization secret
gh secret set API_KEY --org my-org

# Set environment secret
gh secret set API_KEY --env production
```

### GitLab Variables

```bash
# Set variable
glab variable set API_KEY "value"
glab variable set API_KEY --value "value" --protected

# List variables
glab variable list

# Delete variable
glab variable delete API_KEY
```

---

## 4. Debugging Pipelines

### GitHub Actions

```bash
# Enable debug logging
# Set repository secret: ACTIONS_STEP_DEBUG = true
# Set repository secret: ACTIONS_RUNNER_DEBUG = true

# SSH into runner (with action)
- uses: lhotari/action-upterm@v1
  if: failure()
```

### Common Issues

```bash
# Check workflow syntax
gh workflow view <name>

# Validate YAML
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"

# Check triggers
git log --oneline --grep="ci skip"
```

---

## 5. Self-Hosted Runners

### GitHub

```bash
# Create runner directory
mkdir actions-runner && cd actions-runner

# Download latest runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf actions-runner-linux-x64-2.311.0.tar.gz

# Configure
./config.sh --url https://github.com/OWNER/REPO --token <token>

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start

# Remove runner
./config.sh remove --token <token>
```

---

## 6. Pipeline Optimization

### Caching

```yaml
# GitHub Actions cache
- uses: actions/cache@v3
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-

# GitLab cache
cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - node_modules/
```

### Parallel Jobs

```yaml
# GitHub Actions
jobs:
  test-unit:
    runs-on: ubuntu-latest
    steps: [...]
  
  test-integration:
    runs-on: ubuntu-latest
    steps: [...]
  
  test-e2e:
    needs: [test-unit, test-integration]
    runs-on: ubuntu-latest
    steps: [...]
```

---

## 7. Deployment Strategies

### Blue-Green Deployment

```yaml
deploy:
  script:
    - kubectl apply -f k8s/green/
    - kubectl rollout status deployment/app-green
    - kubectl patch service app -p '{"spec":{"selector":{"version":"green"}}}'
    - kubectl delete -f k8s/blue/
```

### Canary Deployment

```yaml
deploy:
  script:
    - kubectl apply -f k8s/canary/
    - sleep 60
    - ./check-metrics.sh
    - kubectl scale deployment app-canary --replicas=10
```

### Rolling Deployment

```yaml
deploy:
  script:
    - kubectl set image deployment/app app=image:tag
    - kubectl rollout status deployment/app
    - kubectl rollout history deployment/app
```
