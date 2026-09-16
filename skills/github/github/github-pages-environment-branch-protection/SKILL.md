---
name: github-pages-environment-branch-protection
description: Fix GitHub Actions deploy-pages failures caused by environment branch protection rules blocking master branch
category: github
---
# GitHub Pages Environment Branch Protection Fix

## Problem
GitHub Actions workflow using `actions/deploy-pages@v4` fails with:
```
Branch "master" is not allowed to deploy to github-pages due to environment protection rules.
The deployment was rejected or didn't satisfy other protection rules.
```

Even after adding `environment: name: github-pages` to the workflow YAML, deployments from `master` branch are rejected.

## Root Cause
The `github-pages` environment has `deployment_branch_policy.custom_branch_policies: true` by default, meaning only explicitly listed branches can deploy. `master` is not on that list.

## Solution
Use the GitHub REST API to update the environment branch policy:

1. Check current config:
```
GET /repos/{owner}/{repo}/environments/github-pages
```

2. Fix — allow all branches including master:
```
PATCH /repos/{owner}/{repo}/environments/github-pages
Body: {"deployment_branch_policy":{"custom_branch_policies":false,"protected_branches":false}}
```

In curl form:
```bash
curl -s -X PATCH "https://api.github.com/repos/{owner}/{repo}/environments/github-pages" \
  -H "Authorization: Bearer {TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"deployment_branch_policy":{"custom_branch_policies":false,"protected_branches":false}}'
```

## Also Required
The workflow YAML must include `environment:` in the job definition:
```yaml
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: https://example.github.io/repo/
    steps:
      - uses: actions/deploy-pages@v4
```

## Key Insight
`actions/deploy-pages@v4` silently requires an environment, but even after adding one, the environment's branch policy can still block the deployment. Both YAML `environment:` AND API branch policy must be correct.

## References
- GitHub Environments: https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment
- actions/deploy-pages: https://github.com/actions/deploy-pages
