---
name: github-pr-governance
description: "Associate PRs with milestones and GitHub Project boards on creation. Never forget governance again."
version: 1.0.0
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, Pull-Requests, Project-Board, Milestones, Governance, gh-CLI]
    related_skills: [github-pr-workflow, github-auth]
---

# GitHub PR Governance — Milestones + Project Boards

Every PR MUST be associated with the correct milestone and project board BEFORE merge.

## Rule

When creating a PR with `gh pr create`, IMMEDIATELY:
1. Set the milestone
2. Add to project board

## Commands

### Set milestone (GraphQL — reliable)

`gh pr edit` fails silently with "Projects classic deprecated" warning. Use GraphQL instead:

```bash
# Get PR node ID
PR_NODE=$(gh api repos/OWNER/REPO/pulls/NUMBER --jq '.node_id')

# Get milestone node ID
MS_NODE=$(gh api repos/OWNER/REPO/milestones/NUMBER --jq '.node_id')

# Set milestone via GraphQL
gh api graphql -f query="mutation { updatePullRequest(input: {pullRequestId: \"$PR_NODE\", milestoneId: \"$MS_NODE\"}) { pullRequest { milestone { title } } } }"
```

Or via REST API (simpler for milestone number):

```bash
gh api repos/OWNER/REPO/issues/PR_NUMBER --method PATCH -f milestone=MILESTONE_NUMBER
```

### Add to project board

```bash
gh project item-add PROJECT_NUMBER --owner OWNER --url https://github.com/OWNER/REPO/pull/PR_NUMBER
```

### Add labels

```bash
gh api repos/OWNER/REPO/issues/PR_NUMBER/labels --method POST --input - <<< '{"labels":["label1","label2"]}'
```

Note: PRs are issues in the REST API, so the endpoint is `/issues/NUMBER/labels`, not `/pulls/`.

## CI Enforcement (PR Governance Workflow)

Add `.github/workflows/pr-governance.yml` to enforce milestone + label at the CI level:

```yaml
name: PR Governance
on:
  pull_request:
    types: [opened, edited, synchronize, labeled, unlabeled, milestoned, demilestoned, ready_for_review]
permissions:
  pull-requests: read
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          PR_NUMBER=${{ github.event.pull_request.number }}
          REPO=${{ github.repository }}
          LABELS=$(gh api repos/$REPO/pulls/$PR_NUMBER --jq '.labels | length')
          MILESTONE=$(gh api repos/$REPO/pulls/$PR_NUMBER --jq '.milestone != null')
          if [ "$LABELS" -eq 0 ]; then echo "::error::This PR must have at least one label."; exit 1; fi
          if [ "$MILESTONE" = "false" ]; then echo "::error::This PR must have a milestone."; exit 1; fi
          echo "Milestone + Labels required — passed"
```

Then: Settings > Branches > main > Required status checks > add the governance check.

## Post-Merge: Branch Cleanup

ALWAYS clean up local branches after merging:

```bash
git checkout main && git pull origin main
git branch | grep -v main | xargs -r git branch -D
```

The `--delete-branch` flag on `gh pr merge` only deletes the REMOTE branch. Local branches linger.

## Project Board Custom Fields

### Discovering fields and options

```bash
# List all fields (returns field IDs)
gh project field-list PROJECT_NUMBER --owner OWNER

# For SingleSelect fields, get option IDs via GraphQL:
gh api graphql -f query="{ node(id: \"FIELD_ID\") { ... on ProjectV2SingleSelectField { name options { name id } } } }"
```

### Setting custom fields

```bash
# Get the real project ID (NOT the number)
PROJECT_ID=$(gh project list --owner OWNER --format json | python3 -c "
import sys,json
for p in json.load(sys.stdin).get('projects',[]):
    if p['number'] == PROJECT_NUMBER: print(p['id'])")

# SingleSelect fields via gh project item-edit:
gh project item-edit \
  --id ITEM_ID \
  --project-id $PROJECT_ID \
  --field-id FIELD_ID \
  --single-select-option-id OPTION_ID

# Text fields via GraphQL mutation:
gh api graphql -f query="mutation {
  updateProjectV2ItemFieldValue(input: {
    projectId: \"$PROJECT_ID\",
    itemId: \"$ITEM_ID\",
    fieldId: \"TEXT_FIELD_ID\",
    value: { text: \"VALUE\" }
  }) { projectV2Item { id } }
}"
```

### Getting the item ID

```bash
gh project item-list PROJECT_NUMBER --owner OWNER --limit 50 --format json | python3 -c "
import sys,json
data = json.load(sys.stdin)
for item in data.get('items',[]):
    content = item.get('content', {})
    url = content.get('url', '') if content else ''
    title = content.get('title', '') if content else ''
    if 'KEYWORD' in url or 'KEYWORD' in title:
        print(f'ID: {item[\"id\"]}')
        print(f'URL: {url}')
"
```

## Full governance checklist for every PR

1. **Milestone** — set via GraphQL or REST API
2. **Labels** (1+) — add via REST API
3. **Project board** — add via `gh project item-add`
4. **Custom fields** — populate all SingleSelect/Text fields
5. **Verify** — `gh pr view NUMBER --json milestone,projectItems,labels`

## Pitfalls

- `gh pr edit --milestone` shows "Projects classic deprecated" warning and may fail. Use REST API or GraphQL instead.
- `gh pr edit --add-label` ALSO fails with the same warning. Use REST API.
- **`GITHUB_TOKEN=` prefix required in Hermes terminal** — the env var overrides stored gh credentials. Prefix with `GITHUB_TOKEN=` for all `gh api` calls that need project/label/milestone scopes.
- `gh api .../labels -f labels='["x"]'` does NOT work — it sends a string, not JSON array. Always use `--input -` with heredoc/pipe.
- `gh project item-add` takes the project NUMBER (e.g. `8`), NOT the global ID.
- Verify with: `gh pr view NUMBER --json milestone,projectItems,labels`
