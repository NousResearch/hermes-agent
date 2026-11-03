# Deploying to the Hugging Face Space

`Sync to Hugging Face` mirrors this repository to
`https://huggingface.co/spaces/cxxa4673/hermes-agent` on every push to `main`
(or via **Actions → Sync to Hugging Face → Run workflow**).

## Why the first attempt never ran

The original file was `deploy.yamal` at the repository root. GitHub Actions only
loads `*.yml` / `*.yaml` files inside `.github/workflows/` on the default
branch, and that file also had broken (flat) YAML indentation, plus a shallow
checkout that cannot be pushed into a new remote. It has been replaced by
`.github/workflows/deploy-huggingface.yml`.

## One-time setup

1. **Create the Space** at <https://huggingface.co/new-space> under the account
   `cxxa4673` with the name `hermes-agent` (or set the `HF_SPACE` repo variable
   to your actual `<user>/<space>`).
2. **Create a Hugging Face access token** with **write** scope at
   <https://huggingface.co/settings/tokens>.
3. In this GitHub repo: **Settings → Secrets and variables → Actions → New
   repository secret**, name `HF_TOKEN`, paste the token. The workflow fails
   with a clear error if the secret is missing.
4. Optional repo variables: `HF_USERNAME` (owner of the token, default
   `cxxa4673`), `HF_SPACE`, `HF_SDK` (`docker` | `gradio` | `static`, default
   `docker`), `HF_APP_PORT` (default `7860`), `HF_SPACE_TITLE`.
5. Merge this branch into `main`. Workflows only run from the default branch.
6. Watch the run at
   <https://github.com/cxxa4673-ux/hermes-agent/actions/workflows/deploy-huggingface.yml>.

The workflow prepends the Space's YAML front matter to the copy of `README.md`
that it pushes (the GitHub README is untouched) and force-pushes `HEAD` to the
Space's `main` branch, because HF creates the Space with its own initial commit.

## Important: the app must actually serve on port 7860

HF Docker Spaces expect a container that listens on `0.0.0.0:7860`. This
repository's root `Dockerfile` runs the Hermes CLI/TUI, has no `EXPOSE 7860`,
and does not serve HTTP, so the Space will sync but fail its health check.
Either pick `HF_SDK: static` and point `app_file` at a built web page, or add an
HF-specific `Dockerfile`/entrypoint that serves the web UI on port 7860.

## The "queued" runs are a different problem

`CI`, `Docker Build, Test, and Publish` and `Nix flake check` come from the
upstream NousResearch repository and request `blacksmith-*` runners, which are
not registered on this personal fork, so those jobs stay `queued` forever. They
do not block the HF sync job (it uses `ubuntu-latest`). Cancel them from the
Actions UI, or disable those three workflows under **Actions → Workflows** in
this fork.

## Workflow file (for reference / manual copy)

```yaml
name: Sync to Hugging Face

# Pushes the whole repository to a Hugging Face Space. The previous version of
# this file lived at the repo root as "deploy.yamal", so GitHub never loaded it:
# workflows must be *.yml / *.yaml inside .github/workflows/ on the default
# branch, and their YAML has to be correctly indented.
#
# Prerequisites (one-time, in the GitHub repo settings):
#   * secret HF_TOKEN  - a Hugging Face access token with WRITE permission.
#   * the Space HF_SPACE must already exist (create it at huggingface.co/new-space).
# Optional repo variables: HF_USERNAME, HF_SPACE, HF_SDK (docker|gradio|static),
# HF_APP_PORT, HF_SPACE_TITLE.
on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read

# Never cancel a push halfway through: a force-push interleaved with another
# run can leave the Space on a half-written commit.
concurrency:
  group: huggingface-space-sync
  cancel-in-progress: false

jobs:
  sync-to-hub:
    name: Push repository to HF Space
    # GitHub-hosted runner. Upstream CI jobs request blacksmith-* runners that
    # are not available on this fork, so they sit in "queued" forever; this job
    # must not use those labels.
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - name: Checkout full history
        uses: actions/checkout@v4
        with:
          # A shallow clone (checkout's default depth 1) cannot be pushed into
          # an unrelated remote: git rejects it with "shallow update not allowed".
          fetch-depth: 0
          lfs: true

      - name: Add Hugging Face Space configuration
        env:
          HF_SDK: ${{ vars.HF_SDK || 'docker' }}
          HF_APP_PORT: ${{ vars.HF_APP_PORT || '7860' }}
          HF_SPACE_TITLE: ${{ vars.HF_SPACE_TITLE || 'Hermes Agent' }}
        run: |
          set -euo pipefail
          # A Space reads its SDK and port from YAML front matter at the top of
          # README.md. This repo's upstream README has none, and a force push
          # would overwrite the README that HF created with the Space, so add
          # the front matter to the copy we ship (the GitHub README stays as is).
          if [ "$(head -n 1 README.md)" = "---" ]; then
            echo "README.md already has front matter; leaving it alone."
            exit 0
          fi
          tmp="$(mktemp)"
          {
            printf -- '---\n'
            printf 'title: %s\n' "$HF_SPACE_TITLE"
            printf 'emoji: 🤖\n'
            printf 'colorFrom: indigo\n'
            printf 'colorTo: gray\n'
            printf 'sdk: %s\n' "$HF_SDK"
            if [ "$HF_SDK" = "docker" ]; then
              printf 'app_port: %s\n' "$HF_APP_PORT"
            fi
            printf 'pinned: false\n'
            printf -- '---\n\n'
            cat README.md
          } > "$tmp"
          mv "$tmp" README.md

      - name: Commit the Space configuration
        run: |
          set -euo pipefail
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git checkout -B hf-space-sync
          git add README.md
          if git diff --cached --quiet; then
            echo "No configuration change to commit."
          else
            git commit -m "chore: add Hugging Face Space configuration"
          fi

      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USERNAME: ${{ vars.HF_USERNAME || 'cxxa4673' }}
          HF_SPACE: ${{ vars.HF_SPACE || 'cxxa4673/hermes-agent' }}
        run: |
          set -euo pipefail
          if [ -z "$HF_TOKEN" ]; then
            echo "::error::HF_TOKEN is empty. Add a Hugging Face token with WRITE access at Settings -> Secrets and variables -> Actions -> New repository secret (name: HF_TOKEN)."
            exit 1
          fi
          # HF_USERNAME must be the account that owns HF_TOKEN (not necessarily
          # the GitHub account name). Override with the HF_USERNAME repo variable
          # if the Space lives under a different user or organisation.
          git remote add space "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE}.git"
          # The Space was created with its own initial commit, so a normal push
          # is rejected as non-fast-forward. HEAD:refs/heads/main also works when
          # checkout leaves us on a detached HEAD.
          git push --force space HEAD:refs/heads/main
```

Reference: <https://huggingface.co/docs/hub/spaces-github-actions> and
<https://huggingface.co/docs/hub/spaces-config-reference>.
