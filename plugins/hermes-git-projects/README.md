# Hermes Git Projects

Hermes Git Projects turns the Hermes dashboard into a lightweight project intake and source-control workbench.

It is built for the workflow where an operator pastes a repository URL, Hermes creates or refreshes a local clone, scans the repository into a useful project card, logs issues against that local workspace, and creates actionable Kanban todos with suggested skills attached.

## What it adds

- Repository URL import from HTTPS or SSH Git remotes.
- Managed local clones under `$HERMES_HOME/hermes-git-projects/repos/`.
- Conservative project scanning with branch, remote, dirty state, ahead/behind, latest commit, and detected stack files.
- Source-control actions: fetch, fast-forward pull, push current branch, checkout branch, create branch.
- Local issue logging for each imported project.
- Automatic Kanban todo creation when an issue is saved.
- Suggested-skill storage so users can review, select, and extend the skills Hermes should attach to project work.
- A dashboard tab called **Git Projects** backed by protected localhost dashboard APIs.

## Why this exists

Hermes can already edit files and use Git, but users need a clean intake layer for real projects. This plugin makes "paste repo URL → ready local workspace → log issue → create work item → branch/push changes" a first-class dashboard flow.

## Storage layout

All plugin state is profile-safe and lives under `get_hermes_home()`:

```text
$HERMES_HOME/hermes-git-projects/
├── repos/                    # managed local clones
├── issues.json               # local issue records + linked Kanban todo metadata
└── suggested-skills.json     # editable suggested-skill catalog
```

No credentials or tokens are stored by the plugin. Git authentication is delegated to the user's existing Git/SSH/GitHub CLI setup.

## Dashboard API

Mounted at `/api/plugins/hermes-git-projects/`:

- `GET /summary` — projects, issues, suggested skills, and storage paths.
- `POST /import` — clone/fetch a repo URL and return scanned project metadata.
- `POST /projects/{project_id}/source-control` — fetch, pull, push, checkout, or create branch.
- `POST /projects/{project_id}/issues` — save a local issue and create a Kanban todo.
- `GET /suggested-skills` — read the skill catalog.
- `PUT /suggested-skills` — replace the skill catalog.

## Suggested skills

The default catalog is intentionally conservative:

- `zeo-development-superpowers`
- `systematic-debugging`
- `test-driven-development`
- `requesting-code-review`
- `github-pr-workflow`
- `github-code-review`
- `github-repo-management`
- `writing-plans`
- `subagent-driven-development`

Users can select a subset per issue in the dashboard. The selected skills are stored on the local issue and sent to Kanban task creation when available.

## Installation

As a bundled plugin, this directory can live under `plugins/hermes-git-projects/` in the Hermes repo. For a user-local install, copy the directory to:

```text
~/.hermes/plugins/hermes-git-projects/
```

Then restart `hermes dashboard` or use the dashboard plugin rescan endpoint.

## Verification

From the Hermes repo root:

```bash
python3 -m py_compile plugins/hermes-git-projects/dashboard/plugin_api.py
hermes dashboard --no-open --skip-build
```

Then open the dashboard and use the **Git Projects** tab.
