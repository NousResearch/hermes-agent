# Projects and folders (`hermes project`)

A **Project** is a named workspace that can span several folders — a repo, a monorepo, a
scratch directory, a sibling repo you always open together. Projects are the named
workspaces the desktop sidebar groups chats by, and the anchor a kanban board uses for
worktrees and branch names.

Projects are **not** git concepts and not config keys: they live in their own SQLite store,
one per profile (`$HERMES_HOME/projects.db`), so two profiles keep separate project lists.
The store is created on first use; nothing to enable.

## Command surface

```
hermes project <create|list|show|add-folder|remove-folder|rename|set-primary|use|archive|restore|bind-board>
```

| Subcommand | What it does |
|---|---|
| `create <name> [folders...]` | New project. First folder becomes the primary; `--slug`, `--primary PATH`, `--description`, `--icon`, `--color`, `--board SLUG`, `--use` (make it active). |
| `list` (alias `ls`) | Projects, oldest first; `*` marks the active one; `--all` includes archived. |
| `show <project>` | Slug, id, name, about/board/primary, folder list (`*` = primary). |
| `add-folder <project> <path>` | Add a folder; `--label`, `--primary` (make it primary). |
| `remove-folder <project> <path>` | Drop a folder (repoints primary if it was primary). |
| `rename <project> <name>` | Change the display name. The slug does **not** change. |
| `set-primary <project> <path>` | Primary folder; the path must already be in the project. |
| `use [<project>]` | Set the active project; with no argument, clear it. |
| `archive <project>` / `restore <project>` | Hide / unhide a project (reversible; nothing on disk moves). |
| `bind-board <project> [<board>]` | Bind a kanban board to the project; omit the board to unbind. |

`<project>` is an id (`p_1a2b3c4d`), a slug, or (case-insensitively) a name.

## Semantics worth knowing

- **Slug** — derived from the name (lowercased, runs of non-alphanumerics collapse to `-`,
  capped at 64 chars) unless `--slug` is given. Must match `[a-z0-9][a-z0-9-_]{0,63}`; an
  invalid slug exits **2** with `project: invalid project slug …`. A colliding slug gets a
  suffix (`team-docs`, `team-docs-2`, …).
- **Primary folder** — folders are stored as absolute paths (`~` expanded). The primary is
  the anchor for workspace moves, kanban worktrees and the desktop project directory; it is
  the first folder given, or `--primary`.
- **One project per primary folder** — `create` refuses a primary path that already belongs
  to another project (`folder already belongs to project '<slug>' …; switch to it instead of
  creating a duplicate`). Reuse the existing project; duplication is what produces several
  identical sidebar subtrees.
- **Active project** — a single per-profile pointer (`project_meta.active_id`) that `use` sets
  and clears. It is what the desktop Projects view highlights and what an unscoped caller
  (CLI, script, messaging platform) sees as "the current project"; a live desktop / TUI chat
  is scoped to its own workspace instead, and switching there never moves the pointer out
  from under other open chats (see below).
- **Archived projects** are hidden from `list` and excluded from path→project resolution, so
  an archived project stops claiming its folders. `restore` brings it back.
- **Path → project resolution** — a folder owns itself and everything under it, and the
  longest matching folder wins, so a nested project resolves to the innermost one. Only
  explicit projects resolve; an ordinary checkout is not "in" a project.
- **No CLI delete.** `archive` is the CLI path to "get rid of it". The store has a hard
  delete (`delete_project`, cascade to folders) but no subcommand exposes it.
- **Exit codes** — `0` ok, `1` no such project / not-a-folder errors, `2` invalid input
  (bad slug, duplicate primary path).

## What projects actually drive

- **Desktop sidebar grouping.** A session's workspace resolves to the owning project
  (`project_for_path` on the session cwd), and the TUI status line, desktop status bar and
  `/status` all name that workspace from the same lookup.
- **The `desktop_project` tool** (toolset `project`; GUI gateways only, off the core tool
  set). `action: create|switch|list` with `name` / `path`. In a live desktop/TUI session
  `create` and `switch` re-anchor *that session's* cwd to the project's primary folder and
  the sidebar follows, which is the intended way to move a chat between workspaces — not
  `cd`. The move is session-scoped on purpose: it never moves the profile-global pointer out
  from under other open chats. In the CLI or a messaging platform there is no workspace
  callback, so the call only writes the DB (and the profile pointer).
- **Kanban anchoring.** A project records the board it is bound to (`projects.board_slug`,
  surfaced in the desktop Projects view), and the board's own metadata records the project
  (`project_id`). `bind-board` also points the board's `default_workdir` at the project's
  primary folder (best effort: only when the board exists or is the default). Tasks created
  on a project-bound board inherit the project, and a project-linked worktree task is created
  under the project's primary repo with a deterministic branch name
  `<project-slug>/<task-id>[-<title-slug>]` instead of the random `wt/<task-id>`.
- **Repo discovery cache.** The desktop "repo-first" project picker scans the filesystem once
  and caches the roots in `discovered_repos` (keyed to a discovery policy, so a policy
  change invalidates the cache) instead of re-walking the disk on every open.

## Common tasks

```bash
# Make a project for a repo and start using it
hermes project create "Hermes Agent" ~/Work/hermes-agent --use

# A workspace spanning two repos (first given = primary)
hermes project create "Payments" ~/Work/payments-api ~/Work/payments-web

# Add a folder later / move the primary
hermes project add-folder payments ~/Work/payments-infra --label infra
hermes project set-primary payments ~/Work/payments-web

# What is active, and what owns a path
hermes project list
hermes project show payments

# Retire one without losing it
hermes project archive payments        # hidden from list / path resolution
hermes project restore payments

# Bind a board so its tasks get project worktrees + branches
hermes project bind-board payments payments-board
```

## Verify it yourself (read-only)

```bash
hermes project --help                       # the live subcommand list
hermes project list --all                   # active marker + archived
sqlite3 "$HERMES_HOME/projects.db" \
  'select slug,name,primary_path,board_slug,archived from projects'
sqlite3 "$HERMES_HOME/projects.db" \
  'select p.slug, f.path, f.is_primary from project_folders f join projects p on p.id=f.project_id'
```

`$HERMES_HOME` defaults to `~/.hermes`; a profile keeps its own `projects.db`, so run the
query with that profile's home (`hermes -p <profile> project list` is the same check).

## Troubleshooting

- **"switched project but this chat didn't move"** — in a desktop/TUI session the workspace
  follows the *session's* project, not the global pointer; switch from inside the chat
  (the `desktop_project` tool) so the session re-anchors, or start the turn in the folder.
- **`create` refused: folder already belongs to project X** — deliberate one-project-per-
  primary-folder guard. `hermes project show X`, then `hermes project use X` (or
  `add-folder` the directory into the project you want).
- **`set-primary` refused** — the path has to be a folder of that project first
  (`hermes project add-folder <project> <path>`), and it must be spelled the way `show`
  prints it.
- **A project's folders still look "unprojected"** — archived projects are skipped by
  path→project resolution; `hermes project restore <project>`.
- **Two profiles show different projects** — expected: `projects.db` is per profile, not
  global.

## See also

- `references/cli-reference.md` — `hermes project` in the CLI command map.
- `references/project-context-files.md` — a different feature: `AGENTS.md` / `.hermes.md` /
  `CLAUDE.md` instruction files that load into a session.
- Docs: `/user-guide/features/kanban` (boards, project binding, workspaces) and
  `/reference/cli-commands#hermes-project`.
