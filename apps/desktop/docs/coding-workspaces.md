# Coding workspaces

Coding controls are optional and scoped to a profile and backend connection. Enable **Settings → Workspace → Show coding controls** for a coding profile. Other profiles remain ordinary chat; **Add context → Work in project…** reveals the controls for one draft only.

## Workflow

1. Start a new chat and choose **Project**.
2. Choose **New worktree**, **Existing worktree**, or **Current checkout**. A non-Git directory uses **Project folder** instead.
3. Send the task. Selecting a project does not create a checkout, navigate into Projects, or change the sidebar's grouping, ordering, or filters.

New worktrees live in the repository's managed `.worktrees` directory. A local Git exclusion keeps this directory out of source status without editing the project's `.gitignore`. The initial base branch is visible and can be changed. Existing checkouts show their actual branch, path, dirty state and active-session usage. Current checkout means the main checkout, including when the chosen project path is itself a linked worktree.

The optional `desktop.coding.default_checkout` preference accepts `worktree` (default) or `current`. It is scoped like the display toggle; it is not a global terminal-directory change.

Folder context remains reference material. **Use as project** is an explicit action, not an automatic retarget on every path mention. Once a chat has a workspace, use a new chat for a different workspace.

## After the first message

The draft selectors give way to a compact summary in the existing composer status strip: **project · checkout kind · branch**. Project folders have no branch label. The summary comes from the conversation's persisted workspace binding, so it remains available when reopening a chat or restarting Desktop.

Click the summary to inspect the full checkout path, copy it, or use the available workspace actions. Native folder reveal requires a verified local descriptor for that exact session owner; it does not borrow the foreground pane's locality. Git status keeps its stricter foreground-route guard. **New chat in another workspace** starts a fresh draft with the coding controls available; it does not retarget the original chat or provision a checkout. Workspace browsing continues to leave the sidebar's grouping, ordering, and filters alone.

## Binding and failure behavior

The first-send transaction captures the draft, connection and profile before asynchronous work. It prepares a checkout, creates or recovers an idempotent session, and verifies the real execution directory before dispatching the prompt. A lost response must not create another checkout or session. Preparation/verification failures retain the draft and references.

The durable session metadata records the workspace identity and a session-owned artifacts directory. Resuming the session restores and validates that binding. Missing checkouts or branch mismatches fail closed rather than silently falling back to the profile's generic CWD. Verification also runs at the turn-worker boundary. References promoted with the selected source project are resolved into the selected checkout before first-send attachment synchronization; unrelated references are not retargeted.

Research and verification outputs have a managed session artifacts location separate from source changes. This is placement guidance and managed state, **not a filesystem sandbox**: an agent with unrestricted filesystem tools can still write an explicitly requested absolute path.

## Regression coverage

Backend coverage lives in `tests/tui_gateway/test_coding_workspaces.py`, `test_coding_workspace_restart.py`, and `test_coding_workspace_worker.py`. Frontend coverage spans the coding-workspaces store, composer controls/selection/controller, scoped setting, and session-create/submit hooks.

For native development verification, use a separate HOME, HERMES_HOME, Electron user-data directory and CDP port, with `HERMES_DESKTOP_HERMES_ROOT` pinned to the feature checkout. Test real temporary Git repositories and read the resulting workspace/session state back. Restart both Vite and Electron after the source settles; importing unversioned store modules into an HMR-versioned renderer can itself create duplicate module state and invalidate routing observations.
