# Thread workspaces: worktree forks and on-demand partial clones

A messaging chat bound with `/workspace` and `gateway.workspaces.per_thread: worktree` runs each thread
in its own fork: every repository there has a git worktree on branch `hermes/thread-...`, shared with
no other chat. The fork root's `AGENTS.md` ("Repositories in this workspace") lists each repository
with its worktree path, or `not cloned` with the clone command. Hermes never clones for you at fork
time, so a thread starts at once; clone a listed repository only when the task needs it.

## Procedure

1. Read the index: `read_file` on `AGENTS.md` in the working directory (it is also in your context).
2. Work in the listed worktree paths; commit on the thread branch, push it with
   `git -C <worktree> push -u origin HEAD` when the user wants a PR (`references/pr-workflow.md`).
3. A repository you need is marked `not cloned`: run its command, exactly as listed, with `terminal`.
   It is a partial clone (`--filter=blob:none`: history and trees now, file contents as checkout
   needs them, so even a large repository is usable in seconds), then the thread's worktree inside
   the clone's `.worktrees/` (where the stale-worktree sweep looks), then its link in the fork root:
   ```bash
   git clone --filter=blob:none https://github.com/acme/web.git /work/acme/web && \
     git -C /work/acme/web worktree add /work/acme/web/.worktrees/<thread> -b hermes/<thread> && \
     ln -s /work/acme/web/.worktrees/<thread> <fork root>/web
   ```
   The next turn records it with the others; later threads fork the clone on their first turn.

## Pitfalls

- Do not edit the shared checkout (`<dir>/<repo>`) from a thread: other threads fork from it.
- A partial clone fetches blobs lazily: `git log -p` or `git blame` over old history downloads
  them one by one. Scope such commands to paths you need.
- The worktree sweep (`worktrees.*` in config.yaml) archives and removes idle forks (a fork is only
  locked while a turn runs); commit or push work you want to keep. The next turn recreates a removed
  tree from its `hermes/<thread>` branch.
