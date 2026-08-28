# Conversation Worktree Isolation Design

## Status

Approved in chat on 2026-08-23. This document specifies implementation; it does not claim the
feature is implemented.

## Problem

Hermes currently has three different workspace behaviors:

- CLI sessions can opt into a disposable worktree with `-w` or top-level `worktree: true`.
- Desktop users can manually create a worktree through the **Start work** flow.
- Gateway sessions, including Discord and iMessage, start in the profile's configured terminal
  directory.

The local configuration already contains a `desktop.conversation_worktree` block, but no runtime
code reads it. Consequently, a fresh chat can execute tools directly in the shared stable checkout.
Concurrent chats can then collide, inherit unrelated changes, or make the stable checkout dirty.

## Goals

1. Give every fresh interactive root conversation a dedicated Git branch and linked worktree.
2. Branch from the committed `HEAD` of a configured stable source checkout.
3. Cover desktop, CLI, Discord `/new`, and iMessage fresh-session creation through one policy.
4. Reuse the same worktree across reconnects, resumes, compression children, and model/profile
   changes.
5. Preserve every conversation worktree until an explicit archive/delete cleanup request.
6. Fail closed when isolation was required but could not be established.
7. Leave cron, Kanban, and delegated-subagent worktree ownership unchanged.

## Non-goals

- Automatically merging, pushing, opening, or closing pull requests.
- Copying dirty files from the stable checkout into a conversation worktree.
- Replacing Kanban task worktrees or delegated-subagent isolation.
- Automatically deleting old branches or worktrees based only on age.
- Inferring that a worktree is safe to delete from chat status alone.
- Making the Obsidian vault, session history, or generated reports authoritative for Git state.

## Terminology

- **Stable source**: the configured checkout whose committed `HEAD` is the base for new
  conversation branches.
- **Interactive root conversation**: a newly created user-facing conversation in desktop, CLI,
  Discord, or iMessage. A chat-history branch is also a new root conversation.
- **Continuation**: a reconnect, resume, compression child, transport reconnect, or model/profile
  change belonging to an existing root conversation.
- **Task-owned worktree**: a worktree whose lifecycle belongs to Kanban, cron dispatch, or delegated
  subagent code rather than the user-facing conversation.

## Configuration

The canonical configuration moves to a platform-neutral top-level section:

```yaml
conversation_worktree:
  enabled: true
  source_worktree: /path/to/stable-checkout
  worktree_root: /path/to/repository/.worktrees
  branch_prefix: hermes/session
  bootstrap: true
  bootstrap_command:
    - python3
    - scripts/bootstrap_agent_workspace.py
  bootstrap_timeout: 300
  create_timeout: 60
  retain_until_explicit_cleanup: true
```

For compatibility, Hermes also reads `desktop.conversation_worktree` when the top-level section is
absent. The top-level section wins when both exist. Startup emits one bounded migration warning
when the legacy location is used; it does not rewrite user configuration automatically.

Profiles inherit the normal Hermes configuration overlay behavior. A profile may disable the
policy explicitly, but an omitted profile section inherits the base policy. The resolver returns a
typed immutable policy object so entry points cannot interpret raw YAML independently.

The existing top-level `worktree` and `worktree_sync` settings retain their CLI meanings. They do
not control conversation isolation. In particular, `worktree_sync` must not fetch a remote or move
the base: conversation worktrees are pinned to the stable source's local committed `HEAD`.

## Stable-Base Contract

Before creating a conversation worktree, Hermes resolves and records:

- the stable checkout's absolute path;
- its working-tree root;
- its common Git directory;
- its current branch, for diagnostics only;
- its exact committed `HEAD` object ID;
- the configured worktree-root path.

The worktree is created from that exact object ID, not from a mutable branch name, remote-tracking
ref, index, or filesystem snapshot. Modified and untracked files in the stable checkout therefore
remain untouched and are not inherited.

Stable source dirtiness is visible in diagnostics but does not block creation because `git worktree
add ... <commit>` is independent of the source index and worktree files. Hermes never runs stash,
reset, clean, checkout, commit, rebase, or merge against the stable source as part of this flow.

Creation fails before agent/tool initialization if the source is missing, has no valid `HEAD`, is
not a Git checkout, the worktree root belongs to a different common repository, or the configured
paths escape their validated roots.

## Branch and Path Identity

Each root conversation receives a collision-resistant identity derived from the durable root
session ID:

```text
branch:   <branch_prefix>/<YYYYMMDD>-<root-session-id>
worktree: <worktree_root>/<YYYYMMDD>-<root-session-id>
```

Names pass through one Git-ref/path sanitizer. Creation occurs under a repository-scoped lock.
Existing matching metadata is treated as an idempotent resume only when the branch, worktree path,
base commit, common Git directory, and root session identity all agree. Any partial or conflicting
state fails closed and produces recovery guidance rather than selecting a different directory.

## Persistent Session Metadata

The session store gains explicit conversation-worktree metadata rather than overloading `cwd` or
`model_config`:

```text
root_session_id
conversation_worktree_path
conversation_branch
conversation_base_commit
conversation_repo_common_dir
conversation_worktree_state
conversation_worktree_created_at
```

`conversation_worktree_state` is one of `creating`, `ready`, `creation_failed`, `retained`, or
`removed`. Database migration is additive and backward compatible.

The root row owns this metadata. Continuation rows resolve their root through the existing session
lineage and borrow the root's ready binding. They never create a new worktree. A chat-history
**Branch in new chat** action creates a new root row and therefore a new stable-based worktree; it
copies conversation history but not uncommitted code from the parent chat.

For gateways that initially use an in-memory runtime ID before lazy row persistence, Hermes reserves
the root identity and writes a `creating` claim transactionally before Git mutation. First-turn row
materialization adopts that claim. A process crash can therefore recover or report the same partial
creation instead of allocating another branch.

## Conversation Worktree Manager

A backend module owns the full lifecycle. Its public boundary is deliberately small:

```python
bind_new_root_session(...)-> ConversationWorktreeBinding
resolve_existing_session(...)-> ConversationWorktreeBinding
inspect_cleanup(...)-> CleanupVerdict
remove_after_explicit_request(...)-> CleanupResult
```

`bind_new_root_session` performs, in order:

1. Resolve the effective profile policy.
2. Verify that the caller is an eligible interactive root session.
3. Claim the root session identity in persistent state.
4. Resolve and pin the stable-base commit.
5. Create the branch and linked worktree with argument-vector subprocess calls.
6. Lock or otherwise mark the worktree as Hermes conversation-owned.
7. Run the configured bootstrap command inside the new worktree when enabled.
8. Verify the resulting Git root, branch, base ancestry, and common repository.
9. Persist `ready` metadata and return the binding.

The manager receives Git/process/filesystem adapters so tests can exercise behavior without mutating
the developer's repositories. Production subprocesses have bounded timeouts and no shell
interpolation.

## Entry-Point Integration

### Desktop

`session.create` invokes the manager before scheduling agent construction. The returned worktree
path becomes the session's explicit `cwd`, terminal scope, file-tool root, Git metadata source, and
workspace label. The desktop renderer does not create the automatic worktree itself; it consumes
the backend's session information. Manual **Start work** remains available for explicitly selected
branches and repositories.

### CLI

A fresh CLI root session uses the same manager after allocating its session identity and before
constructing the agent. The legacy `-w` flow remains available when conversation isolation is
disabled or no conversation policy applies. Hermes must not create both a legacy CLI worktree and a
conversation worktree for the same session.

### Discord and iMessage

The gateway's fresh-session/reset path invokes the manager once when it creates a new root session.
Normal messages in an existing channel/session reuse the stored binding. `/new` creates a new root
and worktree. Transport reconnects, batching, replies, and voice/text switching do not create new
worktrees.

### Cron, Kanban, and Delegation

Sources identified as `cron`, Kanban workers, review workers, or task orchestrators do not enter the
conversation manager. Their existing workspace resolver remains authoritative. Delegated children
continue to use the existing opt-in subagent worktree mechanism; they do not inherit automatic root
conversation creation.

## Tool and Workspace Enforcement

An eligible session cannot become tool-ready until its binding is `ready`. Agent construction,
terminal registration, file tools, LSP, MCP workspace initialization, checkpoints, and session Git
metadata all receive the same bound path.

If binding fails, Hermes returns a concise error containing the failed phase, stable source, root
session identity, and safe recovery command. It must not silently fall back to the stable checkout,
configured terminal cwd, home directory, or a previously active chat's worktree.

The system prompt includes the worktree path, branch, base commit, and stable source plus explicit
instructions not to edit the stable checkout. This prompt is explanatory; the backend binding is
the enforcement mechanism.

## Bootstrap

Bootstrap runs only after Git creation and before the binding becomes ready. It executes inside the
new worktree with the configured timeout. Failure marks the binding `creation_failed` and retains
the new worktree for inspection; it does not delete potentially useful diagnostics or run the agent
without dependencies.

Bootstrap must remain idempotent. For ExampleProject it uses `scripts/bootstrap_agent_workspace.py`,
which supplies the worktree-local/shared environment according to repository policy.

## Retention and Cleanup

Closing, disconnecting, compressing, timing out, or archiving a chat never removes its worktree.
Archive/delete may offer an explicit cleanup action, but cleanup first computes a Git-backed verdict.

Removal is refused if any of the following is true:

- tracked or untracked working-tree changes exist;
- merge/rebase/cherry-pick or another in-progress Git operation exists;
- commits are not reachable from the configured stable base branch or another configured integration
  ref;
- commits have no verified remote reachability when push verification is required;
- the worktree is active in another Hermes process/session;
- metadata identity does not exactly match the requested root session.

An allowed cleanup removes the linked worktree without force, verifies its disappearance from `git
worktree list`, then optionally deletes only the exact Hermes-owned branch. A failed removal leaves
metadata retained and reports the Git error. Hermes never uses broad glob deletion or `git clean`.

## Concurrency and Recovery

Repository-scoped locking serializes branch/worktree metadata mutation while allowing unrelated
repositories to proceed concurrently. Session-state claims make retries idempotent.

At startup and before creation, Hermes inspects `creating` and `creation_failed` records:

- a complete matching worktree is adopted and marked `ready`;
- an empty partial worktree with no branch/work may be repaired under the same identity;
- ambiguous, dirty, or identity-conflicting state is retained and surfaced for manual recovery.

Hermes never guesses a replacement path after a timeout because doing so could split one chat across
two branches.

## Observability

Structured events record policy resolution, base pinning, create/bootstrap phases, ready bindings,
resume reuse, cleanup verdicts, and failures. Logs redact home-relative details where appropriate
and never include credentials or environment contents.

Desktop session information and CLI/gateway status expose:

- isolation state;
- branch and worktree path;
- pinned base commit;
- whether the session is a root or continuation;
- retained/cleanup-blocked reason.

## Testing Strategy

Implementation follows red-green TDD. Required contract coverage includes:

1. Configuration precedence and legacy `desktop.conversation_worktree` compatibility.
2. Exact committed-HEAD pinning from a dirty stable checkout.
3. No mutation of stable index, worktree files, branch, or HEAD.
4. Unique branch/path allocation under concurrent root creation.
5. Idempotent retry after success and deterministic recovery after partial creation.
6. Desktop, CLI, Discord `/new`, and iMessage root-entry integration.
7. Resume, reconnect, compression child, and model/profile changes reuse one binding.
8. Chat-history branching creates a distinct stable-based worktree.
9. Cron, Kanban, review workers, orchestrators, and delegated children bypass the manager.
10. Bootstrap success, timeout, and failure retention.
11. Fail-closed behavior for invalid source, invalid root, Git failure, DB failure, and identity
    conflict.
12. Tool initialization cannot occur before `ready`.
13. Cleanup refuses dirty, active, unmerged, unpushed, or mismatched worktrees.
14. Explicit safe cleanup removes only the exact worktree and owned branch.
15. Existing CLI `-w`, manual desktop **Start work**, Kanban, and subagent tests remain green.

Tests use temporary real Git repositories for Git semantics and focused fakes only at transport or
process boundaries. The full Hermes test runner certifies integration after focused suites pass.

## Rollout

1. Land schema/config resolver and manager behind `conversation_worktree.enabled`.
2. Integrate desktop and CLI root creation, then gateway fresh-session creation.
3. Add inspection/status surfaces and explicit cleanup API/UI.
4. Migrate the local configuration from `desktop.conversation_worktree` to the top-level section.
5. Restart desktop and the managed gateway.
6. Create one smoke conversation per interactive entry point and verify distinct branches,
   worktrees, base commit, and stable-checkout preservation.

Rollout does not alter existing sessions. A session lacking conversation-worktree metadata keeps its
historical cwd; only newly created roots after activation enter the manager.

## Acceptance Criteria

The feature is complete only when fresh desktop, CLI, Discord `/new`, and iMessage conversations each
receive distinct worktrees pinned to the same configured stable committed HEAD; tools execute inside
those paths; resumes reuse them; cron/Kanban ownership remains unchanged; creation failures cannot
fall back to stable; explicit cleanup preserves any work that is dirty, active, unmerged, or
unverified as pushed; and the focused plus full Hermes suites pass.
