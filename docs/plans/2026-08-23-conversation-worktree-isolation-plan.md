# Conversation Worktree Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every fresh interactive Hermes root conversation run in a dedicated Git worktree branched from the configured stable checkout's exact committed HEAD.

**Architecture:** A platform-neutral policy resolver and `ConversationWorktreeManager` own Git lifecycle and fail-closed binding. A durable SQLite binding ledger survives lazy session-row creation and reconnects; desktop/TUI, CLI, and gateway adapters call the same manager while cron, Kanban, and delegated workers remain on their existing workspace owners.

**Tech Stack:** Python 3.11+, SQLite, Git linked worktrees, pytest, existing Hermes TUI/gateway session stores.

**Spec:** `docs/designs/2026-08-23-conversation-worktree-isolation-design.md`

## Global Constraints

- Pin every new branch to the stable source's local committed `HEAD`; do not fetch or derive the base from mutable filesystem state.
- Never stash, reset, clean, checkout, commit, rebase, or merge the stable source.
- Isolation-required creation failures must abort agent/tool initialization; no fallback to stable, home, configured cwd, or another chat's worktree.
- Resume, reconnect, compression, and model/profile changes reuse the root binding.
- Cron, Kanban, review workers, task orchestrators, and delegated children retain their existing worktree ownership.
- Retain worktrees until an explicit cleanup request; refuse dirty, active, unintegrated, unverified-push, or identity-mismatched cleanup.
- Use argument-vector subprocess calls with bounded timeouts and no shell interpolation.
- Preserve legacy CLI `-w` and manual desktop **Start work** behavior without double-creating worktrees.

---

### Task 1: Resolve a Platform-Neutral Conversation Worktree Policy

**Files:**
- Create: `agent/conversation_worktree_policy.py`
- Modify: `hermes_cli/config_defaults.py`
- Test: `tests/agent/test_conversation_worktree_policy.py`

**Interfaces:**
- Consumes: merged Hermes configuration dictionaries.
- Produces: `ConversationWorktreePolicy` and `resolve_conversation_worktree_policy(config)`.

- [ ] **Step 1: Write failing policy tests**

```python
def test_top_level_policy_wins_over_legacy_desktop_block(tmp_path):
    cfg = {
        "conversation_worktree": {"enabled": True, "source_worktree": str(tmp_path / "new")},
        "desktop": {"conversation_worktree": {"enabled": False, "source_worktree": str(tmp_path / "old")}},
    }
    policy = resolve_conversation_worktree_policy(cfg)
    assert policy.enabled is True
    assert policy.source_worktree == (tmp_path / "new").resolve()
    assert policy.legacy_location is False


def test_legacy_desktop_policy_is_read_when_top_level_is_absent(tmp_path):
    cfg = {"desktop": {"conversation_worktree": {"enabled": True, "source_worktree": str(tmp_path)}}}
    policy = resolve_conversation_worktree_policy(cfg)
    assert policy.enabled is True
    assert policy.legacy_location is True


def test_enabled_policy_requires_source_and_root(tmp_path):
    with pytest.raises(ConversationWorktreePolicyError, match="source_worktree"):
        resolve_conversation_worktree_policy({"conversation_worktree": {"enabled": True}})
```

- [ ] **Step 2: Run the policy tests and verify RED**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/agent/test_conversation_worktree_policy.py`

Expected: import failure for `agent.conversation_worktree_policy`.

- [ ] **Step 3: Implement the immutable resolver**

```python
@dataclass(frozen=True)
class ConversationWorktreePolicy:
    enabled: bool
    source_worktree: Path | None
    worktree_root: Path | None
    branch_prefix: str = "hermes/session"
    bootstrap: bool = False
    bootstrap_command: tuple[str, ...] = ()
    bootstrap_timeout: float = 300.0
    create_timeout: float = 60.0
    retain_until_explicit_cleanup: bool = True
    legacy_location: bool = False


def resolve_conversation_worktree_policy(config: Mapping[str, object]) -> ConversationWorktreePolicy:
    """Resolve top-level policy, falling back to desktop.conversation_worktree."""
```

Validate absolute/resolved paths, positive timeouts, non-empty sanitized branch prefix, list-only
bootstrap argv, and mandatory retention when enabled. Add a disabled default block to
`hermes_cli/config_defaults.py`; do not migrate user files yet.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/agent/test_conversation_worktree_policy.py tests/hermes_cli/test_config_defaults.py`

Expected: all pass.

- [ ] **Step 5: Commit the policy unit**

```bash
git add agent/conversation_worktree_policy.py hermes_cli/config_defaults.py tests/agent/test_conversation_worktree_policy.py
git commit -m "feat(agent): resolve conversation worktree policy"
```

### Task 2: Add a Durable Worktree Binding Ledger

**Files:**
- Modify: `hermes_state_common.py`
- Modify: `hermes_state_schema.py`
- Modify: `hermes_state.py`
- Test: `tests/state/test_conversation_worktree_bindings.py`

**Interfaces:**
- Consumes: `root_session_id` plus immutable Git identity fields.
- Produces: `ConversationWorktreeRecord`, `claim_conversation_worktree()`,
  `mark_conversation_worktree_ready()`, `mark_conversation_worktree_failed()`,
  `get_conversation_worktree()`, and `mark_conversation_worktree_removed()`.

- [ ] **Step 1: Write failing persistence and idempotency tests**

```python
def test_binding_claim_precedes_lazy_session_row(db):
    record = db.claim_conversation_worktree(
        root_session_id="root-1",
        worktree_path="/repo/.worktrees/root-1",
        branch="hermes/session/root-1",
        base_commit="a" * 40,
        repo_common_dir="/repo/.git",
    )
    assert record.state == "creating"
    assert db.get_session("root-1") is None


def test_conflicting_second_claim_is_rejected(db):
    db.claim_conversation_worktree(root_session_id="root", worktree_path="/a", branch="b", base_commit="a" * 40, repo_common_dir="/r/.git")
    with pytest.raises(ConversationWorktreeConflict):
        db.claim_conversation_worktree(root_session_id="root", worktree_path="/other", branch="b", base_commit="a" * 40, repo_common_dir="/r/.git")


def test_ready_binding_survives_reopen(tmp_path):
    first = SessionDB(tmp_path / "state.db")
    first.claim_conversation_worktree(root_session_id="root", worktree_path="/w", branch="b", base_commit="a" * 40, repo_common_dir="/r/.git")
    first.mark_conversation_worktree_ready("root")
    first.close()
    assert SessionDB(tmp_path / "state.db").get_conversation_worktree("root").state == "ready"
```

- [ ] **Step 2: Run and verify RED**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/state/test_conversation_worktree_bindings.py`

Expected: missing ledger methods.

- [ ] **Step 3: Add schema and transactional APIs**

Add `conversation_worktree_bindings` to `hermes_state_common.py`:

```sql
CREATE TABLE IF NOT EXISTS conversation_worktree_bindings (
    root_session_id TEXT PRIMARY KEY,
    worktree_path TEXT NOT NULL,
    branch TEXT NOT NULL,
    base_commit TEXT NOT NULL,
    repo_common_dir TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('creating','ready','creation_failed','retained','removed')),
    failure_phase TEXT,
    failure_message TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)
```

Create a frozen `ConversationWorktreeRecord` and use `SessionDB._execute_write` for claims and state
transitions. An identical claim returns the existing row; any identity mismatch raises
`ConversationWorktreeConflict`. Do not add a foreign key because gateway/desktop claims precede lazy
session-row persistence.

- [ ] **Step 4: Run state tests and schema regression tests**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/state/test_conversation_worktree_bindings.py tests/test_hermes_state.py tests/test_state_db_malformed_repair.py`

Expected: all pass.

- [ ] **Step 5: Commit the ledger**

```bash
git add hermes_state_common.py hermes_state_schema.py hermes_state.py tests/state/test_conversation_worktree_bindings.py
git commit -m "feat(state): persist conversation worktree bindings"
```

### Task 3: Build the Git Lifecycle Manager

**Files:**
- Create: `agent/conversation_worktree.py`
- Test: `tests/agent/test_conversation_worktree_manager.py`

**Interfaces:**
- Consumes: `ConversationWorktreePolicy`, `SessionDB`, root identity, and explicit
  `conversation_kind="interactive" | "task"`.
- Produces: `ConversationWorktreeBinding`, `ConversationWorktreeError`, and
  `ConversationWorktreeManager.bind_new_root_session()` / `resolve_existing_session()`.

- [ ] **Step 1: Write real-Git failing tests**

Create temporary repositories with commits plus dirty tracked/untracked stable files. Assert:

```python
def test_binding_pins_committed_head_without_copying_dirty_stable_files(repo, db):
    base = git(repo, "rev-parse", "HEAD").strip()
    (repo / "tracked.txt").write_text("dirty")
    (repo / "untracked.txt").write_text("do not copy")
    binding = manager(repo, db).bind_new_root_session("root-1", conversation_kind="interactive")
    assert binding.base_commit == base
    assert git(binding.path, "rev-parse", "HEAD").strip() == base
    assert not (binding.path / "untracked.txt").exists()
    assert (repo / "tracked.txt").read_text() == "dirty"


def test_task_kind_bypasses_conversation_isolation(repo, db):
    assert manager(repo, db).bind_new_root_session("task", conversation_kind="task") is None


def test_bootstrap_failure_is_retained_and_never_ready(repo, db):
    with pytest.raises(ConversationWorktreeError, match="bootstrap"):
        manager(repo, db, bootstrap_command=(sys.executable, "-c", "raise SystemExit(7)")).bind_new_root_session("root", conversation_kind="interactive")
    record = db.get_conversation_worktree("root")
    assert record.state == "creation_failed"
    assert Path(record.worktree_path).exists()
```

Also test concurrent uniqueness, invalid/common-dir mismatch, idempotent ready reuse, partial-state
conflict, timeout, branch sanitization, and exact environment/cwd passed to bootstrap.

- [ ] **Step 2: Run and verify RED**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/agent/test_conversation_worktree_manager.py`

Expected: missing manager module.

- [ ] **Step 3: Implement minimal manager**

```python
@dataclass(frozen=True)
class ConversationWorktreeBinding:
    root_session_id: str
    path: Path
    branch: str
    base_commit: str
    repo_common_dir: Path


class ConversationWorktreeManager:
    def bind_new_root_session(
        self, root_session_id: str, *, conversation_kind: str
    ) -> ConversationWorktreeBinding | None:
        return self._bind_or_recover(root_session_id, conversation_kind=conversation_kind)

    def resolve_existing_session(
        self, root_session_id: str
    ) -> ConversationWorktreeBinding | None:
        record = self._db.get_conversation_worktree(root_session_id)
        return self._validated_ready_binding(record) if record is not None else None
```

Use `git -C <source> rev-parse HEAD`, `rev-parse --git-common-dir`, and `git worktree add --no-track
-b <branch> <path> <base_sha>` under a repository lock. Verify resulting branch/HEAD/common-dir before
marking ready. Register terminal/file cwd only in adapters after a ready return. Emit structured
`conversation_worktree.policy`, `.create`, `.bootstrap`, `.ready`, `.reuse`, and `.failure` log events
without environment contents or credentials.

- [ ] **Step 4: Run manager and existing worktree tests**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/agent/test_conversation_worktree_manager.py tests/cli/test_worktree.py tests/tools/test_subagent_worktree.py`

Expected: all pass.

- [ ] **Step 5: Commit the manager**

```bash
git add agent/conversation_worktree.py tests/agent/test_conversation_worktree_manager.py
git commit -m "feat(agent): create durable conversation worktrees"
```

### Task 4: Bind Desktop/TUI Root Sessions Before Agent Construction

**Files:**
- Modify: `tui_gateway/methods_session.py`
- Modify: `tui_gateway/server.py`
- Test: `tests/tui_gateway/test_conversation_worktree_sessions.py`
- Test: `tests/tui_gateway/test_session_cwd_follow.py`

**Interfaces:**
- Consumes: `ConversationWorktreeManager.bind_new_root_session()` and
  `resolve_existing_session()`.
- Produces: session dictionaries whose `cwd`, `explicit_cwd`, and session-info Git metadata all
  point at one ready binding.

- [ ] **Step 1: Write failing desktop/TUI contract tests**

```python
def test_session_create_binds_before_agent_build(rpc, fake_manager):
    created = rpc("session.create", {"source": "desktop", "cwd": "/stable"})
    assert fake_manager.calls == [(created["stored_session_id"], "interactive")]
    assert created["info"]["cwd"] == "/repo/.worktrees/root"
    assert agent_build_cwds == ["/repo/.worktrees/root"]


def test_session_branch_gets_distinct_stable_based_binding(rpc, fake_manager, seeded_parent):
    child = rpc("session.branch", {"session_id": seeded_parent})
    assert child["info"]["cwd"] != parent_info["cwd"]
    assert fake_manager.bound_roots[-1] == child["stored_session_id"]


def test_resume_reuses_persisted_binding_without_git_create(rpc, fake_manager, ready_row):
    rpc("session.resume", {"session_key": ready_row.root_session_id})
    assert fake_manager.create_calls == 0
    assert fake_manager.resolve_calls == [ready_row.root_session_id]
```

Add a fail-closed test proving `_schedule_agent_build` is not called when binding raises.

- [ ] **Step 2: Run and verify RED**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/tui_gateway/test_conversation_worktree_sessions.py`

Expected: session cwd remains configured stable cwd or manager is never called.

- [ ] **Step 3: Integrate the manager at root boundaries**

In `session.create`, allocate the durable `key`, bind it, then replace `resolved_cwd` before placing
the session in `_sessions` or scheduling agent construction. In `session.branch`, bind `new_key`
before creating/building the child. Resume resolves existing metadata; compression continues using
parent cwd inheritance and must not invoke creation. Include `conversation_worktree` details in
`_session_info`, and append the binding path, branch, and base commit to the agent system prompt as
explanatory context after backend enforcement succeeds.

- [ ] **Step 4: Run TUI/desktop session suites**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/tui_gateway/test_conversation_worktree_sessions.py tests/tui_gateway/test_session_cwd_follow.py tests/tui_gateway/test_session_git_metadata_generation.py tests/tui_gateway/test_protocol.py`

Expected: all pass.

- [ ] **Step 5: Commit desktop/TUI integration**

```bash
git add tui_gateway/methods_session.py tui_gateway/server.py tests/tui_gateway/test_conversation_worktree_sessions.py tests/tui_gateway/test_session_cwd_follow.py
git commit -m "feat(desktop): isolate every new conversation"
```

### Task 5: Route CLI Roots and `/new` Through the Shared Manager

**Files:**
- Modify: `cli.py`
- Test: `tests/cli/test_conversation_worktree_sessions.py`
- Test: `tests/cli/test_worktree.py`

**Interfaces:**
- Consumes: shared manager and the CLI's allocated `session_id`.
- Produces: initial CLI roots and each `/new` root bound before agent reset/build, with legacy `-w`
  mutually exclusive.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_enabled_conversation_policy_binds_initial_cli_root(monkeypatch):
    cli = build_cli_with_policy(monkeypatch)
    assert cli.working_directory == "/repo/.worktrees/first"
    assert manager.bound_roots == [cli.session_id]


def test_cli_new_allocates_another_worktree(monkeypatch):
    cli = build_cli_with_policy(monkeypatch)
    old = cli.working_directory
    cli.new_session(silent=True)
    assert cli.working_directory != old
    assert len(manager.bound_roots) == 2


def test_legacy_w_does_not_double_create_when_conversation_policy_enabled(monkeypatch):
    run_main(worktree=True)
    assert manager.create_count == 1
    assert legacy_setup_worktree.call_count == 0
```

- [ ] **Step 2: Run and verify RED**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/cli/test_conversation_worktree_sessions.py`

Expected: CLI starts/resets in stable cwd.

- [ ] **Step 3: Integrate initial and `/new` binding**

Resolve the policy before legacy `_setup_worktree`. When enabled and applicable, allocate the CLI
session ID, call the shared manager, set `TERMINAL_CWD`/`working_directory`, and inject the same
system note used by desktop. `new_session()` must bind the newly allocated ID before
`reset_session_state()`; on failure leave the old session usable and report the failed new root.

- [ ] **Step 4: Run CLI lifecycle suites**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/cli/test_conversation_worktree_sessions.py tests/cli/test_worktree.py tests/cli/test_worktree_selfheal.py tests/run_agent/test_session_reset_fix.py`

Expected: all pass.

- [ ] **Step 5: Commit CLI integration**

```bash
git add cli.py tests/cli/test_conversation_worktree_sessions.py tests/cli/test_worktree.py
git commit -m "feat(cli): isolate interactive root sessions"
```

### Task 6: Bind Discord and iMessage Gateway Session Roots

**Files:**
- Modify: `gateway/session.py`
- Modify: `gateway/slash_commands.py`
- Modify: `gateway/run.py`
- Test: `tests/gateway/test_conversation_worktree_sessions.py`
- Test: `tests/gateway/test_new_clears_last_resolved_model.py`

**Interfaces:**
- Consumes: shared manager injected into `SessionStore`/`AsyncSessionStore` and explicit
  `conversation_kind` for root creation.
- Produces: first-contact and `/new` gateway entries whose cwd is ready before agent lookup/build.

- [ ] **Step 1: Write failing gateway tests**

```python
@pytest.mark.parametrize("platform", ["discord", "photon"])
async def test_first_interactive_gateway_session_gets_worktree(platform, store, source):
    source.platform = platform
    entry = await store.get_or_create_session(source)
    assert entry.cwd == "/repo/.worktrees/root"
    assert manager.bound_roots == [entry.session_id]


async def test_new_rotates_to_distinct_worktree(store, source):
    first = await store.get_or_create_session(source)
    second = await store.reset_session(first.session_key)
    assert second.session_id != first.session_id
    assert second.cwd != first.cwd


async def test_cron_session_never_calls_conversation_manager(store, cron_source):
    await store.get_or_create_session(cron_source)
    assert manager.calls == []
```

Add reconnect/reply/voice-text tests asserting reuse, plus failure tests proving no agent-cache entry is
created when binding fails.

- [ ] **Step 2: Run and verify RED**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/gateway/test_conversation_worktree_sessions.py`

Expected: gateway entries use configured stable cwd and manager is not called.

- [ ] **Step 3: Integrate gateway creation/reset**

Inject a manager factory into `SessionStore`. In `_get_or_create_session_impl` and `reset_session`,
classify actual user-facing platform roots as `interactive`, bind the new durable session ID before
DB row/agent creation, and store cwd plus binding metadata on `SessionEntry`. Explicitly classify
cron, Kanban, tool, review, orchestrator, and delegated sources as `task`. Keep transport reconnects
on the existing entry.

- [ ] **Step 4: Run gateway boundary suites**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/gateway/test_conversation_worktree_sessions.py tests/gateway/test_new_clears_last_resolved_model.py tests/gateway/test_restart_resume_pending.py tests/gateway/test_profile_routing.py tests/cron/test_cron_workdir.py`

Expected: all pass.

- [ ] **Step 5: Commit gateway integration**

```bash
git add gateway/session.py gateway/slash_commands.py gateway/run.py tests/gateway/test_conversation_worktree_sessions.py
git commit -m "feat(gateway): isolate interactive conversations"
```

### Task 7: Add Safe Explicit Cleanup and Status Surfaces

**Files:**
- Modify: `agent/conversation_worktree.py`
- Modify: `tui_gateway/methods_session.py`
- Modify: `hermes_cli/worktree_cmd.py`
- Modify: `apps/desktop/src/app/chat/sidebar/session-actions-menu.tsx`
- Modify: `apps/desktop/src/i18n/en.ts`
- Test: `tests/agent/test_conversation_worktree_cleanup.py`
- Test: `tests/tui_gateway/test_conversation_worktree_cleanup.py`
- Test: `apps/desktop/src/app/chat/sidebar/session-actions-menu.test.tsx`

**Interfaces:**
- Consumes: ready/retained ledger records and exact root identity.
- Produces: `CleanupVerdict`, `inspect_cleanup(root_session_id)`,
  `remove_after_explicit_request(root_session_id)`, and `session.worktree_cleanup` RPC.

- [ ] **Step 1: Write failing cleanup tests**

```python
@pytest.mark.parametrize("state", ["dirty", "active", "unintegrated", "unpushed", "mismatched"])
def test_cleanup_refuses_unsafe_worktree(state, prepared_binding):
    verdict = manager.inspect_cleanup(prepared_binding.root_session_id)
    assert verdict.allowed is False
    assert state in verdict.reasons
    assert prepared_binding.path.exists()


def test_explicit_cleanup_removes_only_exact_clean_integrated_worktree(prepared_binding):
    result = manager.remove_after_explicit_request(prepared_binding.root_session_id)
    assert result.removed is True
    assert not prepared_binding.path.exists()
    assert sibling_worktree.exists()
```

- [ ] **Step 2: Run and verify RED**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/agent/test_conversation_worktree_cleanup.py tests/tui_gateway/test_conversation_worktree_cleanup.py`

Expected: cleanup APIs absent.

- [ ] **Step 3: Implement Git-backed verdict and explicit RPC**

```python
@dataclass(frozen=True)
class CleanupVerdict:
    allowed: bool
    reasons: tuple[str, ...]
```

Inspect porcelain status, in-progress Git state, active-session bindings, commit reachability from
configured integration refs, and remote reachability. Remove without `--force`, verify absence from
`git worktree list`, then delete only the exact owned branch when configured. `session.close` and
archive operations must not call cleanup automatically. Add a **Clean up worktree** session action
that first requests `session.worktree_cleanup` in inspect mode, displays every blocking reason, and
only sends the explicit remove request after the existing confirmation-dialog flow. A plain archive
or delete action never implies cleanup.

- [ ] **Step 4: Run cleanup and existing worktree teardown tests**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/agent/test_conversation_worktree_cleanup.py tests/tui_gateway/test_conversation_worktree_cleanup.py tests/hermes_cli/test_kanban_worktree_teardown.py tests/cli/test_worktree.py`

Run: `cd apps/desktop && npm test -- --run src/app/chat/sidebar/session-actions-menu.test.tsx`

Expected: all pass.

- [ ] **Step 5: Commit cleanup/status support**

```bash
git add agent/conversation_worktree.py tui_gateway/methods_session.py hermes_cli/worktree_cmd.py apps/desktop/src/app/chat/sidebar/session-actions-menu.tsx apps/desktop/src/app/chat/sidebar/session-actions-menu.test.tsx apps/desktop/src/i18n/en.ts tests/agent/test_conversation_worktree_cleanup.py tests/tui_gateway/test_conversation_worktree_cleanup.py
git commit -m "feat(worktree): guard explicit conversation cleanup"
```

### Task 8: Add a Guarded Fast Compression Lane

**Files:**
- Modify: `agent/auxiliary_client.py`
- Modify: `agent/context_compressor.py`
- Modify: `agent/conversation_compression.py`
- Modify: `hermes_cli/config_defaults.py`
- Modify: `apps/desktop/src/app/session/hooks/use-message-stream/gateway-event/status.ts`
- Modify: `apps/desktop/src/store/compaction.ts`
- Test: `tests/agent/test_fast_compression_lane.py`
- Test: `tests/agent/test_compression_attempt_telemetry.py`
- Test: `apps/desktop/src/app/session/hooks/use-message-stream/compaction-event.test.tsx`

**Interfaces:**
- Consumes: explicit `auxiliary.compression` route configuration and the existing structured compression template.
- Produces: opt-in non-reasoning compression requests, bounded output only for explicitly certified routes, content-free phase timings, and reconnect-safe compacting-state reconciliation.

- [ ] **Step 1: Write failing fast-lane and telemetry tests**

Cover an explicit compression-only provider/model with `reasoning_effort: none`, an optional output cap that is rejected for inherited or unknown/reasoning routes, queue-wait and first-progress timings, iterative prompt bounds, preservation of active task/constraints/modified-file anchors, and a reconnect lifecycle that clears stale UI state only from trusted terminal/resumed server evidence.

- [ ] **Step 2: Run and verify RED**

Run the focused compressor, auxiliary-client, telemetry, and desktop compaction tests. Expected: missing fast-lane budget/timing and reconnect reconciliation contracts.

- [ ] **Step 3: Implement the guarded lane**

Keep default model inheritance unchanged. Apply `reasoning_effort: none` and `max_output_tokens` only when explicitly configured under `auxiliary.compression`; never change the main chat route or natural-language router. Emit content-free `queue_wait_ms`, `prompt_build_ms`, `time_to_first_progress_ms`, `summary_generation_ms`, and `commit_ms`. Do not log prompt or summary content. Preserve the structured checkpoint template, redaction, tail, durable archive, fallback, cooldown, and logical-root worktree reuse.

- [ ] **Step 4: Run compression and continuation compatibility**

Run focused context-compressor, progress-timeout, worker-isolation, continuity, fallback-budget, telemetry, TUI status, desktop compaction, and conversation-worktree lineage tests. Expected: all pass.

- [ ] **Step 5: Commit the performance lane**

```bash
git add agent/auxiliary_client.py agent/context_compressor.py agent/conversation_compression.py hermes_cli/config_defaults.py tests/agent/test_fast_compression_lane.py tests/agent/test_compression_attempt_telemetry.py apps/desktop/src/app/session/hooks/use-message-stream/gateway-event/status.ts apps/desktop/src/store/compaction.ts apps/desktop/src/app/session/hooks/use-message-stream/compaction-event.test.tsx
git commit -m "perf(compression): add guarded fast summary lane"
```

### Task 9: Migrate Local Configuration, Document, and Certify

**Files:**
- Modify: `hermes_cli/config_migrations.py`
- Modify: `cli-config.yaml.example`
- Modify: `website/docs/user-guide/configuration.md`
- Modify after code merge/deployment: `/Users/mikedemott/.hermes/config.yaml`
- Test: `tests/hermes_cli/test_conversation_worktree_config_migration.py`

**Interfaces:**
- Consumes: completed manager and adapters.
- Produces: config version 39 migration, canonical top-level policy, deployment evidence.

- [ ] **Step 1: Write failing migration test**

```python
def test_v39_moves_legacy_block_without_overwriting_top_level(tmp_path):
    cfg = {"_config_version": 38, "desktop": {"conversation_worktree": {"enabled": True}}}
    migrated = migrate(cfg)
    assert migrated["conversation_worktree"]["enabled"] is True
    assert "conversation_worktree" not in migrated["desktop"]
    assert migrated["_config_version"] == 39
```

Also assert an existing top-level block wins and unrelated desktop/provider fields are byte-equivalent
after structured migration.

- [ ] **Step 2: Run and verify RED**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider tests/hermes_cli/test_conversation_worktree_config_migration.py`

Expected: migration remains at version 38.

- [ ] **Step 3: Implement migration and documentation**

Add `_migrate_to_39`, update examples/docs, and retain runtime legacy fallback for profiles not yet
migrated. Do not modify local config until the code commit is merged into stable Hermes.

- [ ] **Step 4: Run focused integration certification**

Run:

```bash
venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/agent/test_conversation_worktree_policy.py \
  tests/state/test_conversation_worktree_bindings.py \
  tests/agent/test_conversation_worktree_manager.py \
  tests/agent/test_conversation_worktree_cleanup.py \
  tests/tui_gateway/test_conversation_worktree_sessions.py \
  tests/tui_gateway/test_conversation_worktree_cleanup.py \
  tests/cli/test_conversation_worktree_sessions.py \
  tests/gateway/test_conversation_worktree_sessions.py \
  tests/hermes_cli/test_conversation_worktree_config_migration.py \
  tests/cli/test_worktree.py tests/tools/test_subagent_worktree.py \
  tests/cron/test_cron_workdir.py tests/hermes_cli/test_kanban_worktree_teardown.py
```

Expected: all pass with zero failures.

- [ ] **Step 5: Run full Hermes certification**

Run: `scripts/run_tests.sh -j 4`

Expected: terminal summary with zero failures. Record the exact pass/skip counts.

- [ ] **Step 6: Commit migration/docs**

```bash
git add hermes_cli/config_migrations.py cli-config.yaml.example website/docs/user-guide/configuration.md tests/hermes_cli/test_conversation_worktree_config_migration.py
git commit -m "docs(config): enable conversation worktree policy"
```

- [ ] **Step 7: Merge, migrate local config, and restart managed runtimes**

Fast-forward the reviewed implementation branch into stable Hermes. Move the existing local
`desktop.conversation_worktree` block to top-level without changing its values. Restart
`ai.hermes.gateway` through launchd and gracefully relaunch Hermes desktop; never use the updater
path that can replace local commits.

- [ ] **Step 8: Smoke-test each interactive entry point**

Create one new desktop chat, one CLI root, one Discord `/new`, and one iMessage fresh session. For
each, verify the persisted root binding, distinct path/branch, exact stable base commit, tool cwd, and
resume reuse. Recheck stable `git status`, branch, and HEAD to prove its pre-existing dirty artifacts
were neither copied nor changed. Verify a cron/Kanban dispatch still uses its task-owned workspace.

- [ ] **Step 9: Final evidence handoff**

Report exact commits, commands, test totals, four smoke-session bindings, stable source status before
and after, and any retained cleanup blockers. Do not claim existing historical sessions were migrated.
