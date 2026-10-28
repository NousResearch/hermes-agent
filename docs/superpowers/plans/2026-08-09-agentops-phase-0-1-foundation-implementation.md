# AgentOps Phase 0/1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this approved, bounded Phase 0/1 plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在独立 worktree 中交付一个严格 `observe_only` 的 AgentOps 控制面基础：稳定事件与审计契约、SQLite WAL、无密钥 event spool、UDS health API 和无模型 doctor；不接触任何受管 Target。

**Architecture:** `plugins/agentops` 是一个 opt-in Hermes general plugin，只注册 `hermes agentops` CLI，绝不注册 Gateway hook 或 tool。显式启动的 `agentops daemon` 只打开自己的 SQLite 与 UDS；所有状态变更只限于 AgentOps 状态目录。事件先进入原子 JSONL/JSON spool，再在短 SQLite 事务中幂等落库并写 audit chain；API 仅有 `GET /v1/health`。

**Tech Stack:** Python 3.10+ 标准库、PyYAML（现有依赖）、SQLite WAL、Unix Domain Socket、pytest。

## Global Constraints

- 所有 Target 权限固定 `observe_only`；本阶段不实现 R1/R2/R3/R4、Executor、Shell、LLM、Dashboard、Bridge、Collector 或 launchd 安装。
- 只能写入显式传入的 AgentOps state directory 和 pytest `tmp_path`；不得修改 `/Users/molly/Desktop/Hermes` 的 dirty 实时工作目录、`~/.hermes` 既有数据、Gateway、Cron、LaunchAgent 或业务数据。
- 配置缺失、配置无效或路径安全失败时 daemon fail-closed 且不绑定 UDS；迁移、spool 或审计链异常时 health 只能报告 `observe_only` degraded 状态；不得降级为写能力。
- Event 与 Audit 在写入前必须拒绝 Secret；错误和 doctor 输出不得回显输入 payload。
- UDS 目录权限为 `0700`，socket 权限为 `0600`；API 只能提供 `GET /v1/health`。
- 插件不修改 `run_agent.py`、`cli.py`、`gateway/run.py` 或已有 LaunchAgent。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `docs/superpowers/specs/2026-08-09-agentops-phase-0-baseline.md` | 只读盘点、Target/控制器权威关系和 G0 证据 |
| `docs/superpowers/specs/2026-08-09-agentops-phase-1-operations.md` | 迁移、备份/恢复、卸载和未来 launchd 的非执行设计 |
| `plugins/agentops/plugin.yaml` | opt-in standalone plugin manifest |
| `plugins/agentops/__init__.py` | 只注册 CLI，不注册 hook/tool |
| `plugins/agentops/cli.py` | `daemon` 与 `doctor --json` 的 operator CLI |
| `plugins/agentops/control/models.py` | 不可变领域数据类型和 `AuthorityMode.OBSERVE_ONLY` |
| `plugins/agentops/control/config.py` | 严格安全默认配置与 YAML loader |
| `plugins/agentops/control/events.py` | schema-v1 EventEnvelope、canonical hash、secret gate、spool/quarantine |
| `plugins/agentops/control/audit.py` | AuditEvent 和 append-only chain hash |
| `plugins/agentops/control/store.py` | SQLite WAL migration、event idempotency、backup/restore、read-only inspection |
| `plugins/agentops/control/api.py` | UDS HTTP subset 与 health client |
| `plugins/agentops/control/daemon.py` | 组合 config/store/spool/api 的无模型 lifecycle |
| `tests/plugins/agentops/...` | unit、contract、integration 和安全边界测试 |

## Task 1: Phase 0 evidence and operations design

**Files:**
- Create: `docs/superpowers/specs/2026-08-09-agentops-phase-0-baseline.md`
- Create: `docs/superpowers/specs/2026-08-09-agentops-phase-1-operations.md`

**Consumes:** current Git, launchd, process, Cron and config state via read-only commands.

**Produces:** a target registry baseline and a no-install recovery runbook.

- [x] **Step 1: Record deployment and protected-worktree state**

Run:

```bash
git -C /Users/molly/Desktop/Hermes status --short
git -C /Users/molly/Desktop/Hermes rev-parse HEAD
git -C /Users/molly/Desktop/Hermes rev-list --left-right --count main...origin/main
```

Record the exact SHA, dirty-worktree rule and upstream divergence without copying secret config values.

- [x] **Step 2: Record five Profile Targets and existing write controllers**

Map each `ai.hermes.gateway*` launchd Label to its Profile and logs. Read the watchdog source to establish that it currently executes `launchctl kickstart -k`; record it as the existing writer and leave AgentOps as `observe_only`.

- [x] **Step 3: Record Cron and false-green behavior**

Run `hermes cron list`, enumerate active and paused jobs, and state that AIVault’s success exit state is not a future business-health proof.

- [x] **Step 4: Write non-executing operations design**

Document backup-before-migration, restore by stopping the future sidecar and replacing only AgentOps `state.db`, uninstall by removing only AgentOps files after the service is stopped, and a future launchd label that is intentionally not installed in Phase 1.

- [x] **Step 5: Commit Phase 0 evidence and plan**

```bash
git add docs/superpowers/specs/2026-08-09-agentops-phase-0-baseline.md \
  docs/superpowers/specs/2026-08-09-agentops-phase-1-operations.md \
  docs/superpowers/plans/2026-08-09-agentops-phase-0-1-foundation-implementation.md
git commit -m "docs: record agentops phase 0 baseline"
```

## Task 2: Models, safe configuration, and plugin CLI surface

**Files:**
- Create: `plugins/agentops/plugin.yaml`
- Create: `plugins/agentops/__init__.py`
- Create: `plugins/agentops/cli.py`
- Create: `plugins/agentops/control/__init__.py`
- Create: `plugins/agentops/control/models.py`
- Create: `plugins/agentops/control/config.py`
- Test: `tests/plugins/agentops/unit/test_config.py`
- Test: `tests/plugins/agentops/unit/test_models.py`

**Consumes:** `PluginContext.register_cli_command`, `AgentOpsConfig` YAML.

**Produces:** `load_agentops_config(path: Path) -> AgentOpsConfig`, immutable `EventEnvelope`, `AuditEvent`, `ControlPlaneHealth`, and `hermes agentops {daemon,doctor}`.

- [x] **Step 1: Write failing safe-default tests**

```python
def test_missing_config_is_explicitly_observe_only(tmp_path):
    config = load_agentops_config(tmp_path / "missing.yaml")
    assert config.default_authority is AuthorityMode.OBSERVE_ONLY
    assert config.global_write_enabled is False
    assert "config_missing" in config.safe_start_reasons

def test_invalid_config_does_not_raise_write_authority(tmp_path):
    path = tmp_path / "agentops.yaml"
    path.write_text("safety: [")
    config = load_agentops_config(path)
    assert config.default_authority is AuthorityMode.OBSERVE_ONLY
    assert config.safe_start_reasons == ("config_invalid",)
```

- [x] **Step 2: Run the failing tests**

Run: `pytest -q tests/plugins/agentops/unit/test_config.py tests/plugins/agentops/unit/test_models.py`

Expected: import failure until the control package exists.

- [x] **Step 3: Implement only safe models/config**

Use frozen dataclasses, timezone-aware timestamps, `Path.expanduser()`, schema version 1 and an `AuthorityMode` enum containing only `OBSERVE_ONLY` in active configuration. Reject any config attempting to set `global_write_enabled: true` by retaining `False` and adding `write_requested_but_disabled` to safe reasons.

- [x] **Step 4: Add opt-in plugin and CLI parser**

`register(ctx)` calls only:

```python
ctx.register_cli_command(
    name="agentops",
    help="Observe-only AgentOps control-plane diagnostics",
    setup_fn=register_cli,
    handler_fn=agentops_command,
    description="Run the local observe-only AgentOps daemon or diagnostics.",
)
```

The `doctor` parser accepts `--json` and `--config`; the `daemon` parser accepts `--config`. It must never start automatically on import.

- [x] **Step 5: Run models/config tests**

Run: `pytest -q tests/plugins/agentops/unit/test_config.py tests/plugins/agentops/unit/test_models.py`

Expected: all tests pass.

## Task 3: Event envelope, secret gate, spool and audit chain

**Files:**
- Create: `plugins/agentops/control/events.py`
- Create: `plugins/agentops/control/audit.py`
- Test: `tests/plugins/agentops/unit/test_events.py`
- Test: `tests/plugins/agentops/unit/test_audit.py`

**Consumes:** `EventEnvelope`, `AuditEvent`, canonical JSON encoder.

**Produces:** `EventSpool.write/replay`, canonical `sha256:` hashes, secret-safe validation and `append_audit` chain entries.

- [x] **Step 1: Write failing event/audit tests**

```python
def test_event_hash_is_stable_when_mapping_order_changes():
    assert EventEnvelope.from_dict(payload={"b": 2, "a": 1}, **BASE).content_hash == \
           EventEnvelope.from_dict(payload={"a": 1, "b": 2}, **BASE).content_hash

def test_secret_payload_is_rejected_before_spooling_or_storage(tmp_path):
    with pytest.raises(EventValidationError):
        EventEnvelope.from_dict(payload={"token": "sk-test-canary-secret"}, **BASE)
    assert not list(tmp_path.iterdir())
```

Add tests for duplicate replay, unknown schema quarantine, corrupt JSON quarantine, chain verification and chain tampering detection.

- [x] **Step 2: Run failing event/audit tests**

Run: `pytest -q tests/plugins/agentops/unit/test_events.py tests/plugins/agentops/unit/test_audit.py`

Expected: import failure until implementation exists.

- [x] **Step 3: Implement canonical event and audit records**

`canonical_json(value)` serializes only JSON-compatible values with sorted keys and compact separators. `EventEnvelope` requires `schema_version == 1`, UUID event ID, non-empty type/producer/target, aware timestamp and JSON payload. Secret-looking keys and common key/token/cookie/password patterns raise `EventValidationError`; no payload is included in the exception message.

`EventSpool.write(event)` atomically writes `event_id.json` using a same-directory temporary path and `os.replace`. `replay(store)` deletes successfully appended or duplicate entries, moves valid unknown/corrupt non-secret entries to `quarantine`, and writes a redacted metadata-only quarantine record if raw content contains a secret.

`AuditEvent` receives `previous_hash`, computes `entry_hash` from canonical data, and chain verification recomputes every sequence value.

- [x] **Step 4: Run event/audit tests**

Run: `pytest -q tests/plugins/agentops/unit/test_events.py tests/plugins/agentops/unit/test_audit.py`

Expected: all tests pass.

## Task 4: SQLite WAL store, migration and recovery

**Files:**
- Create: `plugins/agentops/control/store.py`
- Test: `tests/plugins/agentops/unit/test_store.py`

**Consumes:** Event and Audit contracts.

**Produces:** `open_store(config: AgentOpsConfig) -> AgentOpsStore`, `append_event(event) -> AppendResult`, `append_audit(event) -> int`, `get_health() -> ControlPlaneHealth`, `backup_to(path)` and `restore_from(path)`.

- [x] **Step 1: Write failing store tests**

```python
def test_store_uses_wal_and_event_idempotency(tmp_path):
    store = open_store(config)
    first = store.append_event(make_event())
    second = store.append_event(make_event())
    assert first.inserted is True
    assert second.inserted is False
    assert store.journal_mode() == "wal"

def test_backup_restore_returns_to_a_verified_snapshot(tmp_path):
    store = open_store(config)
    store.append_event(make_event("a"))
    backup = store.backup_to(tmp_path / "backup.db")
    store.append_event(make_event("b"))
    store.restore_from(backup)
    assert store.event_count() == 1
```

Add tests for migration-version validation, busy timeout, append-only audit chain, and read-only inspection not creating a DB.

- [x] **Step 2: Run failing store tests**

Run: `pytest -q tests/plugins/agentops/unit/test_store.py`

Expected: import failure until implementation exists.

- [x] **Step 3: Implement schema v1 and recovery**

Set `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`, `PRAGMA busy_timeout=5000`, and short `BEGIN IMMEDIATE` transactions. Create only `schema_migrations`, `events`, `audit_events` and `metadata` tables. Before upgrading an existing DB, produce a SQLite backup in the same controlled AgentOps directory. Never migrate an unknown newer version. `restore_from` may replace only the store’s own opened database after a verified SQLite backup; tests run this only in `tmp_path`.

- [x] **Step 4: Run store tests**

Run: `pytest -q tests/plugins/agentops/unit/test_store.py`

Expected: all tests pass.

## Task 5: UDS health API and safe daemon lifecycle

**Files:**
- Create: `plugins/agentops/control/api.py`
- Create: `plugins/agentops/control/daemon.py`
- Test: `tests/plugins/agentops/contract/test_control_api.py`
- Test: `tests/plugins/agentops/integration/test_daemon_restart.py`

**Consumes:** `AgentOpsConfig`, `AgentOpsStore`, `EventSpool`, `ControlPlaneHealth`.

**Produces:** `run_daemon(config: AgentOpsConfig, stop_event: threading.Event) -> int`, UDS `GET /v1/health`, and a client `request_health(socket_path)`.

- [x] **Step 1: Write failing API/lifecycle tests**

```python
def test_uds_exposes_health_and_no_write_routes(tmp_path):
    handle = start_test_daemon(tmp_path)
    assert request_health(handle.socket_path)["authority_mode"] == "observe_only"
    assert uds_request(handle.socket_path, "POST", "/v1/events")["status"] == 405
    assert uds_request(handle.socket_path, "GET", "/v1/fleet")["status"] == 404

def test_restart_replays_spool_once_and_keeps_observe_only(tmp_path):
    spool = EventSpool(tmp_path / "spool")
    spool.write(make_event())
    first = start_test_daemon(tmp_path)
    first.stop()
    second = start_test_daemon(tmp_path)
    assert second.health()["event_count"] == 1
    assert second.health()["authority_mode"] == "observe_only"
```

Add tests for missing config, forced migration failure, audit-chain failure and socket permissions.

- [x] **Step 2: Run failing API/lifecycle tests**

Run: `pytest -q tests/plugins/agentops/contract/test_control_api.py tests/plugins/agentops/integration/test_daemon_restart.py`

Expected: import failure until implementation exists.

- [x] **Step 3: Implement read-only transport and daemon**

Use a `ThreadingMixIn` Unix stream server with a bounded HTTP request parser. Permit exactly `GET /v1/health`; return JSON `404` for unknown paths and `405` for every non-GET request without reading a body. Create the socket’s parent directory at `0700` and socket at `0600`; never unlink a pre-existing socket.

On an explicit daemon start, create only configured AgentOps state directories, open/migrate store, replay spool, verify audit chain, then serve health. A config/store/spool/audit failure adds a `safe_start_reason`, keeps `authority_mode="observe_only"`, and still serves health whenever the UDS can bind. No dependency may open a Gateway connection or a model client.

- [x] **Step 4: Run API/lifecycle tests**

Run: `pytest -q tests/plugins/agentops/contract/test_control_api.py tests/plugins/agentops/integration/test_daemon_restart.py`

Expected: all tests pass.

## Task 6: No-model doctor and security boundary tests

**Files:**
- Modify: `plugins/agentops/cli.py`
- Test: `tests/plugins/agentops/unit/test_cli.py`
- Test: `tests/plugins/agentops/security/test_observe_only_boundaries.py`

**Consumes:** read-only store inspection and UDS health client.

**Produces:** machine-readable `agentops doctor --json` report that never creates state, starts a daemon, or includes secrets.

- [x] **Step 1: Write failing doctor/security tests**

```python
def test_doctor_json_is_machine_readable_and_does_not_create_missing_db(tmp_path, capsys):
    rc = agentops_command(make_args("doctor", tmp_path / "missing.yaml", json=True))
    report = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert report["authority_mode"] == "observe_only"
    assert not (tmp_path / "state.db").exists()

def test_plugin_registers_no_hook_or_tool_and_phase_one_has_no_executor_source():
    ctx = RecordingContext()
    register(ctx)
    assert ctx.commands == ["agentops"]
    assert ctx.hooks == []
    assert ctx.tools == []
```

Also assert synthetic API keys cannot occur in SQLite serialized event/audit rows, spool files, health JSON, or captured log messages.

- [x] **Step 2: Implement doctor**

`doctor_report(config_path)` loads config without creating it, opens an existing store read-only only when it exists, checks schema/audit status, optionally queries existing UDS health, and returns a JSON-safe report. `--json` emits only canonical JSON to stdout; non-JSON emits a compact operator summary. It returns non-zero for missing/degraded state but never performs repair.

- [x] **Step 3: Run doctor/security tests**

Run: `pytest -q tests/plugins/agentops/unit/test_cli.py tests/plugins/agentops/security/test_observe_only_boundaries.py`

Expected: all tests pass.

## Task 7: Full G1 verification, review and commits

**Files:**
- Modify: all Phase 1 implementation/tests as needed to resolve review findings.

**Consumes:** every preceding test and the G1 evidence matrix.

**Produces:** a reviewable branch with only documentation, AgentOps plugin code and tests.

- [x] **Step 1: Run focused suite**

Run: `pytest -q tests/plugins/agentops`

Expected: every unit, contract, integration and security test passes.

- [x] **Step 2: Run static safety scans**

Run:

```bash
rg -n "subprocess|os\.system|shell=True|launchctl|requests\.|httpx\.|openai|gateway" plugins/agentops
rg -n "POST|PUT|PATCH|DELETE" plugins/agentops/control/api.py
git diff --check 39e8b2b2b..HEAD
```

Expected: no executable/Target-control implementation; only API test assertions or explanatory documentation may contain disallowed method names.

- [x] **Step 3: Verify requirement-by-requirement G1 evidence**

Confirm event/API/store contracts, missing/invalid configuration safe mode, migration failure safe mode, audit-chain safe mode, synthetic-secret non-persistence, spool idempotency, backup/restore and health without Gateway/model using actual test output.

- [x] **Step 4: Commit implementation**

```bash
git add plugins/agentops tests/plugins/agentops docs/superpowers/specs/2026-08-09-agentops-phase-1-operations.md
git commit -m "feat: add observe-only agentops foundation"
```

- [ ] **Step 5: Request Sol review before merge/push**

Provide branch SHA, touched files, raw test output, static scan output, G1 matrix, known limitations and the explicit statement that no launchd service or Target write capability was installed.

## G1 remediation evidence (2026-08-09)

| G1 finding | Implemented Phase 1 control | Direct evidence |
|---|---|---|
| 1. Dedicated state boundary | Canonical dedicated state root with marker/owner/symlink/Git/Hermes-root checks; DB/spool/UDS layout must remain inside it; existing DB is read-only preflighted before WAL. | `test_unmanaged_existing_state_dir_is_rejected`, `test_git_worktree_and_symlink_state_dirs_are_rejected`, `test_hermes_root_state_dir_is_rejected`, `test_unrelated_database_is_untouched_when_config_path_is_rejected` |
| 2. Restore safety | Controlled-backup-only restore copies and preflights a read-only candidate, preserves a pre-restore snapshot, atomically replaces only after validation, and rolls back when reopen fails. | `test_restore_rejects_bad_candidates_before_replacing_live_store`, `test_restore_reopen_failure_rolls_back_to_preserved_snapshot`, `test_schema_migration_runner_is_singleton_and_monotonic` |
| 3. Secret gate | All Event/Audit string and metadata fields are validated; invalid UTF-8/raw spool input becomes hash-only quarantine metadata; bounded quarantine may only drop raw data with an explicit count. | `test_event_string_fields_reject_secret_values`, `test_audit_string_fields_reject_secret_values`, `test_invalid_utf8_spool_is_hashed_and_never_persisted_verbatim`, `test_quarantine_budget_drops_untrusted_raw_input_without_retaining_it` |
| 4. Doctor contract | `ok` requires a safe config, read-only store integrity/audit validity, reachable `ready=true` daemon, usable store, valid audit and healthy spool. The plugin wrapper propagates a non-zero degraded exit through the existing Hermes main dispatcher. | `test_real_cli_doctor_exits_nonzero_when_degraded` |
| 5. Crash/singleton safety | A `flock` lock serializes independent daemon processes. A live UDS blocks a second daemon; only a held-lock daemon may reclaim a current-user stale socket after failed health probing. | `test_second_daemon_is_rejected_and_stale_socket_is_reclaimed_after_kill`, `test_daemon_does_not_replace_non_socket_occupant` |
| 6. Audit-chain metadata | Append transaction updates sequence/hash metadata atomically and verification checks first/continuous sequence, count and head/tail agreement. | `test_audit_chain_rejects_head_middle_and_tail_deletion`, `test_audit_chain_rejects_metadata_head_mismatch` |
| 7. UDS fail-closed | State/socket parent must be current-user non-symlink `0700`; bound socket is re-read as `0600`/current-user/socket type. Chmod failure and occupants refuse startup. | `test_uds_refuses_wide_state_dir_and_chmod_failure`, `test_uds_refuses_symlink_socket_occupant` |
| 8. Quarantine crash safety | Quarantine uses UUID temp names, cleans orphan temp artifacts without reading them, fsyncs cleanup, reports `failed`, and makes daemon health fatal on an unredacted/replay failure. | `test_orphaned_quarantine_temp_is_removed_before_replay_and_never_blocks_restart`, `test_quarantine_replace_failure_is_fatal_and_does_not_leave_raw_input`, `test_unremovable_untrusted_spool_input_keeps_daemon_not_ready` |
| 9. Restore atomic recovery | Store RLock spans pre-restore snapshot through replacement, reopen and rollback. Recovery reopens original before replacement or rolls back snapshot after replacement, then verifies audit usability. | `test_restore_faults_leave_a_usable_verified_original_store`, `test_restore_serializes_concurrent_append_across_snapshot_and_replace` |
| P2 durability/budgets | Event/quarantine replacement and orphan cleanup fsync parent; spool `failed` health is fatal; `schema_migrations` is singleton; ProcessLock uses `O_NOFOLLOW` and requires `nlink == 1`. | `test_process_lock_rejects_hard_linked_lockfile` |

### Raw verification output

```text
$ /Users/molly/Desktop/Hermes/venv/bin/python -m compileall -q plugins/agentops && /Users/molly/Desktop/Hermes/venv/bin/python -m pytest -q tests/plugins/agentops
bringing up nodes...
bringing up nodes...

........................................................................ [100%]
72 passed in 3.33s

$ /Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -m pytest -o addopts='' -q tests/plugins/agentops
........................................................................ [100%]
72 passed in 6.17s

$ /Users/molly/Desktop/Hermes/venv/bin/python -m pytest -q tests/hermes_cli/test_plugins.py tests/hermes_cli/test_plugin_cli_registration.py tests/hermes_cli/test_startup_plugin_gating.py tests/hermes_cli/test_plugin_scanner_recursion.py
........................................................................ [ 59%]
..................................................                       [100%]
122 passed in 2.60s

$ rg -n "subprocess|os\.system|shell=True|launchctl|requests\.|httpx\.|openai|gateway" plugins/agentops --glob '*.py'
(no matches)

$ rg -n "POST|PUT|PATCH|DELETE" plugins/agentops/control/api.py
(no matches)

$ git diff --check 34b4513f9..HEAD && git diff --check
(no output; exit 0)
```

### Known limitations / non-authorizations

- This is an isolated Phase 1 implementation only. It has not installed launchd, a scheduler, Gateway hook, collector, Executor, Target write API, or any R1/R2/R3/R4 capability.
- `backup_to` / `restore_from` exist only as controlled local AgentOps-store primitives; no CLI route exposes restore, and they cannot point outside the dedicated `backups/` and state root.
- Crash recovery is validated with an isolated child process and temporary state only. No production `~/.hermes` data, running Gateway or LaunchAgent was accessed.
- Python 3.11 verification used the project test venv (3.11.15); the separately installed `python3.11` had no pytest module, so it was used for `compileall` only. Python 3.14 ran the full Phase 1 suite with `-o addopts=''` because that interpreter lacks pytest-xdist while repository `addopts` requests `-n`.
- This branch is awaiting a fresh Sol G1 review. It must not merge, push or begin Phase 2 until that review accepts the evidence.
