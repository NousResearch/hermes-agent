# Orca–Hermes Account Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make the host Codex account selected in Orca the effective first-choice credential for every local Hermes agent, and reflect Hermes Codex failover back into Orca's account switcher without restarting active tasks.

**Architecture:** Add an Orca-agnostic, idle-only live reload to `CredentialPool`, then place the product-specific integration at the repository edge under `tools/orca_hermes_bridge`. A singleton Python daemon reconciles Orca account snapshots with the persisted Hermes pool, while a minimal Node helper reuses Orca's authenticated local `RuntimeClient` for `accounts.list` and `accounts.selectCodexForTarget`.

**Tech Stack:** Python 3.11+, Hermes credential-pool/auth-store helpers, Node.js 24 CommonJS, Orca 1.4.195 local runtime RPC, PowerShell/.NET Windows notifications, pytest through `scripts/run_tests.sh`.

**Spec:** `docs/superpowers/specs/2026-09-02-orca-hermes-account-bridge-design.md`

## Global Constraints

- Windows host integration; apply the account choice globally to all local Hermes agents.
- Keep `openai-codex` pool strategy `fill_first`; keep OpenRouter `qwen/qwen3-coder-next` as final fallback with reasoning effort `xhigh`.
- Never print, log, transmit, or copy OAuth access tokens, refresh tokens, API keys, or complete credential-file contents.
- Match Orca and Hermes accounts only by the stable ChatGPT provider account ID decoded locally from the OAuth JWT.
- Never reload or reorder an in-memory pool while it has an active credential lease; changes take effect before the next request.
- Mutate Hermes auth state only under `_auth_store_lock()` and persist only through `_save_auth_store()` or `write_credential_pool()`.
- Mutate Orca selection only through authenticated local runtime RPC; never edit Orca persistence files.
- Do not automatically jump back to a recovered account; a user selection in Orca is the recovery action.
- Keep the last Codex account visible when all Codex credentials are unavailable and show one de-duplicated Qwen fallback notification.
- Preserve the unrelated existing modifications in `agent/chat_completion_helpers.py`, `tests/run_agent/test_run_agent.py`, and `ui-tui/src/app/useSubmission.ts`; never stage them.
- Use `apply_patch` for every source/config edit and the canonical `scripts/run_tests.sh` runner for repository tests.

---

## File Map

- Modify `agent/credential_pool.py`: generic persisted-pool fingerprint and idle-only reconciliation.
- Create `tests/agent/test_credential_pool_live_reload.py`: cross-process ordering, lease, status, and token-preservation coverage.
- Create `tools/orca_hermes_bridge/__init__.py`: package marker and public version.
- Create `tools/orca_hermes_bridge/accounts.py`: JWT account-ID extraction, Orca snapshot normalization, Hermes row mapping, availability, and locked reordering.
- Create `tools/orca_hermes_bridge/runtime_rpc.cjs`: strict one-shot adapter to Orca's packaged `RuntimeClient`.
- Create `tools/orca_hermes_bridge/rpc.py`: Python subprocess boundary for the Node adapter.
- Create `tools/orca_hermes_bridge/state.py`: reconciliation state and pure transition decisions.
- Create `tools/orca_hermes_bridge/windows.py`: singleton file lock, hidden process flags, and de-duplicated Windows balloon notification.
- Create `tools/orca_hermes_bridge/bridge.py`: daemon tick/loop, atomic sidecar, logging, backoff, and CLI modes.
- Create `tests/tools/orca_hermes_bridge/test_accounts.py`: pure mapping/reordering tests.
- Create `tests/tools/orca_hermes_bridge/test_rpc.py`: RPC allowlist, serialization, timeout, and error-redaction tests.
- Create `tests/tools/orca_hermes_bridge/test_state.py`: manual selection, echo suppression, failover, recovery, and Qwen transition tests.
- Create `tests/tools/orca_hermes_bridge/test_bridge.py`: daemon orchestration, atomic state, singleton, and notification tests.
- Modify `C:\Users\Afin\AppData\Local\hermes\bin\hermes-orca-resume.py`: idempotently launch the detached singleton before normal Hermes chat/resume execution.

---

### Task 0: Baseline Snapshot and Recoverable Backups

**Files:**
- Read: `agent/credential_pool.py`
- Read: `C:\Users\Afin\AppData\Local\hermes\bin\hermes-orca-resume.py`
- Create outside Git: one timestamped directory under `C:\Users\Afin\AppData\Local\hermes\backups`

**Interfaces:**
- Produces: one exact backup directory path reused by Task 6 and the final rollback handoff.
- Contract: no auth/config credential file is copied into the repository or printed.

- [ ] **Step 1: Confirm the known dirty-worktree baseline**

Run `git status --short` and assert the only pre-existing tracked modifications are:

```text
agent/chat_completion_helpers.py
tests/run_agent/test_run_agent.py
ui-tui/src/app/useSubmission.ts
```

Stop and re-audit scope if any additional pre-existing modification overlaps a planned file.

- [ ] **Step 2: Create the timestamped recovery directory and copy both files that will be modified**

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backupDir = Join-Path 'C:\Users\Afin\AppData\Local\hermes\backups' "orca-account-bridge-$stamp"
New-Item -ItemType Directory -Path $backupDir
Copy-Item -LiteralPath 'C:\Users\Afin\AppData\Local\hermes\hermes-agent\agent\credential_pool.py' -Destination $backupDir
Copy-Item -LiteralPath 'C:\Users\Afin\AppData\Local\hermes\bin\hermes-orca-resume.py' -Destination $backupDir
$backupDir
```

Record the printed directory path in the implementation notes. These are local recovery copies, not files to stage or commit.

---

### Task 1: Generic Idle-Only Credential-Pool Live Reload

**Files:**
- Modify: `agent/credential_pool.py:5-15,582-668,1593-1601,1926-1955`
- Create: `tests/agent/test_credential_pool_live_reload.py`

**Interfaces:**
- Produces: `_credential_rows_fingerprint(rows: list[dict[str, Any]]) -> str`.
- Produces: `CredentialPool._reload_from_auth_store_if_idle_unlocked() -> bool`.
- Contract: returns `True` only when a changed persisted provider slice was adopted; never reloads with a positive value in `_active_leases`.

- [ ] **Step 1: Add failing tests for adoption before the next selection**

```python
def test_select_adopts_external_priority_change(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    write_credential_pool("openai-codex", [_entry("a", 0), _entry("b", 1)])
    pool = load_pool("openai-codex")

    write_credential_pool("openai-codex", [_entry("b", 0), _entry("a", 1)])

    assert pool.select().id == "b"
    assert [entry.id for entry in pool.entries()] == ["b", "a"]
```

Include `_entry()` in the test file with fake JWT-shaped strings and all normal status fields; no real user credentials may be read.

- [ ] **Step 2: Run the new test and confirm the old in-memory order fails**

Run: `bash scripts/run_tests.sh tests/agent/test_credential_pool_live_reload.py -q`

Expected: FAIL because the existing pool still selects `a` after the external write.

- [ ] **Step 3: Add failing lease and persistence-safety cases**

```python
def test_external_change_waits_until_active_lease_released(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    write_credential_pool("openai-codex", [_entry("a", 0), _entry("b", 1)])
    pool = load_pool("openai-codex")
    assert pool.acquire_lease() == "a"
    write_credential_pool("openai-codex", [_entry("b", 0), _entry("a", 1)])

    assert [entry.id for entry in pool.entries()] == ["a", "b"]
    pool.release_lease("a")
    assert pool.acquire_lease() == "b"


def test_reload_adopts_newer_exhaustion_without_changing_token_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    original = [_entry("a", 0, token="token-a"), _entry("b", 1, token="token-b")]
    write_credential_pool("openai-codex", original)
    pool = load_pool("openai-codex")
    changed = [dict(original[0], last_status="exhausted", last_status_at=time.time(),
                    last_error_code=429, last_error_reset_at=time.time() + 3600), original[1]]
    write_credential_pool("openai-codex", changed)

    assert pool.select().id == "b"
    assert [row["access_token"] for row in read_credential_pool("openai-codex")] == ["token-a", "token-b"]
```

- [ ] **Step 4: Implement canonical fingerprinting and idle reconciliation**

Add `hashlib` and `json` imports. Canonicalize only the provider rows, hash the canonical bytes, and never log the digest or source data.

```python
def _credential_rows_fingerprint(rows: List[Dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

Initialize `self._persisted_fingerprint` from `entry.to_dict()` rows. Implement reconciliation under the existing thread lock and auth-store lock:

```python
def _reload_from_auth_store_if_idle_unlocked(self) -> bool:
    if any(count > 0 for count in self._active_leases.values()):
        return False
    with _auth_store_lock():
        raw_rows = read_credential_pool(self.provider)
    fingerprint = _credential_rows_fingerprint(raw_rows)
    if fingerprint == self._persisted_fingerprint:
        return False
    disk_entries = [PooledCredential.from_dict(self.provider, row) for row in raw_rows]
    self._entries = sorted(disk_entries, key=lambda entry: entry.priority)
    self._current_id = None
    self._persisted_fingerprint = fingerprint
    return True
```

After `_persist()`, set the cached fingerprint from the just-persisted in-memory rows. Call reconciliation at the start of `select()` and before the no-ID branch in `acquire_lease()`. Do not call it from `release_lease()` or during an active request.

- [ ] **Step 5: Run focused pool tests**

Run: `bash scripts/run_tests.sh tests/agent/test_credential_pool_live_reload.py tests/agent/test_credential_pool.py tests/agent/test_credential_pool_routing.py -q`

Expected: all selected files PASS; the new lease test proves the persisted switch is delayed until the next lease.

- [ ] **Step 6: Commit only the live-reload files**

```bash
git add agent/credential_pool.py tests/agent/test_credential_pool_live_reload.py
git commit -m "feat: reload idle credential pools from disk"
```

---

### Task 2: Account Identity Mapping and Locked Hermes Reordering

**Files:**
- Create: `tools/orca_hermes_bridge/__init__.py`
- Create: `tools/orca_hermes_bridge/accounts.py`
- Create: `tests/tools/orca_hermes_bridge/test_accounts.py`

**Interfaces:**
- Produces: `OrcaAccount(account_id: str | None, provider_account_id: str, email: str | None)`.
- Produces: `OrcaSnapshot(active: OrcaAccount, accounts_by_provider_id: dict[str, OrcaAccount])`.
- Produces: `parse_orca_accounts(payload: dict[str, Any]) -> OrcaSnapshot`.
- Produces: `chatgpt_account_id(access_token: str) -> str | None`.
- Produces: `mapped_pool_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]`; raises `DuplicateProviderAccountError` on ambiguity.
- Produces: `first_usable_provider_id(rows: list[dict[str, Any]], now: float) -> str | None`.
- Produces: `reorder_codex_pool(provider_account_id: str, *, clear_selected_status: bool) -> bool`.

- [ ] **Step 1: Write failing snapshot and JWT mapping tests**

```python
def test_parse_snapshot_maps_managed_and_system_accounts():
    snapshot = parse_orca_accounts(_orca_payload(active_id="managed-1"))
    assert snapshot.active.provider_account_id == "provider-managed"
    assert snapshot.accounts_by_provider_id["provider-system"].account_id is None
    assert snapshot.accounts_by_provider_id["provider-managed"].account_id == "managed-1"


def test_duplicate_hermes_provider_identity_fails_closed():
    rows = [_row("a", _jwt("same"), 0), _row("b", _jwt("same"), 1)]
    with pytest.raises(DuplicateProviderAccountError):
        mapped_pool_rows(rows)
```

The `_jwt()` test helper must build unsigned base64url test payloads containing only `{"https://api.openai.com/auth": {"chatgpt_account_id": value}}`.

- [ ] **Step 2: Run mapping tests and verify imports fail**

Run: `bash scripts/run_tests.sh tests/tools/orca_hermes_bridge/test_accounts.py -q`

Expected: FAIL because `tools.orca_hermes_bridge.accounts` does not exist.

- [ ] **Step 3: Implement strict normalization and identity extraction**

Use `hermes_cli.auth._decode_jwt_claims()` and accept only a non-empty string at the OpenAI auth namespace. Normalize `accounts.list` responses that are either the RPC `result` or the CLI envelope containing `result`.

```python
@dataclass(frozen=True)
class OrcaAccount:
    account_id: str | None
    provider_account_id: str
    email: str | None


@dataclass(frozen=True)
class OrcaSnapshot:
    active: OrcaAccount
    accounts_by_provider_id: dict[str, OrcaAccount]
```

Reject missing active-account metadata with `InvalidOrcaSnapshotError`; do not guess from email addresses.

- [ ] **Step 4: Add failing reordering, cooldown, and token-invariance tests**

```python
def test_reorder_moves_selected_first_and_clears_only_its_stale_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rows = [
        _row("a", _jwt("provider-a"), 0, status="exhausted", token_marker="A"),
        _row("b", _jwt("provider-b"), 1, status="exhausted", token_marker="B"),
    ]
    _write_store(tmp_path, rows)
    before = _token_tuple(rows)

    assert reorder_codex_pool("provider-b", clear_selected_status=True)
    after = read_credential_pool("openai-codex")

    assert [(row["id"], row["priority"]) for row in after] == [("b", 0), ("a", 1)]
    assert after[0]["last_status"] == "ok"
    assert after[1]["last_status"] == "exhausted"
    assert _token_tuple(after) == before[::-1]
```

Also test that `STATUS_DEAD` and an unexpired `STATUS_EXHAUSTED` row are unavailable, an expired cooldown is usable, no match is a no-op, and duplicate provider IDs raise without writing.

- [ ] **Step 5: Implement stable ordering and one-row status clearing**

Inside one `_auth_store_lock()` transaction, load the `openai-codex` list, resolve exactly one matching row, sort the selected row first and the remaining rows by `(priority, original_index)`, renumber priorities from zero, and clear these fields only on a manual probe:

```python
STATUS_FIELDS = (
    "last_status", "last_status_at", "last_error_code",
    "last_error_reason", "last_error_message", "last_error_reset_at",
)
```

Set `last_status` to `STATUS_OK` and the other status fields to `None`; preserve `access_token`, `refresh_token`, `source`, `auth_type`, and IDs exactly. Persist with `_save_auth_store(store)` while still holding the lock.

- [ ] **Step 6: Run and commit the account-mapping unit**

Run: `bash scripts/run_tests.sh tests/tools/orca_hermes_bridge/test_accounts.py -q`

Expected: PASS.

```bash
git add tools/orca_hermes_bridge/__init__.py tools/orca_hermes_bridge/accounts.py tests/tools/orca_hermes_bridge/test_accounts.py
git commit -m "feat: map Orca accounts to Hermes credentials"
```

---

### Task 3: Authenticated Orca Runtime RPC Boundary

**Files:**
- Create: `tools/orca_hermes_bridge/runtime_rpc.cjs`
- Create: `tools/orca_hermes_bridge/rpc.py`
- Create: `tests/tools/orca_hermes_bridge/test_rpc.py`

**Interfaces:**
- Produces: `OrcaRpcClient(node_executable: Path, resources_path: Path, helper_path: Path, timeout_seconds: float = 10.0)`.
- Produces: `OrcaRpcClient.list_accounts() -> dict[str, Any]`.
- Produces: `OrcaRpcClient.select_host_codex(account_id: str | None) -> dict[str, Any]`.
- Node stdin contract: `{"resourcesPath": string, "method": string, "params": object}`.
- Node stdout contract: one JSON object, `{"ok": true, "response": RpcEnvelope}` or `{"ok": false, "error": {"code": string, "message": string}}`.

- [ ] **Step 1: Write failing Python boundary tests**

```python
def test_select_uses_explicit_host_target(monkeypatch, client):
    completed = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({"ok": True, "response": {"result": {"codex": {}}}}),
        stderr="",
    )
    run = Mock(return_value=completed)
    monkeypatch.setattr(subprocess, "run", run)
    client.select_host_codex("managed-1")
    request = json.loads(run.call_args.kwargs["input"])
    assert request == {
        "resourcesPath": str(client.resources_path),
        "method": "accounts.selectCodexForTarget",
        "params": {"accountId": "managed-1", "target": {"runtime": "host", "wslDistro": None}},
    }


def test_rpc_error_never_includes_stdin_or_tokens(monkeypatch, client):
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("node", 10)))
    with pytest.raises(OrcaRpcError) as exc:
        client.list_accounts()
    assert "access_token" not in str(exc.value)
    assert "stdin" not in str(exc.value).lower()
```

- [ ] **Step 2: Run tests and verify the missing boundary fails**

Run: `bash scripts/run_tests.sh tests/tools/orca_hermes_bridge/test_rpc.py -q`

Expected: FAIL because `OrcaRpcClient` and the helper are absent.

- [ ] **Step 3: Implement the strict CommonJS helper**

`runtime_rpc.cjs` must parse exactly one stdin document, allow only `accounts.list` and `accounts.selectCodexForTarget`, resolve:

```javascript
const runtimeClientPath = path.join(
  input.resourcesPath,
  'app.asar.unpacked', 'out', 'cli', 'runtime-client.js'
)
const { RuntimeClient } = require(runtimeClientPath)
const client = new RuntimeClient(undefined, 10_000, null, null)
const response = await client.call(input.method, input.params)
```

For `accounts.list`, require `params` to equal `{refreshUsage: false}`. For selection, require keys `accountId` and `target`, with target exactly `{runtime: 'host', wslDistro: null}`. Write only the sanitized success/error envelope; never echo stdin or stack traces.

- [ ] **Step 4: Implement the Python client with bounded subprocess behavior**

Resolve defaults to `C:\Program Files\nodejs\node.exe` via `shutil.which("node")` and `%LOCALAPPDATA%\Programs\Orca\resources`. Invoke with `capture_output=True`, `text=True`, `encoding="utf-8"`, `timeout=10`, `check=False`, and `creationflags=hidden_process_flags()`. Reject non-zero exit, malformed JSON, `ok: false`, and missing `response.result` as `OrcaRpcError(code, safe_message)`.

- [ ] **Step 5: Exercise the real helper against a fake packaged RuntimeClient**

Create a temporary `app.asar.unpacked/out/cli/runtime-client.js` in the test. Its `RuntimeClient.call()` records the method/params and returns a synthetic RPC envelope. Run the real `.cjs` with Node when `shutil.which("node")` is available; assert the allowed request succeeds and `accounts.removeCodex` is rejected before loading the client.

- [ ] **Step 6: Run and commit the RPC boundary**

Run: `bash scripts/run_tests.sh tests/tools/orca_hermes_bridge/test_rpc.py -q`

Expected: PASS, including the real Node subprocess test on this Windows host.

```bash
git add tools/orca_hermes_bridge/runtime_rpc.cjs tools/orca_hermes_bridge/rpc.py tests/tools/orca_hermes_bridge/test_rpc.py
git commit -m "feat: add Orca account runtime RPC client"
```

- [ ] **Step 7: Run the required read-only live Orca compatibility probe**

With Orca running, invoke `OrcaRpcClient.list_accounts()` once and assert the returned snapshot contains `codex.activeAccountIdsByRuntime.host`, `codex.systemDefault.providerAccountId`, and the managed account list. Do not invoke either selection RPC in this step. If the installed runtime rejects the read or omits provider IDs, stop before Task 4 and keep write-back disabled.

---

### Task 4: Pure Two-Way Reconciliation State Machine

**Files:**
- Create: `tools/orca_hermes_bridge/state.py`
- Create: `tests/tools/orca_hermes_bridge/test_state.py`

**Interfaces:**
- Consumes: `OrcaSnapshot`, `mapped_pool_rows()`, and `first_usable_provider_id()` from Task 2.
- Produces: `BridgeState(version: int, last_seen_orca_provider_id: str | None, pending_orca_provider_id: str | None, pending_started_at: float | None, qwen_notified: bool)`.
- Produces: `PoolMutation(provider_account_id: str, clear_selected_status: bool)`.
- Produces: `OrcaMutation(account_id: str | None, provider_account_id: str)`.
- Produces: `ReconcileDecision(state: BridgeState, pool_mutation: PoolMutation | None, orca_mutation: OrcaMutation | None, notify_qwen: bool)`.
- Produces: `reconcile(snapshot: OrcaSnapshot, rows: list[dict[str, Any]], state: BridgeState, now: float) -> ReconcileDecision`.

- [ ] **Step 1: Add failing manual-selection and startup tests**

```python
def test_startup_applies_current_orca_selection_as_manual_probe():
    decision = reconcile(_snapshot("provider-b"), _rows(a_ok=True, b_ok=True), BridgeState(), 100.0)
    assert decision.pool_mutation == PoolMutation("provider-b", clear_selected_status=True)
    assert decision.orca_mutation is None


def test_manual_orca_change_reorders_and_clears_only_selected_status():
    state = BridgeState(last_seen_orca_provider_id="provider-a")
    decision = reconcile(_snapshot("provider-b"), _rows(a_ok=True, b_ok=False), state, 100.0)
    assert decision.pool_mutation == PoolMutation("provider-b", clear_selected_status=True)
```

- [ ] **Step 2: Add failing failover, echo, Qwen, and recovery tests**

```python
def test_exhausted_displayed_account_selects_next_usable_account_in_orca():
    state = BridgeState(last_seen_orca_provider_id="provider-a")
    decision = reconcile(_snapshot("provider-a"), _rows(a_ok=False, b_ok=True), state, 100.0)
    assert decision.orca_mutation == OrcaMutation("managed-b", "provider-b")
    assert decision.state.pending_orca_provider_id == "provider-b"


def test_rpc_echo_does_not_clear_exhaustion_as_manual_probe():
    state = BridgeState(last_seen_orca_provider_id="provider-a",
                        pending_orca_provider_id="provider-b", pending_started_at=99.0)
    decision = reconcile(_snapshot("provider-b"), _rows(a_ok=False, b_ok=True), state, 100.0)
    assert decision.pool_mutation == PoolMutation("provider-b", clear_selected_status=False)
    assert decision.state.pending_orca_provider_id is None


def test_all_codex_unavailable_notifies_once_and_keeps_orca_selection():
    first = reconcile(_snapshot("provider-a"), _rows(a_ok=False, b_ok=False), BridgeState(), 100.0)
    second = reconcile(_snapshot("provider-a"), _rows(a_ok=False, b_ok=False), first.state, 101.0)
    assert first.notify_qwen is True and first.orca_mutation is None
    assert second.notify_qwen is False
```

Also cover system default through `OrcaMutation(account_id=None, provider_account_id="provider-system")`, a missing Hermes match, pending-echo expiry after 30 seconds, duplicate identity failure, and reset of `qwen_notified` when a Codex entry becomes effective again.

- [ ] **Step 3: Run the state tests and confirm they fail**

Run: `bash scripts/run_tests.sh tests/tools/orca_hermes_bridge/test_state.py -q`

Expected: FAIL because `state.py` is missing.

- [ ] **Step 4: Implement one-action-per-direction reconciliation**

Use immutable dataclasses. The priority order is:

1. acknowledge a non-expired bridge-originated Orca echo;
2. treat any other changed Orca provider ID as a manual selection;
3. if the displayed account is unavailable, choose the first usable mapped Codex row and plan an Orca RPC;
4. if none is usable, plan one Qwen notification without an Orca mutation;
5. otherwise keep state stable.

Record `pending_orca_provider_id` and `pending_started_at` in the returned state before the caller performs an Orca mutation. Never use email as an identity fallback.

- [ ] **Step 5: Run and commit the pure state machine**

Run: `bash scripts/run_tests.sh tests/tools/orca_hermes_bridge/test_state.py tests/tools/orca_hermes_bridge/test_accounts.py -q`

Expected: PASS.

```bash
git add tools/orca_hermes_bridge/state.py tests/tools/orca_hermes_bridge/test_state.py
git commit -m "feat: reconcile Orca and Hermes account state"
```

---

### Task 5: Singleton Daemon, Atomic State, Logging, and Qwen Notification

**Files:**
- Create: `tools/orca_hermes_bridge/windows.py`
- Create: `tools/orca_hermes_bridge/bridge.py`
- Create: `tests/tools/orca_hermes_bridge/test_bridge.py`

**Interfaces:**
- Consumes: `OrcaRpcClient`, `reorder_codex_pool()`, `reconcile()`, and all Task 4 decision types.
- Produces: `hidden_process_flags() -> int`.
- Produces: `SingletonLock(path: Path)` context manager; raises `AlreadyRunningError` when another process holds the byte-range lock.
- Produces: `show_qwen_notification() -> None`.
- Produces: `load_state(path: Path) -> BridgeState` and `save_state(path: Path, state: BridgeState) -> None`.
- Produces: `Bridge.tick() -> ReconcileDecision`.
- Produces CLI: `python -m tools.orca_hermes_bridge.bridge --daemon|--once|--status`.

- [ ] **Step 1: Write failing atomic-state and orchestration tests**

```python
def test_tick_persists_pending_before_orca_rpc(tmp_path):
    events = []
    bridge = _bridge(tmp_path, rpc=_FakeRpc(on_select=lambda: events.append(_read_state(tmp_path))))
    bridge.tick()
    assert events[0].pending_orca_provider_id == "provider-b"


def test_tick_applies_manual_selection_under_locked_pool_mutator(tmp_path):
    mutate = Mock(return_value=True)
    bridge = _bridge(tmp_path, snapshot=_snapshot("provider-b"), mutate_pool=mutate)
    bridge.tick()
    mutate.assert_called_once_with("provider-b", clear_selected_status=True)


def test_malformed_sidecar_keeps_last_known_good_in_memory(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.state = BridgeState(last_seen_orca_provider_id="provider-a")
    bridge.state_path.write_text("{", encoding="utf-8")
    bridge.tick()
    assert bridge.state.last_seen_orca_provider_id == "provider-a"
```

- [ ] **Step 2: Add failing singleton, notification, and backoff tests**

Test that a second `SingletonLock` on the same path raises, notification is called only on `notify_qwen=True`, an RPC failure does not mutate the Hermes pool, and the daemon backoff sequence is `2, 4, 8, 16, 30` seconds before resetting to `2` after a successful tick.

- [ ] **Step 3: Run the daemon tests and confirm they fail**

Run: `bash scripts/run_tests.sh tests/tools/orca_hermes_bridge/test_bridge.py -q`

Expected: FAIL because `Bridge`, `SingletonLock`, and notification helpers are absent.

- [ ] **Step 4: Implement Windows-safe lifecycle helpers**

Use `msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)` while retaining the open handle for the process lifetime. `hidden_process_flags()` returns `CREATE_NO_WINDOW | DETACHED_PROCESS` when available and zero elsewhere.

For the notification, run hidden Windows PowerShell with a fixed, token-free script using `System.Windows.Forms.NotifyIcon`:

```powershell
Add-Type -AssemblyName System.Windows.Forms
$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Information
$n.BalloonTipTitle = 'Hermes account fallback'
$n.BalloonTipText = 'All Codex accounts are unavailable. Hermes is using OpenRouter/Qwen.'
$n.Visible = $true
$n.ShowBalloonTip(5000)
Start-Sleep -Seconds 6
$n.Dispose()
```

Notification failure is logged once and never fails a tick.

- [ ] **Step 5: Implement atomic sidecar and daemon tick ordering**

Store `orca-account-bridge-state.json`, `orca-account-bridge.lock`, and `logs/orca-account-bridge.log` under `get_hermes_home()`. Write state to a same-directory temporary file, flush and `os.fsync()`, then `os.replace()`.

`Bridge.tick()` must execute in this order:

1. call `accounts.list` with `refreshUsage: false`;
2. read only the `openai-codex` provider slice;
3. compute a pure `ReconcileDecision`;
4. for manual selection, reorder Hermes then persist decision state;
5. for failover, persist pending state, call Orca selection, then reorder without clearing status;
6. show the Qwen notification only after its de-duplication state is persisted.

Log only bridge event names, credential row IDs/labels, provider-account IDs, and safe error codes. Never log subprocess stdin/stdout wholesale.

- [ ] **Step 6: Implement bounded daemon CLI modes**

`--once` performs one tick without acquiring the long-lived lock and returns non-zero on failure. `--daemon` acquires the singleton, polls every two seconds, and applies capped exponential backoff on transient Orca/auth-store failures. `--status` reports only `running`, `last_seen_orca_provider_id`, `pending`, `qwen_notified`, and last safe error code.

- [ ] **Step 7: Run and commit the daemon unit**

Run: `bash scripts/run_tests.sh tests/tools/orca_hermes_bridge/test_bridge.py tests/tools/orca_hermes_bridge/test_state.py tests/tools/orca_hermes_bridge/test_rpc.py -q`

Expected: PASS.

```bash
git add tools/orca_hermes_bridge/windows.py tools/orca_hermes_bridge/bridge.py tests/tools/orca_hermes_bridge/test_bridge.py
git commit -m "feat: run Orca Hermes account bridge daemon"
```

---

### Task 6: Orca Wrapper Integration and Local Installation

**Files:**
- Modify: `C:\Users\Afin\AppData\Local\hermes\bin\hermes-orca-resume.py:20-30,142-176`
- Backup: runtime-generated directory `C:\Users\Afin\AppData\Local\hermes\backups\orca-account-bridge-yyyyMMdd-HHmmss\hermes-orca-resume.py`
- Test: `tests/tools/orca_hermes_bridge/test_bridge.py`

**Interfaces:**
- Consumes CLI: `python -m tools.orca_hermes_bridge.bridge --daemon` from Task 5.
- Produces wrapper helper: `ensure_account_bridge() -> None`.
- Contract: returns immediately, starts at most one effective daemon, never blocks Hermes startup, and has no side effect during `HERMES_ORCA_RESUME_DRY_RUN=1`.

- [ ] **Step 1: Add a failing launch-command test**

Add this to `test_bridge.py` for `build_daemon_launch()` in `bridge.py`:

```python
def test_build_daemon_launch_uses_current_python_and_repo_cwd(tmp_path):
    spec = build_daemon_launch(tmp_path, Path("C:/Python/python.exe"))
    assert spec.argv == [
        "C:/Python/python.exe", "-m", "tools.orca_hermes_bridge.bridge", "--daemon"
    ]
    assert spec.cwd == tmp_path
```

- [ ] **Step 2: Implement and verify the launch specification**

Run: `bash scripts/run_tests.sh tests/tools/orca_hermes_bridge/test_bridge.py -q`

Expected after implementation: PASS.

- [ ] **Step 3: Verify the Task 0 wrapper backup before editing**

Resolve the backup directory recorded by Task 0 and verify it contains both `credential_pool.py` and `hermes-orca-resume.py`. Do not create a second backup set unless the source files changed outside this implementation after Task 0.

```powershell
Get-ChildItem -LiteralPath $backupDir | Select-Object Name,Length
```

- [ ] **Step 4: Patch the wrapper to launch the daemon idempotently**

Add:

```python
REPO_ROOT = REAL_HERMES.parents[2]


def ensure_account_bridge() -> None:
    command = [
        sys.executable, "-m", "tools.orca_hermes_bridge.bridge", "--daemon",
    ]
    try:
        subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=build_child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=hidden_process_flags(),
            close_fds=True,
        )
    except OSError as exc:
        print(f"Could not start Orca account bridge: {exc}", file=sys.stderr)
```

Import `hidden_process_flags` after inserting `REPO_ROOT` into `sys.path`. Call `ensure_account_bridge()` after the dry-run early return and executable validation, immediately before `subprocess.call()` starts Hermes. Multiple wrappers may race; the singleton lock is the authority and losing daemon processes exit immediately.

- [ ] **Step 5: Validate wrapper behavior without launching a new Hermes TUI**

Run the existing wrapper with `HERMES_ORCA_RESUME_DRY_RUN=1` and verify no bridge process count changes. Then invoke `ensure_account_bridge()` once from the Hermes venv, wait up to five seconds for `--status` to report `running: true`, and verify exactly one command line contains `tools.orca_hermes_bridge.bridge --daemon`.

- [ ] **Step 6: Commit only tracked Task 6 changes**

If Task 6 changes only the external wrapper, do not make an empty repository commit. If `bridge.py` and `test_bridge.py` gained `build_daemon_launch()`, commit exactly those tracked files:

```bash
git add tools/orca_hermes_bridge/bridge.py tests/tools/orca_hermes_bridge/test_bridge.py
git commit -m "feat: launch account bridge from Orca Hermes wrapper"
```

---

### Task 7: Security Regression, Full Verification, and Controlled Live Switch

**Files:**
- Modify only if tests expose a defect: files already introduced by Tasks 1-6.
- Verify: `C:\Users\Afin\AppData\Local\hermes\config.yaml`
- Verify without printing secrets: `C:\Users\Afin\AppData\Local\hermes\auth.json`

**Interfaces:**
- Consumes all prior tasks.
- Produces: an installed singleton bridge, preserved token hashes, two-way live account selection, and a rollback record.

- [ ] **Step 1: Run the complete focused regression set**

```bash
bash scripts/run_tests.sh \
  tests/agent/test_credential_pool_live_reload.py \
  tests/agent/test_credential_pool.py \
  tests/agent/test_credential_pool_routing.py \
  tests/tools/orca_hermes_bridge/ -q
```

Expected: all files PASS with zero failed test processes.

- [ ] **Step 2: Run repository static and import checks**

```bash
venv/Scripts/python.exe -m compileall -q agent/credential_pool.py tools/orca_hermes_bridge
git diff --check
rg -n "print\(.*token|logger\..*token|stdout.*auth|access_token.*log|refresh_token.*log" tools/orca_hermes_bridge agent/credential_pool.py
```

Expected: compilation and `git diff --check` succeed; the redaction scan finds no bridge logging/printing of secret values.

- [ ] **Step 3: Verify unchanged routing configuration**

Read only the relevant YAML keys and assert:

```text
model.provider = openai-codex
credential_pool_strategies.openai-codex = fill_first
fallback_providers includes openrouter / qwen/qwen3-coder-next
reasoning_effort = xhigh for the Qwen fallback
```

Do not rewrite `config.yaml` when these values already match.

- [ ] **Step 4: Capture secret-field hashes without displaying secret bytes**

Use a local verification snippet that loads `auth.json`, computes SHA-256 for each tuple `(entry.id, access_token, refresh_token)`, and writes only entry ID plus digest to an in-memory comparison map. Do not redirect the map or token contents to a persistent file. Repeat after every live selection and assert equality.

- [ ] **Step 5: Probe the installed Orca RPC read path**

Run `python -m tools.orca_hermes_bridge.bridge --once`. Expected: success against runtime ID reported by `orca status --json`, no token output, and the currently selected Orca provider account becomes priority zero in the persisted Hermes pool.

- [ ] **Step 6: Perform a controlled switch to system default and back**

Record the current managed Orca account ID from `accounts.list`. Through `OrcaRpcClient.select_host_codex(None)`, select system default and wait until both `accounts.list` and the Hermes pool show provider account `9eb5304a-aaa4-4c49-99a2-6529823c728a` first. Then select managed account `6e1f9037-a07f-415c-bfa5-2415610de5cf` and wait until both sides show provider account `e6cc89de-0381-4dd9-9cc0-8be8c8b9d934` first.

At both points, rerun the in-memory secret hash comparison from Step 4. Restore the account that was active before this test.

- [ ] **Step 7: Verify reverse failover and Qwen behavior without damaging live credentials**

Use temporary `HERMES_HOME` fixtures and the real local Orca fake-runtime adapter tests to drive: displayed account exhausted → other Codex selected; both exhausted → one Qwen notification; repeated poll → no second notification; usable Codex restored → notification flag re-armed without automatic jump. Do not write exhaustion markers into the user's real `auth.json`.

- [ ] **Step 8: Verify daemon and existing agents observe the next-request switch**

Confirm `--status` reports `running: true`. Start two temporary Hermes processes that each load the pool, change Orca selection once, and assert both processes select the new priority-zero credential on their next lease while an intentionally held lease retains its original credential until released.

- [ ] **Step 9: Review repository scope and commit any test-discovered fixes**

Run `git status --short` and confirm the pre-existing unrelated files remain modified but unstaged. Stage only Task 1-6 bridge files if a final corrective patch was required.

If a corrective patch touched bridge production or test files, stage only the explicit bridge paths below, inspect the staged diff, and then commit. Git silently ignores listed files that are unchanged.

```bash
git add agent/credential_pool.py tests/agent/test_credential_pool_live_reload.py tools/orca_hermes_bridge tests/tools/orca_hermes_bridge
git diff --cached --check
git commit -m "test: verify Orca Hermes account bridge"
```

Skip these three commands when no tracked corrective changes exist; never use `git add -A`.

- [ ] **Step 10: Record rollback instructions in the final handoff**

Report the exact wrapper backup path. Rollback is: terminate the singleton bridge PID, restore the backed-up `hermes-orca-resume.py`, revert only the bridge commits if desired, and remove only `orca-account-bridge-state.json` plus `orca-account-bridge.lock`. Do not delete or restore `auth.json`; selection transitions must have left token fields unchanged.

---

## Completion Gate

- [ ] Orca manual selection changes Hermes priority zero for all idle agents before their next request.
- [ ] A held Hermes credential lease is never replaced mid-request.
- [ ] Hermes 401/402/429 failover selects the corresponding account in Orca and updates its status-bar usage source.
- [ ] System default is represented by Orca `accountId: null`, not by an invented managed ID.
- [ ] All-Codex exhaustion leaves the last Codex account visible and emits exactly one Qwen notification.
- [ ] Recovery never causes an automatic jump to an earlier account.
- [ ] Secret-field hashes are identical before and after selection-only transitions.
- [ ] Qwen remains `qwen/qwen3-coder-next` with reasoning `xhigh`.
- [ ] Focused tests, compilation, `git diff --check`, wrapper dry-run, singleton status, and controlled live switching all pass.
- [ ] The three pre-existing unrelated working-tree modifications remain untouched and unstaged.
