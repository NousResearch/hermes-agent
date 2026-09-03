# BWM-796 — Persistent Computers + Human Takeover

**Branch:** `grok/bwm-796-persistent-computers`  
**Worktree:** `/Users/abdulrahman/hermes-worktrees/bwm-796-persistent-computers`  
**BASE_SHA / PRE_IMPLEMENTATION_SHA:** `180291162ff4df0d42b5dc4fecd08005cf7cebf9`  
**CANDIDATE_SHA:** `3ef1caba67bf6b7f1d546a23289ed1718a0be00f`  
**Status:** implementation + tests complete. **Not merged. Not deployed. No real-account canary.**

---

## EXISTING HERMES BROWSER CAPABILITY REUSE

### Exact source files (upstream, unchanged)

| Path | Role |
|---|---|
| `hermes_cli/browser_connect.py` | Default-browser detect, snapshot, pin, lock probe, copy-dir |
| `tools/browser_tool.py` | `_use_real_profile`, `_real_profile_cdp`, launch, session, screenshots |
| `tools/browser_cdp_tool.py` | Raw CDP including `Page.captureScreenshot` |
| `gateway/browser_control_broker.py` | Exclusive controller tickets (inverted: client is the driver) |
| `hermes_cli/config_defaults.py` | `use_real_profile`, `real_profile_pin`, `headed` |
| `tests/tools/test_browser_real_profile.py` | Snapshot/launch/consent |
| `tests/tools/test_browser_real_profile_pin.py` | Pin vs last_used |

### Exact current upstream behavior

Consent (`browser.use_real_profile`, default false) snapshots the OS default
Chromium-family **selected** profile into `{get_hermes_home()}/browser-profile/<browser>/`,
launches the real binary on that copy with `--remote-debugging-port=0`, then
attaches `agent-browser` session `hermes-real-profile`.

### Supported browsers

Stable Chrome, Edge, Brave, Brave Origin, Chromium. Beta/Dev/Canary, Firefox,
Safari fail closed.

### Profile selection

1. OS default browser.
2. `browser.real_profile_pin` if set, else `Local State → profile.last_used`, else `Default`.
3. Missing pin fails closed.

### Snapshot mechanism (Q1)

Selected active-profile state, **not** the entire user-data-dir: `copytree`
minus caches/extensions/IndexedDB/History/Singleton/WAL, then lock-aware
Cookies / Login Data / Web Data / Preferences. `Local State` rewritten to
`Default`.

### Persistence mechanism (Q2)

The copy persists on disk but **re-syncs from the live source on every
launch**. Consent-off deletes the store. This is **not** an independent
BrowserIdentity.

### Managed-profile storage

`{get_hermes_home()}/browser-profile/<browser>/` — one snapshot per browser
family per HERMES_HOME.

### CDP / control path

Real binary → `DevToolsActivePort` → HTTP CDP `127.0.0.1:<port>` →
existing `browser_*` verbs.

### Visible / live-view path (Q6)

Same-host `browser.headed`. On-demand `browser_screenshot` /
`Page.captureScreenshot`. **No** remote spectator WS/SSE/VNC. Human
Takeover is not provided by real-profile import.

### Machine portability (Q3)

Mac snapshots are keychain-bound and cannot become a logged-in VPS profile.
VPS can run **new** headless Chromium profiles locally.

### Multi-agent isolation (Q5)

HERMES_HOME can isolate the snapshot path. Session name
`hermes-real-profile` is still one shared copy-browser. No exclusive
inter-agent identity lock.

### Current locking behavior

No exclusive mount lock between agents. Source Windows deny-all fails
closed; macOS/Linux may copy while running.

### Security limitations

Consent is convenience, not isolation. last_used without pin is
wrong-principal risk. Snapshot holds live cookies. CDP is loopback and
must stay that way.

### Known defects discovered (upstream, not “fixed” by pretending)

1. last_used is incompatible with explicit BrowserIdentity.
2. Re-sync would overwrite an agent-owned identity.
3. Consent-off wipe is the wrong lifecycle for durable identities.
4. No ControlLease / owner-vs-agent fencing.
5. No remote same-environment takeover contract.
6. 120s browser inactivity reaper vs long-lived computers.
7. VPS cannot display headed takeover.

### What BWM-796 reused unchanged

- Chromium launch flag shape (loopback debug port, `--user-data-dir`,
  `--headless=new`, no mock-keychain)
- `browser_*` observation/input verbs as the existing control vocabulary
- Screenshot / headed as the live-view primitive (not a new stream)
- `real_profile_pin` fail-closed **pattern**
- Never open the live default user-data-dir
- file_safety / backup exclusion pattern for cookie stores

### What BWM-796 hardened

- Explicit BrowserIdentity → exact managed dir (never last_used)
- Exclusive identity lock (`BROWSER_IDENTITY_BUSY`, never silent clone)
- AgentComputer bound 1:1 to a permanent profile (not session/run)
- ControlLease + `fencing_epoch`
- Persistence **without** source re-sync
- Owner/agent authorization, takeover tokens, audit
- file_safety + backup exclusion of `agent-computers/`

### What BWM-796 still had to implement

Domain + SQLite store + contract + thin WS/REST/tool adapters + takeover /
give-back / stale rejection + checkpoint hook + recovery + synthetic tests.

### Anything requiring real canary

`[NEEDS_REAL_CANARY]` cookie decrypt on a real Chromium host with
`HERMES_AGENT_COMPUTER_RUNTIME=chromium`.  
`[OWNER AUTHORIZATION REQUIRED]` any import of a credential-bearing profile.

### DID BWM-796 BUILD A NEW BROWSER RUNTIME?

**NO** — reused the existing Hermes Chromium/CDP launch shape and added
only the missing durable ownership, identity isolation, control authority,
takeover, recovery, and audit contracts. Default tests use
`InMemoryRuntime` as a same-environment stand-in. Chromium launch is
opt-in (`HERMES_AGENT_COMPUTER_RUNTIME=chromium`) and does not call
`snapshot_real_profile`.

Q1–Q6 answers are in `docs/bwm-796/COMPUTER_RECON.md`.

---

## Implementation map

| Piece | Path |
|---|---|
| Domain | `gateway/agent_computer/models.py` |
| Errors | `gateway/agent_computer/errors.py` |
| Store | `gateway/agent_computer/store.py` |
| Runtime adapter | `gateway/agent_computer/adapter.py` |
| Control plane | `gateway/agent_computer/service.py` |
| Public contract | `gateway/agent_computer/contract.py` |
| WS | `tui_gateway/methods_agent_computer.py` |
| REST | `hermes_cli/web_routers/agent_computer.py` |
| Agent tools | `tools/agent_computer_tool.py` (opt-in toolset) |
| Tests | `tests/gateway/test_agent_computer.py` |

AgentComputer and BrowserIdentity remain separate objects.

## Independent review repair

First independent review: **REJECT**. Fixed before re-review:

- WS teardown now calls `release_owner_for_transport_if_active` (same path as `browser_control_broker.disconnect_owner`). `wake()` expires a stale owner lease; a live owner lease is not handed to the agent.
- Agent principal is the session profile only. Client/model `profile_id` cannot select another permanent agent.
- `HermesChromiumRuntime.observe` / `act` send real loopback CDP via `tools.browser_cdp_tool._cdp_call`. Non-loopback handles are refused.
- Approved checkpoints are single-use (`consumed`). Identity lock is an atomic SQL update.

## Acceptance evidence (synthetic)

`25 passed` in `tests/gateway/test_agent_computer.py`:

- Persistence across store reopen / sleep+wake
- Same live environment for agent → owner → agent
- Exclusive identity lock; 14-agent isolation
- Stale controller rejected
- Transport TTL + owner-disconnect recovery
- Unauthorized takeover rejected
- Public contract strips CDP/cookies/paths
- Default toolsets do not include computer tools
- file_safety + backup exclude `agent-computers/`

Related regression: `89 passed`
(`test_browser_real_profile_pin`, `test_browser_control_broker`,
`test_file_safety_credentials`, `test_backup`).

## Owner gates (not done)

- Merge to main
- Production deploy
- Live infra change
- Public VNC/RDP/CDP
- Sensitive real-account canary
- Import of the owner’s daily browser

## Verdict

Prior independent re-review: **PASS_WITH_FINDINGS** (no blockers at that
time). The remaining evidence gap was real Chromium/CDP — closed below.
Config selection is now `config.yaml` `agent_computer.runtime` with
`HERMES_AGENT_COMPUTER_RUNTIME` as test/operator override only.

Implementation is isolated on the feature branch. No production action
has been taken. Merge, deploy, and real-account canary remain owner gates.

---

## Addendum — REAL_CHROMIUM_E2E (pre-merge runtime proof)

Host: local Mac. Disposable synthetic profile under pytest `tmp_path`.
No owner browser snapshot. No real credentials. No deploy. No merge.

### REAL_CHROMIUM_E2E

- **runtime:** `HermesChromiumRuntime` (actual Chromium binary + loopback CDP)
- **disposable profile:** `{tmp_path}/identities/<identity_id>/` with `.hermes-identity` marker; never the OS default user-data-dir
- **actual Chromium launched:** yes — process alive, `--user-data-dir` resolved to the identity dir, `--headless=new`, `about:blank`
- **CDP loopback:** `http://127.0.0.1:<port>` from `DevToolsActivePort`; `0.0.0.0` absent; non-loopback refused
- **agent action:** public contract `navigate` → local synthetic page, `type` `#box` `synthetic-796`, `click` `#go` → `clicked-synthetic-796`
- **owner takeover:** `request_takeover` + `connect_takeover` (Service/Contract, not raw CDP)
- **same environment proof:** same `computer.id`, same `BrowserIdentity`, same Chromium `process_id`, same CDP URL; owner observe still showed `clicked-synthetic-796` plus JPEG screenshot
- **stale agent rejected:** old agent lease `type` `#box` `STALE-SHOULD-NOT-APPEAR` → `STALE_CONTROLLER`; page text unchanged
- **owner action:** `click` `#owner` on the same live page → `owner-took-over`
- **give-back:** `OWNER_CONTROLLED` → `AGENT_CONTROLLED`; fencing epoch changed; stale owner click rejected
- **agent re-observe:** required path; agent saw `owner-took-over` and not `clicked-synthetic-796` (no pre-takeover replay)
- **duplicate action:** one continuation after a single re-observe
- **identity contention:** second authorized computer (`abu-saleh`) attach → `BROWSER_IDENTITY_BUSY`; still one runtime handle; profile dir not cloned
- **transport disconnect:** `release_owner_for_transport_if_active` (same helper `tui_gateway/ws.py` teardown calls) → authority `AGENT_CONTROLLED`, `resume_observe_required`, stale owner act rejected; audit `owner_disconnect` + `fencing_recovery`
- **recovery:** `sleep` stops Chromium (CDP down, pid reaped). Reopen of the same store restores the same computer id + identity id + profile marker. New process id + new loopback CDP. DOM/`owner-took-over` is **not** preserved (ephemeral process/CDP/DOM vs durable identity/profile)

REST cannot hold a persistent controller connection. Its disconnect path is TTL + `POST /owner-disconnect`. WS is the authoritative interactive takeover transport.

### REMOTE_TAKEOVER_CONTRACT

- **remotely consumable:** YES — public contract is sufficient for a remote Mobile/Mac client without VNC/WebRTC
- **observation mechanism:** `computer.observe` → url/title/text + `{screenshot: {mime: image/jpeg, data}}` (Page.captureScreenshot). `live_view.kind = screenshot_on_demand`, `remote_stream = false`
- **input mechanism:** `computer.act` (`navigate` / `type` / `click` by selector) under the owner lease — WS JSON-RPC and owner REST both wrap the same contract
- **authority mechanism:** ControlLease `lease_id` + `fencing_epoch`; `computer.takeover` / `connect` / `give_back`
- **backend identifiers exposed:** computer id, identity id, lease id, epoch, workspace url/title, screenshot bytes. CDP URL, cookies, user-data-dir, pid, binary path are stripped
- **remaining limitation:** on-demand JPEG (not a live video stream); input is DOM-targeted, not raw pointer coordinates; no production client UI (not required). REST is stateless per request

### CONFIG

`agent_computer.runtime` in `config.yaml` (default `memory`) is the durable user/server setting. `HERMES_AGENT_COMPUTER_RUNTIME` is test/operator override only.

### TESTS

- **synthetic:** `tests/gateway/test_agent_computer.py` — 26 passed
- **regression:** `test_browser_real_profile_pin`, `test_browser_control_broker`, `test_file_safety_credentials`, `test_backup` — 88 passed, 1 skipped
- **real-runtime:** `tests/gateway/test_agent_computer_chromium.py` — 2 passed (local Chrome, disposable profile)
- **failures:** none

### FINAL_SHA

`3ef1caba67bf6b7f1d546a23289ed1718a0be00f`

### INDEPENDENT_REVIEW

pending

### VERDICT

Prior real-runtime proof was MERGE_READY with findings. The targeted repair
below closes pixel input, wake idempotency, and headed metadata.

---

## Addendum — HUMAN TAKEOVER INPUT + LIFECYCLE HARDENING

**PRE_REPAIR_SHA:** `7cae33abbf06919fe108fd0087ee055fcd65408b`

- Public `computer.act` now accepts `pointer_click` / `pointer_move` / `scroll` / `key` / `text` through the same ControlLease. No CDP exposed.
- `computer.observe` returns screenshot `{mime,data,width,height}` and `viewport {width,height}`. Chromium is forced to `deviceScaleFactor=1`; mapping is `1:1` when sizes match, otherwise scaled.
- `wake()` is idempotent when the runtime is already READY and alive: same handle, same lease, one `runtime_start`.
- `headed_same_host` is `false` (runtime is `--headless=new`).
- Audit redacts `text` / `key` / secrets; accepted input records kind + controller + epoch only.
