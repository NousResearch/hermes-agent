# BWM-796 — Fresh Recon (including real-browser-profile addendum)

**BASE_SHA:** `180291162ff4df0d42b5dc4fecd08005cf7cebf9`  
**HOSTED_SHA (VPS, not on GitHub fork):** `b2bc82cdb44a685eb1ec3c7850e6b09256347bcc`  
**Method:** source read + existing tests + read-only VPS inspect.  
**Claims without live exercise are labelled.**

---

## EXISTING HERMES BROWSER CAPABILITY REUSE

### Exact source files

| Path | Role |
|---|---|
| `hermes_cli/browser_connect.py` | Default-browser detect, snapshot, pin, lock probe, copy-dir |
| `tools/browser_tool.py` | `_use_real_profile`, `_real_profile_cdp`, launch, session, screenshots |
| `tools/browser_use_cli.py` | `_resolve_real_profile_cdp` for Browser Use CLI |
| `tools/browser_cdp_tool.py` | Raw CDP including `Page.captureScreenshot` |
| `gateway/browser_control_broker.py` | Exclusive controller tickets (inverted: client *is* the driver) |
| `hermes_cli/config_defaults.py` | `use_real_profile`, `real_profile_pin`, `headed`, inactivity |
| `apps/desktop/src/app/settings/browser-real-profile-panel.tsx` | “Use My Real Browser Profile” UI |
| `tests/tools/test_browser_real_profile.py` | Snapshot/launch/consent |
| `tests/tools/test_browser_real_profile_pin.py` | Pin vs last_used |
| `website/docs/user-guide/features/browser.md` | Public docs (not evidence by themselves) |
| `agent/file_safety.py` | Denies agent file-read of `browser-profile/` cookies |
| `hermes_cli/backup.py` | Excludes `browser-profile/` from backups |

### Exact current behavior

Consent (`browser.use_real_profile`, default **false**) snapshots the OS **default Chromium-family browser’s selected profile** into `get_hermes_home()/browser-profile/<browser>/`, launches the **real browser binary** on that copy with `--remote-debugging-port=0`, then attaches `agent-browser` session `hermes-real-profile`. Live default user-data-dir is **never** opened (Chrome ≥136 blocks debugging it; SingletonLock).

Headless by default (`--headless=new`). `browser.headed` / `AGENT_BROWSER_HEADED` opens a visible window **on the same host**. Display-less Linux always headless.

All consented local tasks reuse **one** shared copy-browser (`_REAL_PROFILE_SESSION`). Auth files are **re-synced from the live source on every fresh launch**. Turning consent off **deletes** the snapshot store.

### Supported browsers

Stable only: Chrome, Edge, Brave, Brave Origin, Chromium.  
Beta/Dev/Canary → `UNSUPPORTED_CHANNEL`, fail closed (`#95549`).  
Firefox/Safari → fail closed.  
Proven in `_CHROMIUM_BROWSERS` and `_real_profile_cdp`.

### Profile selection mechanism

1. OS default browser (`detect_default_chromium`).
2. Source profile dir: `browser.real_profile_pin` if set, else `Local State → profile.last_used`, else `Default`.
3. Missing pin **fails closed** — never falls back to last_used.  
   Proven: `browser_connect.py:786-805`, `tests/tools/test_browser_real_profile_pin.py`.

### Snapshot mechanism (Q1)

**Selected profile state, not the entire user-data-dir.**

First populate: `copytree` of the **active** profile directory into copy `Default`, ignoring caches, Extensions, IndexedDB, History, Service Worker, Singleton*, WAL/journals, and SQLite auth DBs (`_SNAPSHOT_IGNORES`, `_SQLITE_AUTH_DBS`).

Then lock-aware copy of auth files (`_AUTH_REFRESH_PROFILE_FILES`): Cookies, Network/Cookies, Login Data, Login Data For Account, Web Data, Preferences — SQLite via online-backup API.

`Local State` is rewritten so the copy’s only profile is `Default`. Marker `.hermes-snapshot-complete` stores the source profile name.

### Persistence mechanism (Q2)

The copy **does persist on disk** across Hermes process restarts (`real_profile_copy_dir`, marker).

It is **not** an independent durable BrowserIdentity:

- Every relaunch **re-syncs** cookies/logins from the live source.
- Consent-off **wipes** `~/.hermes/browser-profile/`.
- Lifecycle is tied to the owner’s daily browser, not to a permanent Agent.
- Encrypted cookies need the **same machine’s OS keychain**.

### Managed-profile storage

`{get_hermes_home()}/browser-profile/<browser>/`  
`get_hermes_home()` follows HERMES_HOME / profile override, so a multiplexed profile *can* isolate the dir. There is still only **one snapshot per browser family per HERMES_HOME**, not N named identities.

### CDP / control path

Launch real binary → `DevToolsActivePort` → `agent-browser --session hermes-real-profile --cdp <port> open about:blank` → HTTP CDP `127.0.0.1:<port>`.  
Agent control: existing `browser_*` / `browser_exec` tools (navigate, click, type, snapshot, screenshot).  
Raw CDP: `browser_cdp` when an endpoint is attached.

### Visible / live-view path (Q6)

| Primitive | Exists? | Same live environment? | Remote owner? |
|---|---|---|---|
| `browser.headed` visible window | YES | YES (same host) | NO — must sit at the VPS/Mac display |
| `browser_screenshot` + `/api/files/stream` | YES | After-the-fact frame | YES over authenticated API |
| `Page.captureScreenshot` via `browser_cdp` | YES | On-demand | YES if CDP stays loopback + Hermes-auth |
| CDP screencast / spectator WS | NO | — | ABSENT (capability map §Z) |
| Camofox headed VNC | YES, Camofox only | Camofox only | Must not be public |
| `browser.controller.*` | YES | Inverted: **client drives** a browser | Not owner takeover of the agent browser |
| Desktop preview pane | YES | URL/file preview, not agent Chromium | NO |

**Human Takeover is not provided by “Use My Real Browser Profile.”**  
Smallest delta: fence input on the **existing CDP session** + owner observe via screenshot/headed; do **not** add VNC/WebRTC.

### Machine portability (Q3)

**Tied to the source machine.** Cookie DBs are OS-keychain encrypted; launch deliberately avoids `--use-mock-keychain` so macOS Keychain / gnome-keyring decrypts. Copying `browser-profile/` to the VPS does **not** yield a logged-in session.  
VPS (read-only): Chromium present (`/usr/bin/chromium-browser`), **no DISPLAY**, **no** `browser-profile/` store, `use_real_profile` unset (false). Headed watch on VPS is impossible without a display.

### Multi-agent isolation (Q5)

Profiles are independent HERMES_HOME dirs (`hermes_cli/profiles.py:4-6`). Snapshot *path* can be per-profile if HERMES_HOME is overridden.

**Not sufficient as BrowserIdentity:**

- One shared session name `hermes-real-profile` per process.
- OS-default browser + last_used/pin is **host** config, not AgentComputer attachment.
- No exclusive mount lock between two agents wanting the same identity.
- Tasks share the copy-browser by design (`_REAL_PROFILE_SESSION`).

### Current locking behavior

- Source profile: Windows deny-all lock → fail closed; macOS/Linux copy-while-running.
- Snapshot dir: no inter-agent exclusive lock.
- Live copy-browser: reuse if CDP already on our data dir; snapshot overlay **skipped** while the copy is open (corruption guard).
- `file_safety` blocks *reading* snapshot cookies as files; terminal can still bypass (documented).

### Security limitations

- Consent-gated convenience, **not** an isolation boundary (docs).
- last_used without pin = wrong-principal risk (upstream already added pin for this).
- Snapshot holds live cookies; owner-only perms; excluded from backups.
- CDP bound to 127.0.0.1 in this path. Must stay that way.
- Do not use the owner’s **active daily browser** as the takeover target (upstream already refuses the live dir).

### Known defects / gaps for BWM-796

1. last_used default is incompatible with explicit BrowserIdentity.
2. Re-sync from owner browser would overwrite an agent-owned identity.
3. Consent-off wipe is wrong lifecycle for durable Agent identities.
4. No ControlLease / exclusive identity lock / owner-vs-agent fencing.
5. No remote same-environment takeover contract.
6. 120s inactivity reaper vs long-lived AgentComputer.
7. VPS cannot display headed takeover.

### What BWM-796 should reuse unchanged

- Chromium launch + loopback CDP + agent-browser attach
- `browser_*` observation/input verbs
- Screenshot path (authenticated file stream)
- `real_profile_pin` fail-closed **pattern**
- Never open the live default user-data-dir
- file_safety / backup exclusion of cookie stores
- Headed mode as **same-host** optional watch, not the product takeover path

### What BWM-796 must harden

- Explicit BrowserIdentity → exact managed dir (never last_used)
- Exclusive identity lock (BUSY, never silent clone)
- Bind to permanent Agent/Profile via AgentComputer
- ControlLease / fencing_epoch
- Persistence **without** source re-sync unless owner imports
- Audit

### What BWM-796 still has to implement

- Domain + store + client contract
- Takeover / give-back / stale rejection
- Checkpoint hook
- Recovery across task/session boundaries
- Synthetic multi-agent tests

### Real canary

`[NEEDS_REAL_CANARY]` cookie decrypt on a real Chromium host; `[OWNER AUTHORIZATION REQUIRED]` any import of a credential-bearing profile.

### DID BWM-796 BUILD A NEW BROWSER RUNTIME?

**NO** — if implementation stays an adapter over the existing Chromium/CDP/`browser_*` path.  
A parallel VNC/desktop/cloud-browser platform would be unjustified.

---

## Q1 — What is copied?

Selected active-profile state: Local State + active profile tree minus caches/extensions + lock-aware Cookies/Login Data/Web Data/Preferences. **Not** the entire user-data-dir.  
Evidence: `snapshot_real_profile` docstring and `_SNAPSHOT_IGNORES` / `_AUTH_REFRESH_PROFILE_FILES` (`browser_connect.py:513-566, 922-1097`).

## Q2 — What becomes persistent?

The Hermes copy persists on disk and is reused, but **stays coupled** to the source via per-launch auth re-sync and consent-off deletion. Independent “import once → durable identity” is **not** the current contract.

## Q3 — Tied to the local Mac?

**Yes for logged-in cookies.** Keychain/keyring bound. VPS cannot consume a Mac snapshot. VPS can run **new** headless Chromium profiles locally.

## Q4 — Can profiles be pinned?

**Yes:** `browser.real_profile_pin`. Missing pin fails closed. Unset → last_used (unsafe as the durable contract). Pin is **one global config key**, not per BrowserIdentity.

## Q5 — Namespace per permanent agent?

Storage *can* follow profile HERMES_HOME. Selection, session name, and lock do **not** implement Agent → Computer → Identity. Must add that binding.

## Q6 — Live-view primitive?

Same-host headed window + on-demand screenshots + CDP. **No** remote spectator stream. Takeover = fence the existing CDP session; do not build VNC.

---

## Persistence classification

| State | Persist? | Recreate? | Mechanism |
|---|---|---|---|
| AgentComputer identity | YES | NO | new store, bound to Profile |
| BrowserIdentity | YES | NO | new record → managed user-data-dir |
| Browser auth/profile | YES if identity-owned | NO | Chromium user-data-dir; **no** last_used re-sync |
| Workspace files | YES (min) | — | persistence_ref under profile home |
| Runtime process | NO | YES | lazy Chromium |
| Browser process | NO | YES | same |
| Current tab / DOM | NO | YES | re-observe after wake/takeover |
| Control lease | YES (epoch) | lease row | fencing_epoch |
| Audit | YES | NO | store |
| Transport | NO | YES | WS/HTTP |

---

## Backend decision

**Smallest sufficient:** reuse Hermes local Chromium + CDP + `browser_*` + identity-scoped user-data-dir + ControlLease + owner contract.

Rejected: 14 VMs; VNC/Guacamole; Browser Use/Browserbase as the ownership model (no keys / not same-env takeover); cua-driver desktop (no WS, wrong density); using owner’s live Chrome as takeover browser.
