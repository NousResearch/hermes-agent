# BWM-796 — Design (post-recon)

## Decision

Reuse the current Hermes Chromium launch + loopback CDP shape. Do **not**
build VNC, WebRTC, Guacamole, Browserbase-as-owner, or a second desktop
runtime.

NODE0 proved “Use My Real Browser Profile” is a consent-gated snapshot of
the OS default Chromium profile. It re-syncs from the live source on every
launch and is Mac-keychain bound. It is **not** BrowserIdentity and **not**
Human Takeover.

## Objects (kept separate)

```
Permanent Hermes Profile
        |
        v
   AgentComputer          (1:1, durable, not a session/run)
        |
        +-- may attach --> BrowserIdentity
                           (explicit managed user-data-dir, exclusive lock)
```

A BrowserIdentity may represent a work login, a company login, or another
isolated persistent browser profile. It is never implied from
`profile.last_used`.

## Control

`ControlLease` + monotonic `fencing_epoch`.

```
AGENT_CONTROLLED
    -> TAKEOVER_PENDING / YIELDING
    -> OWNER_CONTROLLED   (same live environment)
    -> RETURNING
    -> AGENT_CONTROLLED   (resume_observe_required)
```

Stale leases are rejected. After give-back or owner disconnect the agent
must observe once. Sensitive action classes require an owner checkpoint.

Identity lock: first valid owner mounts the identity; a second requester
gets `BROWSER_IDENTITY_BUSY`. No silent clone, no last-used fallback, no
profile merge.

## Live view

Smallest existing primitive: on-demand screenshot / same-host headed
window / CDP `Page.captureScreenshot`. No new spectator stream.

## Runtime

- Tests and default process: `InMemoryRuntime` (shared page per identity).
- Opt-in: `HERMES_AGENT_COMPUTER_RUNTIME=chromium` launches the real
  binary with `--user-data-dir=<identity>` and loopback DevTools.
- Never calls `snapshot_real_profile`.
