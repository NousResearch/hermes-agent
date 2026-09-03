# BWM-796 — Client contract

Future clients (Hermes Mobile, Hermes Mac, dashboard) consume only this
shape. The control plane is `gateway.agent_computer.contract.AgentComputerContract`.

## Never returned

- CDP / DevTools URLs
- cookies / auth blobs
- `user-data-dir` / `profile_ref` / `persistence_ref`
- process IDs, browser binaries
- takeover token hashes

The one-time `takeover_token` is returned only to an owner-authenticated
caller on `computer.takeover`.

## JSON-RPC (gateway WS)

| Method | Who | Purpose |
|---|---|---|
| `computer.ensure` | agent or owner | Create/get the profile's computer (no launch) |
| `computer.status` / `computer.list` | agent (own) or owner | Lifecycle + authority |
| `computer.wake` | agent or owner | Start runtime, mint lease |
| `computer.observe` | lease holder | Same-environment observation |
| `computer.act` | lease holder | Fenced input |
| `computer.takeover` | owner only | Request exclusive control |
| `computer.takeover.connect` | owner only | Consume short-lived token |
| `computer.give_back` | owner only | Return control; agent must re-observe |
| `computer.identity.create` | owner only | Explicit BrowserIdentity |
| `computer.identity.attach` | authorized principal | Exclusive mount |

Spoofed `principal` params are ignored. Owner is the server-minted
dashboard identity. Agent is `agent:<profile_id>` from the session profile.

## REST (dashboard, `_require_token` / gated cookie)

`/api/agent-computers`, `/api/browser-identities`, `/api/checkpoints/{id}/approve`.

All owner-authenticated. Same redaction as JSON-RPC.

REST is request/response. It cannot represent a persistent owner
controller connection, so disconnect recovery is TTL expiry plus
explicit `POST .../owner-disconnect`. The authoritative interactive
takeover transport is the gateway WebSocket: `computer.takeover` /
`computer.takeover.connect` bind the owner transport, and WS teardown
calls `release_owner_for_transport_if_active`.

## Agent tools (opt-in toolset `agent_computer`)

`computer_ensure`, `computer_status`, `computer_wake`, `computer_observe`,
`computer_act`. Not in `_HERMES_CORE_TOOLS`. Also gated by
`HERMES_AGENT_COMPUTER_TOOLS`. Takeover is not an agent tool.

## Live view

`computer.observe` returns current URL/title/text plus an optional
on-demand JPEG and the viewport used for input:

```json
"screenshot": {"mime": "image/jpeg", "data": "<base64>", "width": 800, "height": 600},
"viewport": {"width": 800, "height": 600},
"live_view": {
  "kind": "screenshot_on_demand",
  "same_environment": true,
  "remote_stream": false,
  "headed_same_host": false,
  "mapping": "1:1"
}
```

`HermesChromiumRuntime` is `--headless=new` with `deviceScaleFactor=1`.
When screenshot and viewport sizes match, `pointer_click(x, y)` is 1:1
CSS viewport pixels. Otherwise the runtime scales screenshot pixels to
viewport coordinates before `Input.dispatchMouseEvent`.

Owner Human Takeover input (no DOM selectors required):

| kind | fields | meaning |
|---|---|---|
| `pointer_click` | `x`, `y` | click at screenshot/viewport point |
| `pointer_move` | `x`, `y` | move pointer (hover) |
| `scroll` | `delta_x`, `delta_y` | wheel at last pointer or viewport center |
| `key` | `key` / `code` | one key down/up |
| `text` | `text` | insert text at the focused control |

Agent selector actions (`navigate` / `click` / `type`) remain available.

No CDP URL, VNC, WebRTC, or spectator stream. All input is `computer.act`
under the current ControlLease. `text` / `key` values are not written to
audit records.

## Errors

`BROWSER_IDENTITY_BUSY`, `STALE_CONTROLLER`, `OBSERVE_REQUIRED`,
`CHECKPOINT_REQUIRED`, `INVALID_TAKEOVER_TOKEN`, `IDENTITY_REVOKED`,
`FORBIDDEN`, `NOT_FOUND`, `CONFLICT`.
