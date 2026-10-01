# Gateway-owned groups in Desktop

Bot Mode lists gateway rooms separately from classic rooms, which Desktop runs itself. **Refresh gateway groups** reloads the canonical room list for the selected connection and profile. Opening a room captures that exact authority; changing the foreground profile does not retarget its controls.

## Which rooms are gateway rooms

One classifier (`groupExecutionMode`) decides from `groups.capabilities`. A connection is canonical only when its `methods` include `groups.discard`, and it can run rooms only when `driver` is `true`. A `-32601` reply, or any other capability payload (current `main`, standalone `hermes serve` or `hermes dashboard`), means classic rooms. A transport error shows the driver as unavailable with **Retry now**, but a connection already classified classic in this session keeps its classic composer.

Desktop's local connection and SSH connections to Linux/macOS hosts with `gateway ensure` and `gateway ticket` reach the canonical owner. Closing Desktop closes its SSH tunnel without stopping that owner. Update a canonical SSH gateway on its host, then reconnect; Desktop does not run its managed isolated-backend updater against that service. Older SSH runtimes without canonical ensure, Windows SSH, URL remotes and Nous Cloud retain their existing classic connection path. A canonical runtime missing private ticket support asks for an update instead of spawning another backend.

On a canonical connection the creation dialog creates a gateway group when the roster qualifies: two to six members, all on this connection, with unique profiles and non-reserved handles. Other rosters are created as classic rooms, and the dialog says why. On a default install the gateway refuses members that are not listed under `hosted_rooms.profiles`; the dialog explains this and links the hosted profile guide.

An existing classic room whose roster qualifies offers **Start gateway group**, which starts a new gateway room and deliberately does not replay the classic history. A classic room whose roster does not qualify keeps its classic composer.

## The room workspace

The gateway owns the log and the work. Desktop reads `groups.state`, and reads `groups.log` incrementally from the last seen `seq`, starting again from the beginning when the room's `authority_epoch` changes. It polls every two seconds while the room is visible and pauses while it is hidden.

- **Status** comes only from `driver_status`: working, idle or driver stopped, plus blocked, approvals waiting and members that need attention. A member whose turn failed or was deferred is listed beside the live status and never replaces it.
- **Send** goes through `groups.send` with a journaled event id: the native prepared-submission journal, or a browser fallback that survives reloads but not crashes. A refusal with `invalid_params`, `permission_denied`, `unknown_execution` or `stale_generation` hands the text back for editing. Any other refusal, and any transport failure, keeps the exact entry for **Retry**, which resends the same event id. A failed Send only returns to the room it was sent from.
- **Stop** (`groups.stop`) has its own busy state, so it works while a Send is pending, and reports how many tasks it stopped.
- **Retry, Discard and approvals** come only from `driver_status.pending_actions` and send the exact member, task, generation and request identity. Discard requires confirmation that prior side effects are not undone.
- **Attachments** upload with `groups.attachment.upload` and download with `groups.attachment.download`.
- **Rename** (`groups.rename`) keeps one event id per intended name across retries. **Disband** (`groups.disband`) asks for confirmation, and only a confirmed tombstone removes the room from Desktop and closes its tab. Both appear only when the gateway advertises them.

Errors stay visible, and no failed gateway-room action falls back to Desktop-run execution. Membership editing is not offered. Room discovery is durable on the gateway rather than replicated through Desktop `ui_meta`.

## Cross-gateway setup over native connections

Select the always-on gateway as Desktop's current connection before creating the group. Select a local member and a Bot from another supported SSH gateway. Peer members currently use their gateway's default profile and receive text only. Each participating peer gateway must have its API server enabled and advertise a `gateway.room_link_url` that the room home can reach independently of Desktop; a Desktop-owned SSH tunnel is not a durable RoomLink endpoint. HTTPS is required outside loopback. URL/Cloud and Windows SSH do not gain canonical setup through this change.

The native app verifies both installation identities, creates the pinned roster, obtains a room/member grant from the peer and registers it at the home. The gateway owns subsequent execution, approvals, Stop, renewal and Disband. Initial grants renew within a 30-day authorization horizon; renewing beyond that horizon requires fresh authorization.

Setup credentials stay in Electron main and its private per-obligation files; renderer IPC carries intent, status and the public room only. Storage follows the existing Desktop setting: OFF deliberately uses owner-only plaintext native files without calling the OS keychain; ON uses safeStorage and an encryption failure never falls back to plaintext. Changing the setting also rewrites retained setup obligations. Unreadable records remain unknown recovery work.

Interrupted setup is reconciled before another setup starts and when the creation dialog opens. **Retry now** retries cleanup against the original installations. The peer retains the exact invitation receipt by actor, request ID and frozen body, so a lost issuance reply can be replayed without minting a second grant. A replacement gateway never receives an old grant or cleanup mutation. A successful durable registration receipt allows local credential garbage collection; unsuccessful setup revokes its issued grants and disbands its incomplete room.

Run `npm run test:native-gateways` from `apps/desktop` with the prepared `HERMES_PYTHON` test interpreter. This separate Desktop acceptance lane needs installed Node dependencies; ordinary Python CI does not. It exercises native macOS SSH plus two real isolated gateways, a fresh viewer, peer inference, Disband, lost issuance, and replacement refusal. Its injected AES codec verifies custody mechanics; it does not establish OS keychain acceptance.
