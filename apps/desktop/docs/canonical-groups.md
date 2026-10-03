# Gateway-owned groups in Desktop

Bot Mode lists gateway rooms separately from classic rooms, which Desktop runs itself. **Refresh gateway groups** reloads the canonical room list for the selected connection and profile. Opening a room captures that exact authority; changing the foreground profile does not retarget its controls.

## Which rooms are gateway rooms

One classifier (`groupExecutionMode`) decides from `groups.capabilities`. A connection is canonical only when its `methods` include `groups.discard`, and it can run rooms only when `driver` is `true`. A `-32601` reply, or any other capability payload (current `main`, standalone `hermes serve` or `hermes dashboard`), means classic rooms. A transport error shows the driver as unavailable with **Retry now**, but a connection already classified classic in this session keeps its classic composer.

The connection kind plays no part in this choice. Today only Desktop's local gateway connection advertises the canonical surface: SSH, URL remotes and Nous Cloud reach the standalone web server, which answers with the legacy surface, so their rooms stay classic until a remote endpoint advertises the canonical methods.

On a canonical connection the creation dialog creates a gateway group when the roster qualifies: two to six members, all on this connection, with unique profiles and non-reserved handles. Other rosters are created as classic rooms, and the dialog says why. On a default install the gateway refuses members that are not listed under `hosted_rooms.profiles`; the dialog explains this and links the hosted profile guide.

An existing classic room whose roster qualifies offers **Start gateway group**, which starts a new gateway room and deliberately does not replay the classic history. A classic room whose roster does not qualify keeps its classic composer.

## The room workspace

The gateway owns the log and the work. Desktop reads `groups.state`, and reads `groups.log` incrementally from the last seen `seq`, starting again from the beginning when the room's `authority_epoch` changes. It polls every two seconds while the room is visible and pauses while it is hidden.

- **Status** comes only from `driver_status`: working, idle or driver stopped, plus blocked, approvals waiting and members that need attention. A member whose turn failed or was deferred is listed beside the live status and never replaces it.
- **Send** records its event id and attempted state before calling `groups.send`. Desktop uses the native prepared-submission journal with a stable window owner and exact compare-and-set; the browser-only fallback requires Web Locks and guarantees reload recovery, not process-crash safety. Missing native support refuses Send rather than falling back to weaker storage. A first-attempt refusal with `invalid_params`, `permission_denied`, `unknown_execution` or `stale_generation` hands the text back for editing. Once an attempt is uncertain, later refusals keep the original entry for **Retry**, which resends the same event id. Acceptance requires the matching event id and either an absent legacy `accepted` field or `accepted: true`. A failed Send only returns to the group it was sent from. Another window's pending message is offered through **Restore draft**, which preserves its original identity and text without sending it or replacing an occupied composer.
- **Stop** (`groups.stop`) has its own busy state, so it works while a Send is pending, and reports how many tasks it stopped.
- **Retry, Discard and approvals** come only from `driver_status.pending_actions` and send the exact member, task, generation and request identity. Discard requires confirmation that prior side effects are not undone.
- **Attachments** upload with `groups.attachment.upload` and download with `groups.attachment.download`.
- **Rename** (`groups.rename`) keeps one event id per intended name across retries. **Disband** (`groups.disband`) asks for confirmation, and only a confirmed tombstone removes the room from Desktop and closes its tab. Both appear only when the gateway advertises them.

Bookkeeping events with nothing to show stay out of the history: `turn.settled`, `room.activity`, `task.admitted`, `custody.configured` and `succession.state`. When a group moves to another computer, Desktop words the `authority.transition` notice and a `turn.deferred` that is `waiting_for_host` in the reader's language from their display fields (`to_name`, `from_name`, `offline_since`, `resource`, `host_name`), with generic wording when a name is unknown. An older gateway's English `text` remains the fallback. This needs no extra reads.

Errors stay visible, and no failed gateway-room action falls back to Desktop-run execution. Membership editing is not offered. Room discovery is durable on the gateway rather than replicated through Desktop `ui_meta`.
