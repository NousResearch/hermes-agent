# Gateway-owned groups in Desktop

Bot Mode lists gateway rooms separately from legacy renderer rooms. Refresh gateway groups reloads the selected connection/profile's canonical room list. Opening a room captures that exact authority; changing the foreground profile does not retarget its controls.

With `groups.capabilities.driver` enabled, the existing creation dialog creates a canonical group. Only same-authority local-profile members are supported; Desktop connection descriptors are not peer grants. Existing legacy rooms offer **Start gateway group** instead of renderer-owned execution. This starts a new room and deliberately does not replay legacy history.

The gateway owns the log and work. Desktop reads `groups.state` and `groups.log`, sends through `groups.send`, and stops through `groups.stop`. Pending actions come only from `driver_status.pending_actions`. Discard requires explicit confirmation that prior side effects are not undone; Retry and approvals retain the exact member/task/generation/request identity. Errors remain visible and no failed canonical mutation falls back to hidden-session submission.

This initial canonical workspace is text-only. Rename, disband, membership editing and attachments are not offered here. Room discovery is durable on the gateway rather than replicated through Desktop ui_meta. Ambiguous send retries preserve their event identity while the workspace remains mounted; native crash-safe group-send journaling is not implemented.
