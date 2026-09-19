# Forward-only delivery contract

Forward-only mode sends inbound relay events to a separate consumer over a Unix
socket. The consumer owns signature verification, admission policy, and durable
deduplication. The gateway does not run agent tools in this mode.

Each connection carries one 4-byte big-endian length followed by a JSON frame.
The consumer replies with the same framing and an acknowledgement containing
`{"ack": "<event_id>", "status": "accepted|duplicate|rejected"}`.

## Rejection is terminal

`accepted`, `duplicate`, and `rejected` all mark the event as handled and advance
the gateway cursor, subject to earlier deliveries still awaiting acknowledgement.
`rejected` means a final policy decision, not a request to retry. A missing,
invalid, or timed-out acknowledgement leaves delivery unknown and allows retry.

The producer's `BUZZ_CHANNELS` must match channels already admitted by the
consumer before forwarding starts. With buzz-acp, this includes its subscribed
channel set and membership state. If buzz-acp rejects an event with
`channel_not_subscribed`, the gateway will not automatically replay it after
membership catches up. A configuration mismatch or membership propagation race
can therefore discard delivery; the relay's stored event is not deleted.

Operators must coordinate channel membership before enabling intake, monitor
`forward rejected by consumer` logs, and arrange explicit replay for events
rejected during a membership mismatch. This transport does not guarantee
lossless delivery during dynamic membership changes. A future retryable rejection
protocol must be coordinated with consumers; changing only one side is unsafe.

## Outbound restrictions

`BUZZ_ALLOWED_DESTINATIONS` covers messages, edits, deletions, reactions, and file
attachments. An explicitly empty list denies all destinations. Forward-only mode
also denies outbound destinations when the allowlist is absent. Outside
forward-only mode, an absent allowlist preserves existing behavior.

Socket acknowledgements prove consumer admission or rejection, not completion of
an agent turn or successful publication of a final response.
