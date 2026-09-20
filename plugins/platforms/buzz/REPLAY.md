# Buzz reconnect replay

WebSocket reconnects resume with a 1,801-second overlap to accommodate the
Buzz relay's ±900-second event clock allowance. IDs are retained by event time
through the whole active replay window, including across gateway restarts.
Reply/thread metadata keeps its existing 500-entry LRU independently.

A NIP-98 authenticated `/query` backfill walks `until` + `before_id` keysets
while the live subscription stays open. HTTP pages cannot be interleaved with
live WebSocket frames. Even a short page is followed by another query; an
empty page proves exhaustion regardless of a smaller server page clamp.
The replay floor remains pinned until HTTP exhaustion **and** WebSocket EOSE.
The gateway requires a Buzz relay with the `/query` keyset extension. A relay
that ignores the cursor is detected and cannot silently skip the backlog.
This requirement applies from the first connection after seeding, not only
reconnects. Missing `/query` support blocks live dispatch for the affected
channels after the retry bound; it does not fall back to an incomplete replay.
Verify relay support before enabling the WebSocket transport.

Bounds are 50,000 retained IDs per channel, 1,000 pages per attempt and the
existing WebSocket message byte limit per HTTP response. An exhausted budget,
invalid page or three failed attempts blocks that channel and logs an error;
other channels keep running. Its unfinished floor is persisted. After fixing
the cause, an operator can stop the gateway, clear that channel's
`replay_blocked` field in `buzz/channel-cursors.json`, and restart. Preserve
`replay_floor` and `seen_timestamps` when doing so; deleting cursor evidence
can replay already handled messages. A capacity block cannot be recovered
merely by clearing the flag: the pinned ID set is still full. Preserving replay
safety requires a reviewed build with a larger `_REPLAY_SEEN_LIMIT` and enough
memory before unblocking. Trimming IDs or resetting cursors sacrifices that
safety and may dispatch duplicates; this adapter does not do that automatically.
The three-attempt counter survives WebSocket reconnects within a gateway run;
a process restart starts a fresh retry budget.

Newly adopted channels still exclude pre-adoption history. Old cursor files
contain only the most recent 500 IDs, so migration conservatively preserves
the previous replay floor until the new overlap window has accumulated its
own coverage. This cannot recover omissions predating that coverage boundary.
The poll fallback is unchanged. This is bounded replay deduplication, not an
exactly-once guarantee: dispatch and cursor-file writes are not one transaction,
and existing cursor-write failures are best effort.
