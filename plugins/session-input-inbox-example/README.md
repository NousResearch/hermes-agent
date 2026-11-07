# session-input-inbox (example plugin)

A minimal consumer of the three session-input primitives added to core
(`session_input_primitives`, `session_input_checkpoint`,
`gateway.session_input_transfer`). It is **off by default**: core behaves exactly
as before until a profile enables this plugin.

Enable it for one profile only:

```yaml
plugins:
  enabled: [session-input-inbox]
session_input_inbox:
  enabled: true
  first_window_seconds: 2.0
  quiet_seconds: 2.0
  hard_cap_seconds: 8.0
```

What it does:

* folds consecutive plain-text messages from the same sender/session/policy
  into one admission within the windows (a later message restarts the quiet
  window; `hard_cap_seconds` bounds the batch),
* hands an admitted batch to the running generation (tool checkpoint as a
  steer, model request as a ticketed redirect) unless the turn is not
  steerable or FIFO work is ahead,
* renders `[追加消息 n · <timestamp>]` headers. The format is plugin policy;
  core only stores parts and order.

Text that was already accepted stays retryable and settles even if this plugin
is disabled or removed: the versioned batch record and its retry alias live in
the admission transaction, not in the plugin.

Set `session_input_inbox.enabled: false` to stop new batching without a restart;
accepted text still finishes.
