# Custom events and lifecycle

> TypeScript samples below — the per-provider event model and idempotent `stop()` are language-neutral.

## Custom events

Providers can emit events beyond messages — typing, read receipts, delivery status, anything. Each is exposed as a flat async iterable on the app instance:

```ts
for await (const event of app.typing) {
  console.log(`${event.platform}: typing event received`);
}
```

The property name matches the event name the provider declared. Streams are created lazily on first access; subsequent iterations share the same source. Per-platform access is also available on a narrowed instance:

```ts
const im = imessage(app);
for await (const event of im.typing) { /* iMessage-only typing events */ }
```

## Graceful shutdown

```ts
await app.stop();
```

Closes the merged message stream, drains and disposes every custom event stream, and tears down every platform client via its `lifecycle.destroyClient` hook (if defined). Idempotent.

Spectrum registers `SIGINT` and `SIGTERM` handlers on startup. When a signal fires, `stop()` is invoked with a 3-second timeout — exit 0 if cleanup completes, exit 1 if not. You don't need to wire this up yourself.

Call `stop()` manually when embedding Spectrum in a longer-running process, in tests that create and dispose an app per case, or when re-initializing with a different provider set.
