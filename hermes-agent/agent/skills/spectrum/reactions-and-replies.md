# Reactions and replies

> TypeScript samples below — silent no-op on unsupported platforms is part of the model, not a TS detail.

Both `react` and `reply` live directly on an incoming message. Both no-op silently on platforms that don't support them — no `try/catch` needed.

```ts
await message.react("love");
await message.reply("Replying to your message.");
await message.reply("Here's the file:", attachment("/path/to/file.pdf"));
```

On platforms with thread support (iMessage, WhatsApp Business), `reply` sends threaded. **It is not downgraded to a regular send** — if you need guaranteed delivery, use `space.send(...)`.

For iMessage tapback constants (`imessage.tapbacks.love`, `like`, `dislike`, `laugh`, `emphasize`, `question`), see [`providers/imessage.md`](./providers/imessage.md).

| Want to | Use |
|---|---|
| Send fresh content into a conversation | `space.send(...)` |
| Reply in-thread to a specific message | `message.reply(...)` |
| React to a specific message | `message.react(reaction)` |
