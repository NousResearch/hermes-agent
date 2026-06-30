# Spaces and users

> TypeScript samples below — the space/user model is language-neutral.

A **space** is a conversation. A **user** is a participant. Both carry a `__platform` tag.

## Space interface

| Method | Description |
|---|---|
| `send(...content)` | Send one or more content items. |
| `startTyping()` / `stopTyping()` | Show / hide typing indicator. No-op without support. |
| `responding(fn)` | Run `fn` wrapped in typing — guarantees indicator is cleared even on throw. |
| `edit(message, newContent)` | Edit a previously sent message. |
| `getMessage(id)` | Look up a message by ID. |

## Typing indicators

`responding` is the recommended pattern:

```ts
await space.responding(async () => {
  const result = await generateResponse(message);
  await space.send(result);
});
```

Or via the app helper: `await app.responding(space, async () => { ... })`.

## Creating a space

Use [platform narrowing](./platform-narrowing.md) for the platform instance, then pass users:

```ts
import { imessage } from "spectrum-ts/providers/imessage";

const im = imessage(app);
const alice = await im.user("+15551111111");
const bob = await im.user("+15552222222");

const dm = await im.space(alice);
const group = await im.space(alice, bob);

await group.send("Welcome to the group.");
```

The returned space satisfies the generic `Space` interface and carries platform-specific fields (e.g. `type: "dm" | "group"` on iMessage).
