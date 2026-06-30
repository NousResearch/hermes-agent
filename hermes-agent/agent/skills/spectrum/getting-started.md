# Getting started

> TypeScript samples below — primitives and the app instance are language-neutral.

## Installation

```bash
npm install spectrum-ts        # or pnpm / yarn / bun add
```

Requires TypeScript 5 or later.

## Core concepts

| Primitive | What it represents |
|---|---|
| **Message** | An incoming piece of content — text, attachments, or structured data — from any platform. |
| **Space** | A conversation context. A DM, a group chat, a terminal session. You send messages *into* a space. |
| **User** | A participant on a platform, identified by a platform-specific ID. |
| **Platform provider** | A platform adapter (iMessage, terminal, WhatsApp, or your own) that translates platform-specific protocols into Spectrum's unified interface. |

Every message arrives as a `[Space, Message]` pair.

## Quickstart

Find `PROJECT_ID` and `SECRET_KEY` in your project **Settings** on the [dashboard](https://app.photon.codes/).

```ts
import { Spectrum } from "spectrum-ts";
import { imessage } from "spectrum-ts/providers/imessage";

const app = await Spectrum({
  projectId: "your-project-id",
  projectSecret: "your-project-secret",
  providers: [imessage.config()],
});

for await (const [space, message] of app.messages) {
  if (message.content.type !== "text") continue;

  // Use the platform's native vocabulary, not a bare send:
  await message.react("like");                 // tapback to acknowledge instantly
  await space.responding(async () => {         // show a typing indicator while you work
    await message.reply(`echo: ${message.content.text}`);  // threaded reply, not a loose message
  });
}
```

> **Building an agent? Be rich, not robotic.** A bare `space.send(...)` works, but on iMessage (and other rich platforms) it reads like a webhook, not a person. Reach for the native features whenever they fit the moment:
> - **`message.react("love" | "like" | "laugh" | …)`** — acknowledge a message instantly with a tapback before you've composed a full answer.
> - **`message.reply(...)`** — answer *the specific message* in-thread, so the conversation stays legible in a busy chat.
> - **`space.responding(async () => { … })`** — wrap slow work (an LLM call, a fetch) so the recipient sees a typing indicator instead of dead air.
>
> These **no-op silently** on platforms that don't support them, so there's no downside to reaching for the richer call — write the expressive version once and it degrades gracefully everywhere. See [`reactions-and-replies.md`](./reactions-and-replies.md), [`spaces-and-users.md`](./spaces-and-users.md), and the iMessage-only flourishes (message effects, the full tapback set) in [`providers/imessage.md`](./providers/imessage.md).

Projectless providers like `terminal` work without credentials:

```ts
const app = await Spectrum({ providers: [terminal.config()] });
```

## The app instance

```ts
app.messages                       // AsyncIterable<[Space, Message]>
await app.send(space, ...)         // send into a space
await app.responding(space, fn)    // run fn with a typing indicator
await app.stop()                   // graceful shutdown
```

See [`custom-events-and-lifecycle.md`](./custom-events-and-lifecycle.md) for custom event streams and shutdown.

## Multi-platform

Combine providers — `app.messages` merges every source. The `message.platform` field identifies the origin.

```ts
import { Spectrum } from "spectrum-ts";
import { imessage } from "spectrum-ts/providers/imessage";
import { terminal } from "spectrum-ts/providers/terminal";
import { whatsappBusiness } from "spectrum-ts/providers/whatsapp-business";

const app = await Spectrum({
  projectId: process.env.PROJECT_ID!,
  projectSecret: process.env.PROJECT_SECRET!,
  providers: [
    imessage.config(),
    whatsappBusiness.config({
      accessToken: process.env.WA_TOKEN!,
      phoneNumberId: process.env.WA_NUMBER_ID!,
      appSecret: process.env.WA_SECRET!,
    }),
    terminal.config(),
  ],
});

for await (const [space, message] of app.messages) {
  await space.responding(async () => {
    await message.reply("Hello from Spectrum.");
  });
}
```
