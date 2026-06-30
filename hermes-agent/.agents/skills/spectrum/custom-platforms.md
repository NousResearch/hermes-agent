# Building a custom platform

> TypeScript samples below — the authoring contract (config, user/space resolvers, lifecycle, events, actions) is language-neutral; the TS SDK validates with Zod.

`definePlatform` takes a name and a definition object and returns a callable that exposes `.config()` for registration and accepts a Spectrum instance, space, or message for [narrowing](./platform-narrowing.md).

```ts
import { definePlatform } from "spectrum-ts";
import z from "zod";

export const myPlatform = definePlatform("my-platform", {
  config: z.object({ apiKey: z.string() }),

  user: {
    resolve: async ({ input, client }) => ({
      id: input.userID,
      displayName: await client.lookupUser(input.userID),
    }),
  },

  space: {
    resolve: async ({ input, client }) => ({
      id: await client.findOrCreateConversation(input.users.map(u => u.id)),
    }),
  },

  lifecycle: {
    createClient: async ({ config, store }) => new MyPlatformClient(config.apiKey),
    destroyClient: async ({ client }) => { await client.disconnect(); },
  },

  events: {
    async *messages({ client }) {
      for await (const msg of client.onMessage()) {
        yield {
          id: msg.id,
          content: { type: "text", text: msg.body },
          sender: { id: msg.authorId },
          space: { id: msg.channelId },
          timestamp: new Date(msg.ts),
        };
      }
    },
  },

  actions: {
    send: async ({ space, content, client }) => {
      if (content.type === "text") await client.send(space.id, content.text);
    },
    // Optional: startTyping, stopTyping, reactToMessage, replyToMessage, editMessage, getMessage
  },

  static: {
    reactions: { thumbsUp: "+1", thumbsDown: "-1" } as const,
  },
});
```

## Field reference

| Field | Required | Description |
|---|---|---|
| `config` | Yes | Zod schema validating `platform.config()` argument. If every field is optional, `.config()` can be called with no arguments. |
| `user.resolve` | Yes | Resolves a user from a string ID. Returns at minimum `{ id }`. |
| `user.schema` | No | Optional Zod schema for extra user properties. |
| `space.resolve` | Yes | Resolves or creates a conversation. Receives users + optional params. |
| `space.schema` / `space.params` | No | Schemas for the resolved space and for extra creation params. |
| `lifecycle.createClient` | Yes | Creates the platform client. Receives `config`, `projectId`, `projectSecret` (both may be `undefined`), and `store`. |
| `lifecycle.destroyClient` | No | Tears down the client on shutdown. |
| `events.messages` | Yes | Async generator yielding incoming messages. |
| `events.[custom]` | No | Additional generators — exposed on `app.[eventName]`. |
| `actions.send` | Yes | Sends a single content item. Invoked once per item when multiple are passed. |
| `actions.startTyping` / `stopTyping` | No | Typing indicator. |
| `actions.reactToMessage` / `replyToMessage` | No | Missing → corresponding `message.*` becomes a no-op. |
| `actions.editMessage` / `getMessage` | No | Edit and lookup. |
| `message.schema` | No | Zod schema for extra typed fields on incoming messages — surfaced through narrowing. |
| `static` | No | Constants attached to the platform object (e.g. tapback names). |

## Event producers

Every event generator receives `{ client, config, store }` and returns an `AsyncIterable`:

```ts
events: {
  async *messages({ client }) { /* ... */ },
  async *typing({ client }) {
    for await (const ev of client.typing()) {
      yield { spaceId: ev.chatId, userId: ev.user };
    }
  },
},
```

Non-`messages` events are auto-wired as flat properties on both `app` and the narrowed platform instance.

## Registering

```ts
const app = await Spectrum({
  providers: [myPlatform.config({ apiKey: process.env.MY_KEY! })],
});

const mine = myPlatform(app);
const space = await mine.space(await mine.user("user-123"));
await space.send("Hello from my custom platform.");
```
