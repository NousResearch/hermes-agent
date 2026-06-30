# iMessage provider

> TypeScript samples below — connection modes, line model, effects, and tapbacks are platform features that apply across all Spectrum SDKs.

```ts
import { imessage } from "spectrum-ts/providers/imessage";
```

Three connection modes — local, cloud, and dedicated. iMessage-specific features (tapbacks, DM vs group spaces, per-phone routing) come through [platform narrowing](../platform-narrowing.md).

## Connection modes

```ts
// Cloud (default) — managed by Spectrum, full feature set.
// Tokens auto-renew at 80% of TTL. Requires projectId/projectSecret on Spectrum().
imessage.config();

// Local — reads macOS Messages SQLite directly. No network.
// Only supports text + attachments. No reactions, typing, replies, or group creation.
imessage.config({ local: true });

// Dedicated — connect to your own iMessage gRPC endpoints with your own tokens.
// Each entry must include the phone number it serves so Spectrum can route correctly.
imessage.config({
  clients: [
    { address: "instance-1.example.com:443", token: "your-token", phone: "+15551111111" },
    { address: "instance-2.example.com:443", token: "your-token", phone: "+15552222222" },
  ],
});
```

## Line model (cloud mode)

| Plan | Line allocation | What end users see |
|---|---|---|
| **Free / Pro** | Shared pool — each end user routed through a different number from a shared pool | Normal iMessage from a number that may differ across recipients |
| **Business** | Dedicated — all end users text the same number, which belongs to your project | Normal iMessage, always from the same number |

**Auto-scale** is an opt-in Business feature: when traffic to a dedicated line nears its per-line capacity, Spectrum provisions an additional line. Open-source paths (`local: true` or your own dedicated relay) provide their own iCloud account; managed-line concepts don't apply.

## Space types and per-phone routing

iMessage spaces carry `type: "dm" | "group"` and a `phone` field. With multiple dedicated lines, pin a conversation to a specific line:

```ts
const dm = await im.space(alice, { phone: "+15559999999" });
```

When omitted, Spectrum picks at random from available dedicated lines. Per-phone routing applies to **dedicated lines (Business plan) only**; on shared-pool plans the parameter is ignored. Space creation requires cloud or dedicated mode — local mode throws.

## Message effects

Wrap any content with `effect()`. Wrapped content can be a string or `attachment(...)`. Effects only apply on iMessage; other platforms see the inner content unchanged.

```ts
import { effect, imessage } from "spectrum-ts/providers/imessage";

await space.send(effect("Happy birthday!", imessage.effect.message.celebration));
await space.send(effect(attachment("/path/to/photo.jpg"), imessage.effect.message.confetti));
```

| Bubble effects | |
|---|---|
| `imessage.effect.message.slam` | `"com.apple.MobileSMS.expressivesend.impact"` |
| `imessage.effect.message.loud` | `"com.apple.MobileSMS.expressivesend.loud"` |
| `imessage.effect.message.gentle` | `"com.apple.MobileSMS.expressivesend.gentle"` |
| `imessage.effect.message.invisible` | `"com.apple.MobileSMS.expressivesend.invisibleink"` |

| Screen effects | |
|---|---|
| `imessage.effect.message.confetti` | `"com.apple.messages.effect.CKConfettiEffect"` |
| `imessage.effect.message.fireworks` | `"com.apple.messages.effect.CKFireworksEffect"` |
| `imessage.effect.message.balloons` | `"com.apple.messages.effect.CKBalloonEffect"` |
| `imessage.effect.message.heart` | `"com.apple.messages.effect.CKHeartEffect"` |
| `imessage.effect.message.lasers` | `"com.apple.messages.effect.CKLasersEffect"` |
| `imessage.effect.message.celebration` | `"com.apple.messages.effect.CKHappyBirthdayEffect"` |
| `imessage.effect.message.sparkles` | `"com.apple.messages.effect.CKSparklesEffect"` |
| `imessage.effect.message.spotlight` | `"com.apple.messages.effect.CKSpotlightEffect"` |
| `imessage.effect.message.echo` | `"com.apple.messages.effect.CKEchoEffect"` |

## Tapbacks

| Constant | Value |
|---|---|
| `imessage.tapbacks.love` | `"love"` |
| `imessage.tapbacks.like` | `"like"` |
| `imessage.tapbacks.dislike` | `"dislike"` |
| `imessage.tapbacks.laugh` | `"laugh"` |
| `imessage.tapbacks.emphasize` | `"emphasize"` |
| `imessage.tapbacks.question` | `"question"` |

```ts
await message.react(imessage.tapbacks.laugh);
```
