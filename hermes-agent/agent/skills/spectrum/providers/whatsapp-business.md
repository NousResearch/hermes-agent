# WhatsApp Business provider

> TypeScript samples below — the 1:1-only constraint is a platform feature.

```ts
import { whatsappBusiness } from "spectrum-ts/providers/whatsapp-business";
```

Wraps the official WhatsApp Business Cloud API. Reactions and threaded replies map to native WhatsApp features. **1:1 only** — `space(userA, userB)` throws.

## Config

```ts
whatsappBusiness.config({
  accessToken: "...",       // permanent or system-user token from Meta for Developers
  phoneNumberId: "...",     // sender phone number ID
  appSecret: "...",         // verifies webhook payload signatures
});
```

Find these values in your WhatsApp Business app settings on [Meta for Developers](https://developers.facebook.com/).

## Starting a conversation

Resolve a user by their WhatsApp phone number (international format, digits only):

```ts
const wa = whatsappBusiness(app);
const customer = await wa.user("15551234567");
const space = await wa.space(customer);

await space.send("Thanks for reaching out.");
```
