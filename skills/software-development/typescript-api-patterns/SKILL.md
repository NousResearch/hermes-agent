---
name: typescript-api-patterns
description: TypeScript patterns for consuming external APIs - Zod validation, type safety, error handling, and common patterns for OAuth, REST, and GraphQL clients.
triggers:
  - TypeScript errors about unknown types
  - Consuming external APIs (OAuth, REST, GraphQL)
  - Need runtime validation of API responses
  - Setting up error tracking (Sentry) in TypeScript
---

## Patterns

### Zod API Response Validation

Define Zod schemas to validate responses before use. This eliminates `unknown` type errors and provides runtime validation.

#### Example: OAuth Provider

```typescript
import { z } from "zod";

// Define schema for API response
const TokenResponseSchema = z.object({
  access_token: z.string(),
  scope: z.string(),
  token_type: z.literal("bearer"),
});

const UserSchema = z.object({
  login: z.string(),
  name: z.string().nullable(),
  email: z.string().nullable(),
  bio: z.string().nullable(),
  avatar_url: z.string(),
});

export const GitHubOAuth = {
  async exchangeCodeForToken(code: string): Promise<{ access_token: string }> {
    const response = await fetch(TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });

    if (!response.ok) {
      throw new Error(`Token exchange failed: ${response.statusText}`);
    }

    const data = await response.json();
    const parsed = TokenResponseSchema.parse(data);  // Runtime validation + type narrowing
    return { access_token: parsed.access_token };
  },

  async getUserInfo(accessToken: string) {
    const headers = { Authorization: `Bearer ${accessToken}` };

    const userResponse = await fetch(`${API_URL}/user`, { headers });
    const user = UserSchema.parse(await userResponse.json());  // No unknown types

    return {
      name: user.name || user.login,
      email: user.email,
      avatar_url: user.avatar_url,
    };
  },
};
```

#### Handling Optional Fields

```typescript
const Schema = z.object({
  // Field may be missing in response
  description: z.string().optional(),
  // Field may be null in response
  bio: z.string().nullable(),
  // Field may be missing or null
  location: z.string().nullable().optional(),
});

const data = Schema.parse(responseJson);
const displayName = data.name || data.login;  // Fallback for null
```

#### Array Validation

```typescript
const ItemSchema = z.object({ id: z.number(), name: z.string() });
const ResponseSchema = z.object({ items: z.array(ItemSchema) });

const data = ResponseSchema.parse(responseJson);
// data.items is now ItemSchema[] (not any[])
```

#### Literal Types for Enums

```typescript
const TokenResponseSchema = z.object({
  token_type: z.literal("bearer"),  // Ensures exact string match
});

const Schema = z.object({
  status: z.enum(["active", "inactive", "pending"]),
});
```

### Sentry Configuration

#### CLI Apps (non-browser)

Use `@sentry/browser` for CLI apps, NOT `@sentry/react`.

```typescript
import * as Sentry from "@sentry/browser";

type SentryEvent = Parameters<Parameters<typeof Sentry.init>[0]['beforeSend']>[0];

export function initSentry(): void {
  const dsn = process.env.SENTRY_DSN;

  if (!dsn) {
    return;
  }

  Sentry.init({
    dsn,
    environment: process.env.NODE_ENV || "development",
    tracesSampleRate: 0.1,
    sendDefaultPii: false,
    beforeSend(event: SentryEvent) {
      // Sanitize PII
      if (event.user) {
        delete event.user.email;
        delete event.user.username;
        delete event.user.ip_address;
      }
      return event;
    },
  });
}

export function captureException(err: unknown, context?: Record<string, unknown>): void {
  Sentry.captureException(err, { extra: context });
  console.error("[ERROR]:", err, context);
}
```

#### Browser Apps

Use `@sentry/react` for React apps with additional context.

```typescript
import * as Sentry from "@sentry/react";
import { BrowserTracing } from "@sentry/tracing";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  integrations: [
    new BrowserTracing(),
    new Sentry.Replay({
      maskAllText: false,
      blockAllMedia: false,
    }),
  ],
  tracesSampleRate: 0.1,
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,
});
```

## Migration from `any`

### Before (unknown types, TS errors)

```typescript
const data = await response.json();  // type: any

// TypeScript error: 'name' is of type 'unknown'
return {
  name: data.name,
  email: data.email,
};
```

### After (Zod validated)

```typescript
const UserSchema = z.object({
  name: z.string().nullable(),
  email: z.string(),
});

const data = UserSchema.parse(await response.json());  // type: { name: string | null, email: string }

return {
  name: data.name || "Unknown",  // No TS error
  email: data.email,
};
```

## Common Errors

### `Cannot find module 'express'`

Error: TypeScript can't find Express types.

Fix: Add dev dependencies.

```bash
npm install --save-dev express @types/express
```

### `Cannot find module '@sentry/react'` in CLI app

Error: Wrong Sentry package for non-browser environment.

Fix: Use `@sentry/browser` for CLI/Node apps.

```bash
npm install @sentry/browser  # For CLI/Node
# NOT @sentry/react
```

### `error TS18046: '...' is of type 'unknown'`

Error: Accessing properties on untyped JSON response.

Fix: Define Zod schema and parse before use.

```typescript
// BAD
const data = await response.json();
const name = data.name;  // TS18046

// GOOD
const UserSchema = z.object({ name: z.string().nullable() });
const data = UserSchema.parse(await response.json());
const name = data.name || "Unknown";  // No TS error
```

## When NOT to Use Zod

- Internal code - Overkill for data you control
- One-off scripts - TypeScript `any` acceptable for quick prototypes
- Strict performance paths - Zod has runtime overhead (negligible for most API calls)

## Resources

- Zod docs: https://zod.dev
- Pattern examples: `hireme-agent/src/auth/oauth/providers/*.ts`