# Frontend Security — API Keys, CORS & Proxying

## The Core Rule

**NEVER expose your Helius API key in client-side code.** It will be visible in the browser's network tab, source code, and bundle. Anyone can steal it, exhaust your credits, and hit your rate limits.

## Helius Product CORS & Key Requirements

| Product | Browser-Safe? | API Key in Browser? | Recommended Approach |
|---|---|---|---|
| **Sender** (`sender.helius-rpc.com/fast`) | Yes — CORS enabled | **No key needed** | Call directly from browser |
| **RPC** (`mainnet.helius-rpc.com`) | CORS enabled | Key required in URL | **Proxy through backend** |
| **DAS API** | CORS enabled | Key required | **Proxy through backend** |
| **Wallet API** (`api.helius.xyz`) | CORS enabled | Key required | **Proxy through backend** |
| **Enhanced Transactions API** | CORS enabled | Key required | **Proxy through backend** |
| **Priority Fee API** | CORS enabled | Key required | **Proxy through backend** |
| **WebSockets** (`wss://mainnet.helius-rpc.com`) | N/A | Key in URL | **Server relay** (key visible in WS URL) |
| **Webhooks** | N/A | Server-only | Server-only (receives HTTP POSTs) |

**Summary**: Only Helius Sender is safe to call directly from the browser without an API key. Everything else must go through your backend.

## Backend Proxy Patterns

### Next.js App Router — Route Handler

The most common pattern for Next.js apps. Create a catch-all route that proxies Helius API requests:

```typescript
// app/api/helius/[...path]/route.ts
import { NextRequest, NextResponse } from 'next/server';

const HELIUS_API_KEY = process.env.HELIUS_API_KEY!;
const HELIUS_BASE_URL = 'https://mainnet.helius-rpc.com';

// Simple in-memory rate limiter
const rateLimit = new Map<string, { count: number; resetAt: number }>();
const RATE_LIMIT_WINDOW = 60_000; // 1 minute
const RATE_LIMIT_MAX = 60; // requests per window

function checkRateLimit(ip: string): boolean {
  const now = Date.now();
  const entry = rateLimit.get(ip);

  if (!entry || now > entry.resetAt) {
    rateLimit.set(ip, { count: 1, resetAt: now + RATE_LIMIT_WINDOW });
    return true;
  }

  if (entry.count >= RATE_LIMIT_MAX) return false;
  entry.count++;
  return true;
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const ip = request.headers.get('x-forwarded-for') ?? 'unknown';
  if (!checkRateLimit(ip)) {
    return NextResponse.json({ error: 'Rate limit exceeded' }, { status: 429 });
  }

  const { path } = await params;
  const subpath = path.join('/');
  const body = await request.json();

  // Route to the correct Helius endpoint
  let url: string;
  if (subpath.startsWith('v0/') || subpath.startsWith('v1/')) {
    // Enhanced Transactions or Wallet API
    url = `https://api.helius.xyz/${subpath}?api-key=${HELIUS_API_KEY}`;
  } else {
    // RPC / DAS / Priority Fee
    url = `${HELIUS_BASE_URL}/?api-key=${HELIUS_API_KEY}`;
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  return NextResponse.json(data);
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const ip = request.headers.get('x-forwarded-for') ?? 'unknown';
  if (!checkRateLimit(ip)) {
    return NextResponse.json({ error: 'Rate limit exceeded' }, { status: 429 });
  }

  const { path } = await params;
  const subpath = path.join('/');
  const searchParams = request.nextUrl.searchParams.toString();
  const qs = searchParams ? `&${searchParams}` : '';

  const url = `https://api.helius.xyz/${subpath}?api-key=${HELIUS_API_KEY}${qs}`;

  const response = await fetch(url);
  const data = await response.json();
  return NextResponse.json(data);
}
```

**Usage from client:**

```typescript
// Instead of: fetch('https://mainnet.helius-rpc.com/?api-key=SECRET', ...)
// Use:
const response = await fetch('/api/helius/rpc', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: '1',
    method: 'getAssetsByOwner',
    params: { ownerAddress: walletAddress, page: 1, limit: 100 },
  }),
});
```

### Express Proxy

```typescript
import express from 'express';
import rateLimit from 'express-rate-limit';

const app = express();
app.use(express.json());

const HELIUS_API_KEY = process.env.HELIUS_API_KEY!;

const limiter = rateLimit({
  windowMs: 60_000,
  max: 60,
  standardHeaders: true,
  legacyHeaders: false,
});

app.use('/api/helius', limiter);

app.post('/api/helius/rpc', async (req, res) => {
  const response = await fetch(
    `https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    }
  );
  const data = await response.json();
  res.json(data);
});

app.all('/api/helius/v0/*', async (req, res) => {
  const subpath = req.path.replace('/api/helius/', '');
  const url = `https://api.helius.xyz/${subpath}?api-key=${HELIUS_API_KEY}`;

  const response = await fetch(url, {
    method: req.method,
    headers: { 'Content-Type': 'application/json' },
    ...(req.method !== 'GET' && { body: JSON.stringify(req.body) }),
  });
  const data = await response.json();
  res.json(data);
});
```

### Cloudflare Worker Proxy

```typescript
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (!url.pathname.startsWith('/api/helius')) {
      return new Response('Not found', { status: 404 });
    }

    const subpath = url.pathname.replace('/api/helius/', '');
    let targetUrl: string;

    if (subpath.startsWith('v0/') || subpath.startsWith('v1/')) {
      targetUrl = `https://api.helius.xyz/${subpath}?api-key=${env.HELIUS_API_KEY}`;
    } else {
      targetUrl = `https://mainnet.helius-rpc.com/?api-key=${env.HELIUS_API_KEY}`;
    }

    const response = await fetch(targetUrl, {
      method: request.method,
      headers: { 'Content-Type': 'application/json' },
      body: request.method !== 'GET' ? await request.text() : undefined,
    });

    const data = await response.json();

    return new Response(JSON.stringify(data), {
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
      },
    });
  },
};
```

## Environment Variables

### Next.js

```bash
# .env.local (gitignored, dev only)
HELIUS_API_KEY=your-api-key-here

# NEVER use NEXT_PUBLIC_ prefix for API keys!
# NEXT_PUBLIC_HELIUS_API_KEY=xxx  ← DO NOT DO THIS
```

**Rule**: `NEXT_PUBLIC_` prefixed variables are embedded in the client bundle at build time. They are visible to everyone. Only use `NEXT_PUBLIC_` for non-secret values like feature flags.

### Vite

```bash
# .env.local
VITE_HELIUS_API_KEY=xxx  ← DO NOT DO THIS (client-visible)

# Instead, use server-side only:
HELIUS_API_KEY=your-api-key-here  # Only accessible in server code
```

### General Rules

- Store API keys in `.env.local` (gitignored) for development
- Use platform secrets (Vercel, Cloudflare, AWS) for production
- Never commit `.env` files with real keys to git
- Add `.env*.local` to `.gitignore`

## WebSocket Relay Pattern

Helius WebSocket URLs contain the API key (`wss://mainnet.helius-rpc.com/?api-key=KEY`). Opening this connection from the browser exposes the key in the network tab.

**Solution**: Open the WebSocket on your server, relay data to the client via Server-Sent Events (SSE) or your own WebSocket:

```typescript
// Server: connect to Helius WS, relay via SSE
// app/api/stream/route.ts (Next.js App Router)
import { NextRequest } from 'next/server';
import WebSocket from 'ws';

export async function GET(request: NextRequest) {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    start(controller) {
      const ws = new WebSocket(`wss://mainnet.helius-rpc.com/?api-key=${process.env.HELIUS_API_KEY}`);

      ws.on('open', () => {
        ws.send(JSON.stringify({
          jsonrpc: '2.0',
          id: 1,
          method: 'accountSubscribe',
          params: [
            request.nextUrl.searchParams.get('address'),
            { encoding: 'jsonParsed', commitment: 'confirmed' },
          ],
        }));

        // Keep alive
        const pingInterval = setInterval(() => {
          if (ws.readyState === 1) ws.ping();
        }, 30_000);

        ws.on('close', () => clearInterval(pingInterval));
      });

      ws.on('message', (data: Buffer) => {
        const msg = JSON.parse(data.toString());
        if (msg.method) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(msg.params)}\n\n`));
        }
      });

      request.signal.addEventListener('abort', () => {
        ws.close();
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
}
```

```typescript
// Client: consume SSE stream
function useAccountUpdates(address: string) {
  const [data, setData] = useState(null);

  useEffect(() => {
    const eventSource = new EventSource(`/api/stream?address=${address}`);

    eventSource.onmessage = (event) => {
      setData(JSON.parse(event.data));
    };

    return () => eventSource.close();
  }, [address]);

  return data;
}
```

## Rate Limiting Your Proxy

Always rate limit your proxy to prevent abuse:

1. **Per-IP limiting** — prevents a single client from exhausting your Helius credits
2. **Global limiting** — caps total throughput to stay within your Helius plan limits
3. **Method-specific limiting** — apply stricter limits to expensive operations (100-credit Wallet API calls vs 1-credit RPC calls)

```typescript
// Example: different limits per endpoint type
const rpcLimiter = rateLimit({ windowMs: 60_000, max: 120 });   // Standard RPC: generous
const dasLimiter = rateLimit({ windowMs: 60_000, max: 30 });    // DAS: moderate
const walletLimiter = rateLimit({ windowMs: 60_000, max: 10 }); // Wallet API: conservative

app.post('/api/helius/rpc', rpcLimiter, handleRpc);
app.post('/api/helius/das', dasLimiter, handleDas);
app.all('/api/helius/v1/wallet/*', walletLimiter, handleWalletApi);
```

## Common Mistakes

- **API key in `NEXT_PUBLIC_` env var** — this embeds the key in the client bundle. Anyone can extract it from the built JavaScript.
- **API key in browser `fetch()` URL** — visible in the network tab. Use a backend proxy.
- **Opening Helius WebSocket directly from browser** — the API key is in the `wss://` URL, visible in the network tab. Use a server relay.
- **No rate limiting on the proxy** — without limits, anyone can spam your proxy and drain your Helius credits.
- **Using regional HTTP Sender endpoints from browser** — CORS preflight fails on HTTP endpoints. Use `https://sender.helius-rpc.com/fast` (HTTPS) from the browser.
- **Hardcoding API keys in source code** — even in server code, use environment variables. Never commit keys to git.
- **Trusting client input in the proxy** — validate and sanitize the request body before forwarding to Helius. Don't blindly proxy arbitrary requests.
