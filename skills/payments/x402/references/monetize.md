# Build an x402 Payment Server

Create an Express server that charges USDC for API access using the x402 payment protocol. Callers pay per-request in USDC on Base — no accounts, API keys, or subscriptions needed. Your service is automatically discoverable by other agents via the x402 Bazaar.

## How It Works

x402 is an HTTP-native payment protocol. When a client hits a protected endpoint without paying, the server returns HTTP 402 with payment requirements. The client signs a USDC payment and retries with a payment header. The facilitator verifies and settles the payment, and the server returns the response. Services register with the x402 Bazaar so other agents can discover and pay for them automatically.

## Confirm wallet is initialized and authed

```bash
npx awal@latest status
```

If the wallet is not authenticated, see `authenticate.md`.

## Step 1: Get the Payment Address

Run this to get the wallet address that will receive payments:

```bash
npx awal@latest address
```

Use this address as the `payTo` value.

## Step 2: Set Up the Project

```bash
mkdir x402-server && cd x402-server
npm init -y
npm install express @x402/express @x402/core @x402/evm @x402/extensions
```

Create `index.js`:

```js
const express = require("express");
const { paymentMiddleware } = require("@x402/express");
const { x402ResourceServer, HTTPFacilitatorClient } = require("@x402/core/server");
const { ExactEvmScheme } = require("@x402/evm/exact/server");

const app = express();
app.use(express.json());

const PAY_TO = "<address from step 1>";

// Create facilitator client and x402 resource server
const facilitator = new HTTPFacilitatorClient({ url: "https://x402.org/facilitator" });
const server = new x402ResourceServer(facilitator);
server.register("eip155:8453", new ExactEvmScheme());

// x402 payment middleware — protects routes below
app.use(
  paymentMiddleware(
    {
      "GET /api/example": {
        accepts: {
          scheme: "exact",
          price: "$0.01",
          network: "eip155:8453",
          payTo: PAY_TO,
        },
        description: "Description of what this endpoint returns",
        mimeType: "application/json",
      },
    },
    server,
  ),
);

// Protected endpoint
app.get("/api/example", (req, res) => {
  res.json({ data: "This costs $0.01 per request" });
});

app.listen(3000, () => console.log("Server running on port 3000"));
```

## Step 3: Run It

```bash
node index.js
```

Test with curl — you should get a 402 response with payment requirements:

```bash
curl -i http://localhost:3000/api/example
```

## API Reference

### paymentMiddleware(routes, server)

Creates Express middleware that enforces x402 payments.

| Parameter | Type                 | Description                                           |
| --------- | -------------------- | ----------------------------------------------------- |
| `routes`  | object               | Route config mapping route patterns to payment config |
| `server`  | x402ResourceServer   | Pre-configured x402 resource server instance          |

### x402ResourceServer

Created with a facilitator client. Register payment schemes and extensions before passing to middleware.

```js
const { x402ResourceServer, HTTPFacilitatorClient } = require("@x402/core/server");
const { ExactEvmScheme } = require("@x402/evm/exact/server");

const facilitator = new HTTPFacilitatorClient({ url: "https://x402.org" });
const server = new x402ResourceServer(facilitator);
server.register("eip155:8453", new ExactEvmScheme());
```

| Method                      | Description                                               |
| --------------------------- | --------------------------------------------------------- |
| `register(network, scheme)` | Register a payment scheme for a CAIP-2 network identifier |

### Route Config

Each key in the routes object is `"METHOD /path"`. The value is a config object:

```js
{
  "GET /api/data": {
    accepts: {
      scheme: "exact",
      price: "$0.05",
      network: "eip155:8453",
      payTo: "0x...",
    },
    description: "Human-readable description of the endpoint",
    mimeType: "application/json",
    extensions: {
      ...declareDiscoveryExtension({
        output: {
          example: { result: "example response" },
          schema: {
            properties: {
              result: { type: "string" },
            },
          },
        },
      }),
    },
  },
}
```

### Accepts Config Fields

The `accepts` field can be a single object or an array (for multiple payment options):

| Field     | Type   | Description                                       |
| --------- | ------ | ------------------------------------------------- |
| `scheme`  | string | Payment scheme: `"exact"`                         |
| `price`   | string | USDC price (e.g. `"$0.01"`, `"$1.00"`)            |
| `network` | string | CAIP-2 network identifier (e.g. `"eip155:8453"`)  |
| `payTo`   | string | Ethereum address (0x...) to receive USDC payments |

### Route-Level Fields

| Field         | Type            | Description                                |
| ------------- | --------------- | ------------------------------------------ |
| `accepts`     | object or array | Payment requirements (single or multiple)  |
| `description` | string?         | What this endpoint does (shown to clients) |
| `mimeType`    | string?         | MIME type of the response                  |
| `extensions`  | object?         | Extensions config (e.g. Bazaar discovery)  |

### Discovery Extension

The `declareDiscoveryExtension` function registers your endpoint with the x402 Bazaar so other agents can discover it:

```js
const { declareDiscoveryExtension } = require("@x402/extensions/bazaar");

extensions: {
  ...declareDiscoveryExtension({
    output: {
      example: { /* example response body */ },
      schema: {
        properties: {
          /* JSON schema of the response */
        },
      },
    },
  }),
}
```

| Field            | Type   | Description                                |
| ---------------- | ------ | ------------------------------------------ |
| `output.example` | object | Example response body for the endpoint     |
| `output.schema`  | object | JSON schema describing the response format |

### Supported Networks

| Network        | Description                      |
| -------------- | -------------------------------- |
| `eip155:8453`  | Base mainnet (real USDC)         |
| `eip155:84532` | Base Sepolia testnet (test USDC) |

## Patterns

### Multiple endpoints with different prices

```js
app.use(
  paymentMiddleware(
    {
      "GET /api/cheap": {
        accepts: {
          scheme: "exact",
          price: "$0.001",
          network: "eip155:8453",
          payTo: PAY_TO,
        },
        description: "Inexpensive data lookup",
      },
      "GET /api/expensive": {
        accepts: {
          scheme: "exact",
          price: "$1.00",
          network: "eip155:8453",
          payTo: PAY_TO,
        },
        description: "Premium data access",
      },
      "POST /api/query": {
        accepts: {
          scheme: "exact",
          price: "$0.25",
          network: "eip155:8453",
          payTo: PAY_TO,
        },
        description: "Run a custom query",
      },
    },
    server,
  ),
);

app.get("/api/cheap", (req, res) => { /* ... */ });
app.get("/api/expensive", (req, res) => { /* ... */ });
app.post("/api/query", (req, res) => { /* ... */ });
```

### Wildcard routes

```js
app.use(
  paymentMiddleware(
    {
      "GET /api/*": {
        accepts: {
          scheme: "exact",
          price: "$0.05",
          network: "eip155:8453",
          payTo: PAY_TO,
        },
        description: "API access",
      },
    },
    server,
  ),
);

app.get("/api/users", (req, res) => { /* ... */ });
app.get("/api/posts", (req, res) => { /* ... */ });
```

### Health check (no payment)

Register free endpoints before the payment middleware:

```js
app.get("/health", (req, res) => res.json({ status: "ok" }));

// Payment middleware only applies to routes registered after it
app.use(paymentMiddleware({ /* ... */ }, server));
app.get("/api/data", (req, res) => { /* ... */ });
```

### POST with body and discovery extension

```js
app.use(
  paymentMiddleware(
    {
      "POST /api/analyze": {
        accepts: {
          scheme: "exact",
          price: "$0.10",
          network: "eip155:8453",
          payTo: PAY_TO,
        },
        description: "Analyze text sentiment",
        mimeType: "application/json",
        extensions: {
          ...declareDiscoveryExtension({
            output: {
              example: { sentiment: "positive", score: 0.95 },
              schema: {
                properties: {
                  sentiment: { type: "string" },
                  score: { type: "number" },
                },
              },
            },
          }),
        },
      },
    },
    server,
  ),
);

app.post("/api/analyze", (req, res) => {
  const { text } = req.body;
  // ... your logic
  res.json({ sentiment: "positive", score: 0.95 });
});
```

### Multiple payment options per endpoint

Accept payments on multiple networks for the same endpoint:

```js
"GET /api/data": {
  accepts: [
    {
      scheme: "exact",
      price: "$0.01",
      network: "eip155:8453",
      payTo: EVM_ADDRESS,
    },
    {
      scheme: "exact",
      price: "$0.01",
      network: "eip155:84532",
      payTo: EVM_ADDRESS,
    },
  ],
  description: "Data endpoint accepting Base mainnet or testnet",
}
```

### Using the CDP facilitator (authenticated)

For production use with the Coinbase facilitator (supports mainnet):

```bash
npm install @coinbase/x402
```

```js
const { facilitator } = require("@coinbase/x402");
const { HTTPFacilitatorClient } = require("@x402/core/server");

const facilitatorClient = new HTTPFacilitatorClient(facilitator);
const server = new x402ResourceServer(facilitatorClient);
server.register("eip155:8453", new ExactEvmScheme());
```

This requires `CDP_API_KEY_ID` and `CDP_API_KEY_SECRET` environment variables. Get these from https://portal.cdp.coinbase.com.

## Testing with the pay flow

Once the server is running, use the pay commands (see `pay.md`) to test:

```bash
# Check the endpoint's payment requirements
npx awal@latest x402 details http://localhost:3000/api/example

# Make a paid request
npx awal@latest x402 pay http://localhost:3000/api/example
```

## Pricing Guidelines

| Use Case               | Suggested Price |
| ---------------------- | --------------- |
| Simple data lookup     | $0.001 - $0.01  |
| API proxy / enrichment | $0.01 - $0.10   |
| Compute-heavy query    | $0.10 - $0.50   |
| AI inference           | $0.05 - $1.00   |

## Checklist

- [ ] Get wallet address with `npx awal@latest address`
- [ ] Install `express`, `@x402/express`, `@x402/core`, `@x402/evm`, and `@x402/extensions`
- [ ] Create `x402ResourceServer` with facilitator client and register `ExactEvmScheme` for `eip155:8453`
- [ ] Define routes with prices, descriptions, and discovery extensions (Bazaar auto-registers when routes declare it)
- [ ] Register payment middleware before protected routes
- [ ] Keep health/status endpoints before payment middleware
- [ ] Test with `curl` (should get 402) and `npx awal@latest x402 pay` (should get 200)
- [ ] Announce your service so other agents can find and use it
