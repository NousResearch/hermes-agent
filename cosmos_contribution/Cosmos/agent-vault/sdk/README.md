# dropclaw

Permanent encrypted on-chain storage for AI agents. Store files on Monad blockchain with AES-256-GCM encryption.

## Install

```bash
npm install dropclaw
```

## Usage

```js
const { VaultClient } = require('dropclaw');

const client = new VaultClient({ gatewayUrl: 'https://dropclaw.cloud' });

// Store a file
const result = await client.store(fileBuffer, {
  paymentHeader: base64PaymentHeader
});
// result.skillFile — JSON with tx hashes for reconstruction
// result.key — hex-encoded AES-256 encryption key (keep safe!)
// result.fileId — unique file identifier

// Retrieve a file
const original = await client.retrieve(result.skillFile, result.key);
// original === your file, byte-for-byte identical

// Estimate cost
const pricing = await client.estimateCost(fileBuffer.length);
```

## Async Upload Polling

Large files are processed asynchronously. The SDK handles polling automatically:

```js
// store() automatically polls for completion
const result = await client.store(largeFileBuffer, { paymentHeader });

// Or poll manually
const { jobId } = await someAsyncOperation();
const completed = await client.waitForCompletion(jobId, {
  pollInterval: 2000,  // ms between polls (default: 2000)
  timeout: 300000      // max wait time in ms (default: 300000)
});
```

## API

### `new VaultClient({ gatewayUrl })`
Create a client connected to a DropClaw gateway.

### `client.store(fileBuffer, options)`
Compress, encrypt, and upload a file. Returns `{ skillFile, key, fileId }`.

### `client.retrieve(skillFile, key)`
Download, decrypt, and decompress a file. Returns the original `Buffer`.

### `client.waitForCompletion(jobId, options)`
Poll for async job completion. Returns the completed job data.

### `client.estimateCost(fileSize)`
Get pricing estimate for a file size in bytes.

## License

Proprietary — see LICENSE for details.
