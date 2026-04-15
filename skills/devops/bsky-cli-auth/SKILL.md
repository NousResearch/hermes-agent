---
name: bsky-cli-auth
description: >
  Authenticate the bsky CLI for Bluesky when session expires or is missing.
  Creates an encrypted session file matching bsky-cli's AES-256-GCM format
  using the stored app password.
metadata:
  author: Indigo Karasu
  version: "1.0.0"
---

# bsky CLI Authentication

Restores bsky-cli session when `bsky whoami` shows "Not logged in".

## Prerequisites

- App password stored at: `~/.hermes/secrets/bluesky_app_password.txt`
- Handle: `indigokarasu.bsky.social`
- Node.js and Python 3.13+ with `atproto` package installed

## Quick Check

```bash
bsky whoami
# If "Not logged in", proceed with auth restore
```

## Method: Node.js Encrypted Session (Recommended)

bsky-cli stores sessions at `~/.config/bluesky-cli/session.json` encrypted with AES-256-GCM. The encryption key is derived via scrypt from:

```
keyMaterial = "${homedir()}-${hostname()}-${process.env.USER || 'unknown'}-bluesky-cli-v1"
```

With a salt file at `~/.config/bluesky-cli/.salt`.

### Auth Script

```javascript
// /tmp/bsky_login.js
const { createCipheriv, scryptSync, randomBytes } = require('crypto');
const { homedir, hostname } = require('os');
const { writeFileSync, readFileSync, existsSync } = require('fs');
const { join } = require('path');
const https = require('https');

const configDir = join(homedir(), '.config', 'bluesky-cli');
const sessionPath = join(configDir, 'session.json');
const saltPath = join(configDir, '.salt');

// Load existing salt
let salt;
if (existsSync(saltPath)) {
  salt = readFileSync(saltPath);
} else {
  salt = randomBytes(32);
  writeFileSync(saltPath, salt, { mode: 384 });
}

// Derive encryption key (must match bsky-cli)
const keyMaterial = `${homedir()}-${hostname()}-${process.env.USER || 'unknown'}-bluesky-cli-v1`;
const encryptionKey = scryptSync(keyMaterial, salt, 32);

// Read password from secrets file
const fs = require('fs');
const path = require('path');
const password = fs.readFileSync(join(homedir(), '.hermes', 'secrets', 'bluesky_app_password.txt'), 'utf8').trim();
const handle = 'indigokarasu.bsky.social';

// Authenticate via AT Protocol API
const loginData = JSON.stringify({ identifier: handle, password });

const options = {
  hostname: 'bsky.social',
  port: 443,
  path: '/xrpc/com.atproto.server.createSession',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': loginData.length,
    'User-Agent': 'ocas-haiku/1.1'
  }
};

const req = https.request(options, (res) => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    if (res.statusCode !== 200) {
      console.error(`Login failed: ${res.statusCode} ${data}`);
      process.exit(1);
    }
    const session = JSON.parse(data);
    console.log(`Logged in as ${session.handle} (${session.did})`);

    // Encrypt session
    const iv = randomBytes(16);
    const cipher = createCipheriv('aes-256-gcm', encryptionKey, iv);
    let encrypted = cipher.update(JSON.stringify(session), 'utf8', 'hex');
    encrypted += cipher.final('hex');
    const authTag = cipher.getAuthTag();
    const encryptedData = `${iv.toString('hex')}:${authTag.toString('hex')}:${encrypted}`;

    writeFileSync(sessionPath, encryptedData, { mode: 384 });
    console.log(`Session saved to ${sessionPath}`);
  });
});

req.on('error', (e) => { console.error('Error:', e.message); process.exit(1); });
req.write(loginData);
req.end();
```

### Run

```bash
node /tmp/bsky_login.js
bsky whoami  # Verify: should show "Logged in as indigokarasu.bsky.social"
```

## Alternative: Python atproto (for diagnostics only)

Python `atproto` can create sessions but cannot produce bsky-cli-compatible encrypted files. Useful for verifying credentials:

```python
from atproto import Client
client = Client()
client.login("indigokarasu.bsky.social", "<password>")
# Access: client._session.access_jwt, client._session.refresh_jwt
```

## Pitfalls

- **bsky-cli uses encrypted session files** — you cannot just write a plaintext JSON session file. bsky-cli will throw "Invalid encrypted data format". You must use the same AES-256-GCM encryption with the scrypt-derived key.
- **The salt file must be preserved** — `~/.config/bluesky-cli/.salt` is created once and used for all session decryption. If deleted, all existing sessions become unreadable.
- **Key material includes hostname and username** — if the machine hostname or USER env var changes, the derived key changes and sessions break.
- **The "Fetching timeline..." prefix** — `bsky timeline --json` prepends a status line before JSON output. Always strip everything before the first `{` when parsing the JSON response.
- **`bsky notifications --json` produces invalid JSON for `json.loads()`** — The output contains extra data after the JSON array (e.g., status lines or trailing content), causing `json.loads()` to fail with "Extra data" error. **Use `json.JSONDecoder().raw_decode()` in a loop** to stream-parse individual objects from the output instead of loading the entire response at once. Example:
  ```python
  import json
  raw = sys.stdin.read()
  # Strip any prefix before the array
  start = raw.find('[')
  if start > 0:
      raw = raw[start:]
  decoder = json.JSONDecoder()
  pos = 1  # skip opening bracket
  items = []
  while pos < len(raw):
      while pos < len(raw) and raw[pos] in ' \t\n\r,':
          pos += 1
      if pos >= len(raw) or raw[pos] == ']':
          break
      try:
          obj, end_pos = decoder.raw_decode(raw, pos)
          items.append(obj)
          pos = end_pos
      except json.JSONDecodeError:
          pos += 1
  ```