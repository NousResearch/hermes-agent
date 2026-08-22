# Upstream Package Runtime Patching Pattern

Use when a MIXED or CODE_CHANGE mission identifies the root cause in an upstream npm package that cannot be
source-modified (e.g. `openclaw@2026.6.11` from `github.com/openclaw/openclaw`, the canonical repo is only
the application layer, and the dist files in `.npm-global/lib/node_modules/<name>/dist/` are generated
bundles excluded from direct modification).

## Decision tree

```
Bug is in npm package?
├─ No → normal Gate 5 source change
├─ Yes → is the canonical repo a FORK of the upstream?
│        ├─ Yes → fix source in the fork, rebuild, reinstall
│        └─ No → is there a BUILD step from repo?
│                 ├─ Yes → fix source, rebuild, reinstall
│                 └─ No → RUNTIME WRAPPER (this pattern)
```

## Acceptable wrapper techniques (in preference order)

| Technique | Mechanism | Pros | Cons | Example |
|-----------|-----------|------|------|---------|
| **ESM loader hook** | `--experimental-loader` or `register()` intercepts `load()` for the target JS bundle and patches its source text before V8 compiles it | No proxy service, no config change, tested with 14 passing tests | Only works for ESM modules; `--experimental-loader` flag may be removed by Node.js (use `register()` via `--import` for forward compat) | `session-accessor-patch.mjs` — replaces `createReplySessionInitializationRevision` with a stable-revision version that omits metadata fields |
| **`--require` preload** | CommonJS `Module._resolveFilename` hook wraps the target module's exports | Works for CJS modules, stable API | Only for CJS, not ESM | `require('module')._resolveFilename` patching |
| **Systemd unit env injection** | `Environment=NODE_OPTIONS=...` in the service unit to inject the loader | No code change to the service entry point | Requires `systemctl --user daemon-reload && restart` | `Environment=NODE_OPTIONS=--experimental-loader=/opt/scripts/session-accessor-patch.mjs` |
| **Proxy/wrapper binary** | Replace the `ExecStart` in the systemd unit with a wrapper script that sets up the loader then execs the real process | Full control over startup | Adds maintenance burden; the wrapper must stay in sync with real args | `#!/bin/bash\n exec /usr/bin/node --experimental-loader=/opt/scripts/patch.mjs /path/to/index.js "$@"` |

## Implementation steps for ESM loader hook

1. **Identify the target bundle**: Use `grep -rl "functionName" dist/*.js` to find which bundle contains the function to replace.

2. **Read the function source**: Extract the exact function body to create a matching replacement pattern. Test the regex with `python3` to confirm a single match.

3. **Write the loader module** (`scripts/<name>-patch.mjs`):
   ```javascript
   export async function resolve(specifier, context, nextResolve) {
     return nextResolve(specifier, context);
   }
   
   export async function load(url, context, nextLoad) {
     const result = await nextLoad(url, context);
     if (!result.source || !url.includes('target-bundle-name-')) return result;
     
     const source = result.source.toString();
     const oldCode = 'return JSON.stringify(entry ?? null);';
     const newCode = 'return createStableRevision(entry);';
     
     if (!source.includes(oldCode)) {
       console.error('[patch] ERROR: old code not found in', url);
       return result;
     }
     
     const patched = source.replace(oldCode, newCode) + '\n' + stableFnSource;
     return { format: result.format, source: patched };
   }
   ```

4. **Self-test**: Run `node --experimental-loader ./scripts/patch.mjs --input-type=module -e 'import("file:///path/to/bundle.js")'` and verify the patched function is active (check `mod.l.toString()` for the new code).

5. **Deploy via systemd drop-in (preferred)**: Create a drop-in instead of editing the vendor unit directly.
   ```bash
   mkdir -p ~/.config/systemd/user/<service>.service.d
   cat > ~/.config/systemd/user/<service>.service.d/session-repair.conf << 'EOF'
   [Service]
   Environment="NODE_OPTIONS=--experimental-loader=/ABSOLUTE/CANONICAL/PATH/scripts/patch.mjs"
   EOF
   systemctl --user daemon-reload
   systemctl --user show <service>.service -p Environment  # Verify NODE_OPTIONS is present
   systemctl --user restart <service>
   ```

6. **Verify loader activation in the running process**:
   ```bash
   # Method 1: Check process environment
   MAINPID=$(systemctl --user show <service>.service -p MainPID --value)
   cat /proc/$MAINPID/environ | tr '\0' '\n' | grep NODE_OPTIONS
   # Expected: NODE_OPTIONS=--experimental-loader=/path/to/patch.mjs

   # Method 2: Check service logs for loader's console.log output
   # The loader logs "[<name>-patch] Patched <bundle>.js" to stderr
   # For s6-log pipelines: check /opt/data/logs/<service>/default/current
   # For systemd journal: journalctl --user -u <service> | grep "patch"

   # Method 3: Confirm the drop-in is loaded
   systemctl --user status <service>.service  # Shows "Drop-In:" line
   ```

7. **Back up the service unit before any change**:
   ```bash
   cp ~/.config/systemd/user/<service>.service \
      ~/.config/systemd/user/<service>.service.backup.$(date -u +%Y%m%dT%H%M%SZ)
   diff ~/.config/systemd/user/<service>.service ~/.config/systemd/user/<service>.service.backup.*
   # Expected: IDENTICAL (backup OK)
   ```

## Gateway WebSocket protocol specifics (for testing)

When testing the upstream OpenClaw gateway (port 18789) via WebSocket, the protocol is:

1. **Upgrade**: `GET /openclaw/gateway HTTP/1.1` with `Upgrade: websocket`, `Sec-WebSocket-Protocol: gateway`
2. **Server sends**: `connect.challenge` event with `nonce` and `ts`
3. **Client responds**: `{ type: 'req', id: 'connect-1', method: 'connect', params: { minProtocol: 4, maxProtocol: 4, client: { id: '<known-client-id>', version: '1.0', platform: '<os>', mode: '<mode>' }, auth: { token: '<gateway-token>' } } }`

**Known client IDs** (from `client-info-CcqJJIan.js`):
`webchat-ui`, `openclaw-control-ui`, `openclaw-tui`, `webchat`, `cli`, `gateway-client`, `openclaw-macos`, `openclaw-ios`, `openclaw-android`, `node-host`, `test`, `fingerprint`, `openclaw-probe`

**Known client modes**:
`webchat`, `cli`, `ui`, `backend`, `node`, `probe`, `test`

**Critical**: The `test` client does NOT have `operator.write` scope. Use `cli`, `webchat-ui`, or similar for sessions.create/chat.send.

**Important**: The connect format evolves — the gateway rejected `{ version: 4 }` until changed to `{ minProtocol: 4, maxProtocol: 4 }`, the `client.name` was rejected (must use `client.id` from the known set), and `auth.token` goes in a nested `auth` object, not at the root.

## Practical validation after deployment

The ESM loader approach was proven in the CLOE_REPLY_SESSION_RUNTIME_PATCH_DEPLOYMENT mission. Key lessons:

- Always verify with `systemctl --user show <service>.service -p Environment` BEFORE restarting
- After restart, wait 3-4 seconds for the service to bind ports before checking `ss -tlnp | grep <port>`
- The loader writes its activation message to stderr (goes to the service's stdout/s6 pipeline) — grep for `[<name>-patch]`
- To prove the patch works without full end-to-end testing, directly test the patched function against actual production data:
  ```javascript
  node --experimental-loader ./scripts/patch.mjs --input-type=module -e "
  const mod = await import('file:///path/to/dist/target-bundle.js');
  const snap = mod.p({ storePath: '/path/to/sessions.json', sessionKey: '<test-key>' });
  console.log('Has sessionFile:', JSON.parse(snap.revision).sessionFile ? 'YES (bug!)' : 'NO (fixed)');
  "
  ```

## Concurrency safety testing

After deploying a stable-revision patch, verify these properties:
1. **Benign metadata-only writes do NOT conflict** — changing sessionFile/updatedAt alone should not change the revision
2. **Genuine business-logic conflicts ARE detected** — changing sessionId/systemSent should change the revision  
3. **5 rapid sequential inits all see the same stable revision** — if only metadata changes, every init's snapshot should match
4. **Rollback removes the fix** — removing the systemd drop-in and restarting restores the original JSON.stringify revision

## Verification checklist

- [ ] Patch is verified against the actual bundle (not a copy)
- [ ] Original function behavior is preserved for normal cases
- [ ] Tests pass without the loader (baseline) AND with the loader (fix)
- [ ] The loader does NOT modify the bundle on disk — only in memory at load time
- [ ] Rollback is a one-step revert (remove the env var, restart)
- [ ] The patch survives `systemctl --user restart <service>`
- [ ] No secrets or hardcoded credentials are in the loader file

## Pitfalls

| Trap | Symptom | Fix |
|------|---------|-----|
| **Loader regex too narrow** | The loader logs "ERROR: could not find function" but the function exists | Use Python to test the exact regex against the bundle file first. Watch for escaped characters, whitespace variations, and minified vs formatted code. |
| **Loader matches multiple bundles** | Two bundles export the same function; only one is patched | Narrow the URL check in the `load()` hook: check for a unique substring in the bundle filename. |
| **Node.js removes `--experimental-loader`** | Warning about removal; future nodes refuse the flag | Use `--import 'data:text/javascript,import { register } from "node:module"; register("./patch.mjs", import.meta.url);'` instead. |
| **Systemd env not applied to forked children** | Gateway spawns child processes that do NOT inherit `NODE_OPTIONS` | Set `NODE_OPTIONS` explicitly in the systemd unit's `Environment` directive. Some Node.js CLIs unset `NODE_OPTIONS` for subprocesses — test with `process.env.NODE_OPTIONS`. |
| **Loader only works from certain cwd** | The module URL is relative and fails from a different working directory | Use absolute paths in the loader flag. |