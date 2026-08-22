# Tool Invocation in Node Subprocess — Diagnostic & Repair Pattern

When a collector inside `lah-stack-tools` needs to run an external CLI tool
(e.g. Knip, dependency-cruiser, jscpd), `spawnSync('npx', ...)` can hang or
timeout while the same `npx` command works instantly in the terminal.

## Root Cause

- `spawnSync` inherits a restricted `PATH` / `env` compared to the interactive shell.
- `npx` may need to resolve or download the package on first use, which takes
  30–60s inside a subprocess with no visible progress.
- Once the npx cache is hot, the command works — but the first call in a fresh
  session or CI run hits the download.

## Diagnostic Steps

1. Capture the real process environment inside the collector:
   ```js
   console.log('PATH:', process.env.PATH);
   console.log('execPath:', process.execPath);
   console.log('cwd:', process.cwd());
   ```

2. Find the npx cache path for the tool:
   ```bash
   find $HOME/.npm/_npx -type d -name "<tool>" 2>/dev/null
   ```
   This gives you a path like `~/.npm/_npx/<hash>/node_modules/<tool>`.

3. Check the package.json `bin` field to find the CLI entry point:
   ```bash
   cat ~/.npm/_npx/<hash>/node_modules/<tool>/package.json | grep '"bin"'
   ```

## Repair — Call via `process.execPath`

Replace `spawnSync('npx', [...])` with:

```js
import { spawnSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';

function resolveToolPath(toolName, knownHashes = []) {
  const home = process.env.HOME || '/home/deploy';
  for (const hash of knownHashes) {
    const p = resolve(home, '.npm/_npx', hash, `node_modules/${toolName}/bin/${toolName}.js`);
    if (existsSync(p)) return p;
  }
  // Fallback: scan all npx cache hashes
  const npxDir = resolve(home, '.npm/_npx');
  if (existsSync(npxDir)) {
    for (const entry of readdirSync(npxDir)) {
      const p = resolve(npxDir, entry, `node_modules/${toolName}/bin/${toolName}.js`);
      if (existsSync(p)) return p;
    }
  }
  return null;
}

const toolPath = resolveToolPath('knip', ['8d873e0e769cf1b4']);
if (toolPath) {
  const result = spawnSync(process.execPath, [toolPath, ...args], {
    timeout: 30000, cwd: repoPath,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
} else {
  // Fallback: use npx with absolute path and long timeout
  const result = spawnSync('/usr/bin/npx', ['--yes', toolName, ...args], {
    timeout: 120000, cwd: repoPath,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}
```

## Mandatory Diagnostics

Every `spawnSync` for an external tool must log enough to debug a failure:

```js
console.log({
  status: result.status,
  signal: result.signal,
  stdout_len: (result.stdout || '').length,
  stderr_preview: (result.stderr || '').slice(0, 200),
  error_code: result.error?.code || null,
  error_path: result.error?.path || null,
});
```

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Timeout >30s | npx resolving/caching package on first call | Use resolved cache path + `process.execPath`, or extend timeout to 120s for npx fallback |
| status=null, signal='SIGTERM' | Timeout reached | Increase timeout or switch to direct path |
| stdout empty, stderr has npx errors | npx couldn't find/install package | Verify cache path exists, try without `--yes` |
| Different behavior than terminal | PATH/env mismatch in spawnSync | Log `process.env.PATH` from within the collector, use absolute paths |

## What NOT to Do

- Do not silently fabricate tool output when the tool fails — report `UNAVAILABLE`.
- Do not hardcode a single cache path without a fallback scan.
- Do not install tools globally or modify `package.json` without justification.
- Do not assume once-cached means always-cached — npx may create new cache hashes.
