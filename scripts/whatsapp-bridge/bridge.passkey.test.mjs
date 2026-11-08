import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { test } from 'node:test';

test('bridge distinguishes interactive pairing failure from a non-retryable gateway challenge', () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-passkey-'));
  const bridge = new URL('./bridge.js', import.meta.url);
  const stub = path.join(dir, 'socket.mjs');
  const loader = path.join(dir, 'loader.mjs');
  try {
    writeFileSync(stub, `
      export * from ${JSON.stringify(import.meta.resolve('@whiskeysockets/baileys'))};
      import { initAuthCreds } from ${JSON.stringify(import.meta.resolve('@whiskeysockets/baileys'))};
      import { EventEmitter } from 'node:events';
      export async function useMultiFileAuthState() {
        return { state: { creds: initAuthCreds() }, saveCreds() {} };
      }
      export async function fetchLatestBaileysVersion() { return { version: [2, 3000, 1] }; }
      export function makeWASocket({ auth }) {
        const socket = { ev: new EventEmitter(), ws: new EventEmitter(), authState: auth };
        setImmediate(() => socket.ws.emit('CB:notification', {
          tag: 'notification', attrs: { from: 's.whatsapp.net', type: 'passkey_prologue_request' },
        }));
        return socket;
      }
    `);
    writeFileSync(loader, `
      export async function resolve(specifier, context, next) {
        if (specifier === '@whiskeysockets/baileys' && context.parentURL === ${JSON.stringify(bridge.href)}) {
          return { url: ${JSON.stringify(pathToFileURL(stub).href)}, shortCircuit: true };
        }
        if (specifier.endsWith('/passkey_browser.js')) throw new Error('Browser dependency must not load');
        return next(specifier, context);
      }
    `);
    const run = args => spawnSync(process.execPath, [
      '--no-warnings', '--experimental-loader', pathToFileURL(loader).href,
      fileURLToPath(bridge), '--session', path.join(dir, 'session'), '--port', '0', ...args,
    ], { encoding: 'utf8', timeout: 15000, windowsHide: true,
      env: { PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, TEMP: dir, TMP: dir, HERMES_HOME: dir } });
    for (const [args, expected] of [[[], 78], [['--pair-only'], 1]]) {
      const result = run(args);
      assert.equal(result.error, undefined);
      assert.equal(result.status, expected, result.stderr);
      assert.match(result.stderr, /requires passkey verification/);
    }
    const invalid = run(['--passkey-browser', 'unused-browser']);
    assert.equal(invalid.status, 1);
    assert.match(invalid.stderr, /requires interactive --pair-only/);
    assert.doesNotMatch(invalid.stderr, /Browser dependency must not load/);
  } finally {
    assert.equal(path.dirname(dir), tmpdir());
    rmSync(dir, { recursive: true, force: true });
  }
});
