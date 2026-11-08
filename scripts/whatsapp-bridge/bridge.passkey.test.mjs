import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { test } from 'node:test';

test('bridge distinguishes interactive pairing failure from a non-retryable gateway challenge', async () => {
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
        setImmediate(() => {
          if (process.env.STUB_OPEN_FIRST) socket.ev.emit('connection.update', { connection: 'open' });
          socket.ws.emit('CB:notification', {
            tag: 'notification', attrs: { from: 's.whatsapp.net', type: 'passkey_prologue_request' },
          });
        });
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
    const argv = args => ['--no-warnings', '--experimental-loader', pathToFileURL(loader).href,
      fileURLToPath(bridge), '--session', path.join(dir, 'session'), '--port', '0', ...args];
    const env = extraEnv => ({ PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, TEMP: dir, TMP: dir, HERMES_HOME: dir, ...extraEnv });
    const run = args => spawnSync(process.execPath, argv(args), { encoding: 'utf8', timeout: 15000, windowsHide: true, env: env() });
    // Resolves on the marker (then kills the bridge) or on exit, whichever comes first: no fixed sleep.
    const runUntil = (args, extraEnv, marker) => new Promise((resolve, reject) => {
      const child = spawn(process.execPath, argv(args), { windowsHide: true, env: env(extraEnv) });
      let output = '';
      const guard = setTimeout(() => child.kill(), 15000);
      const onData = chunk => {
        output += chunk;
        if (marker.test(output)) { clearTimeout(guard); child.kill(); resolve({ status: null, output }); }
      };
      child.stdout.setEncoding('utf8').on('data', onData);
      child.stderr.setEncoding('utf8').on('data', onData);
      child.on('error', reject);
      child.on('exit', status => { clearTimeout(guard); resolve({ status, output }); });
    });
    for (const [args, expected] of [[[], 78], [['--pair-only'], 1]]) {
      const result = run(args);
      assert.equal(result.error, undefined);
      assert.equal(result.status, expected, result.stderr);
      assert.match(result.stderr, /requires passkey verification/);
    }
    // An already connected gateway session ignores the request instead of parking itself: the guarded exit
    // would happen in the same tick as the request, before the warning, so status null proves it is alive.
    const connected = await runUntil([], { STUB_OPEN_FIRST: '1' }, /already connected session; ignoring it/);
    assert.equal(connected.status, null, connected.output);
    const pairJson = run(['--pair-only', '--pair-json']);
    assert.equal(pairJson.status, 1);
    assert.match(pairJson.stdout, /"event":"error","error":"passkey_required"/);
    const invalid = run(['--passkey-browser', 'unused-browser']);
    assert.equal(invalid.status, 1);
    assert.match(invalid.stderr, /requires interactive --pair-only/);
    assert.doesNotMatch(invalid.stderr, /Browser dependency must not load/);
  } finally {
    assert.equal(path.dirname(dir), tmpdir());
    rmSync(dir, { recursive: true, force: true });
  }
});
