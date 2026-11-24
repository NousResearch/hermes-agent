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
        const creds = initAuthCreds();
        if (process.env.STUB_REGISTERED) creds.me = { id: '15551234567:1@s.whatsapp.net' };
        return { state: { creds }, saveCreds() {} };
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
    const env = extraEnv => ({ PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, TEMP: dir, TMP: dir, HERMES_HOME: dir,
      // A gateway-spawned bridge always gets its session token; without one it refuses to serve (bridge_auth).
      HERMES_WHATSAPP_BRIDGE_TOKEN: 'passkey-test-token', ...extraEnv });
    const run = args => spawnSync(process.execPath, argv(args), { encoding: 'utf8', timeout: 15000, windowsHide: true, env: env() });
    // Resolves once the child exits. `alive` is true only when the marker appeared and the child was
    // still running 300 ms later, when this helper killed it; an exit right after the warning, an exit
    // before it, or the 15 s guard kill all leave it false.
    const runUntil = (args, extraEnv, marker) => new Promise((resolve, reject) => {
      const child = spawn(process.execPath, argv(args), { windowsHide: true, env: env(extraEnv) });
      let output = '';
      let seen = false;
      let killedAlive = false;
      const guard = setTimeout(() => child.kill(), 15000);
      const onData = chunk => {
        output += chunk;
        if (seen || !marker.test(output)) return;
        seen = true;
        setTimeout(() => { killedAlive = child.exitCode === null && child.signalCode === null; child.kill(); }, 300);
      };
      child.stdout.setEncoding('utf8').on('data', onData);
      child.stderr.setEncoding('utf8').on('data', onData);
      child.on('error', error => { clearTimeout(guard); reject(error); });
      child.on('exit', status => { clearTimeout(guard); resolve({ alive: killedAlive, status, output }); });
    });
    // 78 is bridge.js PASSKEY_REQUIRED_EXIT, which the gateway adapter's PASSKEY_REQUIRED_EXIT must match.
    for (const [args, expected] of [[[], 78], [['--pair-only'], 1]]) {
      const result = run(args);
      assert.equal(result.error, undefined);
      assert.equal(result.status, expected, result.stderr);
      assert.match(result.stderr, /requires passkey verification/);
    }
    // An already connected gateway session ignores the request instead of parking itself: the guarded exit
    // would happen in the same tick as the request, before the warning, so status null proves it is alive.
    const connected = await runUntil([], { STUB_OPEN_FIRST: '1' }, /already connected session; ignoring it/);
    assert.equal(connected.alive, true, connected.output);
    // A paired session that is still logging in (Baileys awaits prekey and passive IQs between CB:success and
    // 'open') has the same claim to keep running.
    const reconnecting = await runUntil([], { STUB_REGISTERED: '1' }, /already connected session; ignoring it/);
    assert.equal(reconnecting.alive, true, reconnecting.output);
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
