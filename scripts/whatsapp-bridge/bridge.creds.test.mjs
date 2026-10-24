/**
 * Unit tests for the credential persistence guard.
 *
 * The creds.update handler used to call saveCreds() and drop the promise.
 * Baileys' writeData has no catch, so a failed write (ENOSPC, EROFS, EACCES
 * on the session dir) became an unhandled rejection and Node exited the
 * bridge mid-session, with nothing in the log that named the write.
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';

import * as helpers from './bridge_helpers.js';
import { createCredsSaver } from './bridge_helpers.js';

const tick = () => new Promise(resolve => setImmediate(resolve));

// A failing write is logged and swallowed; the saver reports it.
{
  const logs = [];
  const failing = async () => {
    throw Object.assign(new Error('ENOSPC: no space left on device'), { code: 'ENOSPC' });
  };
  const save = createCredsSaver(failing, { log: line => logs.push(line) });

  assert.equal(await save(), false);
  assert.equal(logs.length, 1);
  assert.match(logs[0], /ENOSPC/);
}

// A successful write resolves true and logs nothing.
{
  const logs = [];
  let writes = 0;
  const save = createCredsSaver(async () => { writes += 1; }, { log: line => logs.push(line) });

  assert.equal(await save(), true);
  assert.equal(writes, 1);
  assert.deepEqual(logs, []);
}

// Called like an event handler, without await, a failing write raises no
// unhandledRejection, so the process keeps running.
{
  let unhandled = 0;
  const onUnhandled = () => { unhandled += 1; };
  process.on('unhandledRejection', onUnhandled);
  const save = createCredsSaver(async () => { throw new Error('EROFS: read-only file system'); }, { log: () => {} });

  save();
  await tick();
  await tick();
  process.off('unhandledRejection', onUnhandled);

  assert.equal(unhandled, 0);
}

// bridge.js imports every helper it calls. These tests never load bridge.js,
// so a missing import would only show up as a ReferenceError at bridge start.
{
  const source = readFileSync(new URL('./bridge.js', import.meta.url), 'utf8');
  const [, importList] = source.match(/import \{([^}]+)\} from '\.\/bridge_helpers\.js';/);
  const imported = new Set(importList.split(',').map(name => name.trim()));
  const missing = Object.keys(helpers).filter(name => source.includes(`${name}(`) && !imported.has(name));

  assert.deepEqual(missing, []);
}

console.log('bridge.creds.test.mjs: all assertions passed');
