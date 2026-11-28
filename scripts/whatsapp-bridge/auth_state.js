/**
 * Crash-safe replacement for Baileys' `useMultiFileAuthState`.
 *
 * WHY THIS EXISTS (confirmed outage, 2026-08-09 09:25 → 2026-08-10 02:07):
 * the host root filesystem hit 100% and the bridge lost its WhatsApp identity
 * for ~17 hours. Baileys persists auth state with a bare
 * `writeFile(path, json)` (lib/Utils/use-multi-file-auth-state.js), which is
 * open(O_TRUNC) followed by write(). When the write fails — ENOSPC here, but
 * EDQUOT/EIO/EFBIG are the same shape — the truncate has ALREADY happened and
 * the file is left at 0 bytes. `creds.json` and 11 key files were destroyed
 * that way.
 *
 * The failure then compounded silently: Baileys' reader treats an unparseable
 * `creds.json` as "no creds" and calls `initAuthCreds()`, minting a BRAND NEW
 * identity. The bridge came up unauthenticated and printed a QR code forever
 * while the gateway logged `whatsapp connect timed out after 30s` every five
 * minutes. Nothing said "your credentials were erased".
 *
 * TWO GUARANTEES, both in the mechanism rather than in a caller remembering:
 *
 *   1. WRITES ARE ATOMIC. Content goes to a temp file in the same directory,
 *      is fsync'd, and is then rename()d over the target. POSIX rename is
 *      atomic, so a reader sees either the complete old file or the complete
 *      new one — never a partial or empty one. If the disk is full the failure
 *      happens on the temp file, which is discarded; the existing credentials
 *      are never even opened for writing. A full disk can no longer cost us
 *      the session.
 *
 *   2. CORRUPTION FAILS LOUDLY. A file that is absent means "first run" and
 *      legitimately mints fresh creds. A file that EXISTS but is empty or
 *      unparseable means real state was damaged, and silently regenerating an
 *      identity there is what made the outage invisible. That case now throws.
 *      Loud beats a QR code nobody is watching.
 */

import { initAuthCreds, BufferJSON, proto } from '@whiskeysockets/baileys';
import * as fsPromises from 'fs/promises';
import { mkdir, readdir, readFile, stat, unlink } from 'fs/promises';
import { dirname, join } from 'path';
import { randomBytes } from 'crypto';

const TMP_PREFIX = '.tmp-';

/**
 * Serialize operations per file path.
 *
 * Baileys keeps a mutex per path for the same reason (their issue #794): the
 * read/write helpers are async, so two overlapping saves on one file can
 * interleave. We reimplement it in five lines rather than importing
 * `async-mutex`, which is a TRANSITIVE dependency here — it is present only
 * because Baileys pulls it in, and depending on somebody else's dependency is
 * how a build breaks on an unrelated upgrade.
 */
function createPathLocks() {
  const chains = new Map();
  return function withLock(key, fn) {
    const prev = chains.get(key) || Promise.resolve();
    const next = prev.then(fn, fn);
    // Keep the chain alive but never let a rejection poison the next caller.
    chains.set(key, next.then(() => {}, () => {}));
    return next;
  };
}

// Codes Windows returns for open()/fsync() on a directory: it has no
// directory fsync at all, and its rename is already durable once it returns.
const WIN32_NO_DIR_FSYNC = new Set(['EISDIR', 'EPERM', 'EACCES']);

/**
 * fsync a directory so a rename() inside it survives a crash.
 *
 * rename() only updates the directory entry, and that entry is itself
 * buffered: without this, a crash right after the rename can bring back the
 * old name (or nothing). On POSIX every failure propagates; only Windows,
 * which cannot open a directory for fsync, is exempt.
 */
async function syncDirectory(dir, fs) {
  let handle = null;
  try {
    handle = await fs.open(dir, 'r');
    await handle.sync();
  } catch (err) {
    if (process.platform === 'win32' && WIN32_NO_DIR_FSYNC.has(err?.code)) return;
    throw err;
  } finally {
    if (handle) await handle.close().catch(() => {});
  }
}

/**
 * Write `contents` to `filePath` atomically and durably.
 *
 * The temp file MUST live in the same directory as the target: rename() is
 * only atomic within a single filesystem, and /tmp is frequently a different
 * mount. If anything fails before the rename, the temp file is removed and the
 * original is left exactly as it was. If the rename landed but the directory
 * fsync failed, the rejection carries `replaced: true`: the new contents are in
 * place but a crash could still revert them.
 *
 * `fs` is the `fs/promises` subset used (open/rename/unlink); tests pass a
 * wrapper to observe call order or inject ENOSPC.
 */
export async function writeFileAtomic(filePath, contents, { fs = fsPromises } = {}) {
  const tmpPath = `${filePath}${TMP_PREFIX}${process.pid}-${randomBytes(6).toString('hex')}`;
  let handle = null;
  try {
    handle = await fs.open(tmpPath, 'wx', 0o600);
    await handle.writeFile(contents, 'utf8');
    // fsync before the rename: rename only orders the directory entry, so
    // without this a crash can leave the new name pointing at unflushed
    // (zero-filled) data — the very outcome we are eliminating.
    await handle.sync();
    await handle.close();
    handle = null;
    await fs.rename(tmpPath, filePath);
  } catch (err) {
    if (handle) await handle.close().catch(() => {});
    await fs.unlink(tmpPath).catch(() => {});
    throw err;
  }
  try {
    await syncDirectory(dirname(filePath), fs);
  } catch (err) {
    throw Object.assign(
      new Error(
        `${filePath} was replaced but its directory could not be fsynced ` +
        `(${err?.code || err?.message || err}); a crash may revert it`,
        { cause: err },
      ),
      { code: err?.code, replaced: true },
    );
  }
}

/**
 * Route `saveCreds()` failures instead of leaving them unhandled.
 *
 * Baileys emits `creds.update` and the bridge cannot await the save inside
 * that event handler, so a rejected save would otherwise be an unhandled
 * rejection, which terminates Node. `save()` never rejects: it resolves
 * `true`/`false` and reports a failure through `onFailure(err, message)`.
 * `settled()` resolves with the outcome of the most recent save, which wrote
 * the full current creds (saves on one path are serialized). `drain()` waits
 * until no save is pending, including saves that start while it waits, so a
 * shutdown does not exit between an accepted `creds.update` and its write.
 */
export function createCredsSaver(saveCreds, { onFailure = (_err, message) => console.error(message) } = {}) {
  let latest = Promise.resolve(true);
  const save = () => {
    latest = Promise.resolve()
      .then(saveCreds)
      .then(
        () => true,
        (err) => {
          const reason = err?.code ? `${err.code}: ${err.message}` : (err?.message || String(err));
          const message = err?.replaced
            ? `WhatsApp creds were written but are not confirmed durable (${reason}). ` +
              'A crash before the next successful save may revert to the previous creds.'
            : `WhatsApp creds save failed (${reason}). creds.json on disk is unchanged ` +
              '(previous state preserved); the next creds.update retries the save.';
          onFailure(err, message);
          return false;
        },
      );
    return latest;
  };
  const drain = async () => {
    for (;;) {
      const pending = latest;
      await pending;
      if (pending === latest) return;
    }
  };
  return { save, settled: () => latest, drain };
}

/**
 * Wait (bounded) for `saver` to drain. Resolves `true` when every pending save
 * finished in time, `false` on timeout. A missing saver (no socket yet) is
 * trivially drained.
 */
export async function drainCredsSaver(saver, timeoutMs) {
  if (!saver) return true;
  let timer;
  const timedOut = new Promise((resolve) => {
    timer = setTimeout(() => resolve(false), timeoutMs);
  });
  try {
    return await Promise.race([saver.drain().then(() => true), timedOut]);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Exit on SIGTERM/SIGINT only after the current creds saver drains (bounded).
 *
 * Node's default is to exit at once, so a `creds.update` accepted just before a
 * stop could die mid-save: the atomic writer keeps the previous creds.json, but
 * the newest creds never reach disk and the next start runs on stale state.
 * Exits 0 once drained, 1 when `timeoutMs` passed first. `started()` lets the
 * bridge stop scheduling reconnects once a shutdown is under way.
 */
export function exitAfterCredsDrain({
  getSaver,
  timeoutMs,
  signals = ['SIGTERM', 'SIGINT'],
  on = (signal, handler) => process.on(signal, handler),
  exit = (code) => process.exit(code),
  log = console.error,
}) {
  let started = false;
  const shutdown = async (signal) => {
    if (started) return;
    started = true;
    const drained = await drainCredsSaver(getSaver(), timeoutMs);
    if (!drained) log(`⚠️  ${signal}: a creds save was still pending after ${timeoutMs}ms; exiting anyway.`);
    exit(drained ? 0 : 1);
  };
  for (const signal of signals) {
    on(signal, () => {
      shutdown(signal);
    });
  }
  return { started: () => started };
}

/** Remove temp files orphaned by a crash or a failed write. */
async function sweepStaleTemps(folder) {
  try {
    const entries = await readdir(folder);
    // Match the exact shape writeFileAtomic produces (`<file>.tmp-<pid>-<hex>`).
    // A substring test would also delete legitimate key files: ids are only
    // mangled for `/` and `:`, so a sender-key whose JID happens to contain
    // ".tmp-" would be swept away as garbage at startup.
    const tempShape = /\.tmp-\d+-[0-9a-f]+$/;
    await Promise.all(
      entries
        .filter((name) => tempShape.test(name))
        .map((name) => unlink(join(folder, name)).catch(() => {})),
    );
  } catch {
    // Sweeping is hygiene, never a startup blocker.
  }
}

/**
 * Drop-in replacement for `useMultiFileAuthState(folder)`.
 *
 * Contract-compatible with Baileys: same `{ state, saveCreds }` shape, same
 * `fixFileName` mangling (so existing session directories keep working with no
 * migration), same BufferJSON encoding, and the same
 * AppStateSyncKeyData rehydration on read.
 */
export async function useAtomicMultiFileAuthState(folder) {
  const folderInfo = await stat(folder).catch(() => undefined);
  if (folderInfo) {
    if (!folderInfo.isDirectory()) {
      throw new Error(
        `found something that is not a directory at ${folder}, either delete it or specify a different location`,
      );
    }
  } else {
    await mkdir(folder, { recursive: true });
  }

  await sweepStaleTemps(folder);

  const withLock = createPathLocks();
  // Identical to Baileys' mangling — session dirs written by the stock
  // implementation must stay readable.
  const fixFileName = (file) => file?.replace(/\//g, '__')?.replace(/:/g, '-');

  const writeData = (data, file) => {
    const filePath = join(folder, fixFileName(file));
    return withLock(filePath, () =>
      writeFileAtomic(filePath, JSON.stringify(data, BufferJSON.replacer)),
    );
  };

  const readData = (file, { strict = false } = {}) => {
    const filePath = join(folder, fixFileName(file));
    return withLock(filePath, async () => {
      let raw;
      try {
        raw = await readFile(filePath, { encoding: 'utf-8' });
      } catch (err) {
        if (err && err.code === 'ENOENT') return null; // never written yet
        if (strict) throw err;
        return null;
      }
      try {
        if (!raw.trim()) throw new Error('file is empty');
        return JSON.parse(raw, BufferJSON.reviver);
      } catch (err) {
        if (strict) {
          // The file exists but its contents are gone or unreadable. Returning
          // null here is what silently minted a new identity during the
          // outage; refuse instead so the operator sees the real fault.
          throw new Error(
            `${filePath} exists but is empty or corrupt (${err.message}). ` +
            'Refusing to silently generate new credentials, which would ' +
            'discard the paired WhatsApp session. Restore this file from a ' +
            'backup, or delete it deliberately to pair again from scratch.',
          );
        }
        // Non-strict path: keys regenerate, so this is recoverable — but an
        // existing-yet-unreadable file is damage, not a first run. Say so, or
        // the key-file half of the outage stays as invisible as the creds
        // half used to be.
        process.emitWarning(
          `${filePath} exists but is empty or unreadable (${err.message}); ` +
          'treating as absent and regenerating.',
        );
        return null;
      }
    });
  };

  const removeData = (file) => {
    const filePath = join(folder, fixFileName(file));
    return withLock(filePath, () => unlink(filePath).catch(() => {}));
  };

  // strict: absent creds are a normal first run; present-but-broken creds are
  // the outage signature and must not be papered over.
  const creds = (await readData('creds.json', { strict: true })) || initAuthCreds();

  return {
    state: {
      creds,
      keys: {
        get: async (type, ids) => {
          const data = {};
          await Promise.all(
            ids.map(async (id) => {
              let value = await readData(`${type}-${id}.json`);
              if (type === 'app-state-sync-key' && value) {
                value = proto.Message.AppStateSyncKeyData.fromObject(value);
              }
              data[id] = value;
            }),
          );
          return data;
        },
        set: async (data) => {
          const tasks = [];
          for (const category in data) {
            for (const id in data[category]) {
              const value = data[category][id];
              const file = `${category}-${id}.json`;
              tasks.push(value ? writeData(value, file) : removeData(file));
            }
          }
          await Promise.all(tasks);
        },
      },
    },
    saveCreds: () => writeData(creds, 'creds.json'),
  };
}
