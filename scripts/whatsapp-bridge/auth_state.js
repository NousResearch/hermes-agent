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
import {
  mkdir, open as fsOpen, readdir, readFile, rename, stat, unlink,
} from 'fs/promises';
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

/**
 * Write `contents` to `filePath` atomically.
 *
 * The temp file MUST live in the same directory as the target: rename() is
 * only atomic within a single filesystem, and /tmp is frequently a different
 * mount. On any failure the temp file is removed and the original is left
 * exactly as it was.
 */
export async function writeFileAtomic(filePath, contents) {
  const tmpPath = `${filePath}${TMP_PREFIX}${process.pid}-${randomBytes(6).toString('hex')}`;
  let handle = null;
  try {
    handle = await fsOpen(tmpPath, 'wx', 0o600);
    await handle.writeFile(contents, 'utf8');
    // fsync before the rename: rename only orders the directory entry, so
    // without this a crash can leave the new name pointing at unflushed
    // (zero-filled) data — the very outcome we are eliminating.
    await handle.sync();
    await handle.close();
    handle = null;
    await rename(tmpPath, filePath);
    // rename() only orders the DIRECTORY ENTRY, and that entry is itself
    // buffered: a crash right after the rename can lose it, leaving the old
    // name or nothing at all. fsync the parent directory so the replacement
    // is durable, not just atomic. Best-effort — some filesystems reject a
    // directory fsync, and failing here would undo a rename that already
    // succeeded.
    let dirHandle = null;
    try {
      dirHandle = await fsOpen(dirname(filePath), 'r');
      await dirHandle.sync();
    } catch {
      /* directory fsync unsupported — the rename still landed */
    } finally {
      if (dirHandle) {
        try { await dirHandle.close(); } catch {}
      }
    }
  } catch (err) {
    if (handle) {
      try { await handle.close(); } catch {}
    }
    try { await unlink(tmpPath); } catch {}
    throw err;
  }
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
