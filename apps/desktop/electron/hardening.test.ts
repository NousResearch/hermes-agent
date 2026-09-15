import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

import { test } from 'vitest'

import {
  ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES,
  clampDataUrlReadMaxMb,
  DATA_URL_READ_DEFAULT_MAX_MB,
  dataUrlReadMaxBytesFromMb,
  DEFAULT_FETCH_TIMEOUT_MS,
  enableBasicPasswordStoreEncryption,
  encryptDesktopSecret,
  readFileDataUrlForIpc,
  resolveDirectoryForIpc,
  resolvePersistedRemoteToken,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc,
  resolveTimeoutMs,
  SAFE_STORAGE_ENCODING,
  SECRET_FILE_MODE,
  sensitiveFileBlockReason,
  stripPathOpenAnnotations,
  tightenSecretFileMode,
  writeSecretFileAtomic
} from './hardening'

/**
 * Real temp dir per test: the property under test IS the on-disk mode after a
 * temp-file-then-rename, which a mocked fs would assert into existence rather
 * than verify. `platform` is still injected so the Windows branch is coverable
 * from a POSIX run.
 */
function withTempDir(run: (dir: string) => void) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-secret-file-'))

  try {
    run(dir)
  } finally {
    fs.rmSync(dir, { force: true, recursive: true })
  }
}

function modeOf(filePath: string) {
  return fs.statSync(filePath).mode & 0o777
}

/**
 * No file other than the target may survive a write, and nothing left in the
 * directory may contain the payload. Asserts the CONTRACT (no readable debris)
 * instead of a literal directory listing, so adding a lock file or renaming the
 * staging file does not break the test.
 */
function assertNoSecretDebris(dir: string, targetName: string, secret: string) {
  for (const name of fs.readdirSync(dir)) {
    if (name === targetName) {
      continue
    }

    // Substring, not RegExp: a real token can contain regex metacharacters.
    assert.equal(
      fs.readFileSync(path.join(dir, name), 'utf8').includes(secret),
      false,
      `leftover file ${name} still contains the secret`
    )
  }
}

async function rejectsWithCode(promise, code: string) {
  await assert.rejects(promise, (error: any) => {
    assert.equal(error?.code, code)

    return true
  })
}

test('clampDataUrlReadMaxMb defaults and bounds the attach size preference', () => {
  assert.equal(clampDataUrlReadMaxMb(undefined), DATA_URL_READ_DEFAULT_MAX_MB)
  assert.equal(clampDataUrlReadMaxMb(0), 1)
  assert.equal(clampDataUrlReadMaxMb(256), 256)
  assert.equal(clampDataUrlReadMaxMb(99999), 4096)
  assert.equal(dataUrlReadMaxBytesFromMb(16), 16 * 1024 * 1024)
})

test('attachment upload cap is bounded above the preview default', () => {
  assert.equal(ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES, 256 * 1024 * 1024)
  assert.ok(ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES > dataUrlReadMaxBytesFromMb(DATA_URL_READ_DEFAULT_MAX_MB))
})

test('attachment data URL helper reads bytes above the preview default without changing that limit', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-large-attachment-'))
  const source = path.join(tempDir, 'large.bin')
  const previewLimit = dataUrlReadMaxBytesFromMb(DATA_URL_READ_DEFAULT_MAX_MB)
  const content = Buffer.alloc(previewLimit + 1024, 0x5a)

  try {
    fs.writeFileSync(source, content)

    await assert.rejects(
      resolveReadableFileForIpc(source, {
        maxBytes: previewLimit,
        purpose: 'File preview'
      }),
      /file is too large/
    )

    const dataUrl = await readFileDataUrlForIpc(source, {
      maxBytes: ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES,
      mimeType: 'application/octet-stream',
      purpose: 'Attachment upload'
    })

    assert.match(dataUrl, /^data:application\/octet-stream;base64,/)
    assert.deepEqual(Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64'), content)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveTimeoutMs falls back to defaults and accepts overrides', () => {
  assert.equal(resolveTimeoutMs(undefined), DEFAULT_FETCH_TIMEOUT_MS)
  assert.equal(resolveTimeoutMs(0), DEFAULT_FETCH_TIMEOUT_MS)
  assert.equal(resolveTimeoutMs(-25), DEFAULT_FETCH_TIMEOUT_MS)
  assert.equal(resolveTimeoutMs('2750'), 2750)
})

test('encryptDesktopSecret requires available secure storage', () => {
  assert.equal(
    encryptDesktopSecret('', { isEncryptionAvailable: () => true, encryptString: () => Buffer.alloc(0) }),
    null
  )

  assert.throws(
    () => encryptDesktopSecret('token', { isEncryptionAvailable: () => false, encryptString: () => Buffer.alloc(0) }),
    /Secure token storage is unavailable/
  )
})

test('encryptDesktopSecret stores safeStorage base64 payload', () => {
  const secret = encryptDesktopSecret('token-123', {
    isEncryptionAvailable: () => true,
    encryptString: value => Buffer.from(`enc:${value}`, 'utf8')
  })

  // Contract: the payload is tagged with the SAME constant main's
  // decryptDesktopSecret dispatches on, and `value` is the keychain ciphertext
  // base64'd — not the token itself.
  assert.equal(secret?.encoding, SAFE_STORAGE_ENCODING)
  assert.equal(Buffer.from(String(secret?.value), 'base64').toString('utf8'), 'enc:token-123')
  assert.doesNotMatch(String(secret?.value), /token-123/, 'the plaintext is not recoverable from the payload')
})

// ─── Owner-only credential files (connection.json) ─────────────────────────

test('writeSecretFileAtomic creates the file owner-only, not at the 0644 umask default', () => {
  withTempDir(dir => {
    const target = path.join(dir, 'connection.json')
    const payload = JSON.stringify({ remote: { token: { encoding: SAFE_STORAGE_ENCODING, value: 'BLOB' } } })

    writeSecretFileAtomic(target, payload)

    assert.equal(modeOf(target), SECRET_FILE_MODE)
    assert.equal(modeOf(target) & 0o077, 0, 'no group/other bits')
    assert.equal(fs.readFileSync(target, 'utf8'), payload, 'content round-trips')
    assertNoSecretDebris(dir, 'connection.json', 'BLOB')
  })
})

test('encryptDesktopSecret allows plain-text opt-in when encryption is unavailable', () => {
  const secret = encryptDesktopSecret(
    'token',
    { isEncryptionAvailable: () => false, encryptString: () => Buffer.alloc(0) },
    { allowPlainText: true }
  )

  assert.deepEqual(secret, { encoding: 'plain', value: 'token' })
})

test('encryptDesktopSecret keeps encrypting when available even with the plain-text opt-in', () => {
  const secret = encryptDesktopSecret(
    'token-123',
    { isEncryptionAvailable: () => true, encryptString: value => Buffer.from(`enc:${value}`, 'utf8') },
    { allowPlainText: true }
  )

  assert.deepEqual(secret, {
    encoding: 'safeStorage',
    value: Buffer.from('enc:token-123', 'utf8').toString('base64')
  })
})

test('encryptDesktopSecret returns null for an empty value even with the plain-text opt-in', () => {
  assert.equal(
    encryptDesktopSecret(
      '',
      { isEncryptionAvailable: () => false, encryptString: () => Buffer.alloc(0) },
      { allowPlainText: true }
    ),
    null
  )
})

test('enableBasicPasswordStoreEncryption flips the flag once on linux with --password-store=basic', () => {
  const calls: boolean[] = []

  const safeStorageApi = {
    setUsePlainTextEncryption: (value: boolean) => calls.push(value)
  }

  const result = enableBasicPasswordStoreEncryption({
    platform: 'linux',
    passwordStoreSwitch: 'basic',
    safeStorageApi
  })

  assert.equal(result, true)
  assert.deepEqual(calls, [true])
})

test('enableBasicPasswordStoreEncryption ignores non-basic password-store values on linux', () => {
  for (const passwordStoreSwitch of ['gnome-libsecret', '', undefined]) {
    const calls: unknown[] = []

    const safeStorageApi = {
      setUsePlainTextEncryption: () => calls.push('called')
    }

    const result = enableBasicPasswordStoreEncryption({
      platform: 'linux',
      passwordStoreSwitch,
      safeStorageApi
    })

    assert.equal(result, false, `value ${JSON.stringify(passwordStoreSwitch)} must not enable plain text`)
    assert.deepEqual(calls, [])
  }
})

test('enableBasicPasswordStoreEncryption never enables plain text off linux even with --password-store=basic', () => {
  for (const platform of ['win32', 'darwin']) {
    const calls: unknown[] = []

    const safeStorageApi = {
      setUsePlainTextEncryption: () => calls.push('called')
    }

    const result = enableBasicPasswordStoreEncryption({
      platform,
      passwordStoreSwitch: 'basic',
      safeStorageApi
    })

    assert.equal(result, false, `platform ${platform} must not enable plain text`)
    assert.deepEqual(calls, [])
  }
})

test('enableBasicPasswordStoreEncryption tolerates a missing setUsePlainTextEncryption method', () => {
  assert.equal(
    enableBasicPasswordStoreEncryption({ platform: 'linux', passwordStoreSwitch: 'basic', safeStorageApi: {} }),
    false
  )
  assert.equal(
    enableBasicPasswordStoreEncryption({ platform: 'linux', passwordStoreSwitch: 'basic', safeStorageApi: undefined }),
    false
  )
})

test('enableBasicPasswordStoreEncryption swallows a throwing setUsePlainTextEncryption', () => {
  const safeStorageApi = {
    setUsePlainTextEncryption: () => {
      throw new Error('backend not ready')
    }
  }

  assert.equal(
    enableBasicPasswordStoreEncryption({ platform: 'linux', passwordStoreSwitch: 'basic', safeStorageApi }),
    false
  )
})

test('resolvePersistedRemoteToken stores plain text end-to-end only with the explicit opt-in', () => {
  const unavailableSafeStorage = { isEncryptionAvailable: () => false, encryptString: () => Buffer.alloc(0) }
  const encryptSecret = (value: string, options: any) => encryptDesktopSecret(value, unavailableSafeStorage, options)

  assert.deepEqual(
    resolvePersistedRemoteToken({
      incomingToken: 'token',
      persistToken: true,
      existingToken: undefined,
      allowPlainText: true,
      encryptSecret
    }),
    { encoding: 'plain', value: 'token' }
  )

  // Only strict boolean true opts in; undefined, false, and truthy-non-true
  // values must all keep the secure-storage requirement (which throws when the
  // keyring is unavailable).
  for (const allowPlainText of [undefined, false, 1, 'yes']) {
    assert.throws(
      () =>
        resolvePersistedRemoteToken({
          incomingToken: 'token',
          persistToken: true,
          existingToken: undefined,
          allowPlainText,
          encryptSecret
        }),
      /Secure token storage is unavailable/,
      `allowPlainText ${JSON.stringify(allowPlainText)} must not enable plain-text storage`
    )
  }
})

test('resolvePersistedRemoteToken keeps encrypting when the keyring is available even with the opt-in', () => {
  const availableSafeStorage = {
    isEncryptionAvailable: () => true,
    encryptString: (value: string) => Buffer.from(`enc:${value}`, 'utf8')
  }

  const encryptSecret = (value: string, options: any) => encryptDesktopSecret(value, availableSafeStorage, options)

  assert.deepEqual(
    resolvePersistedRemoteToken({
      incomingToken: 'token-123',
      persistToken: true,
      existingToken: undefined,
      allowPlainText: true,
      encryptSecret
    }),
    { encoding: 'safeStorage', value: Buffer.from('enc:token-123', 'utf8').toString('base64') }
  )
})

test('resolvePersistedRemoteToken passes the token through untouched on the transient path', () => {
  let called = false

  const encryptSecret = () => {
    called = true

    return null
  }

  assert.deepEqual(
    resolvePersistedRemoteToken({
      incomingToken: 'token',
      persistToken: false,
      existingToken: { encoding: 'safeStorage', value: 'stale' },
      allowPlainText: false,
      encryptSecret
    }),
    { encoding: 'plain', value: 'token' }
  )
  assert.equal(called, false, 'the transient test-connection path must not touch secure storage')
})

test('resolvePersistedRemoteToken keeps the existing token when no new token is supplied', () => {
  let called = false

  const encryptSecret = () => {
    called = true

    return null
  }

  const existingToken = { encoding: 'safeStorage', value: 'kept' }

  assert.equal(
    resolvePersistedRemoteToken({
      incomingToken: '',
      persistToken: true,
      existingToken,
      allowPlainText: true,
      encryptSecret
    }),
    existingToken
  )
  assert.equal(called, false, 'an empty incoming token must not re-encrypt anything')
})

test('writeSecretFileAtomic does not inherit loose bits from a stale temp file', () => {
  // renameSync keeps the TEMP file's permissions, and writeFileSync's `mode`
  // is ignored when the path already exists — so a temp left by a crashed
  // earlier write would otherwise hand 0644 straight to the target.
  withTempDir(dir => {
    const target = path.join(dir, 'connection.json')
    fs.writeFileSync(`${target}.tmp`, 'stale', { mode: 0o666 })
    assert.notEqual(modeOf(`${target}.tmp`), SECRET_FILE_MODE)

    writeSecretFileAtomic(target, 'fresh')

    assert.equal(modeOf(target), SECRET_FILE_MODE)
    assert.equal(fs.readFileSync(target, 'utf8'), 'fresh')
  })
})

/**
 * Owner-only is carried by two independent mechanisms — the create-time `mode`
 * and the chmod before the rename — because each covers a case the other
 * cannot. The next two tests knock out one mechanism at a time (with the REAL
 * fs doing the actual write, so the assertion is still the on-disk mode) and
 * require the survivor to hold the line on its own. Without them, either
 * mechanism could be deleted with every test still green.
 */
function fsWith(overrides: Record<string, unknown>) {
  return { ...fs, ...overrides } as any
}

test('the written file is owner-only even where chmod does nothing', () => {
  // Windows, and any mount that refuses chmod. The create-time `mode` is what
  // covers this — there is no second chance to tighten.
  withTempDir(dir => {
    const target = path.join(dir, 'connection.json')
    // Pin a permissive umask so a mode-less create WOULD land
    // group/other-readable — otherwise a restrictive-umask host could pass this
    // for free. Synchronous and restored in `finally`, and the electron project
    // runs one process per file, so no other test observes it.
    const previousUmask = process.umask(0o022)

    try {
      const witness = path.join(dir, 'witness.json')
      fs.writeFileSync(witness, 'x')
      assert.notEqual(modeOf(witness), SECRET_FILE_MODE, 'the ambient default is NOT already owner-only')

      writeSecretFileAtomic(target, 'tok', { fs: fsWith({ chmodSync: () => void 0 }) })
    } finally {
      process.umask(previousUmask)
    }

    assert.equal(modeOf(target), SECRET_FILE_MODE, 'created owner-only, not tightened after the fact')
  })
})

test('the written file is owner-only even when a stale temp cannot be removed', () => {
  // The unlink is best-effort; if the stale temp survives, writeFileSync's
  // `mode` is ignored on an existing path and only the chmod before the rename
  // can still fix the bits.
  withTempDir(dir => {
    const target = path.join(dir, 'connection.json')
    fs.writeFileSync(`${target}.tmp`, 'stale', { mode: 0o666 })

    writeSecretFileAtomic(target, 'tok', { fs: fsWith({ rmSync: () => void 0 }) })

    assert.equal(modeOf(target), SECRET_FILE_MODE, 'tightened before the rename handed the bits over')
    assert.equal(fs.readFileSync(target, 'utf8'), 'tok')
  })
})

test('writeSecretFileAtomic cannot be redirected through a symlink planted at the temp path', () => {
  // A stale temp path is attacker-controllable in a shared temp/userData dir.
  // Following it would write the token into the victim file AND then rename the
  // link over connection.json, so every later write leaks too.
  withTempDir(dir => {
    const target = path.join(dir, 'connection.json')
    const victim = path.join(dir, 'victim.txt')
    fs.writeFileSync(victim, 'original', { mode: 0o644 })

    try {
      fs.symlinkSync(victim, `${target}.tmp`, 'file')
    } catch (error: any) {
      if (error?.code === 'EPERM' || error?.code === 'EACCES') {
        return
      }

      throw error
    }

    writeSecretFileAtomic(target, 'tok-live-42')

    assert.equal(fs.readFileSync(victim, 'utf8'), 'original', 'the symlink target was not written through')
    assert.equal(modeOf(victim), 0o644, 'the victim file was not chmodded either')
    assert.equal(fs.readFileSync(target, 'utf8'), 'tok-live-42')
    assert.equal(fs.lstatSync(target).isSymbolicLink(), false, 'the target is a real file, not the planted link')
    assert.equal(modeOf(target), SECRET_FILE_MODE)
  })
})

test('tightenSecretFileMode tightens a pre-existing world-readable config in place', () => {
  // The upgrade path: a connection.json written by an older build sits at 0644
  // with a real (encrypted) token in it. Tightening must change the mode and
  // nothing else — the token has to stay readable or the user loses their
  // configured gateway.
  withTempDir(dir => {
    const target = path.join(dir, 'connection.json')

    const legacy = JSON.stringify({
      mode: 'remote',
      remote: {
        url: 'https://gw.example.com',
        authMode: 'token',
        token: { encoding: SAFE_STORAGE_ENCODING, value: 'BLOB' }
      }
    })

    fs.writeFileSync(target, legacy, { mode: 0o644 })
    assert.equal(modeOf(target), 0o644)

    assert.equal(tightenSecretFileMode(target), true)

    assert.equal(modeOf(target), SECRET_FILE_MODE)
    assert.deepEqual(JSON.parse(fs.readFileSync(target, 'utf8')), JSON.parse(legacy), 'contents untouched')
  })
})

test('tightenSecretFileMode leaves a non-safeStorage token payload readable', () => {
  // A hand-edited config (or one from a pre-release build) can hold a
  // non-safeStorage token payload, which decryptDesktopSecret still reads
  // verbatim on purpose. Tightening the mode must not disturb that fallback —
  // it only narrows who can open the file.
  withTempDir(dir => {
    const target = path.join(dir, 'connection.json')

    const legacyPlain = JSON.stringify({
      mode: 'remote',
      remote: { url: 'https://gw.example.com', authMode: 'token', token: { encoding: 'plain', value: 'tok-live-42' } }
    })

    fs.writeFileSync(target, legacyPlain, { mode: 0o644 })

    tightenSecretFileMode(target)

    assert.equal(modeOf(target), SECRET_FILE_MODE)
    assert.equal(JSON.parse(fs.readFileSync(target, 'utf8')).remote.token.value, 'tok-live-42')
  })
})

test('tightenSecretFileMode is idempotent and never throws on an unusable path', () => {
  withTempDir(dir => {
    const target = path.join(dir, 'connection.json')
    writeSecretFileAtomic(target, '{}')

    assert.equal(tightenSecretFileMode(target), true)
    assert.equal(tightenSecretFileMode(target), true)
    assert.equal(modeOf(target), SECRET_FILE_MODE)

    // Missing file (fresh install, nothing saved yet) reports failure quietly
    // instead of breaking the read path it is called from.
    assert.equal(tightenSecretFileMode(path.join(dir, 'absent.json')), false)
  })
})

test('tightenSecretFileMode refuses to chmod a symlink instead of following it to its target', () => {
  // Matches readInstallationId in desktop-installation.ts. Without the lstat
  // guard a link planted at the config path sends the chmod to whatever it
  // resolves to — someone else's file gets its mode rewritten.
  withTempDir(dir => {
    const target = path.join(dir, 'connection.json')
    const victim = path.join(dir, 'victim.txt')
    fs.writeFileSync(victim, 'not mine', { mode: 0o644 })

    try {
      fs.symlinkSync(victim, target, 'file')
    } catch (error: any) {
      if (error?.code === 'EPERM' || error?.code === 'EACCES') {
        return
      }

      throw error
    }

    assert.equal(tightenSecretFileMode(target), false, 'reports "not tightened" rather than acting on the link')
    assert.equal(modeOf(victim), 0o644, 'the symlink target keeps its own mode')
  })
})

test('tightenSecretFileMode only touches a regular file the current user owns', () => {
  // Directories, sockets, fifos and files owned by another account are all
  // "not ours to chmod". Injected lstat so the foreign-owner branch is
  // reachable without a second OS account.
  const chmodded: string[] = []

  const fakeFs = (stat: Record<string, unknown>) =>
    ({
      chmodSync: (filePath: string) => void chmodded.push(filePath),
      lstatSync: () => ({ isFile: () => true, isSymbolicLink: () => false, mode: 0o644, uid: 0, ...stat }),
      renameSync: () => void 0,
      rmSync: () => void 0,
      writeFileSync: () => void 0
    }) as any

  const uid = typeof process.getuid === 'function' ? process.getuid() : 0

  assert.equal(
    tightenSecretFileMode('/x/connection.json', { fs: fakeFs({ isFile: () => false }), platform: 'linux' }),
    false
  )
  assert.equal(
    tightenSecretFileMode('/x/connection.json', { fs: fakeFs({ uid: uid + 1 }), platform: 'linux' }),
    false,
    'a file owned by another user is left alone'
  )
  assert.deepEqual(chmodded, [], 'nothing was chmodded on the rejected paths')

  // The same fs shape, but ours and loose: now it tightens.
  assert.equal(tightenSecretFileMode('/x/connection.json', { fs: fakeFs({ uid }), platform: 'linux' }), true)
  assert.deepEqual(chmodded, ['/x/connection.json'])
})

test('tightenSecretFileMode leaves Windows alone rather than flipping the read-only bit', () => {
  const chmods: string[] = []

  const fakeFs = {
    chmodSync: (filePath: string) => void chmods.push(filePath),
    lstatSync: () => ({
      isFile: () => true,
      isSymbolicLink: () => false,
      mode: 0o644,
      uid: typeof process.getuid === 'function' ? process.getuid() : 0
    }),
    renameSync: () => void 0,
    rmSync: () => void 0,
    writeFileSync: () => void 0
  } as any

  assert.equal(tightenSecretFileMode('C:\\Users\\me\\connection.json', { fs: fakeFs, platform: 'win32' }), true)
  assert.deepEqual(chmods, [], 'no chmod on win32')

  // Same fs, POSIX: the chmod does happen, proving the platform gate is what
  // suppressed it above.
  assert.equal(tightenSecretFileMode('/home/me/connection.json', { fs: fakeFs, platform: 'linux' }), true)
  assert.ok(chmods.includes('/home/me/connection.json'), 'the POSIX path was tightened')
})

test('a token is never persisted in plaintext when safeStorage is unavailable', () => {
  // The defined degradation for `isEncryptionAvailable() === false` (Linux with
  // no keyring): encryptDesktopSecret throws with an actionable message, so the
  // save aborts before any write. It must never fall back to a plaintext
  // payload — the file mode is defense in depth, not a substitute for the
  // keychain.
  const unavailable = {
    isEncryptionAvailable: () => false,
    encryptString: () => Buffer.from('unused', 'utf8')
  }

  assert.throws(
    () => encryptDesktopSecret('tok-live-42', unavailable),
    (error: unknown) => {
      assert.ok(error instanceof Error, 'aborts instead of returning a payload')
      assert.match(String((error as Error).message), /Secure token storage is unavailable/)
      assert.doesNotMatch(String((error as Error).message), /tok-live-42/, 'the secret is not echoed in the error')

      return true
    }
  )

  // And a throwing keychain (available, but encryptString fails) is the same
  // contract — no silent plaintext.
  assert.throws(
    () =>
      encryptDesktopSecret('tok-live-42', {
        isEncryptionAvailable: () => true,
        encryptString: () => {
          throw new Error('keyring locked')
        }
      }),
    /Failed to encrypt the remote gateway token/
  )
})

test('sensitiveFileBlockReason blocks obvious secret file patterns', () => {
  assert.match(String(sensitiveFileBlockReason('/tmp/.env')), /\.env/)
  assert.equal(sensitiveFileBlockReason('/tmp/.env.example'), null)
  assert.match(String(sensitiveFileBlockReason('/Users/me/.ssh/id_ed25519')), /SSH/)
  assert.match(String(sensitiveFileBlockReason('/tmp/server-cert.pem')), /\.pem/)
})

test('path helpers reject blank non-string NUL and Windows device syntax', async () => {
  await rejectsWithCode(resolveReadableFileForIpc('', { purpose: 'File preview' }), 'invalid-path')
  await rejectsWithCode(resolveReadableFileForIpc('   ', { purpose: 'File preview' }), 'invalid-path')
  await rejectsWithCode(resolveReadableFileForIpc(null, { purpose: 'File preview' }), 'invalid-path')
  await rejectsWithCode(resolveReadableFileForIpc(`safe${String.fromCharCode(0)}name.txt`), 'invalid-path')

  const devicePaths = [
    '\\\\?\\C:\\secret.txt',
    '\\\\.\\C:\\secret.txt',
    '\\\\?\\UNC\\server\\share\\secret.txt',
    'GLOBALROOT/Device/HarddiskVolumeShadowCopy1/secret.txt'
  ]

  for (const devicePath of devicePaths) {
    assert.throws(
      () => resolveRequestedPathForIpc(devicePath, { purpose: 'File preview' }),
      (error: any) => {
        assert.equal(error?.code, 'device-path')

        return true
      }
    )
    await rejectsWithCode(resolveReadableFileForIpc(devicePath, { purpose: 'File preview' }), 'device-path')
  }

  assert.throws(
    () => resolveRequestedPathForIpc('file:///%E0%A4%A', { purpose: 'File preview' }),
    (error: any) => {
      assert.equal(error?.code, 'invalid-path')

      return true
    }
  )
  await rejectsWithCode(resolveReadableFileForIpc('file:///%E0%A4%A', { purpose: 'File preview' }), 'invalid-path')
})

test('resolveRequestedPathForIpc resolves relative paths from the trimmed base directory', () => {
  const baseDir = path.join(os.tmpdir(), 'hermes-desktop-base')

  assert.equal(
    resolveRequestedPathForIpc('notes.txt', {
      baseDir: `  ${baseDir}  `,
      purpose: 'File preview'
    }),
    path.resolve(baseDir, 'notes.txt')
  )
})

test('resolveRequestedPathForIpc expands ~ to the home directory', () => {
  assert.equal(resolveRequestedPathForIpc('~', { purpose: 'Directory read' }), path.resolve(os.homedir()))
  assert.equal(
    resolveRequestedPathForIpc('~/www/project', { purpose: 'Directory read' }),
    path.resolve(os.homedir(), 'www/project')
  )
  // `~user` shorthand is NOT expanded — only the caller's own home.
  assert.equal(
    resolveRequestedPathForIpc('~other/secret', { baseDir: os.tmpdir(), purpose: 'Directory read' }),
    path.resolve(os.tmpdir(), '~other/secret')
  )
})

// #95713: chat markdown / @url: wrappers must not leak into the OS open path.
test('stripPathOpenAnnotations drops wrapping markdown and @url: wrappers', () => {
  assert.equal(stripPathOpenAnnotations('**xyz.pdf**'), 'xyz.pdf')
  assert.equal(stripPathOpenAnnotations('__xyz.pdf__'), 'xyz.pdf')
  assert.equal(stripPathOpenAnnotations('C:/docs/xyz.pdf**'), 'C:/docs/xyz.pdf')
  assert.equal(stripPathOpenAnnotations('@url:`C:/docs/xyz.pdf`'), 'C:/docs/xyz.pdf')
  assert.equal(stripPathOpenAnnotations('@url:C:/docs/xyz.pdf'), 'C:/docs/xyz.pdf')
  assert.equal(stripPathOpenAnnotations('report**final.pdf'), 'report**final.pdf')
  // Dunder directories/modules are real paths, not markdown underline.
  assert.equal(stripPathOpenAnnotations('__pycache__'), '__pycache__')
  assert.equal(stripPathOpenAnnotations('__init__'), '__init__')
  assert.equal(stripPathOpenAnnotations('xyz.pdf__'), 'xyz.pdf')
  assert.equal(stripPathOpenAnnotations('file:///C:/docs/xyz.pdf**'), 'file:///C:/docs/xyz.pdf')
  assert.equal(stripPathOpenAnnotations(''), '')
})

test('resolveRequestedPathForIpc strips markdown wrappers before resolving (#95713)', () => {
  const baseDir = path.join(os.tmpdir(), 'hermes-desktop-md-open')
  const clean = resolveRequestedPathForIpc('xyz.pdf', { baseDir, purpose: 'Open external file' })

  assert.equal(
    resolveRequestedPathForIpc('**xyz.pdf**', { baseDir, purpose: 'Open external file' }),
    clean
  )
  assert.equal(
    resolveRequestedPathForIpc('xyz.pdf**', { baseDir, purpose: 'Open external file' }),
    clean
  )

  const abs = path.join(baseDir, 'xyz.pdf')
  const dirtyUrl = `${pathToFileURL(abs).toString()}**`
  assert.equal(
    resolveRequestedPathForIpc(dirtyUrl, { purpose: 'Open external file' }),
    path.resolve(abs)
  )

  assert.throws(
    () => resolveRequestedPathForIpc(['not', 'a', 'path'] as unknown as string, { purpose: 'Open external file' }),
    /file path is required/
  )
})

test('resolveReadableFileForIpc validates existence type size and sensitivity', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-hardening-'))

  try {
    const textPath = path.join(tempDir, 'notes.txt')
    fs.writeFileSync(textPath, 'hello world', 'utf8')

    const fromRelative = await resolveReadableFileForIpc('notes.txt', {
      baseDir: tempDir,
      maxBytes: 256,
      purpose: 'File preview'
    })

    assert.equal(fromRelative.resolvedPath, textPath)
    assert.equal(fromRelative.stat.size, 11)

    const fromFileUrl = await resolveReadableFileForIpc(pathToFileURL(textPath).toString(), {
      purpose: 'File preview'
    })

    assert.equal(fromFileUrl.resolvedPath, textPath)

    const spacedPath = path.join(tempDir, 'notes with spaces.txt')
    fs.writeFileSync(spacedPath, 'space ok', 'utf8')

    const fromSpacedFileUrl = await resolveReadableFileForIpc(pathToFileURL(spacedPath).toString(), {
      purpose: 'File preview'
    })

    assert.equal(fromSpacedFileUrl.resolvedPath, spacedPath)

    await assert.rejects(
      resolveReadableFileForIpc('missing.txt', {
        baseDir: tempDir,
        purpose: 'Text preview'
      }),
      /file does not exist/
    )

    const nestedDir = path.join(tempDir, 'directory')
    fs.mkdirSync(nestedDir)
    await assert.rejects(
      resolveReadableFileForIpc(nestedDir, {
        purpose: 'Text preview'
      }),
      /path points to a directory/
    )

    const largePath = path.join(tempDir, 'large.txt')
    fs.writeFileSync(largePath, 'x'.repeat(40), 'utf8')
    await assert.rejects(
      resolveReadableFileForIpc(largePath, {
        maxBytes: 8,
        purpose: 'File preview'
      }),
      /file is too large/
    )

    const envPath = path.join(tempDir, '.env')
    fs.writeFileSync(envPath, 'SECRET_TOKEN=123', 'utf8')
    await assert.rejects(
      resolveReadableFileForIpc(envPath, {
        purpose: 'File preview'
      }),
      /blocked for sensitive file/
    )

    const envTemplatePath = path.join(tempDir, '.env.example')
    fs.writeFileSync(envTemplatePath, 'EXAMPLE_TOKEN=value', 'utf8')

    const envTemplate = await resolveReadableFileForIpc(envTemplatePath, {
      purpose: 'File preview'
    })

    assert.equal(envTemplate.resolvedPath, envTemplatePath)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveReadableFileForIpc blocks common sensitive files', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-sensitive-'))

  try {
    const sshDir = path.join(tempDir, '.ssh')
    fs.mkdirSync(sshDir)

    const blockedFiles = [
      path.join(tempDir, '.env'),
      path.join(tempDir, '.npmrc'),
      path.join(sshDir, 'id_ed25519'),
      path.join(tempDir, 'cert.pem'),
      path.join(tempDir, 'cert.p12'),
      path.join(tempDir, 'cert.pfx')
    ]

    for (const filePath of blockedFiles) {
      fs.writeFileSync(filePath, 'secret', 'utf8')
      await rejectsWithCode(resolveReadableFileForIpc(filePath, { purpose: 'File preview' }), 'sensitive-file')
    }

    const allowed = path.join(tempDir, '.env.example')
    fs.writeFileSync(allowed, 'EXAMPLE_TOKEN=value', 'utf8')
    assert.equal((await resolveReadableFileForIpc(allowed, { purpose: 'File preview' })).resolvedPath, allowed)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveReadableFileForIpc blocks symlinks whose realpath is sensitive', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-realpath-'))

  try {
    const envPath = path.join(tempDir, '.env')
    const linkPath = path.join(tempDir, 'safe-name.txt')
    fs.writeFileSync(envPath, 'SECRET_TOKEN=123', 'utf8')

    try {
      fs.symlinkSync(envPath, linkPath, 'file')
    } catch (error) {
      if (error?.code === 'EPERM' || error?.code === 'EACCES') {
        // symlink creation is not permitted on this platform — skip
        return
      }

      throw error
    }

    await rejectsWithCode(resolveReadableFileForIpc(linkPath, { purpose: 'File preview' }), 'sensitive-file')
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveDirectoryForIpc accepts directories and rejects invalid directory targets', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-dir-'))

  try {
    const directory = path.join(tempDir, 'project')
    const filePath = path.join(tempDir, 'file.txt')
    fs.mkdirSync(directory)
    fs.writeFileSync(filePath, 'not a directory', 'utf8')

    const resolved = await resolveDirectoryForIpc(directory)
    assert.equal(resolved.resolvedPath, directory)
    assert.equal(resolved.stat.isDirectory(), true)

    await rejectsWithCode(resolveDirectoryForIpc(filePath), 'ENOTDIR')
    await rejectsWithCode(resolveDirectoryForIpc(path.join(tempDir, 'missing')), 'ENOENT')
    await rejectsWithCode(resolveDirectoryForIpc('\\\\?\\C:\\secret'), 'device-path')
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveDirectoryForIpc accepts directory symlinks or junctions', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-dir-link-'))

  try {
    const directory = path.join(tempDir, 'actual-project')
    const linkPath = path.join(tempDir, 'linked-project')
    fs.mkdirSync(directory)

    try {
      fs.symlinkSync(directory, linkPath, process.platform === 'win32' ? 'junction' : 'dir')
    } catch (error) {
      if (error?.code === 'EPERM' || error?.code === 'EACCES') {
        // directory symlink creation is not permitted on this platform — skip
        return
      }

      throw error
    }

    const resolved = await resolveDirectoryForIpc(linkPath)
    assert.equal(resolved.resolvedPath, linkPath)
    assert.equal(resolved.stat.isDirectory(), true)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

// main.ts has no module.exports, so the wiring of the extracted keyring-less
// helpers into the main process follows the repo's source-assertion pattern
// (see windows-hermes-resolution.test.ts). These pin the propagation the PR
// reviewer flagged as untested: the connection-config IPC path forwarding
// allowPlainTextToken through resolvePersistedRemoteToken, and the whenReady
// --password-store=basic startup branch.
const __dirname = path.dirname(fileURLToPath(import.meta.url))

function readMain() {
  return fs.readFileSync(path.join(__dirname, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')
}

test('registry JSON helpers retain native OAuth bearer authentication', () => {
  const source = readMain()
  const postStart = source.indexOf('async function postJsonForBackend(')
  const fetchStart = source.indexOf('async function fetchJsonForBackend(', postStart)
  const helpers = source.slice(postStart, fetchStart)

  assert.notEqual(postStart, -1)
  assert.notEqual(fetchStart, -1)
  assert.match(
    helpers,
    /return fetchJsonForBackend\(descriptor, path, \{ \.\.\.opts, body: body \?\? \{\}, method: 'POST' \}\)/
  )
  assert.match(helpers, /return fetchJsonForBackend\(descriptor, path, opts\)/)
  assert.doesNotMatch(helpers, /fetchJsonViaOauthSession/)
})

test('coerceDesktopConnectionConfig routes token persistence through resolvePersistedRemoteToken', () => {
  const source = readMain()
  const fnStart = source.indexOf('function coerceDesktopConnectionConfig(')
  assert.notEqual(fnStart, -1, 'coerceDesktopConnectionConfig must exist in main.ts')
  const fnEnd = source.indexOf('\nfunction ', fnStart + 1)
  const body = source.slice(fnStart, fnEnd === -1 ? undefined : fnEnd)

  assert.match(
    body,
    /const nextToken = resolvePersistedRemoteToken\(\{/,
    'the persist decision must go through the shared hardening helper'
  )
  // The opt-in must be forwarded RAW (no `=== true` at the call site): the
  // helper owns the strict coercion so it is asserted in exactly one place.
  assert.match(
    body,
    /allowPlainText: input\.allowPlainTextToken\b/,
    'allowPlainTextToken must reach the helper so the IPC opt-in propagates'
  )
  assert.doesNotMatch(
    body,
    /allowPlainText: input\.allowPlainTextToken === true/,
    'the strict coercion must live in the helper, not be duplicated at the call site'
  )
  assert.match(body, /encryptSecret: encryptDesktopSecret\b/, 'the helper must encrypt via encryptDesktopSecret')
})

test('connection-config save and apply IPC handlers route payloads through coerceDesktopConnectionConfig', () => {
  const source = readMain()

  for (const channel of ['hermes:connection-config:save', 'hermes:connection-config:apply']) {
    const handlerStart = source.indexOf(`ipcMain.handle('${channel}'`)
    assert.notEqual(handlerStart, -1, `${channel} handler must exist`)
    const handlerBody = source.slice(handlerStart, handlerStart + 400)
    assert.match(
      handlerBody,
      /coerceDesktopConnectionConfig\(payload(?:, previousConfig)?\)/,
      `${channel} must coerce its payload (the propagation seam) before persisting`
    )
  }
})

test('whenReady enables basic password-store encryption before createWindow', () => {
  const source = readMain()
  const enableIndex = source.indexOf('enableBasicPasswordStoreEncryption({')
  assert.notEqual(enableIndex, -1, 'whenReady must call enableBasicPasswordStoreEncryption')

  const call = source.slice(enableIndex, enableIndex + 240)
  assert.match(call, /platform: process\.platform/, 'the real platform must be forwarded')
  assert.match(
    call,
    /passwordStoreSwitch: app\.commandLine\.getSwitchValue\('password-store'\)/,
    'the real --password-store switch value must be forwarded'
  )
  assert.match(call, /safeStorageApi: safeStorage/, 'the real safeStorage must be forwarded')

  // Ordering matters: the switch must take effect before anything touches
  // safeStorage, so the enable call must precede the first createWindow().
  const createWindowIndex = source.indexOf('createWindow()', enableIndex)
  assert.notEqual(createWindowIndex, -1, 'whenReady must call createWindow after enabling encryption')
  assert.ok(
    enableIndex < createWindowIndex,
    'enableBasicPasswordStoreEncryption must run before createWindow() so the switch is applied first'
  )
})

test('sanitizeDesktopConnectionConfig exposes secureTokenStorage and remoteTokenPlainText', () => {
  const source = readMain()
  const fnStart = source.indexOf('async function sanitizeDesktopConnectionConfig(')
  assert.notEqual(fnStart, -1, 'sanitizeDesktopConnectionConfig must exist in main.ts')
  const fnEnd = source.indexOf('\nfunction ', fnStart + 1)
  const body = source.slice(fnStart, fnEnd === -1 ? undefined : fnEnd)

  const returnIndex = body.indexOf('return {')
  assert.notEqual(returnIndex, -1, 'sanitizeDesktopConnectionConfig must return a sanitized object')
  const returned = body.slice(returnIndex)
  assert.match(returned, /\bsecureTokenStorage\b/, 'the renderer needs the secure-storage availability signal')
  assert.match(returned, /\bremoteTokenPlainText\b/, 'the renderer needs the plain-text token signal')
})

// #95393: connections.save succeeded but the switcher menu (renderer
// $connectionsRegistry snapshot) never refreshed until reload. The registry
// push (broadcastConnectionsChanged) fired only on the dial-material-edit
// branch, so a brand-new connection or a label rename never reached other
// windows — or the switcher's onChanged re-pull. Mirrors the live repro at
// /tmp/mg-ab/w2_95393.py: save → menu (no reload) must include the new row.
test('saveRegistryConnection republishes the registry to renderers on EVERY successful save (#95393)', () => {
  const source = readMain()
  const fnStart = source.indexOf('async function saveRegistryConnection(')
  assert.notEqual(fnStart, -1, 'saveRegistryConnection must exist in main.ts')
  const fnEnd = source.indexOf('\nasync function ', fnStart + 1)
  const body = source.slice(fnStart, fnEnd === -1 ? undefined : fnEnd)

  // The dial-material edit branch keeps its dispose+redial semantics…
  assert.match(
    body,
    /broadcastConnectionsChanged\(\{ connectionId: entry\.id, reason: 'updated' \}\)/,
    'a dial-material edit must still push the dispose+redial signal'
  )
  // …and every OTHER save (new connection, label rename) must still push a
  // registry refresh, or the switcher menu paints stale until reload.
  assert.match(
    body,
    /broadcastConnectionsChanged\(\{ connectionId: entry\.id, reason: 'saved' \}\)/,
    'a non-dial-material save must republish the registry snapshot (#95393)'
  )
})
