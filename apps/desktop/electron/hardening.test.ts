import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

import { test } from 'vitest'

import {
  DEFAULT_FETCH_TIMEOUT_MS,
  enableBasicPasswordStoreEncryption,
  encryptDesktopSecret,
  resolveDirectoryForIpc,
  resolvePersistedRemoteToken,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc,
  resolveTimeoutMs,
  sensitiveFileBlockReason
} from './hardening'

async function rejectsWithCode(promise, code: string) {
  await assert.rejects(promise, (error: any) => {
    assert.equal(error?.code, code)

    return true
  })
}

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

  assert.deepEqual(secret, {
    encoding: 'safeStorage',
    value: Buffer.from('enc:token-123', 'utf8').toString('base64')
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
      /coerceDesktopConnectionConfig\(payload\)/,
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
