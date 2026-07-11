import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { pathToFileURL } from 'node:url'

import {
  createFileSnapshotForIpc,
  DEFAULT_FETCH_TIMEOUT_MS,
  encryptDesktopSecret,
  normalizeFileChunkReadRange,
  readFileChunkForIpc,
  releaseFileSnapshotForIpc,
  resolveDirectoryForIpc,
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

test('normalizeFileChunkReadRange accepts safe integers and rejects unsafe ranges', () => {
  assert.deepEqual(normalizeFileChunkReadRange(0, 512), { offset: 0, maxBytes: 512 })
  assert.deepEqual(normalizeFileChunkReadRange(7, undefined), { offset: 7, maxBytes: undefined })

  for (const offset of [-1, 1.5, Number.MAX_SAFE_INTEGER + 1, Number.NaN]) {
    assert.throws(() => normalizeFileChunkReadRange(offset, 512), /offset must be a non-negative safe integer/)
  }
  for (const maxBytes of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, Number.NaN]) {
    assert.throws(() => normalizeFileChunkReadRange(0, maxBytes), /maxBytes must be a positive safe integer/)
  }
})

test('readFileChunkForIpc returns bounded bytes and stable file metadata', async t => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-chunk-'))
  t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }))
  const filePath = path.join(tempDir, 'notes.txt')
  fs.writeFileSync(filePath, 'hello world', 'utf8')

  const chunk = await readFileChunkForIpc(filePath, 2, 4)

  assert.equal(chunk.buffer.toString('utf8'), 'llo ')
  assert.equal(chunk.byteSize, 11)
  assert.equal(chunk.bytesRead, 4)
  assert.equal(chunk.done, false)
  const sourceStat = fs.statSync(filePath)
  const realStat = fs.statSync(chunk.realPath)
  assert.equal(`${realStat.dev}:${realStat.ino}`, `${sourceStat.dev}:${sourceStat.ino}`)
  assert.equal(typeof chunk.fileId, 'string')
  assert.ok(chunk.fileId.length > 0)
  assert.equal(Number.isFinite(chunk.mtimeMs), true)
})

test('readFileChunkForIpc opens the validated realpath after a symlink swap', async t => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-symlink-swap-'))
  t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }))
  const safePath = path.join(tempDir, 'safe.txt')
  const otherPath = path.join(tempDir, 'other.txt')
  const linkPath = path.join(tempDir, 'selected.txt')
  fs.writeFileSync(safePath, 'SAFE', 'utf8')
  fs.writeFileSync(otherPath, 'OTHER', 'utf8')

  try {
    fs.symlinkSync(safePath, linkPath, 'file')
  } catch (error) {
    const code = error && typeof error === 'object' ? error.code : ''
    if (code === 'EPERM' || code === 'EACCES' || code === 'ENOTSUP') {
      t.skip(`file symlinks unavailable on this host (${code})`)
      return
    }
    throw error
  }

  const originalRealpath = fs.promises.realpath
  let swapped = false
  fs.promises.realpath = (async (requestedPath: fs.PathLike) => {
    const result = await originalRealpath(requestedPath)
    if (!swapped && path.resolve(String(requestedPath)) === path.resolve(linkPath)) {
      swapped = true
      fs.unlinkSync(linkPath)
      fs.symlinkSync(otherPath, linkPath, 'file')
    }
    return result
  }) as typeof fs.promises.realpath
  t.after(() => {
    fs.promises.realpath = originalRealpath
  })

  const chunk = await readFileChunkForIpc(linkPath, 0, 4)
  assert.equal(chunk.buffer.toString('utf8'), 'SAFE')
})

test('file upload snapshot remains immutable after same-inode same-size source rewrite with restored mtime', async t => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-snapshot-'))
  t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }))
  const snapshotRoot = path.join(tempDir, 'snapshots')
  const sourcePath = path.join(tempDir, 'selected.bin')
  fs.writeFileSync(sourcePath, 'ABCDEFGH', 'utf8')
  const sourceStat = fs.statSync(sourcePath)

  const snapshot = await createFileSnapshotForIpc(sourcePath, snapshotRoot)

  fs.writeFileSync(sourcePath, '12345678', 'utf8')
  fs.utimesSync(sourcePath, sourceStat.atime, sourceStat.mtime)
  const first = await readFileChunkForIpc(snapshot.path, 0, 4)
  const second = await readFileChunkForIpc(snapshot.path, 4, 4)

  assert.equal(first.buffer.toString('utf8') + second.buffer.toString('utf8'), 'ABCDEFGH')
  assert.equal(snapshot.byteSize, 8)
  assert.equal(await releaseFileSnapshotForIpc(snapshot.path, snapshotRoot), true)
  assert.equal(fs.existsSync(snapshot.path), false)
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

test('resolveReadableFileForIpc validates existence type size and sensitivity', async t => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-hardening-'))
  t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }))

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
})

test('resolveReadableFileForIpc blocks common sensitive files', async t => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-sensitive-'))
  t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }))

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
})

test('resolveReadableFileForIpc blocks symlinks whose realpath is sensitive', async t => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-realpath-'))
  t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }))

  const envPath = path.join(tempDir, '.env')
  const linkPath = path.join(tempDir, 'safe-name.txt')
  fs.writeFileSync(envPath, 'SECRET_TOKEN=123', 'utf8')

  try {
    fs.symlinkSync(envPath, linkPath, 'file')
  } catch (error) {
    if (error?.code === 'EPERM' || error?.code === 'EACCES') {
      t.skip(`symlink creation is not permitted on this platform (${error.code})`)

      return
    }

    throw error
  }

  await rejectsWithCode(resolveReadableFileForIpc(linkPath, { purpose: 'File preview' }), 'sensitive-file')
})

test('resolveDirectoryForIpc accepts directories and rejects invalid directory targets', async t => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-dir-'))
  t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }))

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
})

test('resolveDirectoryForIpc accepts directory symlinks or junctions', async t => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-dir-link-'))
  t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }))

  const directory = path.join(tempDir, 'actual-project')
  const linkPath = path.join(tempDir, 'linked-project')
  fs.mkdirSync(directory)

  try {
    fs.symlinkSync(directory, linkPath, process.platform === 'win32' ? 'junction' : 'dir')
  } catch (error) {
    if (error?.code === 'EPERM' || error?.code === 'EACCES') {
      t.skip(`directory symlink creation is not permitted on this platform (${error.code})`)

      return
    }

    throw error
  }

  const resolved = await resolveDirectoryForIpc(linkPath)
  assert.equal(resolved.resolvedPath, linkPath)
  assert.equal(resolved.stat.isDirectory(), true)
})
