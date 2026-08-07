import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { rejectSensitiveFilePath, resolveExistingPathForIpc } from './hardening'

// ---------------------------------------------------------------------------
// resolveExistingPathForIpc — the validation pipeline now used by
// hermes:fs:trash and hermes:fs:rename. Tests mirror the existing
// resolveReadableFileForIpc tests in hardening.test.ts but target the
// delete/rename security contract: files and directories must both be
// acceptable (no isFile/isDirectory restriction), but the same
// sensitive-file + syntax + symlink traps must be enforced.
// ---------------------------------------------------------------------------

async function rejectsWithCode(promise, code: string) {
  await assert.rejects(promise, (error: any) => {
    assert.equal(error?.code, code)

    return true
  })
}

test('resolveExistingPathForIpc accepts existing files', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-file-'))

  try {
    const filePath = path.join(tempDir, 'notes.txt')
    fs.writeFileSync(filePath, 'hello', 'utf8')

    const result = await resolveExistingPathForIpc(filePath, { purpose: 'Delete file' })
    assert.equal(result.resolvedPath, filePath)
    assert.equal(result.stat.isFile(), true)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc accepts existing directories', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-dir-'))

  try {
    const subDir = path.join(tempDir, 'subfolder')
    fs.mkdirSync(subDir)

    const result = await resolveExistingPathForIpc(subDir, { purpose: 'Delete folder' })
    assert.equal(result.resolvedPath, subDir)
    assert.equal(result.stat.isDirectory(), true)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc rejects non-existent paths with ENOENT', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-missing-'))

  try {
    await rejectsWithCode(
      resolveExistingPathForIpc(path.join(tempDir, 'nope.txt'), { purpose: 'Delete file' }),
      'ENOENT'
    )
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc rejects blank paths with invalid-path', async () => {
  await rejectsWithCode(resolveExistingPathForIpc('', { purpose: 'Delete file' }), 'invalid-path')
  await rejectsWithCode(resolveExistingPathForIpc('   ', { purpose: 'Delete file' }), 'invalid-path')
  await rejectsWithCode(resolveExistingPathForIpc(null as any, { purpose: 'Delete file' }), 'invalid-path')
})

test('resolveExistingPathForIpc rejects NUL bytes in path', async () => {
  await rejectsWithCode(
    resolveExistingPathForIpc(`safe${String.fromCharCode(0)}name.txt`, { purpose: 'Delete file' }),
    'invalid-path'
  )
})

test('resolveExistingPathForIpc rejects Windows device paths', async () => {
  const devicePaths = [
    '\\\\?\\C:\\secret.txt',
    '\\\\.\\C:\\secret.txt',
    '\\\\?\\UNC\\server\\share\\secret.txt',
    'GLOBALROOT/Device/HarddiskVolumeShadowCopy1/secret.txt'
  ]

  for (const devicePath of devicePaths) {
    await rejectsWithCode(resolveExistingPathForIpc(devicePath, { purpose: 'Delete file' }), 'device-path')
  }
})

test('resolveExistingPathForIpc blocks sensitive files (.env, .ssh, .pem, etc.)', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-sensitive-'))

  try {
    const sshDir = path.join(tempDir, '.ssh')
    fs.mkdirSync(sshDir)

    const blockedFiles = [
      path.join(tempDir, '.env'),
      path.join(tempDir, '.npmrc'),
      path.join(sshDir, 'id_ed25519'),
      path.join(tempDir, 'cert.pem'),
      path.join(tempDir, 'cert.p12'),
      path.join(tempDir, 'cert.pfx'),
      path.join(tempDir, '.netrc'),
      path.join(tempDir, '.pypirc')
    ]

    for (const filePath of blockedFiles) {
      fs.writeFileSync(filePath, 'secret', 'utf8')
      await rejectsWithCode(resolveExistingPathForIpc(filePath, { purpose: 'Delete file' }), 'sensitive-file')
    }
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc blocks case-variant sensitive files (.ENV, .Env)', async () => {
  // sensitiveFileBlockReason lowercases the path before matching, so
  // case variants must also be blocked. This matters on Windows where
  // the filesystem is case-insensitive.
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-case-'))

  try {
    const variants = ['.ENV', '.Env', '.env.local', 'cert.PEM']

    for (const name of variants) {
      const variantPath = path.join(tempDir, name)
      fs.writeFileSync(variantPath, 'secret', 'utf8')
      await rejectsWithCode(resolveExistingPathForIpc(variantPath, { purpose: 'Delete file' }), 'sensitive-file')
    }
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc allows safe env template files', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-env-safe-'))

  try {
    const envExample = path.join(tempDir, '.env.example')
    fs.writeFileSync(envExample, 'EXAMPLE=value', 'utf8')

    const result = await resolveExistingPathForIpc(envExample, { purpose: 'Delete file' })
    assert.equal(result.resolvedPath, envExample)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc blocks symlink to sensitive file (DI fake fs)', async () => {
  // Use dependency injection (options.fs) to deterministically exercise the
  // symlink realpath trap without depending on platform symlink permissions.
  // A real symlink whose realpath resolves to .env must be blocked even
  // though the link name looks innocent.
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-di-link-'))

  try {
    const envPath = path.join(tempDir, '.env')
    const linkPath = path.join(tempDir, 'safe-name.txt')
    fs.writeFileSync(envPath, 'SECRET=123', 'utf8')

    // Fake fs: lstat succeeds (entry exists), realpath resolves the link
    // to .env — triggering the sensitive-file block on realPath.
    const fakeFs = {
      constants: fs.constants,
      promises: {
        lstat: async (p: string) => fs.promises.lstat(p),
        stat: async (p: string) => fs.promises.stat(p),
        realpath: async (p: string) => envPath, // link → .env target
        access: async (p: string, mode?: number) => fs.promises.access(p, mode),
        readFile: async (p: string) => fs.promises.readFile(p)
      }
    }

    // Create a real file at linkPath so stat succeeds; we control what
    // realpath returns via the fake fs.
    fs.writeFileSync(linkPath, 'link-content', 'utf8')

    await rejectsWithCode(
      resolveExistingPathForIpc(linkPath, {
        purpose: 'Delete file',
        fs: fakeFs as any
      }),
      'sensitive-file'
    )
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc accepts non-sensitive symlinks (DI fake fs)', async () => {
  // A symlink whose realpath is NOT a sensitive file should resolve normally.
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-di-safe-link-'))

  try {
    const realTarget = path.join(tempDir, 'real-target.txt')
    const linkPath = path.join(tempDir, 'convenient-link.txt')
    fs.writeFileSync(realTarget, 'data', 'utf8')
    fs.writeFileSync(linkPath, 'link-content', 'utf8')

    const fakeFs = {
      constants: fs.constants,
      promises: {
        lstat: async (p: string) => fs.promises.lstat(p),
        stat: async (p: string) => fs.promises.stat(p),
        realpath: async (p: string) => realTarget, // link → safe target
        access: async (p: string, mode?: number) => fs.promises.access(p, mode),
        readFile: async (p: string) => fs.promises.readFile(p)
      }
    }

    const result = await resolveExistingPathForIpc(linkPath, {
      purpose: 'Delete file',
      fs: fakeFs as any
    })

    assert.equal(result.realPath, realTarget)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc blocks real symlink to sensitive file when permitted', async t => {
  // When the OS permits symlink creation, verify the symlink→sensitive
  // block end-to-end with the real fs. On platforms without symlink
  // permissions (Windows non-Developer-Mode), skip explicitly.
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-real-symlink-'))

  try {
    const envPath = path.join(tempDir, '.env')
    const linkPath = path.join(tempDir, 'safe-name.txt')
    fs.writeFileSync(envPath, 'SECRET=123', 'utf8')

    try {
      fs.symlinkSync(envPath, linkPath, 'file')
    } catch (error) {
      if (error?.code === 'EPERM' || error?.code === 'EACCES') {
        t.skip('symlink creation not permitted on this platform')

        return
      }

      throw error
    }

    await rejectsWithCode(resolveExistingPathForIpc(linkPath, { purpose: 'Delete file' }), 'sensitive-file')
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc allows dangling symlinks (lstat, not stat)', async () => {
  // A dangling symlink — whose target was deleted — is still a valid
  // rename/trash target. lstat (not stat) must be used so the existence
  // check succeeds for the link entry itself.
  // Uses DI fake-fs so it works on platforms without symlink permissions.
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-dangling-'))

  try {
    const linkPath = path.join(tempDir, 'dangling-link.txt')

    // Create a real file so lstat succeeds, then use a fake fs where
    // realpath rejects with ENOENT (simulating a dangling symlink).
    fs.writeFileSync(linkPath, 'link-content', 'utf8')

    const fakeFs = {
      constants: fs.constants,
      promises: {
        lstat: async (p: string) => fs.promises.lstat(p),
        stat: async (p: string) => fs.promises.stat(p),
        realpath: async (_p: string) => {
          const err: any = new Error('ENOENT: no such file or directory')
          err.code = 'ENOENT'

          throw err
        },
        access: async (p: string, mode?: number) => fs.promises.access(p, mode),
        readFile: async (p: string) => fs.promises.readFile(p)
      }
    }

    // The dangling link must NOT be rejected — it's a valid entry to
    // rename/trash. The realpath ENOENT is caught and skipped.
    const result = await resolveExistingPathForIpc(linkPath, {
      purpose: 'Delete file',
      fs: fakeFs as any
    })

    assert.equal(result.resolvedPath, linkPath)
    assert.equal(result.realPath, linkPath) // realpath failed → resolvedPath returned
    assert.equal(result.stat.isFile(), true) // lstat sees the link entry
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc resolves ~ to home directory', async () => {
  await rejectsWithCode(
    resolveExistingPathForIpc('~/hermes-trash-nonexistent-test-file', { purpose: 'Delete file' }),
    'ENOENT'
  )
})

test('resolveExistingPathForIpc resolves relative paths from baseDir', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-basedir-'))

  try {
    const filePath = path.join(tempDir, 'relative.txt')
    fs.writeFileSync(filePath, 'content', 'utf8')

    const result = await resolveExistingPathForIpc('relative.txt', {
      baseDir: tempDir,
      purpose: 'Delete file'
    })

    assert.equal(result.resolvedPath, filePath)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc accepts file URLs', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-fileurl-'))

  try {
    const filePath = path.join(tempDir, 'url-target.txt')
    fs.writeFileSync(filePath, 'data', 'utf8')

    const fileUrl = new URL(`file://${filePath.replace(/\\/g, '/')}`).toString()
    const result = await resolveExistingPathForIpc(fileUrl, { purpose: 'Delete file' })
    assert.equal(result.resolvedPath, filePath)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

test('resolveExistingPathForIpc allows blockSensitive=false override', async () => {
  // This mirrors the same option on resolveReadableFileForIpc — it's an
  // API-consistency feature, not dead code. The default is always true
  // in production IPC handlers.
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-noblock-'))

  try {
    const envPath = path.join(tempDir, '.env')
    fs.writeFileSync(envPath, 'SECRET=123', 'utf8')

    const result = await resolveExistingPathForIpc(envPath, {
      purpose: 'Delete file',
      blockSensitive: false
    })

    assert.equal(result.resolvedPath, envPath)
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

// ---------------------------------------------------------------------------
// Path traversal — resolveRequestedPathForIpc uses path.resolve which
// normalizes .. sequences, so a traversal attempt resolves to a real
// (parent) directory and is then checked by sensitiveFileBlockReason.
// ---------------------------------------------------------------------------

test('resolveExistingPathForIpc resolves .. traversal and applies sensitive-file check', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-trash-traversal-'))
  const subDir = path.join(tempDir, 'subdir')
  const sshDir = path.join(tempDir, '.ssh')

  try {
    fs.mkdirSync(subDir, { recursive: true })
    fs.mkdirSync(sshDir, { recursive: true })
    fs.writeFileSync(path.join(sshDir, 'id_rsa'), 'key', 'utf8')

    // From subdir, ../ reaches parent. The resolved path should be
    // under .ssh, which is then blocked by sensitiveFileBlockReason.
    await rejectsWithCode(
      resolveExistingPathForIpc('../.ssh/id_rsa', {
        baseDir: subDir,
        purpose: 'Delete file'
      }),
      'sensitive-file'
    )
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true })
  }
})

// ---------------------------------------------------------------------------
// rejectSensitiveFilePath — used directly by fs:rename to block destination
// names that would create sensitive files (e.g. renaming to ".env").
// ---------------------------------------------------------------------------

test('rejectSensitiveFilePath blocks destination name ".env"', () => {
  assert.throws(
    () => rejectSensitiveFilePath(path.join(os.tmpdir(), '.env'), 'Rename file'),
    (error: any) => {
      assert.equal(error?.code, 'sensitive-file')

      return true
    }
  )
})

test('rejectSensitiveFilePath blocks destination name "id_ed25519"', () => {
  assert.throws(
    () => rejectSensitiveFilePath(path.join(os.tmpdir(), 'id_ed25519'), 'Rename file'),
    (error: any) => {
      assert.equal(error?.code, 'sensitive-file')

      return true
    }
  )
})

test('rejectSensitiveFilePath blocks case-variant destination names', () => {
  // sensitiveFileBlockReason lowercases before matching
  for (const name of ['.ENV', '.Env', '.ENV.LOCAL']) {
    assert.throws(
      () => rejectSensitiveFilePath(path.join(os.tmpdir(), name), 'Rename file'),
      (error: any) => {
        assert.equal(error?.code, 'sensitive-file')

        return true
      }
    )
  }
})

test('rejectSensitiveFilePath allows safe destination names', () => {
  rejectSensitiveFilePath(path.join(os.tmpdir(), 'notes.txt'), 'Rename file')
  rejectSensitiveFilePath(path.join(os.tmpdir(), 'config.json'), 'Rename file')
  rejectSensitiveFilePath(path.join(os.tmpdir(), '.env.example'), 'Rename file')
})

test('rejectSensitiveFilePath allows .env.example, .env.template suffixes', () => {
  rejectSensitiveFilePath(path.join(os.tmpdir(), '.env.example'), 'Rename file')
  rejectSensitiveFilePath(path.join(os.tmpdir(), '.env.sample'), 'Rename file')
  rejectSensitiveFilePath(path.join(os.tmpdir(), '.env.template'), 'Rename file')
})
