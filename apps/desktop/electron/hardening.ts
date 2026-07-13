import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const DEFAULT_FETCH_TIMEOUT_MS = 15_000
const DATA_URL_READ_MAX_BYTES = 16 * 1024 * 1024
const FILE_CHUNK_READ_MAX_BYTES = 8 * 1024 * 1024
const FILE_UPLOAD_SNAPSHOT_MAX_BYTES = 8 * 1024 * 1024 * 1024
const TEXT_PREVIEW_SOURCE_MAX_BYTES = 64 * 1024 * 1024

const SAFE_ENV_SUFFIXES = new Set(['dist', 'example', 'sample', 'template'])
const SENSITIVE_EXTENSIONS = new Set(['.kdbx', '.p12', '.pem', '.pfx'])

function resolveTimeoutMs(timeoutMs, fallbackMs = DEFAULT_FETCH_TIMEOUT_MS) {
  const fallback =
    Number.isFinite(fallbackMs) && Number(fallbackMs) > 0 ? Math.round(Number(fallbackMs)) : DEFAULT_FETCH_TIMEOUT_MS

  const parsed = Number(timeoutMs)

  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.round(parsed)
  }

  return fallback
}

function normalizeFileChunkReadRange(offset, maxBytes) {
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw ipcPathError('EINVAL', 'File attachment upload failed: offset must be a non-negative safe integer.')
  }
  if (maxBytes !== undefined && (!Number.isSafeInteger(maxBytes) || maxBytes <= 0)) {
    throw ipcPathError('EINVAL', 'File attachment upload failed: maxBytes must be a positive safe integer.')
  }

  return { maxBytes, offset }
}

function encryptDesktopSecret(value, safeStorageApi) {
  const raw = String(value || '')

  if (!raw) {
    return null
  }

  let encryptionAvailable = false

  try {
    encryptionAvailable = Boolean(safeStorageApi?.isEncryptionAvailable?.())
  } catch {
    encryptionAvailable = false
  }

  if (!encryptionAvailable) {
    throw new Error(
      'Secure token storage is unavailable, so Hermes Desktop cannot save remote gateway tokens. ' +
        'Set HERMES_DESKTOP_REMOTE_URL and HERMES_DESKTOP_REMOTE_TOKEN in your environment, or enable OS keychain access and try again.'
    )
  }

  try {
    return {
      encoding: 'safeStorage',
      value: safeStorageApi.encryptString(raw).toString('base64')
    }
  } catch (error) {
    const detail = error instanceof Error && error.message ? ` (${error.message})` : ''
    throw new Error(
      `Failed to encrypt the remote gateway token for secure storage${detail}. ` +
        'Set HERMES_DESKTOP_REMOTE_URL and HERMES_DESKTOP_REMOTE_TOKEN in your environment as a fallback.'
    )
  }
}

function sensitiveFileBlockReason(filePath) {
  const normalized = String(filePath || '')
    .replace(/\\/g, '/')
    .toLowerCase()

  const basename = path.basename(normalized)
  const ext = path.extname(basename)

  if (!basename) {
    return null
  }

  if (normalized.includes('/.ssh/')) {
    return 'SSH key/config files are blocked.'
  }

  if (normalized.includes('/.gnupg/')) {
    return 'GPG key material is blocked.'
  }

  if (normalized.endsWith('/.aws/credentials')) {
    return 'AWS credential files are blocked.'
  }

  if (basename === '.env') {
    return '.env files are blocked because they commonly contain secrets.'
  }

  if (basename.startsWith('.env.')) {
    const suffix = basename.slice('.env.'.length)

    if (!SAFE_ENV_SUFFIXES.has(suffix)) {
      return `${basename} is blocked because it appears to contain environment secrets.`
    }
  }

  if (/^id_(rsa|dsa|ecdsa|ed25519)(?:\..+)?$/.test(basename) && !basename.endsWith('.pub')) {
    return 'SSH private key files are blocked.'
  }

  if (SENSITIVE_EXTENSIONS.has(ext)) {
    return `${ext} key/certificate files are blocked.`
  }

  if (basename === '.npmrc' || basename === '.netrc' || basename === '.pypirc') {
    return `${basename} is blocked because it may include auth credentials.`
  }

  return null
}

function ipcPathError(code: any, message: string): Error & { code: any } {
  const error = new Error(message) as Error & { code: any }
  ;(error as any).code = code

  return error
}

function rejectUnsafePathSyntax(filePath, purpose = 'File read') {
  if (typeof filePath !== 'string') {
    throw ipcPathError('invalid-path', `${purpose} failed: file path is required.`)
  }

  const raw = filePath.trim()

  if (!raw) {
    throw ipcPathError('invalid-path', `${purpose} failed: file path is required.`)
  }

  if (raw.includes('\0')) {
    throw ipcPathError('invalid-path', `${purpose} failed: file path is invalid.`)
  }

  const normalized = raw.replace(/\\/g, '/').toLowerCase()

  if (
    normalized.startsWith('//?/') ||
    normalized.startsWith('//./') ||
    normalized.startsWith('globalroot/device/') ||
    normalized.includes('/globalroot/device/')
  ) {
    throw ipcPathError('device-path', `${purpose} blocked: Windows device paths are not allowed.`)
  }

  return raw
}

function resolveRequestedPathForIpc(filePath, options: { purpose?: string; baseDir?: fs.PathOrFileDescriptor } = {}) {
  const purpose = String(options.purpose || 'File read')
  let raw = rejectUnsafePathSyntax(filePath, purpose)

  // Gateway-reported cwds (config `terminal.cwd`, remote sessions) routinely
  // arrive as `~/...`. Node's fs has no shell — without expansion the path
  // resolves under process.cwd() and every read "ENOENT"s forever.
  if (raw === '~' || raw.startsWith('~/') || raw.startsWith('~\\')) {
    raw = path.join(os.homedir(), raw.slice(1))
  }

  if (/^file:/i.test(raw)) {
    let resolvedPath

    try {
      const parsed = new URL(raw)

      if (parsed.protocol !== 'file:') {
        throw new Error('not a file URL')
      }

      resolvedPath = fileURLToPath(parsed)
    } catch {
      throw ipcPathError('invalid-path', `${purpose} failed: file URL is invalid.`)
    }

    rejectUnsafePathSyntax(resolvedPath, purpose)

    return path.resolve(resolvedPath)
  }

  const baseInput = typeof options.baseDir === 'string' && options.baseDir.trim() ? options.baseDir : process.cwd()
  const safeBaseInput = rejectUnsafePathSyntax(baseInput, purpose)
  const resolvedBase = path.resolve(safeBaseInput)
  rejectUnsafePathSyntax(resolvedBase, purpose)
  const resolvedPath = path.resolve(resolvedBase, raw)
  rejectUnsafePathSyntax(resolvedPath, purpose)

  return resolvedPath
}

async function statForIpc(fsImpl: { promises: { stat: typeof fs.promises.stat } }, resolvedPath, purpose, typeLabel) {
  try {
    return await fsImpl.promises.stat(resolvedPath)
  } catch (error) {
    const code = error && typeof error === 'object' ? error.code : ''

    if (code === 'ENOENT' || code === 'ENOTDIR') {
      throw ipcPathError(code || 'ENOENT', `${purpose} failed: ${typeLabel} does not exist.`)
    }

    throw ipcPathError(
      code || 'read-error',
      `${purpose} failed: ${error instanceof Error ? error.message : String(error)}`
    )
  }
}

async function realpathForIpc(fsImpl, resolvedPath, purpose) {
  if (typeof fsImpl.promises.realpath !== 'function') {
    return resolvedPath
  }

  try {
    const realPath = await fsImpl.promises.realpath(resolvedPath)
    rejectUnsafePathSyntax(realPath, purpose)

    return realPath
  } catch (error) {
    const code = error && typeof error === 'object' ? error.code : ''
    throw ipcPathError(
      code || 'read-error',
      `${purpose} failed: ${error instanceof Error ? error.message : String(error)}`
    )
  }
}

function rejectSensitiveFilePath(filePath, purpose) {
  const blockReason = sensitiveFileBlockReason(filePath)

  if (blockReason) {
    throw ipcPathError('sensitive-file', `${purpose} blocked for sensitive file: ${blockReason}`)
  }
}

async function resolveDirectoryForIpc(
  dirPath,
  options: {
    purpose?: string
    baseDir?: fs.PathOrFileDescriptor
    fs?: { promises: { stat: typeof fs.promises.stat } }
  } = {}
) {
  const purpose = String(options.purpose || 'Directory read')
  const fsImpl = options.fs || fs
  const resolvedPath = resolveRequestedPathForIpc(dirPath, { baseDir: options.baseDir, purpose })
  const stat = await statForIpc(fsImpl, resolvedPath, purpose, 'directory')

  if (!stat.isDirectory()) {
    throw ipcPathError('ENOTDIR', `${purpose} failed: path is not a directory.`)
  }

  const realPath = await realpathForIpc(fsImpl, resolvedPath, purpose)

  return { realPath, resolvedPath, stat }
}

async function resolveReadableFileForIpc(
  filePath,
  options: {
    purpose?: string
    baseDir?: fs.PathOrFileDescriptor
    fs?: typeof fs
    blockSensitive?: boolean
    maxBytes?: number
  } = {}
) {
  const purpose = String(options.purpose || 'File read')
  const fsImpl = options.fs || fs
  const resolvedPath = resolveRequestedPathForIpc(filePath, { baseDir: options.baseDir, purpose })

  if (options.blockSensitive !== false) {
    rejectSensitiveFilePath(resolvedPath, purpose)
  }

  const stat = await statForIpc(fsImpl, resolvedPath, purpose, 'file')

  if (stat.isDirectory()) {
    throw ipcPathError('EISDIR', `${purpose} failed: path points to a directory.`)
  }

  if (!stat.isFile()) {
    throw ipcPathError('EINVAL', `${purpose} failed: only regular files can be read.`)
  }

  const realPath = await realpathForIpc(fsImpl, resolvedPath, purpose)

  if (options.blockSensitive !== false) {
    rejectSensitiveFilePath(realPath, purpose)
  }

  const maxBytes = Number.isFinite(options.maxBytes) && Number(options.maxBytes) > 0 ? Number(options.maxBytes) : null

  if (maxBytes && stat.size > maxBytes) {
    throw ipcPathError('EFBIG', `${purpose} failed: file is too large (${stat.size} bytes; limit ${maxBytes} bytes).`)
  }

  try {
    await fsImpl.promises.access(resolvedPath, fs.constants.R_OK)
  } catch {
    throw ipcPathError('EACCES', `${purpose} failed: file is not readable.`)
  }

  return { realPath, resolvedPath, stat }
}

async function readFileChunkForIpc(filePath, offset, maxBytes = FILE_CHUNK_READ_MAX_BYTES) {
  const range = normalizeFileChunkReadRange(offset, maxBytes)
  const readLimit = Math.min(range.maxBytes ?? FILE_CHUNK_READ_MAX_BYTES, FILE_CHUNK_READ_MAX_BYTES)
  const { realPath } = await resolveReadableFileForIpc(filePath, {
    purpose: 'File attachment upload'
  })
  const handle = await fs.promises.open(realPath, 'r')

  try {
    const before = await handle.stat()

    if (!before.isFile()) {
      throw ipcPathError('EINVAL', 'File attachment upload failed: only regular files can be read.')
    }
    if (!Number.isSafeInteger(before.size)) {
      throw ipcPathError('EFBIG', 'File attachment upload failed: file size exceeds the safe integer range.')
    }

    const remaining = Math.max(0, before.size - range.offset)
    const requested = Math.min(readLimit, remaining)
    const buffer = Buffer.alloc(requested)
    const { bytesRead } = requested > 0 ? await handle.read(buffer, 0, requested, range.offset) : { bytesRead: 0 }
    const after = await handle.stat()
    const sameFile = before.dev === after.dev && before.ino === after.ino
    const unchanged = sameFile && before.size === after.size && before.mtimeMs === after.mtimeMs

    if (!unchanged) {
      throw ipcPathError('ESTALE', 'File attachment upload failed: the file changed while it was being read.')
    }

    return {
      buffer: buffer.subarray(0, bytesRead),
      byteSize: after.size,
      bytesRead,
      done: range.offset + bytesRead >= after.size,
      fileId: `${after.dev}:${after.ino}`,
      mtimeMs: after.mtimeMs,
      realPath
    }
  } finally {
    await handle.close()
  }
}

function fileSnapshotIdentity(stat) {
  return `${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeMs}:${stat.ctimeMs}`
}

async function resolveFileSnapshotRoot(snapshotRoot) {
  const root = path.resolve(String(snapshotRoot || ''))
  await fs.promises.mkdir(root, { mode: 0o700, recursive: true })
  const rootStat = await fs.promises.lstat(root)

  if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) {
    throw ipcPathError('EINVAL', 'File attachment snapshot failed: unsafe snapshot directory.')
  }

  return root
}

async function createFileSnapshotForIpc(filePath, snapshotRoot) {
  const root = await resolveFileSnapshotRoot(snapshotRoot)
  const { realPath } = await resolveReadableFileForIpc(filePath, {
    maxBytes: FILE_UPLOAD_SNAPSHOT_MAX_BYTES,
    purpose: 'File attachment snapshot'
  })
  const snapshotPath = path.join(root, `${randomUUID()}.snapshot`)
  let sourceHandle
  let snapshotHandle

  try {
    sourceHandle = await fs.promises.open(realPath, 'r')
    snapshotHandle = await fs.promises.open(snapshotPath, 'wx', 0o600)
    const before = await sourceHandle.stat()

    if (!before.isFile() || !Number.isSafeInteger(before.size)) {
      throw ipcPathError('EINVAL', 'File attachment snapshot failed: source is not a regular bounded file.')
    }

    const buffer = Buffer.alloc(Math.min(1024 * 1024, Math.max(1, before.size)))
    let offset = 0
    while (offset < before.size) {
      const requested = Math.min(buffer.length, before.size - offset)
      const { bytesRead } = await sourceHandle.read(buffer, 0, requested, offset)
      if (bytesRead <= 0) {
        throw ipcPathError('ESTALE', 'File attachment snapshot failed: source changed while being copied.')
      }
      let written = 0
      while (written < bytesRead) {
        const result = await snapshotHandle.write(buffer, written, bytesRead - written, offset + written)
        if (result.bytesWritten <= 0) {
          throw ipcPathError('EIO', 'File attachment snapshot failed: snapshot write was incomplete.')
        }
        written += result.bytesWritten
      }
      offset += bytesRead
    }

    const overflow = Buffer.alloc(1)
    const extra = await sourceHandle.read(overflow, 0, 1, before.size)
    const after = await sourceHandle.stat()
    if (extra.bytesRead !== 0 || fileSnapshotIdentity(after) !== fileSnapshotIdentity(before)) {
      throw ipcPathError('ESTALE', 'File attachment snapshot failed: source changed while being copied.')
    }

    await snapshotHandle.sync()
    const snapshotStat = await snapshotHandle.stat()
    if (!snapshotStat.isFile() || snapshotStat.size !== before.size) {
      throw ipcPathError('EIO', 'File attachment snapshot failed: snapshot size mismatch.')
    }

    return {
      byteSize: snapshotStat.size,
      fileId: `${snapshotStat.dev}:${snapshotStat.ino}`,
      mtimeMs: snapshotStat.mtimeMs,
      path: snapshotPath
    }
  } catch (error) {
    try {
      await fs.promises.unlink(snapshotPath)
    } catch (cleanupError) {
      if (cleanupError?.code !== 'ENOENT') {
        throw ipcPathError('EIO', 'File attachment snapshot failed and its partial snapshot could not be removed.')
      }
    }
    throw error
  } finally {
    await snapshotHandle?.close()
    await sourceHandle?.close()
  }
}

async function releaseFileSnapshotForIpc(snapshotPath, snapshotRoot) {
  const root = await resolveFileSnapshotRoot(snapshotRoot)
  const candidate = path.resolve(String(snapshotPath || ''))
  if (path.dirname(candidate).toLowerCase() !== root.toLowerCase() || !/^[0-9a-f-]{36}\.snapshot$/i.test(path.basename(candidate))) {
    throw ipcPathError('EINVAL', 'File attachment snapshot cleanup failed: invalid snapshot path.')
  }

  let stat
  try {
    stat = await fs.promises.lstat(candidate)
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return true
    }
    throw error
  }
  if (!stat.isFile() || stat.isSymbolicLink()) {
    throw ipcPathError('EINVAL', 'File attachment snapshot cleanup failed: unsafe snapshot file.')
  }
  await fs.promises.unlink(candidate)
  return true
}

export {
  createFileSnapshotForIpc,
  DATA_URL_READ_MAX_BYTES,
  DEFAULT_FETCH_TIMEOUT_MS,
  encryptDesktopSecret,
  FILE_CHUNK_READ_MAX_BYTES,
  normalizeFileChunkReadRange,
  readFileChunkForIpc,
  rejectUnsafePathSyntax,
  releaseFileSnapshotForIpc,
  resolveDirectoryForIpc,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc,
  resolveTimeoutMs,
  sensitiveFileBlockReason,
  TEXT_PREVIEW_SOURCE_MAX_BYTES
}
