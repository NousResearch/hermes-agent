import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const DEFAULT_FETCH_TIMEOUT_MS = 15_000
const DATA_URL_READ_MAX_BYTES = 16 * 1024 * 1024
const TEXT_PREVIEW_SOURCE_MAX_BYTES = 64 * 1024 * 1024
const DEFAULT_MEMORY_CHAR_LIMIT = 2200
const DEFAULT_USER_CHAR_LIMIT = 1375

const SAFE_ENV_SUFFIXES = new Set(['dist', 'example', 'sample', 'template'])
const SENSITIVE_EXTENSIONS = new Set(['.kdbx', '.p12', '.pem', '.pfx'])

const MEMORY_FILE_LIMITS = Object.freeze({
  'MEMORY.md': {
    configKey: 'memory.memory_char_limit',
    defaultLimit: DEFAULT_MEMORY_CHAR_LIMIT,
    limitKey: 'memory_char_limit'
  },
  'USER.md': {
    configKey: 'memory.user_char_limit',
    defaultLimit: DEFAULT_USER_CHAR_LIMIT,
    limitKey: 'user_char_limit'
  }
})

function resolveTimeoutMs(timeoutMs, fallbackMs = DEFAULT_FETCH_TIMEOUT_MS) {
  const fallback =
    Number.isFinite(fallbackMs) && Number(fallbackMs) > 0 ? Math.round(Number(fallbackMs)) : DEFAULT_FETCH_TIMEOUT_MS

  const parsed = Number(timeoutMs)

  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.round(parsed)
  }

  return fallback
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

function pathIsInside(parent, child) {
  const relative = path.relative(path.resolve(parent), path.resolve(child))

  return relative === '' || (relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative))
}

function configuredMemoryLimit(profileHome, limitKey, fallback) {
  let raw = ''

  try {
    raw = fs.readFileSync(path.join(profileHome, 'config.yaml'), 'utf8')
  } catch {
    return fallback
  }

  const lines = raw.split(/\r?\n/)
  let inMemory = false
  let memoryIndent = -1

  for (const line of lines) {
    const trimmed = line.trim()

    if (!trimmed || trimmed.startsWith('#')) {
      continue
    }

    const indent = line.length - line.trimStart().length

    if (/^memory\s*:\s*(?:#.*)?$/.test(trimmed)) {
      inMemory = true
      memoryIndent = indent

      continue
    }

    if (inMemory && indent <= memoryIndent) {
      inMemory = false
    }

    if (!inMemory || indent <= memoryIndent) {
      continue
    }

    const match = trimmed.match(new RegExp(`^${limitKey}\\s*:\\s*([^#]+)`))

    if (!match) {
      continue
    }

    const normalized = match[1]
      .trim()
      .replace(/^['"]|['"]$/g, '')
      .replace(/_/g, '')

    if (!/^-?\d+$/.test(normalized)) {
      return fallback
    }

    return Number.parseInt(normalized, 10)
  }

  return fallback
}

function hermesMemoryFileTarget(filePath, hermesHome) {
  if (!hermesHome) {
    return null
  }

  const resolvedHome = path.resolve(String(hermesHome))
  const resolvedPath = path.resolve(String(filePath || ''))

  if (!pathIsInside(resolvedHome, resolvedPath)) {
    return null
  }

  const parts = path.relative(resolvedHome, resolvedPath).split(path.sep)
  let profileHome = null
  let fileName = null

  if (parts.length === 2 && parts[0] === 'memories') {
    profileHome = resolvedHome
    fileName = parts[1]
  } else if (parts.length === 4 && parts[0] === 'profiles' && parts[2] === 'memories') {
    profileHome = path.join(resolvedHome, 'profiles', parts[1])
    fileName = parts[3]
  }

  const limitInfo = fileName ? MEMORY_FILE_LIMITS[fileName] : null

  return limitInfo && profileHome ? { ...limitInfo, fileName, profileHome } : null
}

function countUnicodeChars(value) {
  return Array.from(String(value ?? '')).length
}

function validateHermesMemoryFileWrite(filePath, content, options: { hermesHome?: string } = {}) {
  const target = hermesMemoryFileTarget(filePath, options.hermesHome)

  if (!target) {
    return null
  }

  const limit = configuredMemoryLimit(target.profileHome, target.limitKey, target.defaultLimit)
  const current = countUnicodeChars(content)

  if (current <= limit) {
    return { current, limit, target }
  }

  const overBy = current - limit

  throw ipcPathError(
    'memory-limit',
    `${target.fileName} is ${current}/${limit} chars, exceeding ${target.configKey}. ` +
      `Reduce by ${overBy} chars or raise ${target.configKey} in config.yaml before saving.`
  )
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

export {
  DATA_URL_READ_MAX_BYTES,
  DEFAULT_FETCH_TIMEOUT_MS,
  encryptDesktopSecret,
  rejectUnsafePathSyntax,
  resolveDirectoryForIpc,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc,
  resolveTimeoutMs,
  sensitiveFileBlockReason,
  TEXT_PREVIEW_SOURCE_MAX_BYTES,
  validateHermesMemoryFileWrite
}
