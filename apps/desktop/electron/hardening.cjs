const fs = require('node:fs')
const path = require('node:path')
const { fileURLToPath } = require('node:url')

const DEFAULT_FETCH_TIMEOUT_MS = 15_000
const DATA_URL_READ_MAX_BYTES = 16 * 1024 * 1024
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

function resolveRequestedFilePath(filePath, baseDir = process.cwd(), purpose = 'File read') {
  const raw = String(filePath || '').trim()

  if (!raw) {
    throw new Error(`${purpose} failed: file path is required.`)
  }

  if (raw.includes('\0')) {
    throw new Error(`${purpose} failed: file path is invalid.`)
  }

  if (/^file:/i.test(raw)) {
    try {
      return fileURLToPath(raw)
    } catch {
      throw new Error(`${purpose} failed: file URL is invalid.`)
    }
  }

  const resolvedBase = path.resolve(String(baseDir || process.cwd()))
  return path.resolve(resolvedBase, raw)
}

async function resolveReadableFileForIpc(filePath, options = {}) {
  const purpose = String(options.purpose || 'File read')
  const resolvedPath = resolveRequestedFilePath(filePath, options.baseDir, purpose)

  if (options.blockSensitive !== false) {
    const blockReason = sensitiveFileBlockReason(resolvedPath)
    if (blockReason) {
      throw new Error(`${purpose} blocked for sensitive file: ${blockReason}`)
    }
  }

  let stat
  try {
    stat = await fs.promises.stat(resolvedPath)
  } catch (error) {
    const code = error && typeof error === 'object' ? error.code : ''
    if (code === 'ENOENT' || code === 'ENOTDIR') {
      throw new Error(`${purpose} failed: file does not exist.`)
    }
    throw new Error(`${purpose} failed: ${error instanceof Error ? error.message : String(error)}`)
  }

  if (stat.isDirectory()) {
    throw new Error(`${purpose} failed: path points to a directory.`)
  }

  if (!stat.isFile()) {
    throw new Error(`${purpose} failed: only regular files can be read.`)
  }

  const maxBytes = Number.isFinite(options.maxBytes) && Number(options.maxBytes) > 0 ? Number(options.maxBytes) : null
  if (maxBytes && stat.size > maxBytes) {
    throw new Error(`${purpose} failed: file is too large (${stat.size} bytes; limit ${maxBytes} bytes).`)
  }

  try {
    await fs.promises.access(resolvedPath, fs.constants.R_OK)
  } catch {
    throw new Error(`${purpose} failed: file is not readable.`)
  }

  return { resolvedPath, stat }
}

module.exports = {
  DATA_URL_READ_MAX_BYTES,
  DEFAULT_FETCH_TIMEOUT_MS,
  TEXT_PREVIEW_SOURCE_MAX_BYTES,
  encryptDesktopSecret,
  resolveReadableFileForIpc,
  resolveTimeoutMs,
  sensitiveFileBlockReason
}
